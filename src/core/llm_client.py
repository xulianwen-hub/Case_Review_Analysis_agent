"""
统一的 LLM 调用接口
- 支持多厂商：DeepSeek / 豆包(Doubao) / 千问(Qwen)
- 支持自动降级：主模型不可用时，自动切换到备选模型
- 通过 .env 中的 LLM_PROVIDER 和 LLM_FALLBACK_CHAIN 配置
"""
import os
import time
import threading
from enum import Enum
from typing import Optional, List, Dict

from dotenv import load_dotenv
from .logger import logger

load_dotenv()

# 各厂商配置映射
PROVIDER_CONFIG = {
    "deepseek": {
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "base_url": os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1"),
        "model": os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
    },
    "doubao": {
        "api_key": os.getenv("DOUBAO_API_KEY"),
        "base_url": os.getenv("DOUBAO_BASE_URL", "https://ark.cn-beijing.volces.com/api/v3"),
        "model": os.getenv("DOUBAO_MODEL", "doubao-lite-32k"),
    },
    "qwen": {
        "api_key": os.getenv("QWEN_API_KEY"),
        "base_url": os.getenv("QWEN_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        "model": os.getenv("QWEN_MODEL", "qwen-turbo"),
    },
}

# 降级链路：主模型挂了按此顺序依次尝试
# 可通过 .env 配置：LLM_FALLBACK_CHAIN=deepseek,doubao,qwen
_DEFAULT_FALLBACK_CHAIN = ["deepseek", "doubao", "qwen"]
try:
    _chain_env = os.getenv("LLM_FALLBACK_CHAIN", "")
    FALLBACK_CHAIN = [p.strip() for p in _chain_env.split(",") if p.strip()] if _chain_env else _DEFAULT_FALLBACK_CHAIN
except Exception:
    FALLBACK_CHAIN = _DEFAULT_FALLBACK_CHAIN

# 各厂商请求超时（秒），可通过 .env 覆盖，如 DEEPSEEK_TIMEOUT=30
_PROVIDER_TIMEOUTS = {
    "deepseek": int(os.getenv("DEEPSEEK_TIMEOUT", "30")),
    "doubao": int(os.getenv("DOUBAO_TIMEOUT", "60")),
    "qwen": int(os.getenv("QWEN_TIMEOUT", "30")),
}

# 断路器配置
_CB_FAILURE_THRESHOLD = int(os.getenv("CB_FAILURE_THRESHOLD", "3"))
_CB_RECOVERY_TIMEOUT = int(os.getenv("CB_RECOVERY_TIMEOUT", "30"))


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    断路器（生产级熔断器）

    状态机：
        CLOSED ── failure_count >= threshold ──► OPEN
        OPEN   ── recovery_timeout 到期 ──► HALF_OPEN
        HALF_OPEN ── 试探成功 ──► CLOSED（重置计数）
        HALF_OPEN ── 试探失败 ──► OPEN（重新计时）

    面试要点：
    - 这是"断路器模式"的完整实现，Netflix Hystrix 的简化版
    - 三个状态 + 线程安全，是分布式系统容错的经典设计
    - 与降级（Fallback）配合使用：断路器打开 → 跳过该厂商 → 尝试下一个
    """

    def __init__(self, name: str, failure_threshold: int = None, recovery_timeout: int = None):
        self.name = name
        self.failure_threshold = failure_threshold or _CB_FAILURE_THRESHOLD
        self.recovery_timeout = recovery_timeout or _CB_RECOVERY_TIMEOUT

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self._last_failure_time: float = 0.0
        self._lock = threading.Lock()

    def allow_request(self) -> bool:
        with self._lock:
            if self.state == CircuitState.CLOSED:
                return True

            if self.state == CircuitState.OPEN:
                if time.time() - self._last_failure_time >= self.recovery_timeout:
                    self.state = CircuitState.HALF_OPEN
                    logger.info(
                        f"[断路器] {self.name}: OPEN → HALF_OPEN（熔断 {self.recovery_timeout}s 到期，进入试探）"
                    )
                    return True
                return False

            if self.state == CircuitState.HALF_OPEN:
                return True

            return False

    def on_success(self):
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                logger.info(f"[断路器] {self.name}: HALF_OPEN → CLOSED（试探成功，恢复服务）")
            self.state = CircuitState.CLOSED
            self.failure_count = 0

    def on_failure(self):
        with self._lock:
            self.failure_count += 1
            self._last_failure_time = time.time()

            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.OPEN
                logger.warning(f"[断路器] {self.name}: HALF_OPEN → OPEN（试探失败，继续熔断）")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.warning(
                    f"[断路器] {self.name}: CLOSED → OPEN"
                    f"（连续失败 {self.failure_count}/{self.failure_threshold} 次，触发熔断）"
                )

    def get_status(self) -> Dict:
        with self._lock:
            return {
                "name": self.name,
                "state": self.state.value,
                "failure_count": self.failure_count,
                "last_failure_time": self._last_failure_time,
            }


class LLMClient:
    """OpenAI 兼容的 LLM 调用客户端，支持多厂商自动切换"""

    def __init__(
        self,
        provider: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.provider = provider or os.getenv("LLM_PROVIDER", "deepseek").lower()

        if self.provider not in PROVIDER_CONFIG:
            raise ValueError(
                f"不支持的 LLM 厂商: '{self.provider}'。"
                f"可选: {list(PROVIDER_CONFIG.keys())}"
            )

        cfg = PROVIDER_CONFIG[self.provider]
        self.api_key = api_key or cfg["api_key"]
        self.base_url = base_url or cfg["base_url"]
        self.model = model or cfg["model"]

        if not self.api_key:
            env_key_map = {
                "deepseek": "DEEPSEEK_API_KEY",
                "doubao": "DOUBAO_API_KEY",
                "qwen": "QWEN_API_KEY",
            }
            env_key = env_key_map.get(self.provider, f"{self.provider.upper()}_API_KEY")
            raise ValueError(
                f"未找到 {env_key}，请在 .env 中设置 "
                f"或通过 LLM_PROVIDER 切换到其他厂商"
            )

        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
        return self._client

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream,
        )

        if stream:
            full_text = []
            for chunk in response:
                if chunk.choices[0].delta.content:
                    full_text.append(chunk.choices[0].delta.content)
            return "".join(full_text)
        else:
            return response.choices[0].message.content

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", 0.3),
            max_tokens=kwargs.get("max_tokens", 4096),
            stream=True,
        )


class LLMClientWithFallback:
    """
    生产级多厂商 LLM 客户端（断路器 + 超时控制 + 健康检查）

    完整架构：
    ┌──────────────────────────────────────────────────┐
    │                 用户请求                          │
    └─────────────────────┬────────────────────────────┘
                          ▼
    ┌──────────────────────────────────────────────────┐
    │  LLMClientWithFallback                           │
    │                                                  │
    │  for provider in fallback_chain:                 │
    │    ┌──────────────────────────────────────┐      │
    │    │ 1. CircuitBreaker.allow_request()?   │      │
    │    │    ├─ OPEN → 跳过，换下一个厂商        │      │
    │    │    └─ CLOSED/HALF_OPEN → 继续         │      │
    │    │                                       │      │
    │    │ 2. LLMClient.chat(timeout=N)          │      │
    │    │    ├─ 成功 → breaker.on_success()     │      │
    │    │    └─ 失败 → breaker.on_failure()     │      │
    │    └──────────────────────────────────────┘      │
    └──────────────────────────────────────────────────┘
    """

    def __init__(self, fallback_chain: Optional[List[str]] = None):
        self.fallback_chain = fallback_chain or FALLBACK_CHAIN
        self._clients: Dict[str, Optional[LLMClient]] = {}
        self._breakers: Dict[str, CircuitBreaker] = {}
        self._last_used_provider: Optional[str] = None
        self._health_check_thread: Optional[threading.Thread] = None
        self._health_check_stop = threading.Event()

        for provider in self.fallback_chain:
            self._breakers[provider] = CircuitBreaker(name=provider)

    def _get_client(self, provider: str) -> Optional[LLMClient]:
        if provider not in self._clients:
            try:
                self._clients[provider] = LLMClient(provider=provider)
            except ValueError as e:
                logger.warning(f"[LLM] 厂商 '{provider}' 初始化失败（缺少 API Key 或配置错误）: {e}")
                self._clients[provider] = None
        return self._clients[provider]

    def _get_timeout(self, provider: str) -> int:
        return _PROVIDER_TIMEOUTS.get(provider, 30)

    @property
    def provider(self) -> str:
        return self._last_used_provider or self.fallback_chain[0]

    @property
    def model(self) -> str:
        client = self._get_client(self.provider)
        return client.model if client else "unknown"

    @property
    def base_url(self) -> str:
        client = self._get_client(self.provider)
        return client.base_url if client else "unknown"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> str:
        errors = []
        for provider in self.fallback_chain:
            breaker = self._breakers[provider]

            if not breaker.allow_request():
                errors.append(f"[{provider}] 断路器已熔断，跳过")
                continue

            client = self._get_client(provider)
            if client is None:
                errors.append(f"[{provider}] 未配置，跳过")
                continue

            timeout = self._get_timeout(provider)
            try:
                result = client.chat(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=stream,
                )
                breaker.on_success()
                self._last_used_provider = provider
                return result

            except Exception as e:
                breaker.on_failure()
                error_type = type(e).__name__
                error_msg = f"[{provider}] {error_type}: {str(e)[:200]}"
                errors.append(error_msg)
                logger.warning(f"[LLM 降级] {error_msg}（超时={timeout}s）")

        raise RuntimeError(
            f"所有 LLM 厂商均调用失败，降级链路: {' -> '.join(self.fallback_chain)}\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    def chat_stream(self, messages: List[Dict[str, str]], **kwargs):
        errors = []
        for provider in self.fallback_chain:
            breaker = self._breakers[provider]

            if not breaker.allow_request():
                errors.append(f"[{provider}] 断路器已熔断，跳过")
                continue

            client = self._get_client(provider)
            if client is None:
                errors.append(f"[{provider}] 未配置，跳过")
                continue

            try:
                stream = client.chat_stream(messages, **kwargs)
                self._last_used_provider = provider
                yield from stream
                breaker.on_success()
                return

            except Exception as e:
                breaker.on_failure()
                error_type = type(e).__name__
                error_msg = f"[{provider}] {error_type}: {str(e)[:200]}"
                errors.append(error_msg)
                logger.warning(f"[LLM 降级(stream)] {error_msg}")

        raise RuntimeError(
            f"所有 LLM 厂商均调用失败（流式），降级链路: {' -> '.join(self.fallback_chain)}\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    def get_status(self) -> Dict:
        return {
            "fallback_chain": self.fallback_chain,
            "last_used_provider": self._last_used_provider,
            "breakers": {p: b.get_status() for p, b in self._breakers.items()},
        }

    def health_check(self) -> Dict[str, bool]:
        results = {}
        for provider in self.fallback_chain:
            client = self._get_client(provider)
            if client is None:
                results[provider] = False
                continue

            breaker = self._breakers[provider]
            if not breaker.allow_request():
                results[provider] = False
                continue

            try:
                client.chat(
                    messages=[{"role": "user", "content": "ping"}],
                    temperature=0.0,
                    max_tokens=1,
                    stream=False,
                )
                breaker.on_success()
                results[provider] = True
            except Exception as e:
                breaker.on_failure()
                logger.warning(f"[健康检查] {provider} 不可用: {type(e).__name__}")
                results[provider] = False

        return results

    def start_health_check(self, interval: int = 300):
        if self._health_check_thread is not None:
            return

        self._health_check_stop.clear()

        def _run():
            logger.info(f"[健康检查] 后台线程已启动，间隔 {interval}s")
            while not self._health_check_stop.wait(interval):
                try:
                    results = self.health_check()
                    available = [p for p, ok in results.items() if ok]
                    unavailable = [p for p, ok in results.items() if not ok]
                    if unavailable:
                        logger.info(f"[健康检查] 可用: {available}, 不可用: {unavailable}")
                except Exception as e:
                    logger.error(f"[健康检查] 异常: {e}")

        self._health_check_thread = threading.Thread(target=_run, daemon=True)
        self._health_check_thread.start()

    def stop_health_check(self):
        self._health_check_stop.set()
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5)
            self._health_check_thread = None
            logger.info("[健康检查] 后台线程已停止")


# 全局单例
_llm_client: Optional[LLMClientWithFallback] = None


def get_llm_client(provider: Optional[str] = None) -> LLMClientWithFallback:
    global _llm_client
    if _llm_client is None or provider is not None:
        if provider:
            _llm_client = LLMClientWithFallback(fallback_chain=[provider])
        else:
            _llm_client = LLMClientWithFallback()
    return _llm_client