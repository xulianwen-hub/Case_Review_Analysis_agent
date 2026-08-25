"""
统一日志配置（基于 loguru）
- 彩色控制台输出（INFO 级别）
- 文件滚动存储（DEBUG 级别，按天 + 按大小切分）
- 自动创建日志目录
"""
import sys
from pathlib import Path
from loguru import logger

# 移除默认 handler
logger.remove()

# ============================================================
# 控制台输出：彩色、INFO 级别
# ============================================================
logger.add(
    sys.stderr,
    format=(
        "<green>{time:HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    ),
    level="INFO",
    colorize=True,
)

# ============================================================
# 文件输出：按天 + 按 10MB 切分，保留 7 天
# ============================================================
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "agent_{time:YYYY-MM-DD}.log",
    format=(
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} | "
        "{message}"
    ),
    level="DEBUG",
    rotation="10 MB",
    retention="7 days",
    encoding="utf-8",
)

# 导出 logger 供其他模块使用
__all__ = ["logger"]