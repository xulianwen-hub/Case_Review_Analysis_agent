"""采集后台任务状态到 _monitor_report.json（稳定读取）"""
import sys, os, json, glob, subprocess

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
report = {}

# 1. 进程状态
r = subprocess.run(['tasklist', '/FI', 'IMAGENAME eq python.exe'],
                   capture_output=True, text=True, errors='replace')
report['python_running'] = 'python.exe' in r.stdout
report['tasklist_tail'] = r.stdout[-200:] if 'python.exe' in r.stdout else '无 python 进程'

# 2. progress 文件
pf = 'data/processed/vectors/cases_progress.json'
report['progress_exists'] = os.path.exists(pf)
if os.path.exists(pf):
    with open(pf, 'r', encoding='utf-8') as f:
        ids = json.load(f)
    report['progress_count'] = len(ids)
    report['progress_pct'] = round(len(ids)/5487*100, 1)

# 3. 后台日志摘要
logs = glob.glob(r'C:\WINDOWS\TEMP\cline\background-*.log')
logs.sort(key=os.path.getmtime, reverse=True)
report['log_files'] = len(logs)
report['logs'] = []
for log in logs[:4]:
    try:
        size = os.path.getsize(log)
        with open(log, 'r', encoding='utf-8', errors='replace') as f:
            content = f.read()
        # 提取关键进度行
        key_lines = [l for l in content.splitlines()
                     if any(k in l for k in ['[进度]', '[第', '[Milvus]', '[完成]',
                                             '评估汇总', 'Recall@', 'MRR', 'NDCG', '进度'])]
        report['logs'].append({
            'name': os.path.basename(log),
            'size': size,
            'key_lines': key_lines[-15:],
            'tail': content[-300:],
        })
    except Exception as e:
        report['logs'].append({'name': os.path.basename(log), 'error': str(e)})

with open('_monitor_report.json', 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)
print('采集完成')