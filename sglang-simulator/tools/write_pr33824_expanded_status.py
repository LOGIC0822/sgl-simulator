#!/usr/bin/env python3
"""Write progress from artifacts and actual process state; optionally watch."""
from __future__ import annotations
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path


def complete(path: Path) -> bool:
    return path.is_file() and path.read_text().strip() == "complete"


def collected_counts(path: Path) -> tuple[int, int]:
    """Count validated cases, including completed cases in an ongoing group."""
    requests = batches = 0
    for status in path.glob("repeat-*/cases/*/status.txt"):
        if status.read_text().strip() != "ok":
            continue
        manifest = json.loads((status.parent / "TP0.manifest.json").read_text())
        requests += int(manifest["request_count"])
        batches += int(manifest["schedule_batch_count"])
    return requests, batches


def scan_processes(root: Path, proc_root: Path = Path('/proc')) -> dict[str, list[str]]:
    """Identify live actors by exact script/root arguments, not shell text."""
    sim = Path(__file__).resolve().parents[1]
    expected = {
        'comprehensive': sim / 'scripts/run_pr33824_comprehensive_predictor_matrix.sh',
        'native': sim / 'scripts/run_pr33824_data_tools_same_distribution_matrix.sh',
        'resume': sim / 'scripts/resume_pr33824_after_orphan.py',
    }
    actors = {name: [] for name in expected}
    for proc in proc_root.glob('[0-9]*'):
        try:
            stat = (proc / 'stat').read_text()
            fields = stat[stat.rfind(')') + 2:].split()
            if fields[0] == 'Z':
                continue
            argv = (proc / 'cmdline').read_bytes().decode().rstrip('\0').split('\0')
            index = argv.index('--results-root')
            if Path(argv[index + 1]).resolve() != root.resolve():
                continue
            for role, script in expected.items():
                if any(arg.startswith('/') and Path(arg).resolve() == script for arg in argv):
                    actors[role].append(proc.name)
        except (OSError, UnicodeDecodeError, ValueError, IndexError):
            continue
    return {role: sorted(pids, key=int) for role, pids in actors.items()}


def process_state(finished: bool, actors: dict[str, list[str]]) -> str:
    if finished:
        return '全部完成'
    if actors['comprehensive']:
        return '运行中'
    if actors['native']:
        return ('原生采集运行中；后续阶段等待续跑' if actors['resume']
                else '原生采集运行中；未检测到总调度进程')
    if actors['resume']:
        return '续跑监控仍存活；未检测到实验进程，请检查续跑状态'
    return '当前没有实验活进程，请检查日志'


def write_status(root: Path) -> bool:
    metadata = dict(line.split("=", 1) for line in (root / "run_metadata.txt").read_text().splitlines() if "=" in line)
    models = metadata["models"].split(",")
    with (root / "model_manifest.tsv").open() as source:
        planned = {row["model"]: row["mem_fraction_static"] for row in csv.DictReader(source, delimiter="\t")}
    actors = scan_processes(root)
    finished = complete(root / "FINAL_REPORT_COMPLETE")
    rows, completed_models = [], 0
    waits = []
    for model in models:
        model_root = root / "models" / model
        train_done = sum(complete(path) for path in (model_root / "calibration/real").glob("repeat-*/BASELINE_COLLECTION_COMPLETE"))
        test_done = sum(complete(path) for path in (model_root / "holdout/real").glob("repeat-*/BASELINE_COLLECTION_COMPLETE"))
        ml = complete(model_root / "REAL_ML_COMPLETE")
        aic = complete(model_root / "AIC_PREDICTOR_COMPLETE")
        completed_models += int(ml and aic)
        settings_path = model_root / "model_settings.txt"
        settings = dict(line.split("=", 1) for line in settings_path.read_text().splitlines() if "=" in line) if settings_path.exists() else {}
        train_requests, train_batches = collected_counts(model_root / "calibration/real")
        test_requests, test_batches = collected_counts(model_root / "holdout/real")
        fraction = settings.get("mem_fraction_static") if train_requests else planned.get(model, "待记录")
        rows.append(f"| {model} | {train_done}/5 | {train_requests} / {train_batches} | {test_done}/5 | {test_requests} / {test_batches} | {'完成' if ml else '未完成'} | {'完成' if aic else '未完成'} | {fraction} |")
        for log in model_root.glob("**/hardware/gpu_capacity.jsonl"):
            lines = log.read_text().splitlines()
            if lines:
                value = json.loads(lines[-1])
                if value["state"] == "waiting":
                    waits.append(f"- {model}：最近检查 {value['timestamp']}，空闲 {value['free_mib']} MiB，门槛 {value['required_mib']} MiB。")
    text = ["# 全模型扩充实验状态", "", f"更新时间：{datetime.now().astimezone().isoformat(timespec='seconds')}",
            f"状态：{process_state(finished, actors)}；ML/AIC 均完成模型 {completed_models}/{len(models)}。",
            f"总调度进程 PID：{', '.join(actors['comprehensive']) or '无'}。",
            f"原生采集矩阵 PID：{', '.join(actors['native']) or '无'}。",
            f"续跑监控 PID：{', '.join(actors['resume']) or '无'}。", "",
            f"每模型计划：训练 {metadata['training_requests_total']} 请求；独立测试 {metadata['heldout_requests_total']} 请求。五组训练每组包含三类负载，五组测试每组包含五类负载。", "",
            "| 模型 | 训练组 | 训练请求 / 时延条数 | 测试组 | 测试请求 / 时延条数 | ML采集/训练/评分/回放 | AIC采集/评分/回放 | 显存比例（未采集时为计划值） |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |", *rows, "", "## 显存等待", "", *(waits or ["暂无等待记录。"]), "",
            "runner.log、status.tsv 和 comprehensive_status.tsv 保存运行记录。组完成须有完整 completion marker；请求/时延数只统计已通过计数检查的用例，包含正在执行组内已经完成的用例。目录存在不代表完成，预检和 attempts 不计入。报告仅在全部通过后生成。", ""]
    (root / "status.md").write_text("\n".join(text), encoding="utf-8")
    return finished


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--watch-interval", type=int, default=0)
    args = parser.parse_args()
    if args.watch_interval < 0:
        parser.error("watch interval must be nonnegative")
    while True:
        finished = write_status(args.results_root)
        if finished or not args.watch_interval:
            return
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
