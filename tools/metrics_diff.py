# -*- coding: utf-8 -*-
"""
tools/metrics_diff.py —— batch_report.json 指标对比工具（重构回归用，零依赖）
========================================================
用途：
    重构前后各跑一次 run_batch（同 --count --seed），对比两份
    data/batch_report.json 的算法指标是否逐位一致。自动剔除墙钟噪声：
    takt_ms 整段与 gates[].actual（其中嵌着节拍实测字符串）。

用法：
    python tools/metrics_diff.py BASELINE.json CURRENT.json
    例：
      python run_batch.py --count 100 --seed 42        # 重构前
      cp data/batch_report.json %TEMP%/baseline.json   # 留黄金样本
      # ……重构……
      python run_batch.py --count 100 --seed 42        # 重构后
      python tools/metrics_diff.py %TEMP%/baseline.json data/batch_report.json

退出码：0 = METRICS-IDENTICAL；1 = 存在差异（逐条列出差异路径）。

注意：对比必须同 count/defect_rate/seed（meta 里可肉眼复核）；
      JSON 解析显式 utf-8——不要用 PowerShell 5.1 的 ConvertFrom-Json
      对中文 UTF-8 做这件事（会静默解析失败造成"空比空"假一致）。
"""
import argparse
import json
import sys

VOLATILE_KEYS = {"takt_ms",          # 整段剔除：纯墙钟计时
                 "generated_at",     # 报告生成时刻（每次必不同）
                 "wall_time_s"}      # 总耗时（墙钟）
GATE_VOLATILE_KEYS = {"actual"}      # 门槛行里嵌着节拍实测字符串


def normalize(obj, path="$"):
    """递归返回剔除易失字段的规范化副本"""
    if isinstance(obj, dict):
        skip = GATE_VOLATILE_KEYS if path.endswith(".gates[]") else VOLATILE_KEYS
        return {k: normalize(v, f"{path}.{k}")
                for k, v in obj.items() if k not in skip}
    if isinstance(obj, list):
        return [normalize(v, f"{path}[]") for v in obj]
    return obj


def deep_diff(a, b, path="$", out=None):
    """收集 a/b 规范化后的全部差异路径（叶子级）"""
    out = [] if out is None else out
    if type(a) is not type(b) and not (
            isinstance(a, (int, float)) and isinstance(b, (int, float))):
        out.append(f"{path}: 类型 {type(a).__name__} != {type(b).__name__}")
        return out
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}.{k}: 仅存在于 CURRENT ({b[k]!r})")
            elif k not in b:
                out.append(f"{path}.{k}: 仅存在于 BASELINE ({a[k]!r})")
            else:
                deep_diff(a[k], b[k], f"{path}.{k}", out)
    elif isinstance(a, list):
        if len(a) != len(b):
            out.append(f"{path}: 长度 {len(a)} != {len(b)}")
        for i, (x, y) in enumerate(zip(a, b)):
            deep_diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(f"{path}: {a!r} != {b!r}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="batch_report 指标对比（剔除节拍噪声）")
    ap.add_argument("baseline")
    ap.add_argument("current")
    args = ap.parse_args()

    base = json.loads(open(args.baseline, encoding="utf-8").read())
    curr = json.loads(open(args.current, encoding="utf-8").read())

    nb, nc = normalize(base), normalize(curr)
    diffs = deep_diff(nb, nc)
    if not diffs:
        print("METRICS-IDENTICAL")
        return 0
    print(f"METRICS-DIFF ({len(diffs)} 处):")
    for d in diffs[:40]:
        print(" ", d)
    if len(diffs) > 40:
        print(f"  … 其余 {len(diffs) - 40} 条略")
    return 1


if __name__ == "__main__":
    sys.exit(main())
