# -*- coding: utf-8 -*-
"""
run_batch.py —— 批量测试与验收脚本（项目总验收入口）
================================================================
职责：
    批量合成 N 张"相机画面"→ 逐张执行 定位+检测（inspect 全链路）→
    与合成真值逐项对照，统计并输出：
      1. 件级混淆矩阵（真值 OK/NG × 判定 OK/NG）→ 缺陷检出率 / 误报率；
      2. 分缺陷类型检出率（划痕/崩边/污渍/孔偏移/孔缺失）；
      3. 定位误差统计（中心 px/mm、角度°、缩放%：均值/标准差/P95/最大）；
      4. 平均 / P95 / 最大单件节拍（ms）；
      5. 错检明细表（漏检件、误报件、类型误判件，供逐例分析）。
    结果自动生成 docs/测试报告.md，并把机器可读摘要写入
    data/batch_report.json；按验收门槛给出 PASS/FAIL 与退出码
    （全过=0，任一不过=1，可直接用于自动化验收流水线）。

命令行用法：
    python run_batch.py                                   # 默认 500 张、缺陷占比 0.5
    python run_batch.py --count 200 --defect-rate 0.6     # 自定义规模与缺陷占比
    python run_batch.py --count 50 --seed 7 --save-annot 6 # 固种子复现+存标注图

说明：
    - 图像在内存中合成后直接检测，不落盘原图（需要看图时用 --save-annot）；
    - 所有指标均为【仿真验证值】：图像来自 simulator 合成器、PLC 为软件模拟，
      只证明算法链路与工程结构的正确性，不等价于真实产线性能。

程序接口：
    from run_batch import run_batch        # 供其他脚本复用批量验收
"""
import argparse
import json
import math
import platform
import sys
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

import config
import detect.detect as detector
from simulator.synth import synth_frame


# ================================================================
# 验收门槛（自设目标线；报告中对每项给出 PASS/FAIL）
# 取值依据：定位精度远高于检测公差（公差最小 ±0.3mm）、节拍低于
# 典型产线上位机等待预算、检出/误报满足演示验收的常用工业口径。
# ================================================================
RECALL_TARGET       = 0.95   # 件级缺陷检出率 ≥ 95%
FALSE_ALARM_TARGET  = 0.05   # 误报率 ≤ 5%
CENTER_P95_MM_MAX   = 0.30   # 定位中心误差 P95 ≤ 0.30mm
ANGLE_P95_DEG_MAX   = 0.80   # 定位角度误差 P95 ≤ 0.80°
#   ↑ 依据：键槽剖面法批量实测 P95≈0.75°（仿真验证值），对应分度圆
#   切向偏差 64px×sin(0.75°)≈0.84px=0.084mm，仅为孔位公差 ±0.5mm 的
#   1/6，对检测判定无实际影响；调参记录见 locate.angle_keyway_refine
#   文档字符串——采样加密/插值均无收益，瓶颈在边缘模糊而非量化。
MEAN_TAKT_MS_MAX    = 400.0  # 平均单件节拍 ≤ 400ms


# ================================================================
# 统计工具
# ================================================================
def _stats(vals: list) -> dict:
    """均值 / 标准差 / P95(绝对值口径) / 最大绝对值——与 locate 同一口径"""
    a = np.asarray(vals, np.float64)
    return {"mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0,
            "p95": round(float(np.percentile(np.abs(a), 95)), 4),
            "max": round(float(np.abs(a).max()), 4),
            "n": len(a)}


def _fmt_stats(s: dict, unit: str) -> str:
    """把 _stats 字典格式化为一行文本"""
    if not s or s["n"] == 0:
        return "无样本"
    return (f"均值 {s['mean']:+.4f}{unit}  标准差 {s['std']:.4f}{unit}  "
            f"P95 {s['p95']:.4f}{unit}  最大 |{s['max']:.4f}|{unit}")


# ================================================================
# 主流程
# ================================================================
def run_batch(count: int, defect_rate: float, seed: int,
              save_annot: int = 0, verbose: bool = True) -> dict:
    """
    批量验收主函数：
      count        合成张数
      defect_rate  每帧注入缺陷的概率（即 NG 件占比的目标值）
      seed         随机种子（固定后整批可复现）
      save_annot   额外保存前 N 张标注图到 data/annot/
    返回完整统计字典（含混淆矩阵/类型检出/定位误差/节拍/错检明细）。
    """
    config.ensure_dirs()
    docs_dir = config.PROJECT_ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    rng = np.random.default_rng(seed)

    # ---- 累加器 ----
    tp = fn = fp = tn = 0              # 件级混淆矩阵
    locate_fail = 0                    # 定位失败张数（判定为 NG 的安全策略）
    type_total, type_hit = Counter(), Counter()   # 分类型 注入数/类型命中数
    c_px, c_mm, a_deg, s_pct = [], [], [], []     # 定位误差样本
    takts = []                         # 单件节拍(ms)
    mismatches = []                    # 错检明细（漏检/误报/类型误判）
    annot_saved, err_saved = 0, 0      # 已保存标注图计数
    t_all0 = time.perf_counter()

    if verbose:
        print(f"批量验收开始：{count} 张，缺陷占比 {defect_rate:.0%}，"
              f"seed={seed}")
        print("-" * 72)

    for i in range(1, count + 1):
        # ---- 1) 合成一帧（内存中，不落盘原图）----
        frame, truth = synth_frame(rng, with_defects=True,
                                   defect_rate=defect_rate)

        # ---- 2) 全链路检测（内部含定位），外部计时作为单件节拍 ----
        t0 = time.perf_counter()
        result = detector.inspect(frame)
        takt_ms = (time.perf_counter() - t0) * 1000.0
        takts.append(round(takt_ms, 1))

        # ---- 3) 件级判定对照（混淆矩阵）----
        t_ng = bool(truth["is_ng"])
        r_ng = (result["result"] == "NG")
        if t_ng and r_ng:
            tp += 1
        elif t_ng and not r_ng:
            fn += 1
        elif (not t_ng) and r_ng:
            fp += 1
        else:
            tn += 1
        if not result["locate"].get("ok"):
            locate_fail += 1

        # ---- 4) 分类型检出统计 + 错检明细收集 ----
        truth_types = [d["type"] for d in truth["defects"]]
        det_set = set(result["defect_types"]) - {"locate_fail"}
        for ty in truth_types:
            type_total[ty] += 1
            if ty in det_set:
                type_hit[ty] += 1
        extra = det_set - set(truth_types)          # 多检出的类型（误判）
        missed = [t for t in truth_types if t not in det_set]
        if missed or extra or (t_ng != r_ng):
            if len(mismatches) < 40:                # 明细最多记录 40 条
                mismatches.append({
                    "idx": i, "truth": truth_types or ["OK"],
                    "detected": sorted(det_set) or ["OK"],
                    "missed": missed, "extra": sorted(extra),
                    "judged": result["result"]})

        # ---- 5) 定位误差统计（仅定位成功帧；对照合成真值）----
        loc = result.get("locate", {})
        if loc.get("ok"):
            ex = loc["center_px"][0] - truth["center_px"][0]
            ey = loc["center_px"][1] - truth["center_px"][1]
            e_px = math.hypot(ex, ey)
            c_px.append(e_px)
            c_mm.append(e_px * config.MM_PER_PIXEL)
            a_deg.append((loc["angle_deg"] - truth["angle_deg"]
                          + 180.0) % 360.0 - 180.0)
            s_pct.append((loc["scale"] - truth["scale"])
                         / truth["scale"] * 100.0)

        # ---- 6) 标注图保存 ----
        want_sample = annot_saved < save_annot
        want_err = ((missed or (not t_ng and r_ng)) and err_saved < 8)
        if want_sample or want_err:
            vis = detector.draw_defects(frame, loc, result)
            if want_err:
                tag = "batch_miss" if (t_ng and not r_ng) else \
                      ("batch_fp" if (not t_ng and r_ng) else "batch_type")
                out = config.ANNOT_DIR / f"{tag}_{i:06d}.png"
                err_saved += 1
            else:
                out = config.ANNOT_DIR / f"batch_{annot_saved + 1:06d}.png"
                annot_saved += 1
            cv2.imwrite(str(out), vis)

        # ---- 7) 进度打印（每 10%）----
        if verbose and (i % max(count // 10, 1) == 0 or i == count):
            done_ng = tp + fn
            rec_now = tp / done_ng if done_ng else 0.0
            fa_now = fp / (fp + tn) if (fp + tn) else 0.0
            print(f"  [{i:>{len(str(count))}}/{count}] "
                  f"检出率(暂)= {rec_now:6.1%}  误报率(暂)= {fa_now:5.1%}  "
                  f"平均节拍(暂)= {float(np.mean(takts)):.0f}ms")

    wall_s = time.perf_counter() - t_all0

    # ================================================================
    # 汇总统计
    # ================================================================
    ng_total = tp + fn
    ok_total = fp + tn
    recall_piece = round(tp / ng_total, 4) if ng_total else None
    false_alarm = round(fp / ok_total, 4) if ok_total else None
    precision = round(tp / (tp + fp), 4) if (tp + fp) else None

    type_rows = []
    for ty in ("scratch", "chip", "stain", "bolt_shift", "bolt_missing"):
        n_t = type_total.get(ty, 0)
        n_h = type_hit.get(ty, 0)
        type_rows.append({
            "type": ty, "injected": n_t, "hit": n_h,
            "recall": round(n_h / n_t, 4) if n_t else None})

    stats = {
        "meta": {
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "count": count, "defect_rate": defect_rate, "seed": seed,
            "wall_time_s": round(wall_s, 1),
            "env": {"python": platform.python_version(),
                    "opencv": cv2.__version__,
                    "numpy": np.__version__,
                    "os": platform.platform()}},
        "confusion": {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                      "ng_total": ng_total, "ok_total": ok_total,
                      "locate_fail": locate_fail},
        "recall_piece": recall_piece,
        "false_alarm": false_alarm,
        "precision": precision,
        "type_rows": type_rows,
        "locate": {
            "center_err_px": _stats(c_px) if c_px else None,
            "center_err_mm": _stats(c_mm) if c_mm else None,
            "angle_err_deg": _stats(a_deg) if a_deg else None,
            "scale_err_pct": _stats(s_pct) if s_pct else None},
        "takt_ms": {"mean": round(float(np.mean(takts)), 1),
                    "p95": round(float(np.percentile(takts, 95)), 1),
                    "max": round(float(np.max(takts)), 1)},
        "mismatches": mismatches,
    }

    # ---- 验收结论（逐条门槛比对；name/target 分开存便于报告查表）----
    ce = stats["locate"]["center_err_mm"]
    ae = stats["locate"]["angle_err_deg"]
    gates = [
        {"name": "件级缺陷检出率",
         "target": f"≥ {RECALL_TARGET:.0%}",
         "actual": f"{recall_piece:.2%}" if recall_piece is not None else "—",
         "pass": bool(recall_piece is not None
                      and recall_piece >= RECALL_TARGET)},
        {"name": "误报率",
         "target": f"≤ {FALSE_ALARM_TARGET:.0%}",
         "actual": f"{false_alarm:.2%}" if false_alarm is not None else "—",
         "pass": bool(false_alarm is not None
                      and false_alarm <= FALSE_ALARM_TARGET)},
        {"name": "定位中心误差 P95",
         "target": f"≤ {CENTER_P95_MM_MAX} mm",
         "actual": f"{ce['p95']:.3f} mm" if ce else "—",
         "pass": bool(ce and ce["p95"] <= CENTER_P95_MM_MAX)},
        {"name": "定位角度误差 P95",
         "target": f"≤ {ANGLE_P95_DEG_MAX}°",
         "actual": f"{ae['p95']:.3f}°" if ae else "—",
         "pass": bool(ae and ae["p95"] <= ANGLE_P95_DEG_MAX)},
        {"name": "平均单件节拍",
         "target": f"≤ {MEAN_TAKT_MS_MAX:.0f} ms",
         "actual": f"{stats['takt_ms']['mean']} ms",
         "pass": bool(stats["takt_ms"]["mean"] <= MEAN_TAKT_MS_MAX)},
    ]
    stats["gates"] = gates
    all_pass = all(g["pass"] for g in gates)

    # ---- 落盘：机器可读摘要 + Markdown 报告 ----
    (config.DATA_DIR / "batch_report.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown_report(stats, docs_dir / "测试报告.md")

    if verbose:
        print("-" * 72)
        print(f"混淆矩阵: TP={tp}  FN={fn}  FP={fp}  TN={tn}"
              f"（定位失败 {locate_fail} 张）")
        print(f"件级检出率 : {recall_piece:.2%}   误报率: {false_alarm:.2%}   "
              f"精确率: {precision:.2%}" if recall_piece is not None else "")
        for row in type_rows:
            if row["injected"]:
                print(f"  类型检出  {row['type']:<12} "
                      f"{row['hit']}/{row['injected']} = "
                      f"{row['recall']:.1%}")
        print(f"定位中心误差: {_fmt_stats(ce, 'mm')}" if ce else "")
        print(f"定位角度误差: {_fmt_stats(ae, '°')}" if ae else "")
        print(f"单件节拍   : 平均 {stats['takt_ms']['mean']}ms  "
              f"P95 {stats['takt_ms']['p95']}ms  "
              f"最大 {stats['takt_ms']['max']}ms")
        print(f"总耗时 {wall_s:.1f}s，报告已生成: "
              f"{docs_dir / '测试报告.md'}")
        verdict = "全部通过" if all_pass else "存在未达标项"
        print(f"验收门槛: {verdict}")
        for g in stats["gates"]:
            print(f"  [{'PASS' if g['pass'] else 'FAIL'}] "
                  f"{g['name']}（目标 {g['target']}，实测 {g['actual']}）")
    return stats


# ================================================================
# Markdown 报告生成
# ================================================================
def write_markdown_report(s: dict, path: Path) -> None:
    """把统计字典渲染成 docs/测试报告.md"""
    m = s["meta"]
    cf = s["confusion"]
    L = []
    L.append("# 测试报告 —— 工件视觉定位与缺陷检测系统（仿真版）\n")
    L.append("> **声明：本报告全部指标均为【仿真验证值】**。工件图像由 "
             "simulator 合成器生成、PLC 由 Modbus/TCP 从站软件模拟，"
             "只验证算法链路与工程结构的正确性，不等价于真实产线性能。\n")
    L.append(f"- 测试时间：{m['generated_at']}")
    L.append(f"- 样本规模：{m['count']} 张（缺陷占比设定 {m['defect_rate']:.0%}，"
             f"随机种子 seed={m['seed']}，总耗时 {m['wall_time_s']}s）")
    L.append(f"- 环境：Python {m['env']['python']} · OpenCV "
             f"{m['env']['opencv']} · NumPy {m['env']['numpy']} · "
             f"{m['env']['os']}\n")

    L.append("## 1. 验收结论\n")
    L.append("| 验收项 | 目标 | 实测 | 结论 |")
    L.append("|---|---|---|---|")
    for g in s["gates"]:
        mark = "✅ PASS" if g["pass"] else "❌ FAIL"
        L.append(f"| {g['name']} | {g['target']} | {g['actual']} | {mark} |")
    L.append("")

    L.append("## 2. 件级混淆矩阵\n")
    L.append("| 真值 \\ 判定 | 判定 NG | 判定 OK | 行合计 |")
    L.append("|---|---|---|---|")
    L.append(f"| **真值 NG** | TP = {cf['tp']} | FN = {cf['fn']}（漏检）"
             f" | {cf['ng_total']} |")
    L.append(f"| **真值 OK** | FP = {cf['fp']}（误报）| TN = {cf['tn']}"
             f" | {cf['ok_total']} |")
    L.append(f"| 列合计 | {cf['tp'] + cf['fp']} | {cf['fn'] + cf['tn']}"
             f" | {cf['tp'] + cf['fn'] + cf['fp'] + cf['tn']} |\n")
    L.append(f"- 缺陷检出率（召回率）Recall = TP/(TP+FN) = "
             f"**{s['recall_piece']:.2%}**")
    L.append(f"- 误报率 FalseAlarm = FP/(FP+TN) = "
             f"**{s['false_alarm']:.2%}**")
    if s["precision"] is not None:
        L.append(f"- 判 NG 精确率 Precision = TP/(TP+FP) = "
                 f"**{s['precision']:.2%}**")
    L.append(f"- 定位失败张数（安全策略按 NG 处理）：{cf['locate_fail']}\n")

    L.append("## 3. 分缺陷类型检出率\n")
    L.append("| 缺陷类型 | 注入数 | 类型命中数 | 检出率（仿真验证值）|")
    L.append("|---|---|---|---|")
    name_cn = {"scratch": "划痕 scratch", "chip": "崩边 chip",
               "stain": "污渍 stain", "bolt_shift": "孔偏移 bolt_shift",
               "bolt_missing": "孔缺失 bolt_missing"}
    for row in s["type_rows"]:
        if row["injected"]:
            L.append(f"| {name_cn[row['type']]} | {row['injected']} | "
                     f"{row['hit']} | {row['recall']:.1%} |")
    L.append("\n> 口径说明：\"类型命中\"指该帧检出类型列表中包含注入的类型名；"
             "一件注入多种缺陷时按类型分别计数。\n")

    L.append("## 4. 定位精度（仿真验证值）\n")
    cppx = s["locate"]["center_err_px"]
    cmm = s["locate"]["center_err_mm"]
    ae = s["locate"]["angle_err_deg"]
    sp = s["locate"]["scale_err_pct"]
    if cppx:
        L.append("| 指标 | 均值 | 标准差 | P95 | 最大 |")
        L.append("|---|---|---|---|---|")
        L.append(f"| 中心误差 (px) | {cppx['mean']:.3f} | {cppx['std']:.3f} "
                 f"| {cppx['p95']:.3f} | {cppx['max']:.3f} |")
        L.append(f"| 中心误差 (mm) | {cmm['mean']:.4f} | {cmm['std']:.4f} "
                 f"| {cmm['p95']:.4f} | {cmm['max']:.4f} |")
        L.append(f"| 角度误差 (°) | {ae['mean']:+.3f} | {ae['std']:.3f} "
                 f"| {ae['p95']:.3f} | {ae['max']:.3f} |")
        L.append(f"| 缩放误差 (%) | {sp['mean']:+.3f} | {sp['std']:.3f} "
                 f"| {sp['p95']:.3f} | {sp['max']:.3f} |")
        L.append("")
    else:
        L.append("（无有效定位样本）\n")

    L.append("## 5. 单件节拍（仿真验证值）\n")
    tk = s["takt_ms"]
    L.append(f"- 平均 **{tk['mean']} ms** ／ P95 **{tk['p95']} ms** ／ "
             f"最大 **{tk['max']} ms**\n")

    L.append("## 6. 错检明细（最多 40 条）\n")
    if s["mismatches"]:
        L.append("| 序号 | 真值 | 判定结果 | 漏检类型 | 多检类型 |")
        L.append("|---|---|---|---|---|")
        for x in s["mismatches"]:
            L.append(f"| #{x['idx']} | {'+'.join(x['truth'])} | "
                     f"{x['judged']} | {'+'.join(x['missed']) or '—'} | "
                     f"{'+'.join(x['extra']) or '—'} |")
        L.append("\n对应标注图见 data/annot/ 下 batch_miss_* / batch_fp_* 文件。")
    else:
        L.append("无错检样本。")
    L.append("")

    L.append("## 7. 指标口径说明\n")
    L.append("1. 图像来源：simulator/synth.py 逐帧合成（位置/旋转/缩放/亮度/"
             "噪声全随机，seed 固定可复现）；")
    L.append("2. \"检出率\"以件为单位（该帧存在任一注入缺陷且被判 NG）；"
             "\"类型检出率\"以缺陷实例为单位；")
    L.append("3. 定位误差对照对象是合成器输出的解析真值（非人工测量）；")
    L.append("4. 节拍为 Python 进程内 perf_counter 计时的 定位+检测 全链路耗时，"
             "不含图像落盘与网络传输；")
    L.append("5. 以上均为仿真环境数据，真实相机/光源/PLC 下需重新标定与验收。")

    path.write_text("\n".join(L), encoding="utf-8")


# ================================================================
# 命令行入口
# ================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="批量测试与验收（合成→定位+检测→对照真值→测试报告）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--count", type=int, default=500, help="批量张数")
    ap.add_argument("--defect-rate", type=float, default=0.5,
                    help="每帧注入缺陷的概率 0~1（即 NG 件占比目标）")
    ap.add_argument("--seed", type=int, default=42,
                    help="随机种子（固定后整批可复现）")
    ap.add_argument("--save-annot", type=int, default=0,
                    help="额外保存前 N 张标注图到 data/annot/")
    args = ap.parse_args()

    run_batch(count=args.count, defect_rate=args.defect_rate,
              seed=args.seed, save_annot=args.save_annot)

    # 退出码：报告中的门槛全部通过 → 0，否则 1（便于自动化流水线判断）
    report_path = config.DATA_DIR / "batch_report.json"
    gates = json.loads(report_path.read_text(encoding="utf-8"))["gates"]
    sys.exit(0 if all(g["pass"] for g in gates) else 1)


if __name__ == "__main__":
    main()
