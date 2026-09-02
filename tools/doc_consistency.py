# -*- coding: utf-8 -*-
"""
tools/doc_consistency.py —— 工件↔文档数字一致性自检（防口径漂移门禁，零依赖）
================================================================
背景：
    第一轮审查的 P0（README 宣称 500 帧而工件只有 100 帧）本质是
    "数字口径漂移"；复审报告 N3 进一步指出 docs/batch_report.json 快照
    靠手动同步进 README / docs/测试报告.md / docs/系统设计说明书.md，
    重跑批量后忘记同步就会再次打开漂移窗口。本工具把"文档声称 vs
    工件字段"变成机器可判定的断言：凡不一致即 FAIL，并逐条指出
    「哪个文件的哪一句话 与 工件的哪个字段 差多少」。
    纪律：重跑 run_batch 后必须先跑本自检（见 README 快速开始 3.6 与
    docs/验收清单.md §5），全绿才允许提交。

覆盖范围（以 docs/batch_report.json 实际字段为准）：
    - meta：count（帧数）/ defect_rate / seed / wall_time_s /
      generated_at / env（python·opencv·numpy·os）；
    - confusion：TP / FN / FP / TN / ng_total / ok_total / locate_fail，
      以及测试报告"列合计/总样本"的自洽数字；
    - recall_piece / false_alarm / precision（百分比）；
    - type_rows：五类缺陷的 注入数 / 命中数 / 检出率；
    - locate：center_err_px / center_err_mm / angle_err_deg /
      scale_err_pct 的 均值 / 标准差 / P95 / 最大；
    - takt_ms：mean / p95 / max；
    - gates：五条门槛行的 target / actual 原文。
    校验对象：README.md、docs/测试报告.md、docs/系统设计说明书.md 中
    所有可归属到上述字段的数字声称（正则逐条提取、逐处比对）。
    batch_report.json ↔ 测试报告.md 的一致性由"两者对同一工件逐字段
    比对 + generated_at/wall_time_s 同源校验"共同保证。
    明确不在范围（无 batch_report.json 字段可对照，强行校验只会误报）：
    标定值（0.0919px / 0.10001mm·px⁻¹，其工件为 calib 产物不入库）、
    Modbus 自测 10/10、历史调参记录（"误报 13→0"、"+20ms/件"等）。

容差口径（避免脆弱误报，均已在代码中注明）：
    - 整数字段（帧数 / TP / FN / FP / TN / 注入数 / 命中数 / seed）：
      0 容差，差 1 即漂移；
    - 浮点字段：容差 = 文档显示小数位的"半个最小刻度"换算到工件单位
      （文档按自身精度舍入后不可分辨的区间）+ 1e-9 浮点余量；
    - 格式容错：千分位逗号、全角负号 − / 加号 ＋、数值前后空格、
      单位（ms / mm / % / °）均按格式差异解析，不判为漂移；
    - 如确需对个别易浮动字段放宽，可对单条 Claim 设 rel_tol（相对
      容差，上限 5%，须在 note 写明理由）——当前全部为 0：墙钟类
      数字（节拍/耗时）一旦入库就是口径本身，放宽反而会漏报真漂移。

用法：
    python tools/doc_consistency.py             # 校验仓库现状，退出码 0/1
    python tools/doc_consistency.py --root DIR  # 校验指定目录（篡改演练/测试用）
    from tools.doc_consistency import check     # 供 tests/test_doc_consistency.py 调用

退出码：0 = DOC-ARTIFACT CONSISTENT；1 = 存在漂移或锚点缺失（逐条列出）。
"""
import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_RELPATH = "docs/batch_report.json"
DOC_RELPATHS = ("README.md", "docs/测试报告.md", "docs/系统设计说明书.md")

# 数值片段（带可选符号与千分位；−/＋ 为全角正负号，设计说明书在用）
NUM_SIGNED = r"[+\-−＋]?[\d,]+(?:\.\d+)?"
NUM_UNSIGNED = r"[\d,]+(?:\.\d+)?"


@dataclass(frozen=True)
class Claim:
    """一条"文档数字声称 ↔ 工件字段"校验规则"""
    doc: str            # 相对路径（DOC_RELPATHS 之一）
    field: str          # 工件字段路径，如 confusion.tp / gates[name=X].actual
    pattern: str        # 正则，第一个捕获组为文档中的数字/文本
    scale: float = 1.0  # 文档单位 → 工件单位（百分比声称 = 0.01）
    kind: str = "number"   # number | text（版本号/时间戳等非数值用 text）
    rel_tol: float = 0.0   # 额外相对容差（上限 0.05，须在 note 说明理由）
    note: str = ""         # 容差/口径说明


def _resolve(artifact: dict, field: str):
    """按点分路径取工件字段值；列表段支持 list[key]（按 name 或 type 匹配）"""
    cur = artifact
    for seg in field.split("."):
        m = re.fullmatch(r"(\w+)\[(?:name=)?([^\]]+)\]", seg)
        if m:
            lst = cur[m.group(1)]
            key = m.group(2)
            cur = next(it for it in lst
                       if it.get("name") == key or it.get("type") == key)
        else:
            cur = cur[seg]
    return cur


def _num(text: str) -> float:
    """解析文档数字：容忍千分位与全角正负号"""
    return float(text.replace(",", "")
                    .replace("−", "-").replace("＋", "+").strip())


def _decimals(text: str) -> int:
    """文档数字显示的小数位数（决定舍入容差的半个刻度）"""
    s = text.lstrip("+-−＋").replace(",", "").strip()
    return len(s.split(".", 1)[1]) if "." in s else 0


# ================================================================
# 校验规则清单（静态部分：README / 设计说明书 / 测试报告头部与结论段）
# 模式锚点均带足够上下文，避免误捕历史调参记录（如设计说明书 §8 的
# "误报 13→0"、"+20ms/件"）——改动文档措辞时同步维护此清单。
# ================================================================
def _static_specs() -> list:
    rd, tr, dd = DOC_RELPATHS
    specs = []

    # ---- README.md ----
    specs += [
        (rd, "meta.count", r"批量验收[：（(]\s*(\d[\d,]*)\s*帧"),
        (rd, "meta.defect_rate",
         r"批量验收[：（(]\s*\d[\d,]*\s*帧，缺陷占比\s*(\d+(?:\.\d+)?)\s*%",
         0.01),
        (rd, "meta.seed",
         r"批量验收[：（(]\s*\d[\d,]*\s*帧，缺陷占比\s*\d+(?:\.\d+)?\s*%，"
         r"seed=(\d+)"),
        (rd, "confusion.tp", r"TP\s*=\s*(\d[\d,]*)"),
        (rd, "confusion.fn", r"FN\s*=\s*(\d[\d,]*)"),
        (rd, "confusion.fp", r"FP\s*=\s*(\d[\d,]*)"),
        (rd, "confusion.tn", r"TN\s*=\s*(\d[\d,]*)"),
        (rd, "recall_piece", r"件级召回）\s*\|\s*\*\*([\d.,]+)\s*%", 0.01),
        (rd, "false_alarm", r"误报率\s*\|\s*\*\*([\d.,]+)\s*%", 0.01),
        (rd, "precision", r"判 NG 精确率\s*\|\s*([\d.,]+)\s*%", 0.01),
        (rd, "type_rows[chip].recall", r"崩边\s*([\d.,]+)\s*%", 0.01),
        (rd, "type_rows[bolt_missing].recall", r"孔缺失\s*([\d.,]+)\s*%", 0.01),
        (rd, "type_rows[bolt_shift].recall", r"孔偏移\s*([\d.,]+)\s*%", 0.01),
        (rd, "type_rows[scratch].recall", r"划痕\s*([\d.,]+)\s*%", 0.01),
        (rd, "type_rows[stain].recall", r"污渍\s*([\d.,]+)\s*%", 0.01),
        (rd, "locate.center_err_mm.p95", r"P95\s*\*\*([\d.,]+)\s*mm"),
        (rd, "locate.center_err_mm.max", r"mm\*\*（最大\s*([\d.,]+)）"),
        (rd, "locate.angle_err_deg.p95", r"P95\s*\*\*([\d.,]+)\s*°"),
        (rd, "takt_ms.mean", r"平均\s*\*\*([\d.,]+)\s*ms"),
        (rd, "takt_ms.p95", r"P95\s*([\d.,]+)\s*ms"),
        (rd, "meta.env.python", r"环境：Windows 11 · Python\s*([\d.]+)",
         1.0, "text"),
        (rd, "meta.env.opencv", r"OpenCV\s*([\d.]+)\s*·\s*CPU", 1.0, "text"),
    ]

    # ---- docs/系统设计说明书.md（§7.2 汇总表 + §4.3 正文引用）----
    specs += [
        (dd, "meta.count", r"批量验收（(\d[\d,]*)\s*帧"),
        (dd, "meta.defect_rate",
         r"批量验收（\d[\d,]*\s*帧，缺陷占比\s*(\d+(?:\.\d+)?)\s*%", 0.01),
        (dd, "meta.seed",
         r"批量验收（\d[\d,]*\s*帧，缺陷占比\s*\d+(?:\.\d+)?\s*%，seed=(\d+)"),
        (dd, "confusion.tp", r"TP\s*=\s*(\d[\d,]*)"),
        (dd, "confusion.fn", r"FN\s*=\s*(\d[\d,]*)"),
        (dd, "confusion.fp", r"FP\s*=\s*(\d[\d,]*)"),
        (dd, "confusion.tn", r"TN\s*=\s*(\d[\d,]*)"),
        (dd, "recall_piece", r"件级检出率\s*\|\s*\*\*([\d.,]+)\s*%", 0.01),
        (dd, "false_alarm", r"误报率\s*\|\s*\*\*([\d.,]+)\s*%", 0.01),
        (dd, "precision", r"判 NG 精确率\s*\|\s*([\d.,]+)\s*%", 0.01),
        (dd, "type_rows[chip].recall",
         r"崩边/孔偏移/孔缺失\s*\|\s*([\d.,]+)\s*%", 0.01),
        (dd, "type_rows[bolt_shift].recall",
         r"崩边/孔偏移/孔缺失\s*\|\s*[\d.,]+%\s*/\s*([\d.,]+)\s*%", 0.01),
        (dd, "type_rows[bolt_missing].recall",
         r"崩边/孔偏移/孔缺失\s*\|\s*[\d.,]+%\s*/\s*[\d.,]+%\s*/\s*"
         r"([\d.,]+)\s*%", 0.01),
        (dd, "type_rows[scratch].recall",
         r"划痕/污渍\s*\|\s*([\d.,]+)\s*%", 0.01),
        (dd, "type_rows[stain].recall",
         r"划痕/污渍\s*\|\s*[\d.,]+%\s*/\s*([\d.,]+)\s*%", 0.01),
        (dd, "locate.center_err_mm.mean",
         r"中心误差\s*\|\s*均值\s*(" + NUM_SIGNED + r")\s*mm"),
        (dd, "locate.center_err_mm.p95", r"P95\s*\*\*([\d.,]+)\s*mm"),
        (dd, "locate.center_err_mm.p95", r"P95=([\d.,]+)mm"),   # §4.3 正文
        (dd, "locate.center_err_mm.max", r"最大\s*([\d.,]+)\s*mm"),
        (dd, "locate.angle_err_deg.mean",
         r"角度误差\s*\|\s*均值\s*(" + NUM_SIGNED + r")\s*°"),
        (dd, "locate.angle_err_deg.p95", r"P95\s*\*\*([\d.,]+)\s*°"),
        (dd, "locate.angle_err_deg.max", r"最大\s*([\d.,]+)\s*°"),
        # scale_err_pct 工件字段本身就是百分数单位，scale=1
        (dd, "locate.scale_err_pct.p95", r"缩放误差\s*\|\s*P95\s*([\d.,]+)\s*%"),
        (dd, "takt_ms.mean", r"平均\s*\*\*([\d.,]+)\s*ms"),
        (dd, "takt_ms.mean", r"([\d.,]+)\s*ms/件的 CPU 节拍"),   # §4.3 正文
        (dd, "takt_ms.p95", r"P95\s*([\d.,]+)\s*ms"),
        (dd, "takt_ms.max", r"最大\s*([\d.,]+)\s*ms"),
        (dd, "meta.env.python", r"环境：Windows 11 · Python\s*([\d.]+)",
         1.0, "text"),
        (dd, "meta.env.opencv", r"OpenCV\s*([\d.]+)\s*·\s*NumPy", 1.0, "text"),
        (dd, "meta.env.numpy", r"NumPy\s*([\d.]+)\s*·\s*CPU", 1.0, "text"),
    ]

    # ---- docs/测试报告.md（头部信息 + 结论段；表体由 _generated_specs 生成）----
    specs += [
        (tr, "meta.generated_at",
         r"测试时间：(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", 1.0, "text"),
        (tr, "meta.count", r"样本规模：(\d[\d,]*)\s*张"),
        (tr, "meta.defect_rate",
         r"缺陷占比设定\s*(\d+(?:\.\d+)?)\s*%", 0.01),
        (tr, "meta.seed", r"seed=(\d+)"),
        (tr, "meta.wall_time_s", r"总耗时\s*([\d.,]+)\s*s"),
        (tr, "meta.env.python", r"Python\s*([\d.]+)\s*·", 1.0, "text"),
        (tr, "meta.env.opencv", r"OpenCV\s*([\d.]+)\s*·", 1.0, "text"),
        (tr, "meta.env.numpy", r"NumPy\s*([\d.]+)\s*·", 1.0, "text"),
        (tr, "meta.env.os", r"NumPy\s*[\d.]+\s*·\s*(\S+)", 1.0, "text"),
        (tr, "confusion.tp", r"TP\s*=\s*(\d[\d,]*)"),
        (tr, "confusion.fn", r"FN\s*=\s*(\d[\d,]*)"),
        (tr, "confusion.fp", r"FP\s*=\s*(\d[\d,]*)"),
        (tr, "confusion.tn", r"TN\s*=\s*(\d[\d,]*)"),
        (tr, "confusion.ng_total", r"FN\s*=\s*\d[\d,]*（漏检）\s*\|\s*(\d[\d,]*)"),
        (tr, "confusion.ok_total", r"TN\s*=\s*\d[\d,]*\s*\|\s*(\d[\d,]*)"),
        (tr, "meta.count",   # 混淆矩阵总计 = TP+FN+FP+TN，回挂 count
         r"列合计\s*\|\s*\d[\d,]*\s*\|\s*\d[\d,]*\s*\|\s*(\d[\d,]*)\s*\|"),
        (tr, "recall_piece",
         r"Recall\s*=\s*TP/\(TP\+FN\)\s*=\s*\*\*([\d.,]+)\s*%", 0.01),
        (tr, "false_alarm",
         r"FalseAlarm\s*=\s*FP/\(FP\+TN\)\s*=\s*\*\*([\d.,]+)\s*%", 0.01),
        (tr, "precision",
         r"Precision\s*=\s*TP/\(TP\+FP\)\s*=\s*\*\*([\d.,]+)\s*%", 0.01),
        (tr, "confusion.locate_fail",
         r"定位失败张数（安全策略按 NG 处理）：(\d[\d,]*)"),
        (tr, "takt_ms.mean", r"平均\s*\*\*([\d.,]+)\s*ms"),
        (tr, "takt_ms.p95", r"P95\s*\*\*([\d.,]+)\s*ms"),
        (tr, "takt_ms.max", r"最大\s*\*\*([\d.,]+)\s*ms"),
    ]
    return specs


# ================================================================
# 生成式规则：测试报告的表格体（类型行/定位表/门槛行）按工件字段派生，
# 工件新增类型/门槛时校验清单自动跟进，不需手写。
# ================================================================
# 与 config.DEFECT_CN 同值（工具保持零依赖故此处复制；改动时须同步）
_DEFECT_CN = {"scratch": "划痕", "chip": "崩边", "stain": "污渍",
              "bolt_shift": "孔偏移", "bolt_missing": "孔缺失"}
_LOCATE_ROWS = (("中心误差 (px)", "center_err_px"),
                ("中心误差 (mm)", "center_err_mm"),
                ("角度误差 (°)", "angle_err_deg"),
                ("缩放误差 (%)", "scale_err_pct"))
_STAT_KEYS = ("mean", "std", "p95", "max")


def _generated_specs(artifact: dict) -> list:
    tr = "docs/测试报告.md"
    specs = []

    # 分缺陷类型行：| 划痕 scratch | 75 | 65 | 86.7% |
    for row in artifact.get("type_rows", []):
        if not row.get("injected"):
            continue                                  # 报告会略去 0 注入行
        label = re.escape(f"{_DEFECT_CN[row['type']]} {row['type']}")
        specs += [
            (tr, f"type_rows[{row['type']}].injected",
             label + r"\s*\|\s*(\d[\d,]*)\s*\|"),
            (tr, f"type_rows[{row['type']}].hit",
             label + r"\s*\|\s*\d[\d,]*\s*\|\s*(\d[\d,]*)\s*\|"),
            (tr, f"type_rows[{row['type']}].recall",
             label + r"\s*\|\s*\d[\d,]*\s*\|\s*\d[\d,]*\s*\|\s*([\d.,]+)\s*%",
             0.01),
        ]

    # 定位精度表：| 中心误差 (mm) | 均值 | 标准差 | P95 | 最大 |
    for label, key in _LOCATE_ROWS:
        stats = artifact.get("locate", {}).get(key)
        if not stats:
            continue
        esc = re.escape(label)
        for i, stat in enumerate(_STAT_KEYS):
            if stats.get(stat) is None:
                continue
            pre = r"\s*\|\s*".join([esc] + [NUM_SIGNED] * i)
            specs.append((tr, f"locate.{key}.{stat}",
                          pre + r"\s*\|\s*(" + NUM_SIGNED + r")\s*\|"))

    # 门槛行：| 件级缺陷检出率 | ≥ 95% | 100.00% | ✅ PASS |（原文逐字比对）
    for g in artifact.get("gates", []):
        esc = re.escape(g["name"])
        specs += [
            (tr, f"gates[name={g['name']}].target",
             r"\|\s*" + esc + r"\s*\|\s*([^|]+?)\s*\|", 1.0, "text"),
            (tr, f"gates[name={g['name']}].actual",
             r"\|\s*" + esc + r"\s*\|\s*[^|]+?\|\s*([^|]+?)\s*\|", 1.0, "text"),
        ]
    return specs


def build_claims(artifact: dict) -> list:
    """把规则清单实例化为 Claim；工件中无值（None）的声称自动豁免"""
    claims = []
    for spec in _static_specs() + _generated_specs(artifact):
        doc, field, pattern = spec[0], spec[1], spec[2]
        scale = spec[3] if len(spec) > 3 else 1.0
        kind = spec[4] if len(spec) > 4 else "number"
        try:
            expected = _resolve(artifact, field)
        except (KeyError, StopIteration) as exc:
            raise RuntimeError(
                f"工件 {ARTIFACT_RELPATH} 缺少校验清单引用的字段 {field}"
                f"（{exc!r}）——字段结构变更时请同步更新 "
                f"tools/doc_consistency.py 的规则清单") from exc
        if expected is None:
            continue            # 报告生成器会略去无值项，文档同样不含
        claims.append(Claim(doc=doc, field=field, pattern=pattern,
                            scale=scale, kind=kind))
    return claims


# ================================================================
# 执行校验
# ================================================================
def check_texts(artifact: dict, texts: dict, claims: list = None):
    """对给定的文档文本执行全部声称校验。

    返回 (failures, summary)。failures 每条都是可直接定位的人读消息；
    summary 记录 声称数/比对处数，供测试与 CLI 汇总。
    """
    if claims is None:
        claims = build_claims(artifact)
    failures = []
    n_match = 0
    for c in claims:
        text = texts[c.doc]
        matches = list(re.finditer(c.pattern, text, re.MULTILINE))
        if not matches:
            failures.append(
                f"[锚点缺失] {c.doc}: 模式 /{c.pattern}/ 零命中——文档中已无"
                f"工件字段 {c.field} 的声称。若为文档改版，请同步更新 "
                f"tools/doc_consistency.py 的规则清单；若是漏写数字，请补齐。")
            continue
        expected = _resolve(artifact, c.field)
        lines = text.splitlines()
        for m in matches:
            n_match += 1
            ln = text.count("\n", 0, m.start())
            line = lines[ln].strip()
            if len(line) > 72:
                line = line[:69] + "..."
            if c.kind == "text":
                got = m.group(1).strip()
                if got != str(expected):
                    failures.append(
                        f"[漂移] {c.doc}:{ln + 1}: 「{line}」 文档声称 "
                        f"{got!r} ↔ 工件 {c.field} = {expected!r}（原文须一致）")
                continue
            raw = m.group(1)
            val = _num(raw) * c.scale
            # 容差 = 文档显示小数位半个刻度（换算到工件单位）+ 浮点余量
            #      （+ 相对容差，仅当该条 Claim 显式声明并注明理由）
            tol = (0.5 * 10.0 ** (-_decimals(raw)) * c.scale + 1e-9
                   + abs(expected) * c.rel_tol)
            diff = val - expected
            if abs(diff) > tol:
                tol_note = "文档小数位半个刻度"
                if c.rel_tol:
                    tol_note += f" + 相对容差 {c.rel_tol:.1%}（{c.note}）"
                failures.append(
                    f"[漂移] {c.doc}:{ln + 1}: 「{line}」 文档声称 {raw}"
                    f"（换算后 {val:g}）↔ 工件 {c.field} = {expected:g}，"
                    f"差 {diff:+g}，超出容差 ±{tol:g}（{tol_note}）")
    return failures, {"claims": len(claims), "matches": n_match}


def check(root: Path = PROJECT_ROOT):
    """校验 root 目录下的工件与三份文档（生产入口/测试入口）"""
    root = Path(root)
    missing = [p for p in (ARTIFACT_RELPATH, *DOC_RELPATHS)
               if not (root / p).exists()]
    if missing:
        return ([f"[缺失] {root} 下未找到: {', '.join(missing)}"],
                {"claims": 0, "matches": 0})
    artifact = json.loads(
        (root / ARTIFACT_RELPATH).read_text(encoding="utf-8"))
    texts = {d: (root / d).read_text(encoding="utf-8") for d in DOC_RELPATHS}
    return check_texts(artifact, texts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="工件↔文档数字一致性自检（batch_report.json vs "
                    "README/测试报告/设计说明书）")
    ap.add_argument("--root", default=str(PROJECT_ROOT),
                    help="项目根目录（默认为本工具所在仓库根）")
    args = ap.parse_args()

    failures, summary = check(args.root)
    if not failures:
        print(f"共校验 {summary['claims']} 条数字声称、逐处比对 "
              f"{summary['matches']} 处：README.md / docs/测试报告.md / "
              f"docs/系统设计说明书.md 与 {ARTIFACT_RELPATH} 逐字段一致。")
        print("DOC-ARTIFACT CONSISTENT")
        return 0
    print(f"DOC-ARTIFACT DRIFT（{len(failures)} 处）——以下文档声称与工件不一致：")
    for f in failures:
        print(" ", f)
    print("修复方式：重跑批量后以新工件为准同步文档（或按工件更正声称），"
          "再跑本自检至全绿。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
