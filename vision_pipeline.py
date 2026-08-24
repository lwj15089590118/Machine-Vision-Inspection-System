# -*- coding: utf-8 -*-
"""
vision_pipeline.py —— 视觉流水线共享层（批量验收与 PLC 循环的唯一驱动）
========================================================
职责：
    把 run_batch 与 plc_server 曾经各自复制的「驱动检测流水线」知识
    收敛为一个深模块，调用方退化为薄 driver：
      1. inspect_frame(frame)     定位+检测一帧，返回 (结果JSON, 耗时ms)；
      2. build_record(...)        records.jsonl 单行记录的**唯一构造点**
                                 （FAULT 记录与 OK/NG 满键同构，空值不缺键）；
      3. defect_code_of(result)   HR3 缺陷位组合编码（含定位失败位）；
      4. save_latest_assets(...)  看板最新标注帧 + 最新结果 JSON 落盘。

口径说明：
    - duration_ms 只计 detect.detect.inspect 的纯算法耗时（不含合成/落盘）；
      plc_server 此前把合成时间也计入 records.duration_ms，统一后略有
      变小，属口径修正而非回归。
    - 记录字段契约以本模块 build_record 为准：dashboard/app.py 的
      trim_record 只做读侧兜底，不再是契约的一部分。
"""
import json
import time
from datetime import datetime

import cv2
import numpy as np

import config
import detect.detect as detector


def inspect_frame(frame: np.ndarray) -> tuple:
    """定位+检测一帧。返回 (result_dict, duration_ms)，耗时为纯算法口径。"""
    t0 = time.perf_counter()
    result = detector.inspect(frame)
    return result, round((time.perf_counter() - t0) * 1000.0, 1)


def defect_code_of(result: dict) -> int:
    """HR3 缺陷位组合：按类型查位表 + 定位失败置 bit5，限 16bit"""
    code = 0
    for t in result["defect_types"]:
        code |= config.DEFECT_BIT.get(t, 0)
    if not result["locate"].get("ok"):
        code |= config.DEFECT_BIT["locate_fail"]
    return code & 0xFFFF


def build_record(seq: int, result: dict = None, *, fault: bool = False,
                 duration_ms: float = None, truth_types=None) -> dict:
    """
    records.jsonl 单行的唯一构造点。
    正常记录传 (seq, result, duration_ms=..., truth_types=[...])；
    看门狗故障记录传 (seq, fault=True, duration_ms=...)——与正常记录
    **满键同构**（缺测字段填 None/[]，前端 .length / ?? 兜底从"必需"
    降级为"保险"）。
    """
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    if fault:
        return {"seq": seq, "ts": ts, "result": "FAULT",
                "defect_types": ["watchdog"], "defect_code": 0,
                "center_mm": None, "angle_deg": None,
                "hole_max_offset_mm": None, "confidence": None,
                "duration_ms": round(duration_ms, 1),
                "truth_defects": [], "fault": True}
    return {"seq": seq, "ts": ts,
            "result": result["result"],
            "defect_types": list(result["defect_types"]),
            "defect_code": defect_code_of(result),
            "center_mm": result["locate"].get("center_mm"),
            "angle_deg": result["locate"].get("angle_deg"),
            "hole_max_offset_mm": result["hole_max_offset_mm"],
            "confidence": result["confidence"],
            "duration_ms": round(duration_ms, 1),
            "truth_defects": list(truth_types or []),
            "fault": False}


def public_result(result: dict) -> dict:
    """latest.json 的对外裁剪层：只保留看板实际消费的字段（防泄漏内部键）"""
    return {"result": result.get("result"),
            "defect_types": list(result.get("defect_types") or []),
            "confidence": result.get("confidence"),
            "duration_ms": result.get("duration_ms"),
            "hole_max_offset_mm": result.get("hole_max_offset_mm"),
            "locate": {"angle_deg":
                       (result.get("locate") or {}).get("angle_deg")}}


def save_latest_assets(frame: np.ndarray, result: dict) -> None:
    """看板最新资产落盘：data/annot/latest.png + latest.json（经裁剪层）。
    展示失败不影响主流程（异常在此吞掉并打印）。"""
    try:
        png = config.ANNOT_DIR / "latest.png"
        cv2.imwrite(str(png),
                    detector.draw_defects(frame, result["locate"], result))
        (config.ANNOT_DIR / "latest.json").write_text(
            json.dumps(public_result(result), ensure_ascii=False,
                       default=str),
            encoding="utf-8")
    except Exception as e:
        print(f"[VISION] 保存最新帧失败(不影响检测): {e}")
