# -*- coding: utf-8 -*-
"""
tests/test_vision_pipeline.py —— 视觉流水线共享层单元测试
========================================================
覆盖：
    - build_record 正常记录与 FAULT 记录**满键同构**（字段集合完全相同）；
    - defect_code_of 位编码（多缺陷或运算、定位失败位、16bit 截断）；
    - public_result 裁剪层不泄漏内部键、缺键输入不抛异常；
    - inspect_frame 返回结构与耗时口径。
运行：python -m unittest discover -s tests
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

import config
import vision_pipeline as vp


def _fake_result(result="NG", types=("scratch",), ok=True):
    return {"result": result,
            "defect_types": list(types),
            "confidence": 0.93,
            "hole_max_offset_mm": 0.12,
            "hole_max_dia_dev_mm": 0.05,
            "holes_found": 4,
            "defects": [{"type": "scratch", "area_px": 88.0}],
            "duration_ms": 105.2,
            "locate": {"ok": ok, "center_mm": [1.5, -2.0], "angle_deg": 12.3,
                       "scale": 1.01, "score": 0.88}}


class RecordShapeTests(unittest.TestCase):
    """记录 schema 单点定义的核心保证：两种记录字段集合一致"""

    def test_fault_record_is_key_isomorphic_with_normal(self):
        normal = vp.build_record(1, _fake_result(), duration_ms=105.2,
                                 truth_types=["scratch"])
        fault = vp.build_record(2, fault=True, duration_ms=2000.0)
        self.assertEqual(set(normal), set(fault),
                         "FAULT 记录必须与正常记录满键同构（前端无需容错缺键）")

    def test_normal_record_values(self):
        rec = vp.build_record(7, _fake_result("NG", ("scratch", "stain")),
                              duration_ms=99.44, truth_types=["stain"])
        self.assertEqual(rec["seq"], 7)
        self.assertEqual(rec["result"], "NG")
        self.assertEqual(rec["defect_types"], ["scratch", "stain"])
        self.assertEqual(rec["truth_defects"], ["stain"])
        self.assertEqual(rec["duration_ms"], 99.4)      # 一位小数
        self.assertFalse(rec["fault"])

    def test_fault_record_values(self):
        rec = vp.build_record(9, fault=True, duration_ms=2000.0)
        self.assertEqual(rec["result"], "FAULT")
        self.assertEqual(rec["defect_types"], ["watchdog"])
        self.assertTrue(rec["fault"])
        self.assertIsNone(rec["center_mm"])
        self.assertEqual(rec["truth_defects"], [])

    def test_record_defect_types_is_copy_not_alias(self):
        res = _fake_result("NG", ("chip",))
        rec = vp.build_record(1, res, duration_ms=1.0, truth_types=[])
        res["defect_types"].append("stain")
        self.assertEqual(rec["defect_types"], ["chip"],
                         "记录不应与检测结果共享同一列表对象")


class DefectCodeTests(unittest.TestCase):

    def test_single_type_matches_config_bit(self):
        for name, bit in config.DEFECT_BIT.items():
            if name == "locate_fail":
                continue
            code = vp.defect_code_of(_fake_result("NG", (name,)))
            self.assertEqual(code, bit, f"{name} 位编码应与 config 一致")

    def test_multiple_types_are_ored(self):
        code = vp.defect_code_of(_fake_result("NG", ("scratch", "stain")))
        self.assertEqual(code, config.DEFECT_BIT["scratch"] |
                         config.DEFECT_BIT["stain"])

    def test_locate_fail_sets_its_bit(self):
        code = vp.defect_code_of(_fake_result("NG", (), ok=False))
        self.assertEqual(code, config.DEFECT_BIT["locate_fail"])

    def test_unknown_type_ignored_and_16bit_clamped(self):
        code = vp.defect_code_of(_fake_result("NG", ("不存在的类型",)))
        self.assertEqual(code, 0)
        self.assertEqual(code & 0xFFFF, code)


class PublicResultTests(unittest.TestCase):

    def test_only_whitelisted_keys(self):
        pub = vp.public_result(_fake_result())
        self.assertEqual(set(pub), {"result", "defect_types", "confidence",
                                    "duration_ms", "hole_max_offset_mm",
                                    "locate"})
        self.assertEqual(set(pub["locate"]), {"angle_deg"},
                         "locate 只对外暴露角度，score/scale 等内部量不泄漏")

    def test_tolerates_missing_keys(self):
        pub = vp.public_result({})       # 不应抛异常
        self.assertIsNone(pub["result"])
        self.assertEqual(pub["defect_types"], [])
        self.assertIsNone(pub["locate"]["angle_deg"])


class InspectFrameTests(unittest.TestCase):

    def test_returns_result_and_positive_duration(self):
        import part_model
        frame, _ = part_model.make_reference()
        result, ms = vp.inspect_frame(frame)
        self.assertIn("result", result)
        self.assertIn(result["result"], ("OK", "NG"))
        self.assertIsInstance(ms, float)
        self.assertGreater(ms, 0.0)
        self.assertEqual(ms, round(ms, 1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
