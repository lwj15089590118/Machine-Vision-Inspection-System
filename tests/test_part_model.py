# -*- coding: utf-8 -*-
"""
tests/test_part_model.py —— 工件基准模型单元测试（零依赖，标准库 unittest）
========================================================
覆盖：
    1. 规范几何性质：螺栓孔数量/分度圆半径/方位角；键槽中心与多边形；
       位姿矩阵与仿射变换的一致性；
    2. 名义外观：build_part 形状/确定性（两次调用逐位相等）；
    3. 黄金资产：make_reference 可复现（两次调用逐位相等）；
    4. 模板落盘缓存：首跑生成文件、命中读盘（改缓存内容能被读到，
       证明走的是缓存而非重渲染）、指纹失效自动重渲染、损坏自愈；
    5. 指纹纯函数：对几何/外观参数敏感、重复调用稳定。

运行：
    python -m unittest discover -s tests        # 或
    python tests/test_part_model.py
"""
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

# 允许直接 `python tests/test_part_model.py` 运行：项目根加入 sys.path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402  （仅用于测试中读写 PNG）

import config  # noqa: E402
import part_model as pm  # noqa: E402


class GeometryTests(unittest.TestCase):
    """规范几何性质"""

    def test_bolt_centers_count_and_radius(self):
        holes = pm.bolt_centers_canonical()
        self.assertEqual(len(holes), len(config.BOLT_ANGLES_DEG))
        cx, cy = config.CANON_CENTER
        for (x, y), a_deg in zip(holes, config.BOLT_ANGLES_DEG):
            r = math.hypot(x - cx, y - cy)
            self.assertAlmostEqual(r, config.BOLT_PC_R_PX, places=6)
            a = math.degrees(math.atan2(y - cy, x - cx))
            self.assertAlmostEqual(a % 360.0, a_deg % 360.0, places=6)

    def test_keyway_center_on_designated_azimuth(self):
        cx, cy = config.CANON_CENTER
        r_mid = config.FLANGE_R_PX - config.KEYWAY_D_PX / 2.0
        kx, ky = pm.keyway_center_canonical()
        self.assertAlmostEqual(kx, cx + r_mid * math.cos(
            math.radians(config.KEYWAY_ANGLE_DEG)), places=6)
        self.assertAlmostEqual(ky, cy + r_mid * math.sin(
            math.radians(config.KEYWAY_ANGLE_DEG)), places=6)

    def test_keyway_polygon_shape_and_span(self):
        poly = pm.keyway_polygon()
        self.assertEqual(poly.dtype, np.int32)
        self.assertEqual(poly.shape, (4, 2))
        # 默认 0° 方位：矩形沿 +x 展开，y 跨键槽宽，x 覆盖 [r_in, r_out]
        cx, cy = config.CANON_CENTER
        if config.KEYWAY_ANGLE_DEG == 0.0:
            xs, ys = poly[:, 0], poly[:, 1]
            self.assertGreaterEqual(xs.min(), cx + config.FLANGE_R_PX
                                    - config.KEYWAY_D_PX - 1.0)
            self.assertLessEqual(xs.max(), cx + config.FLANGE_R_PX + 4.0)
            self.assertAlmostEqual(ys.max() - ys.min(),
                                   config.KEYWAY_W_PX, delta=2.0)
            self.assertAlmostEqual(float(ys.mean()), cy, delta=1.0)

    def test_pose_and_affine_roundtrip(self):
        theta, scale = 17.0, 0.96
        M = pm.pose_matrix(512.0, 288.0, theta, scale)
        base = [(config.CANON_CENTER[0] + config.BOLT_PC_R_PX,
                 config.CANON_CENTER[1])]
        moved = pm.apply_affine(M, base)[0]
        # 基准点绕基准中心旋转 θ 后平移到新中心（图像坐标 y 向下）
        ex = 512.0 + scale * config.BOLT_PC_R_PX * math.cos(math.radians(-theta))
        ey = 288.0 + scale * config.BOLT_PC_R_PX * math.sin(math.radians(-theta))
        self.assertAlmostEqual(float(moved[0]), ex, places=3)
        self.assertAlmostEqual(float(moved[1]), ey, places=3)


class NominalAppearanceTests(unittest.TestCase):
    """名义外观：无缺陷工件图层"""

    def test_build_part_shapes(self):
        part, mask = pm.build_part()
        self.assertEqual(part.shape, (config.IMG_H, config.IMG_W))
        self.assertEqual(part.dtype, np.float32)
        self.assertEqual(mask.dtype, np.uint8)

    def test_build_part_deterministic(self):
        p1, m1 = pm.build_part()
        p2, m2 = pm.build_part()
        self.assertTrue(np.array_equal(p1, p2))
        self.assertTrue(np.array_equal(m1, m2))

    def test_material_mask_is_disk_minus_keyway(self):
        _, mask = pm.build_part()
        area = float((mask > 0).sum())
        full_disk = math.pi * config.FLANGE_R_PX ** 2
        keyway_cut = config.KEYWAY_W_PX * (config.KEYWAY_D_PX + 3.0)
        # 掩膜语义 = "工件材料圆盘"（孔属于盘内特征，只挖灰度层不挖掩膜；
        # 键槽挖穿露出皮带才从掩膜去除）。外扩 3px 圆弧用 1% 容差吸收。
        self.assertAlmostEqual(area, full_disk - keyway_cut,
                               delta=full_disk * 0.01)


class GoldenAssetTests(unittest.TestCase):
    """黄金资产可复现性"""

    def test_make_reference_reproducible(self):
        r1, m1 = pm.make_reference()
        r2, m2 = pm.make_reference()
        self.assertEqual(r1.dtype, np.uint8)
        self.assertTrue(np.array_equal(r1, r2))
        self.assertTrue(np.array_equal(m1, m2))

    def test_template_equals_reference_crop(self):
        tpl = pm.make_template()
        ref, _ = pm.make_reference()
        m = int(config.FLANGE_R_PX + config.TEMPLATE_MARGIN_PX)
        cx, cy = int(config.CANON_CENTER[0]), int(config.CANON_CENTER[1])
        self.assertTrue(np.array_equal(
            tpl, ref[cy - m:cy + m, cx - m:cx + m]))


class TemplateCacheTests(unittest.TestCase):
    """模板落盘缓存：命中读盘 / 指纹失效重建 / 损坏自愈（隔离到临时目录）"""

    def setUp(self):
        self._old_path = config.TEMPLATE_PATH
        self._tmp = tempfile.TemporaryDirectory()
        config.TEMPLATE_PATH = Path(self._tmp.name) / "template.png"

    def tearDown(self):
        config.TEMPLATE_PATH = self._old_path
        self._tmp.cleanup()

    def test_first_call_persists_png_and_meta(self):
        tpl = pm.make_template()
        self.assertTrue(config.TEMPLATE_PATH.exists())
        meta = pm._meta_path()
        self.assertTrue(meta.exists())
        self.assertIn("fingerprint", meta.read_text(encoding="utf-8"))
        self.assertEqual(tpl.dtype, np.uint8)

    def test_cache_hit_reads_from_disk(self):
        t1 = pm.make_template()
        # 篡改缓存内容（指纹仍匹配）：若下次返回被篡改版本 ⇒ 走了读盘路径
        bumped = cv2.imread(str(config.TEMPLATE_PATH),
                            cv2.IMREAD_GRAYSCALE)
        bumped = np.clip(bumped.astype(np.int16) + 7, 0, 255).astype(np.uint8)
        cv2.imwrite(str(config.TEMPLATE_PATH), bumped)
        t2 = pm.make_template()
        self.assertTrue(np.array_equal(t2, bumped))
        self.assertFalse(np.array_equal(t2, t1))

    def test_fingerprint_miss_regenerates(self):
        old_gray = config.FACE_BASE_GRAY
        try:
            t1 = pm.make_template()
            config.FACE_BASE_GRAY = old_gray + 3.0        # 外观参数变更
            t2 = pm.make_template()
            self.assertEqual(t2.shape, t1.shape)
            self.assertFalse(np.array_equal(t2, t1))      # 用新参数重渲染
        finally:
            config.FACE_BASE_GRAY = old_gray

    def test_corrupt_png_self_heals(self):
        t1 = pm.make_template()
        config.TEMPLATE_PATH.write_bytes(b"not a png")    # 缓存损坏
        t2 = pm.make_template()
        self.assertTrue(np.array_equal(t2, t1))           # 自动重建


class FingerprintTests(unittest.TestCase):
    """指纹纯函数：稳定且对像素相关参数敏感"""

    def test_stable_across_calls(self):
        self.assertEqual(pm._template_fingerprint(),
                         pm._template_fingerprint())

    def test_sensitive_to_geometry_and_appearance(self):
        fp0 = pm._template_fingerprint()
        patches = [("TEMPLATE_MARGIN_PX", 34),
                   ("KEYWAY_ANGLE_DEG", 31.0),
                   ("BOLT_PC_R_PX", 66.0),
                   ("RING_PERIOD_PX", 20.0),
                   ("BRUSHED_SIGMA", 3.5)]
        for name, val in patches:
            old = getattr(config, name)
            try:
                setattr(config, name, val)
                self.assertNotEqual(pm._template_fingerprint(), fp0,
                                    f"指纹未感知 {name} 变化")
            finally:
                setattr(config, name, old)


if __name__ == "__main__":
    unittest.main(verbosity=2)
