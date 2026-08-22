# -*- coding: utf-8 -*-
"""
simulator/synth.py —— 工件图像合成器（模拟工业相机俯拍画面）
================================================================
职责：
    1. 合成 800×600 灰度"相机画面"：传送带纹理背景 + 圆形法兰盘工件
       （外圆、键槽、中心孔、4 个螺栓孔、环带表面纹理）；
    2. 每张随机化：中心位置 ±100px、旋转 ±30°、缩放 0.9~1.1、
       亮度增益 ±20%、高斯噪声 σ=4；
    3. 支持注入缺陷：划痕 / 崩边 / 污渍 / 螺栓孔偏移 / 螺栓孔缺失；
    4. 同帧输出真值 JSON（中心、角度、缩放、缺陷列表及位置）。

合成原理（成像链路仿真）：
    传送带背景(随机纹理) + 法兰盘图层(基准位姿绘制) --仿射变换(平移/旋转/缩放)-->
    图层合成 --> ×固定光照不均匀场 --> ×整帧亮度增益 --> +高斯噪声 --> 8bit 画面
    缺陷在法兰盘图层上、仿射变换之前绘制，因此真值坐标 = 仿射变换(缺陷基准坐标)，
    与像素级完全一致。

命令行用法：
    python simulator/synth.py --count 20 --defects --seed 42   # 带缺陷，可复现
    python simulator/synth.py --count 5                        # 无缺陷
输出：
    data/images/frame_000001.png ...   合成画面
    data/truth/frame_000001.json ...   真值 JSON

程序接口（供 locate / inspect / run_batch / plc_server 调用）：
    from simulator.synth import synth_frame, make_reference, make_template
"""
import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2

# 允许直接 `python simulator/synth.py` 运行：把项目根目录加入 sys.path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

# ----------------------------------------------------------------
# 全局缓存网格（避免每帧重复构造 800×600 坐标矩阵，批量生成时提速）
# ----------------------------------------------------------------
_G_XX = None    # 每像素 x 坐标（float32）
_G_YY = None    # 每像素 y 坐标（float32）
_G_RR = None    # 每像素到基准中心(400,300)的距离（float32）
_G_ILLUM = None  # 固定光照增益场


def _grids():
    """惰性创建并缓存整幅画面的坐标网格与径向距离矩阵"""
    global _G_XX, _G_YY, _G_RR
    if _G_XX is None:
        yy, xx = np.mgrid[0:config.IMG_H, 0:config.IMG_W].astype(np.float32)
        _G_XX, _G_YY = xx, yy
        _G_RR = np.hypot(xx - config.CANON_CENTER[0],
                         yy - config.CANON_CENTER[1])
    return _G_XX, _G_YY, _G_RR


def illumination_field() -> np.ndarray:
    """
    固定光照增益场：模拟灯罩下左亮右暗、上暗下亮的轻微不均匀照度。
    注意：该场逐帧不变（真实产线相机与光源固定），属于"固定成像条件"，
    因此基准比对时它对基准图与测试图的影响完全相同，可被增益归一化抵消。
    """
    global _G_ILLUM
    if _G_ILLUM is None:
        xx, yy, _ = _grids()
        _G_ILLUM = (1.0
                    + 0.06 * (xx / config.IMG_W - 0.5)      # 水平方向 ±3%
                    + 0.04 * (0.5 - yy / config.IMG_H)      # 垂直方向 ±2%
                    ).astype(np.float32)
    return _G_ILLUM


def face_ring_value(rr: np.ndarray) -> np.ndarray:
    """
    盘面环带纹理灰度：半径的确定函数（车削纹理）。
    回填材料（处理"孔偏移/孔缺失"缺陷）时按该公式逐像素重建，
    保证与基准图的纹理严格一致，不引入额外差异。
    """
    return (config.FACE_BASE_GRAY
            + config.RING_AMP * np.sin(2.0 * np.pi * rr / config.RING_PERIOD_PX))


# ----------------------------------------------------------------
# 传送带背景
# ----------------------------------------------------------------
def make_belt(rng: np.random.Generator) -> np.ndarray:
    """
    生成传送带背景（float32 灰度）：
    基础灰度 + 行向明暗条带（皮带接缝/磨损）+ 若干条纵向刮痕。
    rng 不同则纹理不同（模拟皮带运动带来的背景变化）。
    """
    belt = np.full((config.IMG_H, config.IMG_W),
                   config.BELT_BASE_GRAY, np.float32)
    # 行向条带：沿 y 方向的正弦明暗变化，相位随机
    phase = rng.uniform(0.0, 2.0 * math.pi)
    rows = np.arange(config.IMG_H, dtype=np.float32)[:, None]
    belt *= 1.0 + 0.04 * np.sin(rows / 9.0 + phase)
    # 纵向刮痕：3~6 条略暗的斜线
    for _ in range(int(rng.integers(3, 7))):
        x0 = rng.uniform(0.0, config.IMG_W)
        drift = rng.uniform(-40.0, 40.0)
        gray = config.BELT_BASE_GRAY - rng.uniform(12.0, 26.0)
        cv2.line(belt,
                 (int(round(x0)), 0),
                 (int(round(x0 + drift)), config.IMG_H - 1),
                 float(gray), 1)
    return belt


# ----------------------------------------------------------------
# 法兰盘图层（基准位姿绘制，缺陷也在这一层注入）
# ----------------------------------------------------------------
def bolt_centers_canonical() -> list:
    """4 个螺栓孔在基准位姿下的圆心坐标 [(x,y)]×4"""
    cx, cy = config.CANON_CENTER
    pc = config.BOLT_PC_R_PX
    return [(cx + pc * math.cos(math.radians(a)),
             cy + pc * math.sin(math.radians(a)))
            for a in config.BOLT_ANGLES_DEG]


def keyway_center_canonical() -> tuple:
    """键槽中心在基准位姿下的坐标（用于真值输出与定位角度解算）"""
    cx, cy = config.CANON_CENTER
    r_mid = config.FLANGE_R_PX - config.KEYWAY_D_PX / 2.0
    return (cx + r_mid, cy)


def keyway_polygon() -> np.ndarray:
    """键槽矩形四角（int32，供 fillPoly 挖空材料）"""
    cx, cy = config.CANON_CENTER
    r_out = config.FLANGE_R_PX + 3.0          # 键槽向外略越过外圆，保证挖穿
    r_in = config.FLANGE_R_PX - config.KEYWAY_D_PX
    hw = config.KEYWAY_W_PX / 2.0
    return np.array([[cx + r_out, cy - hw],
                     [cx + r_out, cy + hw],
                     [cx + r_in, cy + hw],
                     [cx + r_in, cy - hw]], np.int32)


def _paint_face(part: np.ndarray, center: tuple, radius: float) -> None:
    """
    在指定圆域内按环带纹理公式回填盘面材料灰度（不含拉丝噪声，
    残差仅为 ±3 灰度级的噪声差，远低于缺陷判定阈值）。
    用途："螺栓孔偏移/缺失"缺陷需要先把原孔恢复成材料面。
    """
    h, w = part.shape
    cxp, cyp = center
    y0 = max(0, int(cyp - radius - 2))
    y1 = min(h, int(cyp + radius + 3))
    x0 = max(0, int(cxp - radius - 2))
    x1 = min(w, int(cxp + radius + 3))
    yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
    rr = np.hypot(xx - cxp, yy - cyp)
    disc = rr <= radius
    patch = part[y0:y1, x0:x1]
    patch[disc] = face_ring_value(rr[disc])


def _draw_defect(part: np.ndarray, mask: np.ndarray, d: dict) -> None:
    """
    在基准位姿的法兰盘图层上绘制单个缺陷。
    part: 盘图层灰度；mask: 材料区域掩膜（255=材料）。
    """
    t = d["type"]
    if t == "scratch":
        # 划痕：一条暗色随机折线，宽度 2~3px
        pts = np.round(np.asarray(d["points"], np.float32)).astype(np.int32)
        cv2.polylines(part, [pts], False,
                      config.FACE_BASE_GRAY - 75.0, d["width"])
    elif t == "chip":
        # 崩边：外圆处挖掉一块材料（盘图层清零 + 材料掩膜挖空 → 露出皮带）
        p = (int(round(d["center"][0])), int(round(d["center"][1])))
        cv2.circle(part, p, int(round(d["radius"])), 0.0, -1)
        cv2.circle(mask, p, int(round(d["radius"])), 0, -1)
    elif t == "stain":
        # 污渍：4~7 个相互重叠的暗色圆斑，构成不规则暗块
        for (qx, qy), rad in d["blobs"]:
            cv2.circle(part, (int(round(qx)), int(round(qy))),
                       int(round(rad)), config.FACE_BASE_GRAY - d["depth"], -1)
    elif t == "bolt_shift":
        # 螺栓孔偏移：原孔位置回填材料 → 在偏移后的新位置画孔
        _paint_face(part, d["orig"], config.BOLT_HOLE_R_PX + 1.5)
        p = (int(round(d["new"][0])), int(round(d["new"][1])))
        cv2.circle(part, p, int(config.BOLT_HOLE_R_PX), config.HOLE_GRAY, -1)
    elif t == "bolt_missing":
        # 螺栓孔缺失：原孔位置整块回填材料
        _paint_face(part, d["center"], config.BOLT_HOLE_R_PX + 1.5)
    else:
        raise ValueError(f"未知缺陷类型: {t}")


def build_part(defects: list) -> tuple:
    """
    在基准位姿下绘制法兰盘图层（含缺陷）。
    返回 (part float32 灰度层, mask uint8 材料掩膜)。
    表面拉丝噪声使用固定种子 → 基准图与各帧纹理完全一致。
    """
    part = np.zeros((config.IMG_H, config.IMG_W), np.float32)
    mask = np.zeros((config.IMG_H, config.IMG_W), np.uint8)
    cc = (int(config.CANON_CENTER[0]), int(config.CANON_CENTER[1]))

    # 1) 材料区域 = 外圆整圆
    cv2.circle(mask, cc, int(config.FLANGE_R_PX), 255, -1)
    region = mask > 0

    # 2) 盘面 = 环带纹理 + 固定种子拉丝噪声
    _, _, rr = _grids()
    part[region] = face_ring_value(rr[region])
    tex_rng = np.random.default_rng(20260101)      # 固定种子：表面纹理逐帧一致
    tex = tex_rng.standard_normal((config.IMG_H, config.IMG_W)).astype(np.float32)
    part[region] += tex[region] * config.BRUSHED_SIGMA

    # 3) 外圆暗环（一圈倒角阴影，形成强边缘）
    cv2.circle(part, cc,
               int(config.FLANGE_R_PX - config.RIM_RING_W / 2),
               config.RIM_GRAY, config.RIM_RING_W)

    # 4) 中心孔与 4 个螺栓孔
    cv2.circle(part, cc, int(config.CENTER_HOLE_R_PX), config.HOLE_GRAY, -1)
    for (bx, by) in bolt_centers_canonical():
        cv2.circle(part, (int(round(bx)), int(round(by))),
                   int(config.BOLT_HOLE_R_PX), config.HOLE_GRAY, -1)

    # 5) 键槽：挖空材料，合成时露出皮带（形成用于角度定位的明暗缺口）
    poly = keyway_polygon()
    cv2.fillPoly(part, [poly], 0.0)
    cv2.fillPoly(mask, [poly], 0)

    # 6) 注入缺陷（基准坐标系，之后统一随图层一起做仿射变换）
    for d in defects:
        _draw_defect(part, mask, d)

    return part, mask


# ----------------------------------------------------------------
# 缺陷规划（只生成"参数"，不绘制；真值与绘制共用同一份参数）
# ----------------------------------------------------------------
def _plan_scratch_pts(rng: np.random.Generator) -> list:
    """规划划痕折线的基准坐标点列（保证落在盘面内）"""
    cx, cy = config.CANON_CENTER
    r0 = rng.uniform(20.0, config.FLANGE_R_PX - 45.0)
    a0 = rng.uniform(0.0, 2.0 * math.pi)
    x = cx + r0 * math.cos(a0)
    y = cy + r0 * math.sin(a0)
    direction = rng.uniform(0.0, 2.0 * math.pi)
    pts = [(x, y)]
    for _ in range(int(rng.integers(3, 6))):       # 3~5 段折线
        length = rng.uniform(25.0, 50.0)
        direction += rng.uniform(-math.pi / 4.0, math.pi / 4.0)
        x += length * math.cos(direction)
        y += length * math.sin(direction)
        r_now = math.hypot(x - cx, y - cy)
        if r_now > config.FLANGE_R_PX - 28.0:      # 越界则拉回盘面内
            a_now = math.atan2(y - cy, x - cx)
            x = cx + (config.FLANGE_R_PX - 28.0) * math.cos(a_now)
            y = cy + (config.FLANGE_R_PX - 28.0) * math.sin(a_now)
        pts.append((x, y))
    return pts


def plan_defects(rng: np.random.Generator) -> list:
    """
    随机规划 1~2 个类型不重复的缺陷（全部为基准坐标系参数）。
    返回缺陷参数字典列表，绘制与真值输出共用。
    """
    all_types = ["scratch", "chip", "stain", "bolt_shift", "bolt_missing"]
    n = int(rng.integers(1, 3))                    # 1~2 个缺陷
    types = list(rng.choice(all_types, size=n, replace=False))

    defects = []
    first_bolt_idx = None                          # 避免两种孔缺陷打在同一个孔上
    for t in types:
        if t == "scratch":
            defects.append({"type": "scratch",
                            "points": _plan_scratch_pts(rng),
                            "width": int(rng.integers(2, 4))})
        elif t == "chip":
            # 崩边：避开键槽方位 ±25°，防止与键槽特征混淆
            a = rng.uniform(25.0, 335.0)
            cx, cy = config.CANON_CENTER
            rc = config.FLANGE_R_PX - 2.0
            defects.append({"type": "chip",
                            "center": (cx + rc * math.cos(math.radians(a)),
                                       cy + rc * math.sin(math.radians(a))),
                            "radius": float(rng.uniform(14.0, 22.0))})
        elif t == "stain":
            cx, cy = config.CANON_CENTER
            r0 = rng.uniform(25.0, config.FLANGE_R_PX - 45.0)
            a0 = rng.uniform(0.0, 2.0 * math.pi)
            sc = (cx + r0 * math.cos(a0), cy + r0 * math.sin(a0))
            blobs = [((sc[0] + rng.uniform(-12.0, 12.0),
                       sc[1] + rng.uniform(-12.0, 12.0)),
                      float(rng.uniform(5.0, 12.0)))
                     for _ in range(int(rng.integers(4, 8)))]
            defects.append({"type": "stain",
                            "center": sc,
                            "blobs": blobs,
                            "depth": float(rng.uniform(40.0, 60.0))})
        elif t == "bolt_shift":
            idx = int(rng.integers(0, 4))
            if first_bolt_idx is None:
                first_bolt_idx = idx
            # 偏移量 0.8~1.5mm（超过 ±0.5mm 公差），方向随机
            mag_mm = rng.uniform(0.8, 1.5)
            ang = rng.uniform(0.0, 2.0 * math.pi)
            dx_px = mag_mm / config.MM_PER_PIXEL * math.cos(ang)
            dy_px = mag_mm / config.MM_PER_PIXEL * math.sin(ang)
            orig = bolt_centers_canonical()[idx]
            defects.append({"type": "bolt_shift",
                            "hole_index": idx,
                            "orig": orig,
                            "new": (orig[0] + dx_px, orig[1] + dy_px),
                            "shift_mm": [mag_mm * math.cos(ang),
                                         mag_mm * math.sin(ang)]})
        elif t == "bolt_missing":
            idx = int(rng.integers(0, 4))
            if first_bolt_idx is None:
                first_bolt_idx = idx
            elif idx == first_bolt_idx:
                idx = (idx + 2) % 4                # 错开已占用的孔
            defects.append({"type": "bolt_missing",
                            "hole_index": idx,
                            "center": bolt_centers_canonical()[idx]})
    return defects


# ----------------------------------------------------------------
# 位姿变换与坐标工具
# ----------------------------------------------------------------
def pose_matrix(cx: float, cy: float, angle_deg: float, scale: float) -> np.ndarray:
    """
    构造"基准位姿 → 实际位姿"的 2×3 仿射矩阵：
    先绕基准中心旋转+缩放，再平移到实际中心 (cx, cy)。
    """
    M = cv2.getRotationMatrix2D(config.CANON_CENTER, angle_deg, scale)
    M[0, 2] += cx - config.CANON_CENTER[0]
    M[1, 2] += cy - config.CANON_CENTER[1]
    return M


def apply_affine(M: np.ndarray, pts) -> np.ndarray:
    """把基准坐标点列 (N×2) 变换为实际画面像素坐标 (N×2)"""
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    return p @ M[:, :2].T + M[:, 2]


# ----------------------------------------------------------------
# 真值输出
# ----------------------------------------------------------------
def _f2(v: float) -> float:
    """float → 保留 2 位小数（并转为 Python 原生 float，保证可 JSON 序列化）"""
    return round(float(v), 2)


def _build_truth(cx: float, cy: float, angle: float, scale: float,
                 bright: float, defects: list, M: np.ndarray) -> dict:
    """组装单帧真值字典（缺陷坐标已变换为最终像素坐标）"""
    mm = config.MM_PER_PIXEL
    holes_px = apply_affine(M, bolt_centers_canonical())
    key_px = apply_affine(M, [keyway_center_canonical()])[0]

    truth = {
        "file": None,                              # 由调用方/CLI 回填
        "width": config.IMG_W,
        "height": config.IMG_H,
        "center_px": [_f2(cx), _f2(cy)],
        "center_mm": [_f2((cx - config.CANON_CENTER[0]) * mm),
                      _f2((cy - config.CANON_CENTER[1]) * mm)],
        "angle_deg": round(float(angle), 3),
        "scale": round(float(scale), 4),
        "brightness": round(float(bright), 3),
        "holes_px": [[_f2(x), _f2(y)] for x, y in holes_px],
        "keyway_center_px": [_f2(key_px[0]), _f2(key_px[1])],
        "defects": [],
        "is_ng": len(defects) > 0,
    }
    for d in defects:
        t = d["type"]
        out = {"type": t}
        if t == "scratch":
            pts = apply_affine(M, d["points"])
            out["points_px"] = [[_f2(x), _f2(y)] for x, y in pts]
            out["width_px"] = d["width"]
        elif t == "chip":
            c = apply_affine(M, [d["center"]])[0]
            out["center_px"] = [_f2(c[0]), _f2(c[1])]
            out["radius_px"] = _f2(d["radius"])
        elif t == "stain":
            c = apply_affine(M, [d["center"]])[0]
            out["center_px"] = [_f2(c[0]), _f2(c[1])]
            out["radius_px"] = _f2(max(r for _, r in d["blobs"]))
        elif t == "bolt_shift":
            nc = apply_affine(M, [d["new"]])[0]
            out["hole_index"] = d["hole_index"]
            out["shift_mm"] = [_f2(d["shift_mm"][0]), _f2(d["shift_mm"][1])]
            out["new_center_px"] = [_f2(nc[0]), _f2(nc[1])]
        elif t == "bolt_missing":
            c = apply_affine(M, [d["center"]])[0]
            out["hole_index"] = d["hole_index"]
            out["center_px"] = [_f2(c[0]), _f2(c[1])]
        truth["defects"].append(out)
    return truth


# ----------------------------------------------------------------
# 主合成接口
# ----------------------------------------------------------------
def synth_frame(rng: np.random.Generator, with_defects: bool = False,
                defect_rate: float = 1.0) -> tuple:
    """
    合成一帧"相机画面"。
    参数：
        rng           np.random.default_rng 实例（相同种子序列 → 整批可复现）
        with_defects  是否允许注入缺陷
        defect_rate   单帧注入缺陷的概率（0~1）
    返回：
        (img_uint8, truth_dict)
    """
    defect_rate = min(max(defect_rate, 0.0), 1.0)

    # 1) 位姿与成像参数随机化
    cx = config.CANON_CENTER[0] + rng.uniform(-config.POSE_JITTER_PX,
                                              config.POSE_JITTER_PX)
    cy = config.CANON_CENTER[1] + rng.uniform(-config.POSE_JITTER_PX,
                                              config.POSE_JITTER_PX)
    angle = rng.uniform(-config.ANGLE_JITTER_DEG, config.ANGLE_JITTER_DEG)
    scale = rng.uniform(*config.SCALE_RANGE)
    bright = rng.uniform(*config.BRIGHTNESS_RANGE)

    # 2) 缺陷规划
    defects = []
    if with_defects and rng.random() < defect_rate:
        defects = plan_defects(rng)

    # 3) 图层合成：皮带背景 + 仿射变换后的法兰盘图层
    belt = make_belt(rng)
    part, mask = build_part(defects)
    M = pose_matrix(cx, cy, angle, scale)
    part_w = cv2.warpAffine(part, M, (config.IMG_W, config.IMG_H),
                            flags=cv2.INTER_LINEAR)
    mask_w = cv2.warpAffine(mask, M, (config.IMG_W, config.IMG_H),
                            flags=cv2.INTER_NEAREST)
    img = np.where(mask_w > 0, part_w, belt)

    # 4) 成像链路：固定光照场 → 整帧亮度增益 → 高斯噪声 → 8bit 量化
    img = img * illumination_field() * bright
    img = img + rng.normal(0.0, config.NOISE_SIGMA,
                           size=img.shape).astype(np.float32)
    img_u8 = np.clip(img, 0.0, 255.0).astype(np.uint8)

    truth = _build_truth(cx, cy, angle, scale, bright, defects, M)
    return img_u8, truth


def make_reference() -> tuple:
    """
    生成基准图（基准位姿、无缺陷、无随机增益/噪声；皮带纹理用固定种子）。
    用途：locate 的匹配模板、inspect 的比对基准。重复调用结果完全一致。
    返回：(img_uint8, mask_uint8)   mask 为基准材料区域
    """
    rng = np.random.default_rng(20260822)          # 固定种子：基准图可复现
    belt = make_belt(rng)
    part, mask = build_part([])
    img = np.where(mask > 0, part, belt) * illumination_field()
    return np.clip(img, 0.0, 255.0).astype(np.uint8), mask


def make_template() -> np.ndarray:
    """从基准图裁出含 30px 余量的工件模板（用于 cv2.matchTemplate）"""
    ref, _ = make_reference()
    cx, cy = int(config.CANON_CENTER[0]), int(config.CANON_CENTER[1])
    m = int(config.FLANGE_R_PX + 30)
    return ref[cy - m:cy + m, cx - m:cx + m].copy()


# ----------------------------------------------------------------
# 命令行入口
# ----------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="工件图像合成器（模拟工业相机俯拍画面）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--count", type=int, default=10, help="合成张数")
    ap.add_argument("--defects", action="store_true", help="启用缺陷注入")
    ap.add_argument("--defect-rate", type=float, default=1.0,
                    help="每张图注入缺陷的概率（0~1）")
    ap.add_argument("--seed", type=int, default=None,
                    help="随机种子（固定后同参数重跑结果完全一致）")
    ap.add_argument("--out-dir", type=Path, default=config.IMAGE_DIR,
                    help="图像输出目录")
    ap.add_argument("--truth-dir", type=Path, default=config.TRUTH_DIR,
                    help="真值 JSON 输出目录")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.truth_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.seed is not None:
        print(f"随机种子 seed={args.seed}（同参数重跑结果完全一致）")

    stat_ng = 0
    type_count = {}
    for i in range(1, args.count + 1):
        img, truth = synth_frame(rng, with_defects=args.defects,
                                 defect_rate=args.defect_rate)
        stem = f"frame_{i:06d}"
        png_path = args.out_dir / f"{stem}.png"
        json_path = args.truth_dir / f"{stem}.json"
        cv2.imwrite(str(png_path), img)
        truth["file"] = png_path.name
        json_path.write_text(json.dumps(truth, ensure_ascii=False, indent=2),
                             encoding="utf-8")
        types = [d["type"] for d in truth["defects"]]
        for t in types:
            type_count[t] = type_count.get(t, 0) + 1
        if truth["is_ng"]:
            stat_ng += 1
        print(f"[{i:>{len(str(args.count))}}/{args.count}] {png_path.name}  "
              f"中心=({truth['center_px'][0]:.1f},{truth['center_px'][1]:.1f})  "
              f"角度={truth['angle_deg']:+7.2f}°  缩放={truth['scale']:.3f}  "
              f"亮度={truth['brightness']:.2f}  缺陷={types if types else '无'}")

    print("-" * 64)
    print(f"输出目录: {args.out_dir}")
    print(f"真值目录: {args.truth_dir}")
    print(f"合计 {args.count} 张：OK {args.count - stat_ng} 张 / NG {stat_ng} 张")
    if type_count:
        detail = "，".join(f"{k}×{v}" for k, v in sorted(type_count.items()))
        print(f"缺陷统计: {detail}")


if __name__ == "__main__":
    main()
