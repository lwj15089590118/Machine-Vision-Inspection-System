# -*- coding: utf-8 -*-
"""
detect/detect.py —— 缺陷检测与 NG 判定模块（系统核心）
================================================================
职责：
    输入一帧灰度"相机画面"，先定位（locate 模块），再用三个互补的
    检测分支找缺陷，按 NG 判定规则表输出结果 JSON，并支持输出标注图。
    供 plc_link（检测主循环）、dashboard（看板）、run_batch（批量验收）调用。

检测分支设计（为什么是三个分支）：
    A. 基准比对分支（面域缺陷：划痕/污渍/孔异常）
       按定位结果把工件仿射矫正回基准位姿 → 与无缺陷基准图做增益归一化
       → absdiff → 自适应阈值 → 形态学去噪 → 连通域提取
       → 按面积/长宽比/所在半径分类。
       增益归一化：用盘面环形区域的中位数比值消除 ±20% 亮度增益；
       双边高斯模糊(3×3)：抑制 0.3~0.7px 矫正残差在孔边缘产生的细环假差。
    B. 外圆轮廓分支（崩边）
       键槽和崩边都是"基准比对不可靠"的区域（边缘对齐误差大），
       改为角向半径剖面：对矫正图逐角度求材料边界半径，与基准剖面
       比较，局部内凹 > 6px(0.6mm) 且不在键槽扇区 → 崩边。
    C. 几何测量分支（螺栓孔）
       霍夫圆在螺栓孔分度圆环带内检测 4 孔 → 孔位偏移(mm)与孔径
       偏差(mm)与公差比较；孔缺失（找不到圆且基准比对确认被填实）。

NG 判定规则表（config 中集中定义）：
    1) 任一缺陷连通域面积 > 30px²(0.3mm²)
    2) 外圆局部内凹 > 6px（崩边）
    3) 螺栓孔位偏移 > ±0.5mm
    4) 螺栓孔径偏差 > ±0.3mm
    5) 螺栓孔缺失
    6) 定位失败（安全策略：看不见工件按 NG 处理）

输出 JSON 字段：
    result(OK/NG) / defect_types / defects[{type,area_px,area_mm2,
    center_px,bbox_px}] / holes_found / hole_offsets_mm / hole_max_offset_mm /
    hole_max_dia_dev_mm / confidence / duration_ms / locate

命令行用法：
    python detect/detect.py --image data/images/frame_000001.png
        （若 data/truth 下有同名真值 JSON，自动附带真值对照）

程序接口：
    from detect.detect import inspect, draw_defects
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

# 允许直接 `python detect/detect.py` 运行：把项目根目录加入 sys.path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import cv2

import config
import part_model
from locate.locate import locate, draw_result as draw_locate
from part_model import make_reference, pose_matrix, apply_affine, \
    bolt_centers_canonical

# ---- 分支内部工程常数（公差类阈值在 config，此处为算法窗口/几何先验）----
RIM_KEYWAY_GUARD_DEG = 12.0   # 键槽扇区保护角（±12°内不算崩边）
RIM_RAY_R0, RIM_RAY_R1 = 60.0, 112.0   # 轮廓射线采样半径范围(px)
RIM_MIN_ARC_DEG = 2.0         # 崩边连续弧长下限
FACE_MASK_R_IN = 40.0         # 增益估计环内径（避开中心孔）
FACE_MASK_R_OUT = 85.0        # 增益估计环外径（避开暗环/键槽）
HOLE_MATCH_WIN_PX = 18.0      # 期望孔位与检测圆的匹配窗口（含偏移量上限）
HOLE_RING_BAND = (44.0, 82.0) # 螺栓孔霍夫搜索环带（基准系半径）
HOLE_EDGE_GUARD_PX = 3.0      # 未判异常孔的边缘保护圈外扩量（像素）：
#   亚像素矫正残差会在高对比孔缘产生细长月牙差分（批量调参实测为 OK 件
#   误报主源，质心集中在孔心 ±10px），在比对掩膜上把"孔半径+3px"圆域
#   抠掉即可在源头抑制；保护圈只抠未判异常的孔——缺失/偏移孔需保留
#   回填/新孔特征供佐证与分类。

# 基准资产缓存（基准图 / 模糊基准 / 基准材料掩膜 / 基准外圆剖面）
_REF_IMG = None
_REF_BLUR = None
_REF_FACE = None
_REF_RIM_PROFILE = None
_GRID_RR = None               # 每像素到基准中心(400,300)的距离


def _get_ref_assets() -> tuple:
    """惰性生成基准比对所需的全部基准资产（重复调用结果一致）"""
    global _REF_IMG, _REF_BLUR, _REF_FACE, _REF_RIM_PROFILE, _GRID_RR
    if _REF_IMG is None:
        ref, _mask = make_reference()
        _REF_IMG = ref
        _REF_BLUR = cv2.GaussianBlur(ref, (3, 3), 0).astype(np.float32)
        if _GRID_RR is None:
            yy, xx = np.mgrid[0:config.IMG_H, 0:config.IMG_W].astype(np.float32)
            _GRID_RR = np.hypot(xx - config.CANON_CENTER[0],
                                yy - config.CANON_CENTER[1])
        # 基准"亮盘面"掩膜（外圆轮廓分支的参照）
        _REF_FACE = _face_mask(ref)
        _REF_RIM_PROFILE = _rim_profile(_REF_FACE)
    return _REF_IMG, _REF_BLUR, _REF_FACE, _REF_RIM_PROFILE


def _grid_rr() -> np.ndarray:
    """每像素到基准中心的距离矩阵（缓存）"""
    global _GRID_RR
    if _GRID_RR is None:
        yy, xx = np.mgrid[0:config.IMG_H, 0:config.IMG_W].astype(np.float32)
        _GRID_RR = np.hypot(xx - config.CANON_CENTER[0],
                            yy - config.CANON_CENTER[1])
    return _GRID_RR


# ================================================================
# 预处理：仿射矫正回基准位姿
# ================================================================
def warp_to_canonical(img: np.ndarray, loc: dict) -> tuple:
    """
    按定位结果把测试图仿射矫正回基准位姿。
    返回 (矫正图 float32, 正向矩阵M)；M 用于把基准系坐标映射回原图坐标。
    """
    M = pose_matrix(loc["center_px"][0], loc["center_px"][1],
                    loc["angle_deg"], loc["scale"])
    M3 = np.vstack([M, [0.0, 0.0, 1.0]])
    Minv = np.linalg.inv(M3)[:2]
    warped = cv2.warpAffine(img, Minv, (config.IMG_W, config.IMG_H),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return warped.astype(np.float32), M


def _face_mask(warped: np.ndarray) -> np.ndarray:
    """
    矫正图中的"亮盘面"掩膜（外圆轮廓分支用）。
    只在工件邻域 ROI 内做 Otsu——矫正图四周边界是仿射填充的黑边，
    若对整幅图做 Otsu，阈值会落在"黑边 vs 皮带"之间导致皮带被误并入前景。
    """
    cx, cy = int(config.CANON_CENTER[0]), int(config.CANON_CENTER[1])
    m = 115                                        # ROI 半边长（> 外圆+余量）
    x0, y0 = cx - m, cy - m
    roi = warped[y0:y0 + 2 * m, x0:x0 + 2 * m].astype(np.uint8)
    _thr, binary = cv2.threshold(roi, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    n, labels, stats, _c = cv2.connectedComponentsWithStats(binary, 8)
    mask_roi = np.zeros_like(binary)
    if n >= 2:
        face = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mask_roi = np.where(labels == face, 255, 0).astype(np.uint8)
    full = np.zeros(warped.shape, np.uint8)
    full[y0:y0 + 2 * m, x0:x0 + 2 * m] = mask_roi
    return full


def _rim_profile(face_mask: np.ndarray) -> np.ndarray:
    """
    外圆角向半径剖面：360 个 1° 扇区，每个扇区取材料掩膜沿射线的
    最大半径（外边界）。正常≈亮盘面半径，键槽/崩边处内凹。
    返回长度 360 的数组（下标=扇区号，0°=+x 方向，逆时针图像系）。
    """
    cx, cy = config.CANON_CENTER
    prof = np.zeros(360, np.float64)
    radii = np.arange(RIM_RAY_R0, RIM_RAY_R1, 0.5)
    for k in range(360):
        phi = math.radians(k)
        xi = np.round(cx + radii * math.cos(phi)).astype(int)
        yi = np.round(cy + radii * math.sin(phi)).astype(int)
        on = face_mask[yi, xi] > 0
        prof[k] = radii[on].max() if on.any() else 0.0
    return prof


# ================================================================
# 分支 A：基准比对（面域缺陷）
# ================================================================
def diff_branch(warped: np.ndarray, exclude_pts: list = None,
                exclude_r: float = 20.0,
                hole_mask_pts: list = None) -> list:
    """
    基准比对：增益归一化 → 双边模糊 → absdiff → 自适应阈值 →
    形态学去噪 → 连通域提取 → 碎片聚类。
    exclude_pts:   需要在碎片阶段直接剔除的基准系坐标点列表（孔事件
                   位置的碎片不参与聚类，避免与路过的划痕碎片合并后被
                   整体跳过）；exclude_r 为剔除判定半径（px）。
    hole_mask_pts: 在比对掩膜上抠掉的圆域圆心列表（未判异常螺栓孔的
                   孔缘保护圈，半径 = BOLT_HOLE_R_PX + HOLE_EDGE_GUARD_PX），
                   用于在源头抑制孔缘亚像素矫正残差月牙——按像素抠掩膜
                   只损失缺陷与保护圈重叠的部分，不会像"按碎片质心剔除"
                   那样把恰好压在孔附近的大块真实缺陷整团误吞（批量调参
                   实测案例：距孔心仅 11px 的 807px² 污渍被整团剔除）。
    返回面域缺陷 blob 列表（基准系坐标）：
      {centroid(x,y), area_px, bbox(x,y,w,h), aspect, width_est, r_c}
    """
    ref, ref_blur, _, _ = _get_ref_assets()
    rr = _grid_rr()

    # 1) 增益归一化：盘面环形区（避开孔带/边缘）的中位数比值
    ring = (rr >= FACE_MASK_R_IN) & (rr <= FACE_MASK_R_OUT)
    g_ref = float(np.median(ref[ring]))
    g_tst = float(np.median(warped[ring]))
    gain = g_tst / max(g_ref, 1.0)
    test_norm = warped / max(gain, 0.2)

    # 2) 双边模糊（抑制亚像素矫正残差在锐利孔边缘的细环假差）
    test_blur = cv2.GaussianBlur(test_norm, (3, 3), 0)

    # 3) 比较掩膜：盘面内（避开外圆暗环 ±10px）；中心孔与未判异常
    #    螺栓孔统一用"半径+3px 边缘保护圈"从源头抑制矫正残差月牙，
    #    内边界收窄到 +3px 可显著减少中心孔邻域污渍落入盲区
    compare_mask = ((rr <= config.FLANGE_R_PX - 10.0) &
                    (rr >= config.CENTER_HOLE_R_PX)).astype(np.uint8)
    cv2.circle(compare_mask,
               (int(config.CANON_CENTER[0]), int(config.CANON_CENTER[1])),
               int(round(config.CENTER_HOLE_R_PX + HOLE_EDGE_GUARD_PX)),
               0, -1)
    if hole_mask_pts:
        guard_r = int(round(config.BOLT_HOLE_R_PX + HOLE_EDGE_GUARD_PX))
        for (px, py) in hole_mask_pts:
            cv2.circle(compare_mask, (int(round(px)), int(round(py))),
                       guard_r, 0, -1)

    # 4) absdiff + 自适应阈值（鲁棒统计：中位数+MAD）
    #    不能用 均值+kσ：缺陷自身的高差异像素会抬高档位，把弱缺陷
    #    （如细划痕）压到阈值以下（自污染）。MAD=绝对中位差，
    #    ×1.4826 折算为正态等效标准差，对离群缺陷像素不敏感。
    diff = cv2.absdiff(test_blur.astype(np.float32), ref_blur)
    vals = diff[compare_mask > 0]
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med))) * 1.4826
    thr = max(config.DIFF_ABS_FLOOR, med + config.DIFF_ADAPT_K * mad)
    binary = np.where((diff > thr) & (compare_mask > 0), 255, 0).astype(np.uint8)

    # 5) 形态学：闭运算连接划痕断裂段，开运算去孤立噪点
    k = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k,
                              iterations=config.MORPH_CLOSE_ITER)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, k)

    # 6) 连通域提取（原始碎片；孔事件位置的碎片直接剔除）
    n, labels, stats, cent = cv2.connectedComponentsWithStats(binary, 8)
    frags = []
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < config.MIN_BLOB_AREA_PX:
            continue                       # 小于最小面积按噪声丢弃
        fx, fy = float(cent[i][0]), float(cent[i][1])
        if exclude_pts and any(math.hypot(fx - px, fy - py) < exclude_r
                               for px, py in exclude_pts):
            continue                       # 孔事件特征碎片（含偏移后的新孔位，
                                             # 注入偏移上限 15px+测量余量）
        frags.append((i, fx, fy, area))
    if not frags:
        return []

    # 7) 碎片聚类：同一缺陷（划痕断裂段/污渍团块）的碎片质心距离 < 40px
    #    合并为一个整体再分类（单链聚类，并查集实现）
    parent = {f[0]: f[0] for f in frags}

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for ai in range(len(frags)):
        for bi in range(ai + 1, len(frags)):
            if math.hypot(frags[ai][1] - frags[bi][1],
                          frags[ai][2] - frags[bi][2]) < 40.0:
                parent[find(frags[ai][0])] = find(frags[bi][0])
    groups = {}
    for f in frags:
        groups.setdefault(find(f[0]), []).append(f)

    # 8) 每个聚类整体计算几何特征（最小外接矩形主轴/次轴/等效宽度）
    blobs = []
    cx, cy = config.CANON_CENTER
    for members in groups.values():
        gm = np.zeros(binary.shape, np.uint8)
        for (iid, _x, _y, _a) in members:
            gm[labels == iid] = 255
        area = float(np.count_nonzero(gm))
        conts, _ = cv2.findContours(gm, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)
        # 聚类内可能含多个不连通碎片：最小外接矩形须覆盖全部碎片
        (_rx, _ry), (rw, rh), _ra = cv2.minAreaRect(np.vstack(conts))
        maj, mn = max(rw, rh, 1.0), min(rw, rh, 1.0)
        mx = sum(m[1] * m[3] for m in members) / area
        my = sum(m[2] * m[3] for m in members) / area
        # 外接框 = 各碎片外接框的并集（用于标注与输出）
        bx0 = min(stats[m[0], cv2.CC_STAT_LEFT] for m in members)
        by0 = min(stats[m[0], cv2.CC_STAT_TOP] for m in members)
        bx1 = max(stats[m[0], cv2.CC_STAT_LEFT] +
                  stats[m[0], cv2.CC_STAT_WIDTH] for m in members)
        by1 = max(stats[m[0], cv2.CC_STAT_TOP] +
                  stats[m[0], cv2.CC_STAT_HEIGHT] for m in members)
        blobs.append({
            "centroid": (float(mx), float(my)),
            "area_px": area,
            "bbox": (float(bx0), float(by0),
                     float(bx1 - bx0), float(by1 - by0)),
            "aspect": float(maj / mn),
            "width_est": float(area / maj),   # 等效宽度=面积/主轴长
            "r_c": float(math.hypot(mx - cx, my - cy)),
        })
    return blobs


# ================================================================
# 分支 B：外圆轮廓（崩边）
# ================================================================
def rim_branch(warped: np.ndarray) -> list:
    """
    外圆轮廓分支：测试图材料边界剖面与基准剖面的差值，
    内凹 > RIM_DEVIATION_NG_PX 且不在键槽保护扇区、连续弧长达标 → 崩边。
    返回 [{angle_deg, depth_px, arc_deg, area_px}]（基准系坐标）。
    """
    _, _, _, ref_prof = _get_ref_assets()
    face = _face_mask(warped)
    prof = _rim_profile(face)
    dev = np.clip(ref_prof - prof, 0.0, None)    # 内凹量（外凸按 0 处理）

    chips = []
    k = 0
    while k < 360:
        if dev[k] > config.RIM_DEVIATION_NG_PX and not _in_keyway_sector(k):
            j = k
            while j < 360 and dev[j] > config.RIM_DEVIATION_NG_PX and \
                    not _in_keyway_sector(j):
                j += 1
            arc_deg = j - k
            if arc_deg >= RIM_MIN_ARC_DEG:
                seg_dev = dev[k:j]
                seg_prof = prof[k:j]           # 该段的测试剖面半径
                angle_mid = (k + j) / 2.0
                depth = float(seg_dev.max())
                # 面积估计：弧长(px)×深度(px)，弧长=角度×半径
                area = float(np.mean(seg_prof)) * math.radians(arc_deg) * depth
                chips.append({"angle_deg": angle_mid, "depth_px": depth,
                              "arc_deg": float(arc_deg),
                              "area_px": round(area, 1)})
            k = j
        else:
            k += 1
    return chips


def _in_keyway_sector(bin_idx: int) -> bool:
    """扇区是否落在键槽保护扇区（键槽方位±RIM_KEYWAY_GUARD_DEG，含环绕）

    键槽方位单源自 config.KEYWAY_ANGLE_DEG（经 part_model 的图像极角
    约定），默认 0° 时行为与旧版逐位一致。
    """
    a = (bin_idx - config.KEYWAY_ANGLE_DEG) % 360.0
    return a <= RIM_KEYWAY_GUARD_DEG or a >= 360.0 - RIM_KEYWAY_GUARD_DEG


# ================================================================
# 分支 C：几何测量（螺栓孔）
# ================================================================
def _bilinear_gray(img: np.ndarray, x: float, y: float):
    """灰度图双线性插值采样（img 可为 uint8 或 float）；越界返回 None"""
    h, w = img.shape[:2]
    x0, y0 = int(math.floor(x)), int(math.floor(y))
    if not (0 <= x0 < w - 1 and 0 <= y0 < h - 1):
        return None
    fx, fy = x - x0, y - y0
    a = float(img[y0, x0])
    b = float(img[y0, x0 + 1])
    c = float(img[y0 + 1, x0])
    d = float(img[y0 + 1, x0 + 1])
    return (a * (1.0 - fx) * (1.0 - fy) + b * fx * (1.0 - fy) +
            c * (1.0 - fx) * fy + d * fx * fy)


def _refine_circle(warped: np.ndarray, c0: tuple, r0: float) -> tuple:
    """
    亚像素圆参数精修：霍夫粗值 → 径向灰度剖面 50% 交叉点 → 最小二乘圆拟合。
    霍夫累加器只给整像素级（±1~2px）的圆心/半径，直接与 ±0.3mm
    （直径 3px）公差比较时量化噪声即超差——批量验收中这是 OK 件误报的
    第一大来源。本函数：
      1) 以霍夫圆为初值，沿 36 条射线在 [r0-4, r0+4]px 范围采样灰度；
      2) 每条射线找"孔内暗→盘面亮"的灰度 50% 交叉点，线性内插到亚像素；
      3) 对全部交叉点做 Kasa 代数圆拟合，输出精确 (cx, cy, r)。
    有效交叉点不足 12 个（污渍压孔/对比度异常）时退回霍夫粗值，保证鲁棒。
    返回 (cx, cy, r)，坐标与半径单位均为像素。
    """
    rs = np.arange(max(r0 - 4.0, 3.0), r0 + 4.0 + 1e-6, 0.25)
    pts = []
    for k in range(36):
        phi = 2.0 * math.pi * k / 36.0
        dx, dy = math.cos(phi), math.sin(phi)
        vals = []
        for r in rs:
            vals.append(_bilinear_gray(warped,
                                       c0[0] + r * dx, c0[1] + r * dy))
        if any(v is None for v in vals):
            continue
        vmin, vmax = min(vals), max(vals)
        if vmax - vmin < 40.0:                 # 该方向无足够明暗过渡
            continue
        thr = 0.5 * (vmin + vmax)
        for i in range(1, len(vals)):          # 由内向外找暗→亮 50% 交叉
            if vals[i] >= thr > vals[i - 1]:
                f = (thr - vals[i - 1]) / max(vals[i] - vals[i - 1], 1e-6)
                rr = rs[i - 1] + f * (rs[i] - rs[i - 1])
                pts.append((c0[0] + rr * dx, c0[1] + rr * dy))
                break
    if len(pts) < 12:
        return (float(c0[0]), float(c0[1]), float(r0))
    p = np.asarray(pts, np.float64)
    # Kasa 圆拟合：最小化 Σ(x²+y² − 2cx·x − 2cy·y + c)² 的线性最小二乘形式
    A = np.column_stack([2.0 * p[:, 0], 2.0 * p[:, 1], np.ones(len(p))])
    b = np.einsum("ij,ij->i", p, p)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = D, E
    return (float(cx), float(cy),
            float(math.sqrt(max(F + cx * cx + cy * cy, 1.0))))


def holes_branch(warped: np.ndarray) -> dict:
    """
    霍夫圆粗检 + 亚像素精修测量 4 个螺栓孔（矫正图分度圆环带内搜索），
    与基准孔位逐一匹配，输出：
      holes_found / hole_offsets_mm(4项,缺失记None) / hole_max_offset_mm /
      hole_max_dia_dev_mm / missing_idx / shifted_idx
    测量链路：霍夫圆给粗位置 → _refine_circle 用径向剖面把圆心/半径修到
    亚像素级 → 再与公差比较。直接用霍夫整数量化结果判定 ±0.5mm 孔位、
    ±0.3mm 孔径公差时量化噪声即超差（批量验收实测为 OK 件误报主源）。
    "孔被填实"佐证（filled_confirmed）统一由 inspect 主流程在基准比对
    分支完成后回填——此处不做，避免同一判定逻辑两处维护。
    """
    rr = _grid_rr()
    band = (rr >= HOLE_RING_BAND[0]) & (rr <= HOLE_RING_BAND[1])
    img8 = np.clip(warped, 0, 255).astype(np.uint8)
    roi = img8.copy()
    roi[~band] = 128                           # 环带外填中灰，防边界假圆

    circles = _hough_holes(roi, config.HOUGH_PARAM2)
    if len(circles) < 4:                       # 一次降阈值重试，抗漏检
        circles = _hough_holes(roi, config.HOUGH_PARAM2 - 6)
    if len(circles) == 0:
        circles = []

    expected = bolt_centers_canonical()
    used = [False] * len(circles)
    holes, missing_idx, shifted_idx = [], [], []
    offsets_mm = []
    max_off, max_dia = 0.0, 0.0
    for idx, (ex, ey) in enumerate(expected):
        best, best_d = None, 1e9
        for ci, (x, y, r) in enumerate(circles):
            if used[ci]:
                continue
            d = math.hypot(x - ex, y - ey)
            if d < best_d:
                best_d, best = d, ci
        if best is not None and best_d <= HOLE_MATCH_WIN_PX:
            used[best] = True
            x, y, r = circles[best]
            # 亚像素精修：匹配窗口判断用霍夫粗值即可，几何量必须精测
            fx, fy, fr = _refine_circle(warped, (x, y), r)
            off_px = math.hypot(fx - ex, fy - ey)
            off_mm = off_px * part_model.mm_per_pixel()
            dia_dev_mm = abs(2.0 * fr - 2.0 * config.BOLT_HOLE_R_PX) * \
                part_model.mm_per_pixel()
            offsets_mm.append(round(off_mm, 3))
            max_off = max(max_off, off_mm)
            max_dia = max(max_dia, dia_dev_mm)
            if off_mm > config.HOLE_POS_TOL_MM:
                shifted_idx.append(idx)
            holes.append({"index": idx, "detected_px": (round(fx, 1),
                                                        round(fy, 1)),
                          "offset_mm": round(off_mm, 3),
                          "dia_dev_mm": round(dia_dev_mm, 3)})
        else:
            # 未匹配到圆：filled_confirmed 由 inspect 主流程回填（此处
            # 基准比对分支尚未运行，拿不到 blobs）
            missing_idx.append(idx)
            offsets_mm.append(None)
            holes.append({"index": idx, "detected_px": None,
                          "offset_mm": None, "dia_dev_mm": None})

    return {"holes_found": 4 - len(missing_idx),
            "holes": holes,
            "missing_idx": missing_idx,
            "shifted_idx": shifted_idx,
            "hole_offsets_mm": offsets_mm,
            "hole_max_offset_mm": round(max_off, 3),
            "hole_max_dia_dev_mm": round(max_dia, 3)}


def _hough_holes(roi: np.ndarray, param2: float) -> list:
    """霍夫圆检测（参数来自 config），返回 [(x, y, r), ...]"""
    found = cv2.HoughCircles(roi, cv2.HOUGH_GRADIENT,
                             dp=1,
                             minDist=config.HOUGH_MIN_DIST,
                             param1=config.HOUGH_PARAM1,
                             param2=param2,
                             minRadius=config.HOUGH_MIN_R,
                             maxRadius=config.HOUGH_MAX_R)
    if found is None:
        return []
    return [(float(x), float(y), float(r)) for x, y, r in found[0]]


# ================================================================
# 汇总判定
# ================================================================
def inspect(img: np.ndarray, loc: dict = None) -> dict:
    """
    完整检测：定位（可用外部传入的定位结果）→ 三分支检测 → NG 判定。
    返回结果字典（见文件头"输出 JSON 字段"）。
    """
    t0 = time.perf_counter()
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ---- 定位（失败按 NG 安全策略）----
    if loc is None:
        loc = locate(gray)
    if not loc.get("ok"):
        return {"result": "NG", "defect_types": ["locate_fail"],
                "defects": [], "holes_found": 0,
                "hole_offsets_mm": [None] * 4, "hole_max_offset_mm": None,
                "hole_max_dia_dev_mm": None, "confidence": 0.0,
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
                "locate": loc, "reason": loc.get("reason", "定位失败")}

    # ---- 仿射矫正回基准位姿 ----
    warped, M = warp_to_canonical(gray, loc)

    # ---- 三分支检测 ----
    # 顺序：几何测量先行——孔事件位置作为排除点传给基准比对分支。
    # 未判缺失/偏移的孔：其邻域碎片直接剔除。批量实测（run_batch 调参）
    # 表明亚像素矫正残差会在孔边缘产生细长月牙状差分（质心集中在孔心
    # ±10px），是 OK 件误报的主要来源；孔周边事件一律归几何分支管辖。
    # 已判异常的孔不剔除：保留"回填/新孔"特征 blob，供填实佐证
    # （filled_confirmed）与后续分类守卫使用。
    holes = holes_branch(warped)
    canon_holes = bolt_centers_canonical()
    flagged = set(holes["missing_idx"]) | set(holes["shifted_idx"])
    # 未判缺失/偏移的孔：孔缘保护圈在比对掩膜上抠掉（源头抑制亚像素
    # 矫正残差月牙——批量实测 OK 件误报主源）。已判异常的孔不抠：
    # 保留"回填/新孔"特征 blob，供填实佐证（filled_confirmed）与
    # 后续分类守卫使用。
    hole_mask_pts = [canon_holes[i] for i in range(len(canon_holes))
                     if i not in flagged]
    blobs = diff_branch(warped, hole_mask_pts=hole_mask_pts)
    # 用基准比对结果回填"孔被填实"佐证（缺失判定的辅助信息）
    for h in holes["holes"]:
        if h["detected_px"] is None:
            h["filled_confirmed"] = any(
                abs(b["centroid"][0] - canon_holes[h["index"]][0]) < 14 and
                abs(b["centroid"][1] - canon_holes[h["index"]][1]) < 14 and
                b["area_px"] > 100 for b in blobs)
    chips = rim_branch(warped)

    # ---- blob 分类 ----
    # 孔位环带上的孔缺陷 blob 交给几何分支命名（避免重复计数/误分类）：
    # blob 靠近某个"基准孔位"且该孔已被判 缺失/偏移，或 blob 面积≈孔面积，
    # 都视为孔事件的特征，不进入划痕/污渍分类。
    cx, cy = config.CANON_CENTER
    flagged_holes = set(holes["missing_idx"]) | set(holes["shifted_idx"])
    defects = []
    blob_types = []
    for b in blobs:
        nearest_hole, nearest_d = None, 1e9
        for hi, (hx, hy) in enumerate(canon_holes):
            d = math.hypot(b["centroid"][0] - hx, b["centroid"][1] - hy)
            if d < nearest_d:
                nearest_d, nearest_hole = d, hi
        if nearest_d < 20.0 and nearest_hole in flagged_holes:
            continue                       # 已判缺失/偏移的孔：孔事件特征
        if (nearest_d < 15.0 and
                b["aspect"] < 2.0 and b["width_est"] > 10.0 and
                180.0 < b["area_px"] < 700.0):
            continue                       # 圆盘状孔残留（几何分支口径），
                                             # 细长划痕即使路过孔位也不误吞
        if b["r_c"] > config.FLANGE_R_PX - 18.0 and b["area_px"] > 40:
            dtype = "chip"                 # 面域分支兜底识别的崩边残留
        elif (b["aspect"] >= config.CLASS_SCRATCH_ASPECT and
              b["width_est"] <= config.CLASS_SCRATCH_W_MAX_PX):
            dtype = "scratch"              # 细长（长宽比大且等效宽度≤6px）
        else:
            dtype = "stain"
        if dtype not in blob_types:
            blob_types.append(dtype)
        defects.append(_to_defect_entry(dtype, b, M))

    for ch in chips:
        if "chip" not in blob_types:
            blob_types.append("chip")
        px = (cx + config.FLANGE_R_PX * math.cos(math.radians(ch["angle_deg"])),
              cy + config.FLANGE_R_PX * math.sin(math.radians(ch["angle_deg"])))
        op = apply_affine(M, [px])[0]
        defects.append({"type": "chip",
                        "area_px": ch["area_px"],
                        "area_mm2": round(part_model.px_area_to_mm2(
                            ch["area_px"]), 2),
                        "center_px": [round(float(op[0]), 1),
                                      round(float(op[1]), 1)],
                        "bbox_px": None})

    if holes["missing_idx"]:
        canon_holes = bolt_centers_canonical()
        for idx in holes["missing_idx"]:
            if "bolt_missing" not in blob_types:
                blob_types.append("bolt_missing")
            op = apply_affine(M, [canon_holes[idx]])[0]
            defects.append({"type": "bolt_missing",
                            # 孔缺失是"该有而没有"的事件，无真实面积——
                            # 不伪造 area_px（314=π·10² 的假圆面积，曾误导
                            # 下游把测量值当真值）
                            "center_px": [round(float(op[0]), 1),
                                          round(float(op[1]), 1)],
                            "bbox_px": None})
    if holes["shifted_idx"]:
        for idx in holes["shifted_idx"]:
            if "bolt_shift" not in blob_types:
                blob_types.append("bolt_shift")
            h = holes["holes"][idx]
            op = apply_affine(M, [h["detected_px"]])[0]
            defects.append({"type": "bolt_shift",
                            "center_px": [round(float(op[0]), 1),
                                          round(float(op[1]), 1)],
                            "offset_mm": h["offset_mm"],
                            "bbox_px": None})

    # ---- NG 判定规则表 ----
    reasons = []
    big_blobs = [d for d in defects
                 if d["type"] in config.AREA_BLOB_TYPES and
                 d["area_px"] > config.DEFECT_AREA_NG_PX]
    if big_blobs:
        reasons.append(f"缺陷面积超阈值({len(big_blobs)}处)")
    if holes["missing_idx"]:
        reasons.append(f"孔缺失{holes['missing_idx']}")
    if holes["hole_max_offset_mm"] > config.HOLE_POS_TOL_MM:
        reasons.append(f"孔位偏移{holes['hole_max_offset_mm']}mm")
    if holes["hole_max_dia_dev_mm"] > config.HOLE_DIA_TOL_MM:
        reasons.append(f"孔径偏差{holes['hole_max_dia_dev_mm']}mm")
    result = "NG" if reasons else "OK"

    # ---- 置信度（定位质量 × 角度相关 × 孔检出率的加权）----
    hole_score = holes["holes_found"] / 4.0
    conf = round(0.35 * loc.get("match_score", 0.0) +
                 0.35 * max(loc.get("angle_score", 0.0), 0.0) +
                 0.30 * hole_score, 3)

    return {"result": result,
            "defect_types": blob_types,
            "defects": defects,
            "holes_found": holes["holes_found"],
            "hole_offsets_mm": holes["hole_offsets_mm"],
            "hole_max_offset_mm": holes["hole_max_offset_mm"],
            "hole_max_dia_dev_mm": holes["hole_max_dia_dev_mm"],
            "confidence": conf,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
            "locate": loc,
            "reasons": reasons}


def _to_defect_entry(dtype: str, b: dict, M: np.ndarray) -> dict:
    """把基准系 blob 转换为输出条目（坐标映射回原图）"""
    op = apply_affine(M, [b["centroid"]])[0]
    x, y, w, h = b["bbox"]
    corners = apply_affine(M, [(x, y), (x + w, y), (x + w, y + h), (x, y + h)])
    return {"type": dtype,
            "area_px": round(b["area_px"], 1),
            "area_mm2": round(part_model.px_area_to_mm2(b["area_px"]), 2),
            "center_px": [round(float(op[0]), 1), round(float(op[1]), 1)],
            "bbox_px": [[round(float(px), 1), round(float(py), 1)]
                        for px, py in corners]}


# ================================================================
# 标注图（看板/调试）
# ================================================================
def draw_defects(img: np.ndarray, loc: dict, result: dict) -> np.ndarray:
    """
    输出标注图：定位框（绿圆+十字+角度）+ 缺陷红框 + 孔位标记 + 结论文字。
    """
    if loc.get("ok"):
        canvas = draw_locate(img if img.ndim == 2 else
                             cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), loc)
    else:
        canvas = img.copy() if img.ndim == 3 else \
            cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for d in result.get("defects", []):
        if d.get("bbox_px"):
            pts = np.array(d["bbox_px"], np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], True, (0, 0, 255), 2)
        elif d.get("center_px"):
            p = (int(d["center_px"][0]), int(d["center_px"][1]))
            cv2.drawMarker(canvas, p, (0, 0, 255), cv2.MARKER_TILTED_CROSS,
                           16, 2)
            cv2.putText(canvas, d["type"], (p[0] + 8, p[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    color = (0, 0, 255) if result["result"] == "NG" else (0, 200, 0)
    text = f"{result['result']} conf={result['confidence']:.2f} " \
           f"{','.join(result['defect_types']) or 'CLEAN'} " \
           f"{result['duration_ms']:.0f}ms"
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(canvas, text, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                color, 2)
    return canvas


# ================================================================
# 命令行入口
# ================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="缺陷检测与 NG 判定（单张演示，批量验收用 run_batch.py）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image", type=Path, required=True, help="图像路径")
    ap.add_argument("--no-annot", action="store_true", help="不保存标注图")
    args = ap.parse_args()
    config.ensure_dirs()

    gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        sys.exit(f"无法读取图像: {args.image}")
    result = inspect(gray)
    print(json.dumps({k: v for k, v in result.items() if k != "locate"},
                     ensure_ascii=False, indent=2))

    # 若存在同名真值则打印对照（便于演示）
    truth_path = config.TRUTH_DIR / (args.image.stem + ".json")
    if truth_path.exists():
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        truth_types = sorted({d["type"] for d in truth["defects"]})
        det_types = sorted(set(result["defect_types"]))
        print(f"真值对照: 真值NG={truth['is_ng']} 真值类型={truth_types} | "
              f"检测={result['result']} 检测类型={det_types}")

    if not args.no_annot:
        out = config.ANNOT_DIR / ("insp_" + args.image.name)
        cv2.imwrite(str(out), draw_defects(gray, result["locate"], result))
        print(f"标注图已保存: {out}")


if __name__ == "__main__":
    main()
