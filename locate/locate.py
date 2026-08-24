# -*- coding: utf-8 -*-
"""
locate/locate.py —— 工件定位模块（模板匹配 + 精修）
================================================================
职责：
    输入一帧灰度"相机画面"，输出工件位姿：中心 (x,y)、旋转角 θ、缩放 s。
    供 inspect（仿射矫正基准比对）、plc_link（写回定位寄存器）、
    run_batch（定位误差统计）调用。

定位流水线（粗→精三级，与真实工程做法一致）：
    1. 粗定位：cv2.matchTemplate（TM_CCOEFF_NORMED）+ 基准模板
       → 工件中心粗略位置（圆盘特征对 ±30° 旋转和 0.9~1.1 缩放不敏感，
         归一化相关分数仍能给出可靠峰位）；
    2. 中心/缩放精修：粗位 ROI 内 Otsu 二值化提取工件盘面 → 凸包填充
       （补回孔/键槽/崩边等凹缺）→ 质心=精确中心，最小外接矩形尺寸
       → 等效半径 → 缩放 s = r / 基准半径；
       说明：圆形工件的最小外接矩形角度无意义（近正方形），角度必须
       另行解算——这正是引入键槽特征的原因；
    3. 角度解算：
       3.1 粗角度：把 ROI 旋转到各候选角（-46°~+46°，步进 2°），
           与基准盘面小图做掩膜归一化互相关，取相关峰（键槽+孔系
           共同贡献，旋转对称性被键槽破坏，峰唯一）；
       3.2 精角度：键槽处"外圆暗环(灰度≈80×增益)"被键槽露出的
           皮带(灰度≈110×增益)打断——在预测方位 ±22° 窗口内做
           角向灰度剖面，Otsu 分离"提亮段"，其加权质心方位即键槽
           方向，精度可达 0.1° 级。崩边与键槽相距 ≥25°，不会入窗。

角度约定（与 simulator 真值严格一致）：
    θ 为 getRotationMatrix2D 意义下的旋转角；键槽图像坐标方位角 φ 与
    θ 互为相反数（φ = -θ），代码中统一用 φ=-θ 换算。

命令行用法：
    python locate/locate.py --image data/images/frame_000001.png   # 单张
    python locate/locate.py --image-dir data/images --truth-dir data/truth
                                                                   # 批量误差统计

程序接口：
    from locate.locate import locate, batch_locate, draw_result
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import cv2

# 允许直接 `python locate/locate.py` 运行：把项目根目录加入 sys.path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

# ================================================================
# 模板与基准小图缓存（首次调用时生成，之后复用）
# ================================================================
_TEMPLATE = None      # 匹配模板（260×260，含 30px 皮带余量）
_REF_SMALL = None     # 基准盘面小图（124×124，用于角度相关）
_REF_MASK = None      # 小图上的盘面掩膜（只比较工件区域，屏蔽皮带）
_REF_FACE_R = None    # 基准图"亮盘面"等效半径（半径自校准用）
_REF_HALF = int(config.FLANGE_R_PX + 24)          # 基准裁剪半边长 = 124
_REF_MASK_R = (config.FLANGE_R_PX + 6) / 2.0     # 掩膜圆半径（缩小一倍后）= 53


def _get_assets() -> tuple:
    """惰性生成并缓存：匹配模板、基准盘面小图、盘面掩膜、基准亮盘面半径"""
    global _TEMPLATE, _REF_SMALL, _REF_MASK, _REF_FACE_R
    if _TEMPLATE is None:
        from part_model import make_template, make_reference
        _TEMPLATE = make_template()
        ref, _ = make_reference()
        # 基准图自校准：同一套掩膜算法在基准图上测得的"亮盘面"半径
        # （亮盘面不含外圆暗环，系统性地小于材料外缘半径，用同一把
        #   "尺子"测量基准与测试，系统偏差相互抵消）
        _, r_face, _, _ = extract_part(ref, config.CANON_CENTER)
        _REF_FACE_R = r_face
        cx, cy = int(config.CANON_CENTER[0]), int(config.CANON_CENTER[1])
        crop = ref[cy - _REF_HALF:cy + _REF_HALF,
                   cx - _REF_HALF:cx + _REF_HALF]
        _REF_SMALL = cv2.pyrDown(crop)            # 248×248 → 124×124
        h, w = _REF_SMALL.shape
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        _REF_MASK = np.hypot(xx - w / 2, yy - h / 2) <= _REF_MASK_R
    return _TEMPLATE, _REF_SMALL, _REF_MASK


# ================================================================
# 二级精修：工件掩膜 / 中心 / 缩放
# ================================================================
def extract_part(gray: np.ndarray, coarse_xy: tuple) -> tuple:
    """
    在粗定位中心的 ROI 内提取工件：
      Otsu 二值化(亮=盘面) → 形态学开噪 → 最大连通域 → 凸包填充
    返回 (中心(x,y), 等效半径px, 缩放, 材料掩膜(全图尺寸,255=材料))
    """
    r_roi = int(config.FLANGE_R_PX * 1.25) + 40   # ROI 半边长 ≈ 165
    cx0, cy0 = int(round(coarse_xy[0])), int(round(coarse_xy[1]))
    x0, x1 = max(0, cx0 - r_roi), min(gray.shape[1], cx0 + r_roi)
    y0, y1 = max(0, cy0 - r_roi), min(gray.shape[0], cy0 + r_roi)
    roi = gray[y0:y1, x0:x1]

    # 1) Otsu 自动阈值：盘面亮(~185×增益)、皮带暗(~110×增益)，分离稳定
    _thr, binary = cv2.threshold(roi, 0, 255,
                                 cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                              np.ones((5, 5), np.uint8))

    # 2) 最大连通域 = 工件盘面（画面中只有一个工件）
    n, labels, stats, _cent = cv2.connectedComponentsWithStats(binary, 8)
    if n < 2:
        raise RuntimeError("未在 ROI 中找到工件连通域")
    area = stats[1:, cv2.CC_STAT_AREA]
    face = (np.argmax(area) + 1)
    face_mask = np.where(labels == face, 255, 0).astype(np.uint8)

    # 3) 凸包填充：补回孔/键槽/崩边等凹缺，得到完整"材料圆盘"
    contours, _ = cv2.findContours(face_mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    hull = cv2.convexHull(contours[0])
    disk = np.zeros_like(face_mask)
    cv2.fillPoly(disk, [hull], 255)

    # 4) 中心 = 凸包盘面质心（亚像素）；半径/缩放 = 最小外接矩形等效边长
    m = cv2.moments(disk)
    center = (m["m10"] / m["m00"] + x0, m["m01"] / m["m00"] + y0)
    (rcx, rcy), (rw, rh), _r = cv2.minAreaRect(hull)
    radius = (rw + rh) / 4.0                      # 近正方形 → 等效半径
    scale = radius / config.FLANGE_R_PX

    # 5) 材料掩膜回贴到全图坐标（键槽角度精修要用）
    mask_full = np.zeros(gray.shape, np.uint8)
    hull_shift = hull + np.array([x0, y0])
    cv2.fillPoly(mask_full, [hull_shift], 255)
    return center, radius, scale, mask_full


# ================================================================
# 三级 a：角度粗解算（旋转归一化互相关）
# ================================================================
def _masked_ncc(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> float:
    """掩膜内归一化互相关（两边同时去均值，等价 TM_CCOEFF_NORMED）"""
    va = a[mask].astype(np.float64)
    vb = b[mask].astype(np.float64)
    sa, sb = va.std(), vb.std()
    if sa < 1e-6 or sb < 1e-6:
        return -1.0
    return float(np.mean((va - va.mean()) * (vb - vb.mean())) / (sa * sb))


def angle_coarse(gray: np.ndarray, center: tuple, radius: float) -> tuple:
    """
    角度粗解算：把以工件为中心的 ROI 旋转到各候选角，与基准盘面小图
    做掩膜归一化互相关，相关峰对应的旋转量即键槽方位。
    返回 (θ粗值deg, 相关峰分数)
    """
    _, ref_small, ref_mask = _get_assets()
    r_crop = int(round(radius)) + 24              # 与基准裁剪半径同构
    cx, cy = int(round(center[0])), int(round(center[1]))
    x0, x1 = max(0, cx - r_crop), min(gray.shape[1], cx + r_crop)
    y0, y1 = max(0, cy - r_crop), min(gray.shape[0], cy + r_crop)
    roi = gray[y0:y1, x0:x1]
    roi_small = cv2.pyrDown(roi)
    h, w = roi_small.shape
    sc = (w / 2.0, h / 2.0)                       # 旋转中心=小图中心

    best_a, best_score = 0.0, -2.0
    for a_deg in range(-46, 47, 2):
        M = cv2.getRotationMatrix2D(sc, float(a_deg), 1.0)
        rot = cv2.warpAffine(roi_small, M, (w, h))
        if rot.shape != ref_small.shape:          # 统一到基准小图尺寸
            rot = cv2.resize(rot, (ref_small.shape[1],
                                   ref_small.shape[0]))
        score = _masked_ncc(rot, ref_small, ref_mask)
        if score > best_score:
            best_score, best_a = score, float(a_deg)
    # 最优旋转 a* 把键槽转到基准方位，θ = -a*（角度约定见文件头）
    return -best_a, best_score


# ================================================================
# 三级 b：角度精修（键槽角向灰度剖面）
# ================================================================
def angle_keyway_refine(gray: np.ndarray, center: tuple,
                        radius_true: float, theta_coarse: float) -> float:
    """
    键槽精修：外圆处正常材料是暗环(≈80×增益)，键槽挖空后露出皮带(≈110×增益)。
    在预测方位 ±22° 窗口内沿 [0.965,0.995]×R_true 径向带做角向剖面（0.5°/bin）。
    关键抗干扰设计：每个采样点用"同一图像行"上、材料圆外的皮带像素作光度
    参考（皮带行条带亮度只是 y 的函数，同行比值可将其严格抵消；再取 3 像素
    中值抗皮带纵向刮痕线）。正常段比值≈0.73，键槽段≈1.0，Otsu 分离后对
    超阈值部分做加权质心即键槽方位。
    调参记录（run_batch 批量实测，seed=7）：曾试验"两级细化"（第二级 ±6°/0.25°
    复测窗），因窗口完全落入键槽亮段内部（键槽角向半宽约 8°）、Otsu 退化为
    单总体导致质心失锁，P95 由 0.72° 恶化到 2.84° 并连带引发基准比对误报，
    已回退；也试验过"单窗 0.25° 加密 + 双线性插值 + 12 条径向采样"，P95
    持平（0.75°）无收益——说明误差瓶颈不在采样量化而在边缘模糊与 Otsu 阈值
    抖动。0.75° 对应分度圆切向偏差仅 64px×sin(0.75°)≈0.08mm（孔位公差的
    1/6），对检测判定无实际影响，不再追加复杂度。崩边与键槽相距 ≥25°，
    不会入窗。异常（对比度不足/无超阈值 bin）时回退粗角度。
    """
    # 键槽基准方位单源自 config.KEYWAY_ANGLE_DEG（经 part_model 渲染进
    # 基准图）；默认 0° 时以下两处换算与旧版逐位一致。
    phi_c = math.radians(config.KEYWAY_ANGLE_DEG)
    phi_pred = phi_c - math.radians(theta_coarse)  # 预测键槽图像方位角
    half_win = 22.0                               # 搜索窗口半宽（度）
    bin_deg = 0.5
    bins_deg = np.arange(-half_win, half_win + 1e-6, bin_deg)
    rads = np.linspace(radius_true * 0.965, radius_true * 0.995, 8)
    r_esc = radius_true + 7.0                     # 水平逃逸半径（材料圆外）
    cx, cy = center
    h, w = gray.shape

    profile = np.zeros(len(bins_deg), np.float64)
    for k, bd in enumerate(bins_deg):
        phi = phi_pred + math.radians(float(bd))
        ratios = []
        for r in rads:
            x = cx + r * math.cos(phi)
            y = cy + r * math.sin(phi)
            # 同行皮带参考点：沿水平方向逃逸出材料圆（条带亮度同行抵消）
            dy2 = (y - cy) ** 2
            dx_exit = math.sqrt(max(r_esc * r_esc - dy2, 1.0))
            xb = cx + (dx_exit if x >= cx else -dx_exit)
            xi, yi = int(round(x)), int(round(y))
            xbi, ybi = int(round(xb)), int(round(y))
            if not (0 <= xi < w and 0 <= yi < h and 1 <= xbi < w - 1
                    and 0 <= ybi < h):
                continue
            g_ref = float(np.median([gray[ybi, xbi - 1],
                                     gray[ybi, xbi],
                                     gray[ybi, xbi + 1]]))
            if g_ref < 5.0:
                continue
            ratios.append(min(max(gray[yi, xi] / g_ref, 0.4), 1.6))
        profile[k] = float(np.mean(ratios)) if ratios else 0.5

    if profile.max() - profile.min() < 0.06:      # 对比度不足（异常）
        return theta_coarse
    # Otsu 自适应阈值分离 暗环(~0.73) / 键槽(~1.0) 两个总体
    prof_u8 = np.clip((profile - 0.5) * 200.0, 0, 255).astype(
        np.uint8).reshape(1, -1)
    thr, _ = cv2.threshold(prof_u8, 0, 255,
                           cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    weights = np.clip(profile - (float(thr) / 200.0 + 0.5), 0.0, None)
    if weights.sum() < 1e-6:
        return theta_coarse
    delta = float(np.sum(weights * bins_deg) / weights.sum())
    psi = phi_pred + math.radians(delta)          # 键槽质心方位角
    return config.KEYWAY_ANGLE_DEG - math.degrees(psi)   # θ = φc − ψ


# ================================================================
# 主定位接口
# ================================================================
def locate(img: np.ndarray) -> dict:
    """
    单帧定位。img: 灰度图（uint8）。
    返回字典：
      ok / center_px [x,y] / center_mm [x,y] / angle_deg / scale /
      radius_px / match_score / angle_score / duration_ms
    """
    t0 = time.perf_counter()
    gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template, _, _ = _get_assets()

    # 1) 模板匹配粗定位
    res = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
    _minv, score, _minl, top_left = cv2.minMaxLoc(res)
    if score < config.TM_SCORE_MIN:
        return {"ok": False, "reason": f"模板匹配分数过低 {score:.3f}",
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2)}

    th, tw = template.shape
    coarse = (top_left[0] + tw / 2.0, top_left[1] + th / 2.0)

    # 2) 中心/缩放精修（亮盘面半径 → 基准自校准 → 材料外缘半径与缩放）
    try:
        center, r_face, _s_raw, mask = extract_part(gray, coarse)
    except RuntimeError as e:
        return {"ok": False, "reason": str(e),
                "duration_ms": round((time.perf_counter() - t0) * 1000, 2)}
    _template, _, _ = _get_assets()
    k_calib = config.FLANGE_R_PX / _REF_FACE_R   # 亮盘面半径→材料外缘半径
    radius = r_face * k_calib
    scale = radius / config.FLANGE_R_PX

    # 3) 角度粗解算 + 键槽剖面精修
    theta_c, angle_score = angle_coarse(gray, center, radius)
    theta = angle_keyway_refine(gray, center, radius, theta_c)

    # 4) 毫米坐标（相对基准位姿中心；标定结果存在则用实测像素当量）
    try:
        from calib.calibrate import px_to_mm
        mmx, mmy = px_to_mm(center[0], center[1])
        mmp = None
    except Exception:
        mmp = config.MM_PER_PIXEL
        mmx = (center[0] - config.CANON_CENTER[0]) * mmp
        mmy = (center[1] - config.CANON_CENTER[1]) * mmp

    return {
        "ok": True,
        "center_px": [round(center[0], 2), round(center[1], 2)],
        "center_mm": [round(mmx, 3), round(mmy, 3)],
        "angle_deg": round(theta, 3),
        "scale": round(scale, 4),
        "radius_px": round(radius, 2),
        "match_score": round(float(score), 4),
        "angle_score": round(float(angle_score), 4),
        "duration_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


# ================================================================
# 标注图（调试 / 看板用）
# ================================================================
def draw_result(img: np.ndarray, loc: dict) -> np.ndarray:
    """在原图上绘制定位结果：绿色外圆 + 十字中心 + 红色键槽方向线 + 文字"""
    canvas = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if not loc.get("ok"):
        cv2.putText(canvas, f"LOCATE FAIL: {loc.get('reason','')}",
                    (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        return canvas
    cx, cy = loc["center_px"]
    r = loc["radius_px"]
    cv2.circle(canvas, (int(cx), int(cy)), int(r), (0, 255, 0), 2)
    cv2.drawMarker(canvas, (int(cx), int(cy)), (0, 255, 0),
                   cv2.MARKER_CROSS, 14, 2)
    phi = math.radians(-loc["angle_deg"])         # 键槽图像方位角
    ex = int(cx + r * math.cos(phi))
    ey = int(cy + r * math.sin(phi))
    cv2.line(canvas, (int(cx), int(cy)), (ex, ey), (0, 0, 255), 2)
    cv2.putText(canvas,
                f"({loc['center_px'][0]:.1f},{loc['center_px'][1]:.1f}) "
                f"th={loc['angle_deg']:+.2f} s={loc['scale']:.3f}",
                (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
    return canvas


# ================================================================
# 批量定位与误差统计（对比合成真值）
# ================================================================
def _stats(vals: list) -> dict:
    """均值 / 标准差 / P95 / 最大绝对值（误差统计口径：最大值取绝对值）"""
    a = np.asarray(vals, np.float64)
    return {"mean": round(float(a.mean()), 4),
            "std": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0,
            "p95": round(float(np.percentile(np.abs(a), 95)), 4),
            "max": round(float(np.abs(a).max()), 4)}


def batch_locate(image_dir=None, truth_dir=None, limit: int = None,
                 verbose: bool = True) -> dict:
    """
    批量定位并与真值逐张对比。
    返回：{n, fail, center_err_px/mm, angle_err_deg, scale_err_pct,
           mean_ms, records:[...]}（记录含逐张误差，供 run_batch 复用）
    """
    image_dir = Path(image_dir) if image_dir else config.IMAGE_DIR
    truth_dir = Path(truth_dir) if truth_dir else config.TRUTH_DIR
    files = sorted(image_dir.glob("*.png"))
    if limit:
        files = files[:limit]
    if not files:
        raise FileNotFoundError(f"{image_dir} 下没有图像，请先运行 simulator/synth.py")

    s_mm = config.MM_PER_PIXEL                  # 误差换算用（像素当量）
    c_err, a_err, s_err, durs, records = [], [], [], [], []
    fail = 0
    for f in files:
        truth = json.loads(
            (truth_dir / (f.stem + ".json")).read_text(encoding="utf-8"))
        gray = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        loc = locate(gray)
        dur = loc["duration_ms"]
        if not loc["ok"]:
            fail += 1
            records.append({"file": f.name, "ok": False,
                            "reason": loc.get("reason"), "ms": dur})
            if verbose:
                print(f"  {f.name}  定位失败: {loc.get('reason')}")
            continue
        ex = loc["center_px"][0] - truth["center_px"][0]
        ey = loc["center_px"][1] - truth["center_px"][1]
        e_px = math.hypot(ex, ey)
        e_ang = (loc["angle_deg"] - truth["angle_deg"] + 180.0) % 360.0 - 180.0
        e_scl = (loc["scale"] - truth["scale"]) / truth["scale"] * 100.0
        c_err.append(e_px)
        a_err.append(e_ang)
        s_err.append(e_scl)
        durs.append(dur)
        records.append({"file": f.name, "ok": True,
                        "center_err_px": round(e_px, 3),
                        "center_err_mm": round(e_px * s_mm, 3),
                        "angle_err_deg": round(e_ang, 3),
                        "scale_err_pct": round(e_scl, 3), "ms": dur})
        if verbose:
            print(f"  {f.name}  中心误差 {e_px:5.2f}px({e_px*s_mm:.2f}mm)  "
                  f"角度误差 {e_ang:+6.2f}°  缩放误差 {e_scl:+5.2f}%  {dur:.0f}ms")

    out = {
        "n": len(files), "fail": fail,
        "center_err_px": _stats(c_err) if c_err else None,
        "center_err_mm": _stats([v * s_mm for v in c_err]) if c_err else None,
        "angle_err_deg": _stats(a_err) if a_err else None,
        "scale_err_pct": _stats(s_err) if s_err else None,
        "mean_ms": round(float(np.mean(durs)), 1) if durs else None,
        "records": records,
    }
    if verbose:
        print("-" * 64)
        print(f"批量定位: {out['n']} 张，失败 {fail} 张，"
              f"平均耗时 {out['mean_ms']}ms")
        if out["center_err_mm"]:
            c = out["center_err_mm"]
            a = out["angle_err_deg"]
            print(f"中心误差   : 均值 {c['mean']:.3f}mm  标准差 {c['std']:.3f}mm  "
                  f"P95 {c['p95']:.3f}mm  最大 {c['max']:.3f}mm")
            print(f"角度误差   : 均值 {a['mean']:.3f}°  标准差 {a['std']:.3f}°  "
                  f"P95 {a['p95']:.3f}°  最大 {a['max']:.3f}°")
            s = out["scale_err_pct"]
            print(f"缩放误差   : 均值 {s['mean']:+.2f}%  P95 {s['p95']:+.2f}%")
    return out


# ================================================================
# 命令行入口
# ================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="工件定位（模板匹配+精修）与批量误差统计",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--image", type=Path, default=None,
                    help="单张图像路径（指定则只处理这一张）")
    ap.add_argument("--image-dir", type=Path, default=config.IMAGE_DIR,
                    help="批量模式图像目录")
    ap.add_argument("--truth-dir", type=Path, default=config.TRUTH_DIR,
                    help="真值 JSON 目录")
    ap.add_argument("--limit", type=int, default=None, help="最多处理张数")
    ap.add_argument("--save-annot", type=int, default=0,
                    help="批量模式下保存前 N 张标注图到 data/annot/")
    args = ap.parse_args()
    config.ensure_dirs()

    if args.image:
        gray = cv2.imread(str(args.image), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            sys.exit(f"无法读取图像: {args.image}")
        loc = locate(gray)
        print(json.dumps(loc, ensure_ascii=False, indent=2))
        out = config.ANNOT_DIR / ("loc_" + args.image.name)
        cv2.imwrite(str(out), draw_result(gray, loc))
        print(f"标注图已保存: {out}")
        return

    print(f"批量定位: {args.image_dir}")
    result = batch_locate(args.image_dir, args.truth_dir, args.limit)
    for rec in result["records"][:max(args.save_annot, 0)]:
        if not rec["ok"]:
            continue
        gray = cv2.imread(str(Path(args.image_dir) / rec["file"]),
                          cv2.IMREAD_GRAYSCALE)
        loc = locate(gray)
        cv2.imwrite(str(config.ANNOT_DIR / ("loc_" + rec["file"])),
                    draw_result(gray, loc))
    if args.save_annot > 0:
        print(f"前 {args.save_annot} 张标注图已保存到 {config.ANNOT_DIR}")


if __name__ == "__main__":
    main()
