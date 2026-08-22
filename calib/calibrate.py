# -*- coding: utf-8 -*-
"""
calib/calibrate.py —— 相机标定模块（独立模块，仿真验证版）
================================================================
职责：
    1. 用"已知真实相机参数"生成 20 张带畸变的棋盘格图像（正向仿真成像）；
    2. cv2.findChessboardCorners 提取角点 → cv2.calibrateCamera 标定；
    3. 输出内参矩阵、畸变系数、重投影误差（验收门槛 ≤ 0.5 px），
       并与仿真真值对照，验证"标定流程"本身的正确性；
    4. 提供去畸变函数（undistort_image / undistort_points）；
    5. 像素当量标定：在标称工作距离下实测 mm_per_pixel，
       提供像素坐标 ↔ 毫米坐标换算函数（px_to_mm / mm_to_px）。

仿真成像链路（正向）：
    棋盘格平面 --位姿(旋转rvec/平移tvec)--> 三维点
        --针孔投影 K--> 理想象素坐标 --真实畸变系数--> 畸变图像
    即：真实图像 = remap(理想透视渲染图, 畸变逆映射)，角点真值可解析计算。
    标定时走正常逆向流程：检测角点 → calibrateCamera 拟合参数。

说明：
    主链路（simulator 合成画面）按"已校正的理想成像"仿真，不含镜头畸变，
    因此 locate / inspect 不需要去畸变预处理；本模块独立验证标定方法与
    像素当量，其结果保存在 data/calib/calibration.json 供全项目查询。

命令行用法：
    python calib/calibrate.py                 # 生成20张→标定→验证→保存结果
    python calib/calibrate.py --count 20 --seed 3
输出：
    data/calib/chess/board_01.png ...         合成棋盘格图像（带畸变）
    data/calib/calibration.json               标定结果
    data/calib/undistort_demo.png             去畸变前后对比图

程序接口：
    from calib.calibrate import (load_calibration, undistort_image,
                                 undistort_points, px_to_mm, mm_to_px,
                                 get_mm_per_pixel)
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import cv2

# 允许直接 `python calib/calibrate.py` 运行：把项目根目录加入 sys.path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config

# ================================================================
# 一、仿真"真实相机"参数（生成图像用，标定前不可见）
# ================================================================
TRUE_FX = TRUE_FY = 800.0                  # 焦距（像素）
TRUE_CX, TRUE_CY = 400.0, 300.0            # 主点（画面中心）
TRUE_DIST = np.array([-0.06, 0.012,        # k1, k2（径向）
                      0.0006, -0.0005,     # p1, p2（切向）
                      0.0005], np.float64)  # k3（径向）

# 标定板参数：9×6 内角点，格边长 5 mm
BOARD_INNER = (9, 6)
SQUARE_MM = 5.0
BOARD_W_MM = (BOARD_INNER[0] - 1) * SQUARE_MM   # 角点跨度 40 mm
BOARD_H_MM = (BOARD_INNER[1] - 1) * SQUARE_MM   # 角点跨度 25 mm
SQUARE_PX_FLAT = 40                             # 平面板渲染时格边长（像素）
FLAT_SCALE = SQUARE_PX_FLAT / SQUARE_MM         # 平面图比例尺（像素/毫米）=8
RMS_GATE_PX = 0.5                               # 重投影误差验收门槛

_CHESS_DIR = config.CALIB_DIR / "chess"         # 棋盘格图像输出目录


# ================================================================
# 二、正向成像仿真：棋盘格生成
# ================================================================
def make_flat_board() -> np.ndarray:
    """
    生成平面棋盘格图（10×7 个方格，含一圈边框，白色为底）。
    内角点 (i,j) 位于平面图坐标 (SQUARE_PX_FLAT*(i+1), SQUARE_PX_FLAT*(j+1))。
    """
    cols, rows = BOARD_INNER[0] + 1, BOARD_INNER[1] + 1
    w, h = cols * SQUARE_PX_FLAT, rows * SQUARE_PX_FLAT
    board = np.full((h, w), 255, np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:               # 左上角起黑白交替
                board[r * SQUARE_PX_FLAT:(r + 1) * SQUARE_PX_FLAT,
                      c * SQUARE_PX_FLAT:(c + 1) * SQUARE_PX_FLAT] = 0
    return board


def object_points_mm() -> np.ndarray:
    """
    标定板内角点的三维坐标（Z=0 平面，单位 mm，行优先排列）。
    注意：平面板最外圈是边框方格，内角点 (i,j) 位于平面图第 (i+1,j+1) 格
    交点，因此物点坐标从 (5,5) 起算，与渲染图严格对齐。
    """
    objp = np.zeros((BOARD_INNER[0] * BOARD_INNER[1], 3), np.float32)
    idx = 0
    for j in range(BOARD_INNER[1]):            # 行
        for i in range(BOARD_INNER[0]):        # 列
            objp[idx] = ((i + 1) * SQUARE_MM, (j + 1) * SQUARE_MM, 0.0)
            idx += 1
    return objp


def true_camera_matrix() -> np.ndarray:
    """仿真真相机内参矩阵 K"""
    return np.array([[TRUE_FX, 0.0, TRUE_CX],
                     [0.0, TRUE_FY, TRUE_CY],
                     [0.0, 0.0, 1.0]], np.float64)


def _distort_normalized(xn: np.ndarray, yn: np.ndarray,
                        dist: np.ndarray) -> tuple:
    """
    按OpenCV畸变模型对"归一化坐标"施加正向畸变。
    xn/yn: 归一化平面坐标（可以是任意形状的数组）
    """
    k1, k2, p1, p2, k3 = (float(v) for v in dist[:5])
    r2 = xn * xn + yn * yn
    radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = xn * radial + 2.0 * p1 * xn * yn + p2 * (r2 + 2.0 * xn * xn)
    yd = yn * radial + p1 * (r2 + 2.0 * yn * yn) + 2.0 * p2 * xn * yn
    return xd, yd


def _invert_distortion(xd: np.ndarray, yd: np.ndarray,
                       dist: np.ndarray, iters: int = 15) -> tuple:
    """
    数值求逆：给定畸变后的归一化坐标，迭代解出畸变前（理想）坐标。
    原理：寻找 xi 满足 distort(xi) = xd，用不动点迭代
          xi ← xi − (distort(xi) − xd)，弱畸变下收敛迅速。
    """
    xi, yi = xd.copy(), yd.copy()
    for _ in range(iters):
        dx, dy = _distort_normalized(xi, yi, dist)
        xi = xi - (dx - xd)
        yi = yi - (dy - yd)
    return xi, yi


def distort_image(ideal: np.ndarray, K: np.ndarray,
                  dist: np.ndarray) -> np.ndarray:
    """
    对"理想成像图"施加镜头畸变，返回畸变图。
    方法：对输出图每个像素 p_d，数值反解其理想来源 p_i（畸变逆映射），
    再 cv2.remap(理想图, 逆映射) —— 与真实镜头成像过程一致。
    """
    h, w = ideal.shape[:2]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float64),
                         np.arange(h, dtype=np.float64))
    xd = (xs - cx) / fx
    yd = (ys - cy) / fy
    xi, yi = _invert_distortion(xd, yd, dist)
    map_x = (xi * fx + cx).astype(np.float32)
    map_y = (yi * fy + cy).astype(np.float32)
    return cv2.remap(ideal, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=128)


def generate_board_image(rng: np.random.Generator) -> tuple:
    """
    随机生成一张"带真实畸变"的棋盘格图像。
    返回 (图像uint8, 角点真值Nx2, 检测角点Nx2)；角点均为畸变图像坐标系。
    """
    K = true_camera_matrix()
    objp = object_points_mm()
    flat = make_flat_board()
    subpix_crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                   30, 1e-5)

    for _attempt in range(200):                # 随机位姿重试，直到成功
        # 随机位姿：姿态角 ±17°/±17°/±26°，距离 100~150mm，平移小幅偏移
        rvec = np.array([rng.uniform(-0.30, 0.30),
                         rng.uniform(-0.30, 0.30),
                         rng.uniform(-0.45, 0.45)], np.float64)
        tvec = np.array([-BOARD_W_MM / 2 + rng.uniform(-12.0, 12.0),
                         -BOARD_H_MM / 2 + rng.uniform(-8.0, 8.0),
                         rng.uniform(100.0, 150.0)], np.float64)

        # 1) 平面板 → 理想透视视图：
        #    H = K·[r1/s, r2/s, t]（s=平面图比例尺 px/mm，把平面像素换算成毫米）
        R, _ = cv2.Rodrigues(rvec)
        H = K @ np.column_stack([R[:, 0] / FLAT_SCALE,
                                 R[:, 1] / FLAT_SCALE, tvec])
        ideal = cv2.warpPerspective(flat, H, (config.IMG_W, config.IMG_H),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=128)

        # 2) 理想角点（无畸变投影）——检查是否都在画面内（留 30px 边距）
        corners_ideal = cv2.projectPoints(
            objp, rvec, tvec, K, np.zeros(5, np.float64)
        )[0].reshape(-1, 2)
        if (corners_ideal[:, 0].min() < 30 or
                corners_ideal[:, 0].max() > config.IMG_W - 30 or
                corners_ideal[:, 1].min() < 30 or
                corners_ideal[:, 1].max() > config.IMG_H - 30):
            continue

        # 3) 角点真值：同一投影 + 真实畸变系数
        corners_true = cv2.projectPoints(objp, rvec, tvec, K,
                                         TRUE_DIST)[0].reshape(-1, 2)

        # 4) 整图施加畸变 + 灰度高斯噪声（模拟传感器噪声）
        distorted = distort_image(ideal, K, TRUE_DIST)
        distorted = np.clip(
            distorted.astype(np.float64)
            + rng.normal(0.0, 1.5, distorted.shape), 0, 255).astype(np.uint8)

        # 5) 正常流程检测角点（自适应阈值 + 亚像素细化）
        found, corners_det = cv2.findChessboardCorners(
            distorted, BOARD_INNER,
            flags=(cv2.CALIB_CB_ADAPTIVE_THRESH
                   + cv2.CALIB_CB_NORMALIZE_IMAGE))
        if not found:
            continue
        corners_det = cv2.cornerSubPix(
            distorted, corners_det, (11, 11), (-1, -1), subpix_crit)
        return distorted, corners_true, corners_det.reshape(-1, 2)

    raise RuntimeError("200 次随机位姿均未生成可检测的棋盘格图像（不应发生）")


# ================================================================
# 三、标定流程
# ================================================================
def run_calibration(images_dir: Path = None, count: int = 20,
                    seed: int = 3) -> dict:
    """
    完整标定流程：生成（或复用）棋盘格图像 → 检测角点 → calibrateCamera
    → 与真值对照 → 实测像素当量 → 保存 calibration.json。
    返回标定结果字典。
    """
    config.ensure_dirs()
    images_dir = images_dir or _CHESS_DIR
    images_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    objp = object_points_mm()
    obj_points, img_points, det_errors = [], [], []
    print(f"生成并检测 {count} 张棋盘格图像（目录 {images_dir}）")
    for i in range(1, count + 1):
        img, corners_true, corners_det = generate_board_image(rng)
        png = images_dir / f"board_{i:02d}.png"
        cv2.imwrite(str(png), img)
        obj_points.append(objp)
        img_points.append(corners_det.astype(np.float32))
        # 检测角点 vs 解析真值的偏差（验证仿真与检测链路本身）
        det_err = np.linalg.norm(corners_det - corners_true, axis=1)
        det_errors.append(det_err.mean())
        print(f"  [{i:>2}/{count}] {png.name}  "
              f"角点检测误差均值 {det_err.mean():.3f}px（最大 "
              f"{det_err.max():.3f}px）")

    # ---- calibrateCamera：内参 / 畸变 / 重投影误差 ----
    # 注：OpenCV 5.0 默认返回 14 个畸变系数（含薄棱镜项），统一展平为一维
    rms, K_est, dist_est, _rvecs, _tvecs = cv2.calibrateCamera(
        obj_points, img_points, (config.IMG_W, config.IMG_H),
        None, None)
    dist_est = np.ravel(np.asarray(dist_est, np.float64))

    # ---- 与仿真真值对照 ----
    K_true = true_camera_matrix()
    print("\n===== 标定结果 vs 仿真真值 =====")
    print(f"重投影误差 RMS : {rms:.4f} px   （验收门槛 ≤ {RMS_GATE_PX} px）")
    print(f"fx: 标定 {K_est[0, 0]:8.2f} | 真值 {TRUE_FX:.2f} | "
          f"偏差 {K_est[0, 0] - TRUE_FX:+.2f}")
    print(f"fy: 标定 {K_est[1, 1]:8.2f} | 真值 {TRUE_FY:.2f} | "
          f"偏差 {K_est[1, 1] - TRUE_FY:+.2f}")
    print(f"cx: 标定 {K_est[0, 2]:8.2f} | 真值 {TRUE_CX:.2f} | "
          f"偏差 {K_est[0, 2] - TRUE_CX:+.2f}")
    print(f"cy: 标定 {K_est[1, 2]:8.2f} | 真值 {TRUE_CY:.2f} | "
          f"偏差 {K_est[1, 2] - TRUE_CY:+.2f}")
    for name, est, tru in zip(("k1", "k2", "p1", "p2", "k3"),
                               dist_est[:5], TRUE_DIST[:5]):
        print(f"{name}: 标定 {est:+.5f} | 真值 {tru:+.5f} | "
              f"偏差 {est - tru:+.5f}")

    if rms > RMS_GATE_PX:
        raise RuntimeError(f"重投影误差 {rms:.4f}px 超过门槛 {RMS_GATE_PX}px")

    # ---- 像素当量实测：正对标定板 @ 标称工作距离 ----
    mm_per_px = measure_mm_per_pixel(K_est, dist_est)
    print(f"\n像素当量实测  : {mm_per_px:.5f} mm/px "
          f"（config 设定 {config.MM_PER_PIXEL}，偏差 "
          f"{mm_per_px - config.MM_PER_PIXEL:+.5f}）")

    result = {
        "image_size": [config.IMG_W, config.IMG_H],
        "camera_matrix": [[float(v) for v in row] for row in K_est],
        "dist_coeffs": [float(v) for v in dist_est[:5]],
        "rms_reproj_px": round(float(rms), 4),
        "corner_detect_err_px": round(float(np.mean(det_errors)), 4),
        "mm_per_pixel": round(mm_per_px, 5),
        "working_distance_mm": round(K_est[0, 0] * config.MM_PER_PIXEL, 2),
        "board": {"inner_corners": list(BOARD_INNER),
                  "square_mm": SQUARE_MM},
        "num_images": count,
        "true_camera_matrix": [[float(v) for v in row] for row in K_true],
        "true_dist_coeffs": [float(v) for v in TRUE_DIST],
        "calibrated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config.CALIB_JSON.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"标定结果已保存: {config.CALIB_JSON}")
    return result


def measure_mm_per_pixel(K_est: np.ndarray, dist_est: np.ndarray) -> float:
    """
    像素当量标定：生成一张"正对相机(rvec=0)、位于标称工作距离"的棋盘格，
    用标定结果去畸变后检测角点，统计相邻角点平均间距（像素），
    mm_per_pixel = 格边长(mm) / 平均间距(px)。
    标称工作距离 Z = fx × config.MM_PER_PIXEL = 80mm，
    此时视场恰为 800px×0.1mm = 80mm，与主链路设定一致。
    """
    K = true_camera_matrix()
    Z = K[0, 0] * config.MM_PER_PIXEL          # 80 mm
    rvec = np.zeros(3, np.float64)
    tvec = np.array([-BOARD_W_MM / 2, -BOARD_H_MM / 2, Z], np.float64)
    objp = object_points_mm()

    R, _ = cv2.Rodrigues(rvec)
    H = K @ np.column_stack([R[:, 0] / FLAT_SCALE,
                             R[:, 1] / FLAT_SCALE, tvec])
    ideal = cv2.warpPerspective(make_flat_board(), H,
                                (config.IMG_W, config.IMG_H),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=128)
    distorted = distort_image(ideal, K, TRUE_DIST)

    # 用"标定得到的参数"去畸变（模拟真实使用流程）
    undist = cv2.undistort(distorted, K_est, dist_est)
    found, corners = cv2.findChessboardCorners(
        undist, BOARD_INNER,
        flags=(cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE))
    if not found:
        raise RuntimeError("像素当量标定图角点检测失败（不应发生）")
    corners = cv2.cornerSubPix(
        undist, corners, (11, 11), (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-5)
    ).reshape(-1, 2)

    # 相邻角点间距：水平方向（同行 i→i+1）与垂直方向（同列 j→j+1）
    w, h = BOARD_INNER
    dists = []
    for j in range(h):
        for i in range(w - 1):
            a = corners[j * w + i]
            b = corners[j * w + i + 1]
            dists.append(np.hypot(*(b - a)))
    for j in range(h - 1):
        for i in range(w):
            a = corners[j * w + i]
            b = corners[(j + 1) * w + i]
            dists.append(np.hypot(*(b - a)))
    return SQUARE_MM / float(np.mean(dists))


# ================================================================
# 四、标定结果加载与坐标换算（供其他模块调用）
# ================================================================
_CALIB_CACHE = None


def load_calibration() -> dict:
    """
    加载标定结果 JSON（模块级缓存）。
    若尚未标定，抛出异常并提示先运行 `python calib/calibrate.py`。
    """
    global _CALIB_CACHE
    if _CALIB_CACHE is None:
        if not config.CALIB_JSON.exists():
            raise FileNotFoundError(
                f"未找到标定结果 {config.CALIB_JSON}，"
                f"请先运行: python calib/calibrate.py")
        _CALIB_CACHE = json.loads(
            config.CALIB_JSON.read_text(encoding="utf-8"))
    return _CALIB_CACHE


def _K_of(calib: dict) -> np.ndarray:
    return np.asarray(calib["camera_matrix"], np.float64)


def undistort_image(img: np.ndarray, calib: dict = None) -> np.ndarray:
    """用标定结果对整幅图像去畸变"""
    calib = calib or load_calibration()
    return cv2.undistort(img, _K_of(calib),
                         np.asarray(calib["dist_coeffs"], np.float64))


def undistort_points(pts, calib: dict = None) -> np.ndarray:
    """
    对像素坐标点列 (N×2) 去畸变，返回去畸变后的像素坐标 (N×2)。
    """
    calib = calib or load_calibration()
    K = _K_of(calib)
    p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    out = cv2.undistortPoints(p, K,
                              np.asarray(calib["dist_coeffs"], np.float64), P=K)
    return out.reshape(-1, 2)


def get_mm_per_pixel(calib: dict = None) -> float:
    """返回标定实测的像素当量（mm/像素）"""
    calib = calib or load_calibration()
    return float(calib["mm_per_pixel"])


def px_to_mm(x_px: float, y_px: float, calib: dict = None) -> tuple:
    """
    像素坐标 → 毫米坐标（相对主点/光轴中心，X 向右、Y 向下为正）。
    输入可以是绝对像素坐标，输出为该点相对画面中心的毫米偏移。
    """
    calib = calib or load_calibration()
    K = _K_of(calib)
    s = get_mm_per_pixel(calib)
    return ((x_px - K[0, 2]) * s, (y_px - K[1, 2]) * s)


def mm_to_px(x_mm: float, y_mm: float, calib: dict = None) -> tuple:
    """毫米坐标 → 像素坐标（px_to_mm 的逆变换）"""
    calib = calib or load_calibration()
    K = _K_of(calib)
    s = get_mm_per_pixel(calib)
    return (x_mm / s + K[0, 2], y_mm / s + K[1, 2])


# ================================================================
# 五、命令行入口
# ================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description="相机标定（仿真验证版：合成畸变棋盘格 → 标定 → 验证）",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--count", type=int, default=20, help="棋盘格图像张数")
    ap.add_argument("--seed", type=int, default=3, help="随机种子")
    args = ap.parse_args()

    result = run_calibration(count=args.count, seed=args.seed)

    # 去畸变演示：畸变图 vs 去畸变图 并排保存
    first = (_CHESS_DIR / "board_01.png")
    if first.exists():
        distorted = cv2.imread(str(first), cv2.IMREAD_GRAYSCALE)
        fixed = undistort_image(distorted, result)
        demo = np.hstack([distorted, np.full((config.IMG_H, 4), 128, np.uint8),
                          fixed])
        demo_path = config.CALIB_DIR / "undistort_demo.png"
        cv2.imwrite(str(demo_path), demo)
        print(f"去畸变对比图已保存: {demo_path}（左:畸变图 右:去畸变图）")

    print("\n===== 验收结论 =====")
    ok_rms = result["rms_reproj_px"] <= RMS_GATE_PX
    print(f"[{'PASS' if ok_rms else 'FAIL'}] 重投影误差 "
          f"{result['rms_reproj_px']} px ≤ {RMS_GATE_PX} px")
    print(f"[{'PASS' if abs(result['mm_per_pixel'] - config.MM_PER_PIXEL) < 0.002 else 'FAIL'}] "
          f"像素当量实测 {result['mm_per_pixel']} ≈ 设定 {config.MM_PER_PIXEL}")
    if not ok_rms:
        sys.exit(1)


if __name__ == "__main__":
    main()
