# -*- coding: utf-8 -*-
"""
part_model.py —— 工件基准模型（生产期唯一的"工件事实来源"）
========================================================
职责：
    集中承载法兰盘工件的名义定义，供 定位(locate)、检测(inspect) 与
    合成器(simulator) 共同引用，使生产算法不再依赖测试数据生成器：
      1. 规范几何：螺栓孔分布 / 键槽几何 / 位姿矩阵与仿射坐标变换；
      2. 标准成像环境：固定光照增益场、传送带背景纹理（rng 参数化，
         黄金资产以固定种子取得可复现背景）；
      3. 名义外观：盘面环带纹理公式 + 基准位姿工件图层渲染
         （无缺陷、内部固定种子、确定性）；
      4. 黄金资产：基准图 make_reference（标准工位成像：皮带+光照+
         无缺陷件）与匹配模板 make_template，重复调用结果完全一致。

边界约定（详见 docs/adr/0001，ADR-0001 的代码体现）：
    - locate / inspect 一律从本模块获取几何期望值与基准资产；
    - simulator/synth 是本模块的「下游组合者」：在名义图层上叠加缺陷
      注入与随机化，合成测试帧；任何生产模块不得反向 import simulator。

使用示例：
    import part_model
    holes = part_model.bolt_centers_canonical()   # 规范孔位
    ref, mask = part_model.make_reference()       # 黄金基准图
"""
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import cv2

# 允许直接 `python part_model.py` 自检运行：把项目根目录加入 sys.path
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import config


# ================================================================
# 零、像素当量权威（px↔mm 换算的唯一入口）
# ================================================================
_MMP_CACHE = None    # 进程内缓存（标定结果静态，无需每帧重读）


def mm_per_pixel() -> float:
    """
    像素当量（mm/像素）唯一出处。优先级：新鲜标定 > config 设定值。
      - data/calib/calibration.json 存在且可解析 → 用标定实测值；
      - 否则回退 config.MM_PER_PIXEL（未标定的仿真/新产线场景）。
    locate 原有的 try-import calib 写法即此语义，现收敛为本函数；
    结果进程内缓存（标定文件不会在运行中变化）。
    """
    global _MMP_CACHE
    if _MMP_CACHE is None:
        try:
            from calib.calibrate import get_mm_per_pixel
            _MMP_CACHE = get_mm_per_pixel()
        except Exception:
            _MMP_CACHE = float(config.MM_PER_PIXEL)
    return _MMP_CACHE


def px_area_to_mm2(area_px: float) -> float:
    """像素面积 → 平方毫米（同一权威的平方，禁止使用点再写 ×0.01）"""
    return area_px * mm_per_pixel() ** 2


# ================================================================
# 一、全局缓存网格（避免每帧重复构造 800×600 坐标矩阵，批量生成时提速）
# ================================================================
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


# ================================================================
# 二、标准成像环境（固定条件：真实产线相机与光源固定不动）
# ================================================================
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


# ================================================================
# 三、规范几何（真值、定位角度解算、检测期望值的共同出处）
# ================================================================
def bolt_centers_canonical() -> list:
    """4 个螺栓孔在基准位姿下的圆心坐标 [(x,y)]×4"""
    cx, cy = config.CANON_CENTER
    pc = config.BOLT_PC_R_PX
    return [(cx + pc * math.cos(math.radians(a)),
             cy + pc * math.sin(math.radians(a)))
            for a in config.BOLT_ANGLES_DEG]


def keyway_center_canonical() -> tuple:
    """键槽中心在基准位姿下的坐标（用于真值输出与定位角度解算）

    方位由 config.KEYWAY_ANGLE_DEG 唯一决定（图像极角约定，
    与 bolt_centers_canonical 同口径）；默认 0° 时位于正东 (cx+r_mid, cy)。
    """
    cx, cy = config.CANON_CENTER
    a = math.radians(config.KEYWAY_ANGLE_DEG)
    r_mid = config.FLANGE_R_PX - config.KEYWAY_D_PX / 2.0
    return (cx + r_mid * math.cos(a), cy + r_mid * math.sin(a))


def keyway_polygon() -> np.ndarray:
    """键槽矩形四角（int32，供 fillPoly 挖空材料）

    矩形沿键槽方位 config.KEYWAY_ANGLE_DEG 展开：u 轴=键槽朝向，
    v 轴=垂直方向；方位为 0° 时与旧版逐位一致。
    """
    cx, cy = config.CANON_CENTER
    a = math.radians(config.KEYWAY_ANGLE_DEG)
    ca, sa = math.cos(a), math.sin(a)
    r_out = config.FLANGE_R_PX + 3.0          # 键槽向外略越过外圆，保证挖穿
    r_in = config.FLANGE_R_PX - config.KEYWAY_D_PX
    hw = config.KEYWAY_W_PX / 2.0

    def _pt(u: float, v: float) -> list:
        return [cx + u * ca - v * sa, cy + u * sa + v * ca]

    return np.array([_pt(r_out, -hw), _pt(r_out, hw),
                     _pt(r_in, hw), _pt(r_in, -hw)], np.int32)


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


# ================================================================
# 四、名义外观（车削纹理公式 + 无缺陷工件图层，全部确定性）
# ================================================================
def face_ring_value(rr: np.ndarray) -> np.ndarray:
    """
    盘面环带纹理灰度：半径的确定函数（车削纹理）。
    回填材料（处理"孔偏移/孔缺失"缺陷）时按该公式逐像素重建，
    保证与基准图的纹理严格一致，不引入额外差异。
    """
    return (config.FACE_BASE_GRAY
            + config.RING_AMP * np.sin(2.0 * np.pi * rr / config.RING_PERIOD_PX))


def build_part() -> tuple:
    """
    在基准位姿绘制**名义**法兰盘图层（无缺陷、确定性）。
    返回 (part float32 灰度层, mask uint8 材料掩膜)。
    表面拉丝噪声使用内部固定种子 → 逐次调用与各帧纹理完全一致；
    缺陷注入由 simulator/synth 在返回的图层上叠加（见其 _draw_defect）。
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

    return part, mask


# ================================================================
# 五、黄金资产（locate 的模板 / inspect 的比对基准，固定种子可复现）
# ================================================================
def make_reference() -> tuple:
    """
    生成基准图（基准位姿、无缺陷、无随机增益/噪声；皮带纹理用固定种子）。
    用途：locate 的匹配模板、inspect 的比对基准。重复调用结果完全一致。
    返回：(img_uint8, mask_uint8)   mask 为基准材料区域
    """
    rng = np.random.default_rng(20260822)          # 固定种子：基准图可复现
    belt = make_belt(rng)
    part, mask = build_part()
    img = np.where(mask > 0, part, belt) * illumination_field()
    return np.clip(img, 0.0, 255.0).astype(np.uint8), mask


def make_template() -> np.ndarray:
    """
    工件匹配模板（用于 cv2.matchTemplate）：基准图中心裁剪，余量
    config.TEMPLATE_MARGIN_PX（与缓存校验、参数指纹共用同一出处）。

    带落盘缓存：首次渲染后写入 config.TEMPLATE_PATH，并附带"几何+外观
    参数指纹" sidecar（template.meta.json）。后续进程指纹一致则直接读盘
    （PNG 无损，与现算逐像素相等）；任一影响模板像素的参数变化或文件
    损坏 → 指纹不符/校验失败，自动重生成。资产渲染逻辑本身变更时，
    手动递增 _ASSET_VERSION 使旧缓存失效。
    """
    fp = _template_fingerprint()
    cached = _load_cached_template(fp)
    if cached is not None:
        return cached

    ref, _ = make_reference()
    cx, cy = int(config.CANON_CENTER[0]), int(config.CANON_CENTER[1])
    m = int(config.FLANGE_R_PX + config.TEMPLATE_MARGIN_PX)
    tpl = ref[cy - m:cy + m, cx - m:cx + m].copy()
    _save_cached_template(tpl, fp)
    return tpl


# ----------------------------------------------------------------
# 模板落盘缓存（指纹 = 影响模板像素的全部 config 参数 + 资产版本号）
# ----------------------------------------------------------------
_ASSET_VERSION = "1"   # part_model 黄金资产渲染代码版本：改渲染逻辑须递增


def _template_fingerprint() -> str:
    """对影响模板像素的全部参数做 SHA256 短摘要（16 hex 字符）"""
    payload = {
        "version": _ASSET_VERSION,
        "img": [config.IMG_W, config.IMG_H],
        "canon_center": list(config.CANON_CENTER),
        "template_margin_px": config.TEMPLATE_MARGIN_PX,   # 裁剪余量入指纹
        "geometry": {
            "flange_r_px": config.FLANGE_R_PX,
            "rim_ring_w": config.RIM_RING_W,
            "center_hole_r_px": config.CENTER_HOLE_R_PX,
            "bolt_hole_r_px": config.BOLT_HOLE_R_PX,
            "bolt_pc_r_px": config.BOLT_PC_R_PX,
            "bolt_angles_deg": list(config.BOLT_ANGLES_DEG),
            "keyway_angle_deg": config.KEYWAY_ANGLE_DEG,
            "keyway_w_px": config.KEYWAY_W_PX,
            "keyway_d_px": config.KEYWAY_D_PX,
        },
        "appearance": {
            "belt_base_gray": config.BELT_BASE_GRAY,
            "face_base_gray": config.FACE_BASE_GRAY,
            "rim_gray": config.RIM_GRAY,
            "hole_gray": config.HOLE_GRAY,
            "ring_amp": config.RING_AMP,
            "ring_period_px": config.RING_PERIOD_PX,
            "brushed_sigma": config.BRUSHED_SIGMA,
        },
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _meta_path():
    """指纹 sidecar 路径：data/template.meta.json"""
    return config.TEMPLATE_PATH.with_name(
        config.TEMPLATE_PATH.stem + ".meta.json")


def _load_cached_template(fingerprint: str):
    """指纹一致且 PNG 可读、尺寸吻合时返回缓存模板；否则 None"""
    meta, png = _meta_path(), config.TEMPLATE_PATH
    if not (meta.exists() and png.exists()):
        return None
    try:
        info = json.loads(meta.read_text(encoding="utf-8"))
        if info.get("fingerprint") != fingerprint:
            return None
        tpl = cv2.imread(str(png), cv2.IMREAD_GRAYSCALE)
        if tpl is None or tpl.dtype != np.uint8:
            return None
        m = int(config.FLANGE_R_PX + config.TEMPLATE_MARGIN_PX)
        if tpl.shape != (2 * m, 2 * m):       # 尺寸与当前几何不符
            return None
        return tpl
    except (OSError, ValueError, json.JSONDecodeError):
        return None                            # 损坏/半写：按失效处理


def _save_cached_template(tpl: np.ndarray, fingerprint: str) -> None:
    """写入模板 PNG 与指纹 sidecar（失败不抛出：缓存只是加速）"""
    try:
        config.TEMPLATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(config.TEMPLATE_PATH), tpl):
            return
        _meta_path().write_text(
            json.dumps({"fingerprint": fingerprint,
                        "asset_version": _ASSET_VERSION},
                       ensure_ascii=False, indent=2),
            encoding="utf-8")
    except OSError as e:
        print(f"[part_model] 模板缓存写入失败(不影响功能): {e!r}")
