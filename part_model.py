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
    """从基准图裁出含 30px 余量的工件模板（用于 cv2.matchTemplate）"""
    ref, _ = make_reference()
    cx, cy = int(config.CANON_CENTER[0]), int(config.CANON_CENTER[1])
    m = int(config.FLANGE_R_PX + 30)
    return ref[cy - m:cy + m, cx - m:cx + m].copy()
