# -*- coding: utf-8 -*-
"""
config.py —— 全局配置中心（全项目唯一的参数入口）
=========================================================
职责：
    集中管理所有跨模块共用参数：数据目录、像素当量、工件几何尺寸、
    合成随机化范围、检测公差与 NG 判定阈值、Modbus 寄存器规划、
    看板服务参数。任何模块需要参数时一律 `import config`，
    禁止在其他文件里散落硬编码——调参只需改本文件。

使用示例：
    import config
    config.ensure_dirs()
    print(config.MM_PER_PIXEL)      # 0.1
"""
from pathlib import Path

# ================================================================
# 一、目录与文件路径
#     data/ 及其子目录无需手工创建，各模块调用 config.ensure_dirs() 自动建立
# ================================================================
PROJECT_ROOT  = Path(__file__).resolve().parent          # 项目根目录
DATA_DIR      = PROJECT_ROOT / "data"                    # 运行数据总目录
IMAGE_DIR     = DATA_DIR / "images"                      # 合成"相机画面"输出
TRUTH_DIR     = DATA_DIR / "truth"                       # 每张画面的真值 JSON
ANNOT_DIR     = DATA_DIR / "annot"                       # 检测标注图输出
CALIB_DIR     = DATA_DIR / "calib"                       # 相机标定结果
TEMPLATE_PATH = DATA_DIR / "template.png"                # 定位模板缓存（part_model 首次生成后落盘，指纹不符自动重建）
CALIB_JSON    = CALIB_DIR / "calibration.json"           # 标定结果 JSON


def ensure_dirs() -> None:
    """创建全部数据子目录（幂等：已存在则跳过）"""
    for d in (DATA_DIR, IMAGE_DIR, TRUTH_DIR, ANNOT_DIR, CALIB_DIR):
        d.mkdir(parents=True, exist_ok=True)


# ================================================================
# 二、相机与像素当量（仿真设定）
#     800×600 像素 ↔ 80×60 mm 视场 → 1 像素 = 0.1 mm
# ================================================================
IMG_W, IMG_H = 800, 600        # 相机画面分辨率（像素）
MM_PER_PIXEL = 0.1             # 像素当量（mm/像素），标定模块会复检该值


# ================================================================
# 三、工件几何（全部以"基准位姿"下的像素尺寸给出）
#     基准位姿 = 中心(400,300) + 旋转 0° + 缩放 1.0
# ================================================================
CANON_CENTER     = (400.0, 300.0)                        # 基准中心（像素）
FLANGE_R_PX      = 100.0                                 # 法兰外圆半径（=10.0 mm）
RIM_RING_W       = 4                                     # 外圆暗环宽度（倒角阴影）
CENTER_HOLE_R_PX = 30.0                                  # 中心孔半径（=3.0 mm）
BOLT_HOLE_R_PX   = 10.0                                  # 螺栓孔半径（=1.0 mm）
BOLT_PC_R_PX     = 64.0                                  # 螺栓孔分度圆半径（=6.4 mm）
BOLT_ANGLES_DEG  = (45.0, 135.0, 225.0, 315.0)           # 4 孔基准方位角
KEYWAY_ANGLE_DEG = 0.0                                   # 键槽方位角（破坏旋转对称的定位特征）
KEYWAY_W_PX      = 26.0                                  # 键槽宽度
KEYWAY_D_PX      = 16.0                                  # 键槽深度

# 外观灰度（8bit 灰度级）
BELT_BASE_GRAY = 110.0    # 传送带基础灰度
FACE_BASE_GRAY = 185.0    # 法兰盘面基础灰度
RIM_GRAY       = 80.0     # 外圆暗环灰度
HOLE_GRAY      = 62.0     # 孔（中心孔/螺栓孔）灰度
RING_AMP       = 6.0      # 盘面环带纹理振幅
RING_PERIOD_PX = 18.0     # 盘面环带纹理周期（像素）
BRUSHED_SIGMA  = 3.0      # 盘面径向拉丝噪声标准差（固定种子，保证各帧一致）

# 每帧随机化范围（合成器扰动量）
POSE_JITTER_PX   = 100.0          # 中心位置偏移 ±100 px（=±10 mm）
ANGLE_JITTER_DEG = 30.0           # 旋转 ±30°
SCALE_RANGE      = (0.9, 1.1)     # 缩放范围
BRIGHTNESS_RANGE = (0.8, 1.2)     # 亮度增益范围（±20%）
NOISE_SIGMA      = 4.0            # 高斯噪声标准差（灰度级）

# ================================================================
# 四、locate 定位参数
# ================================================================
TM_SCORE_MIN       = 0.30    # 模板匹配最低得分，低于此值判"定位失败"
KEYWAY_AREA_MIN_PX = 250.0   # 键槽连通域面积下限（用于区分键槽与崩边缺口）
KEYWAY_AREA_MAX_PX = 620.0   # 键槽连通域面积上限

# ================================================================
# 五、inspect 缺陷检测与 NG 判定参数
# ================================================================
# —— 基准比对分支（划痕 / 污渍 / 孔缺陷）——
DIFF_ADAPT_K     = 3.0      # 差异图自适应阈值：thr = max(下限, 均值 + k×标准差)
DIFF_ABS_FLOOR   = 20.0     # 差异灰度下限（干净件实测余量>10级，低于它视为噪声）
MIN_BLOB_AREA_PX = 12       # 连通域面积小于此值按噪声丢弃
MORPH_CLOSE_ITER = 1        # 形态学闭运算次数（连接断裂区域）
CLASS_SCRATCH_ASPECT = 3.0  # 聚类整体长宽比 ≥3 → 线状缺陷
CLASS_SCRATCH_W_MAX_PX = 8.0  # 且等效宽度 ≤8px → 划痕，否则污渍
#   ↑ 依据批量实测（run_batch 调参记录）：划痕标注宽 2~3px，经阈值膨胀+
#   形态学闭运算+折线弯曲后等效宽度最大约 7.1px；污渍团块 ≥15px，
#   取 8px 两类间隔充足。

# —— 外圆轮廓分支（崩边检测）——
RIM_PROFILE_BINS    = 360   # 外圆轮廓采样角度数
RIM_DEVIATION_NG_PX = 6.0   # 外圆半径局部削减 > 6 px（=0.6 mm）→ 崩边

# —— 几何测量分支（螺栓孔）——
HOLE_POS_TOL_MM = 0.5       # 螺栓孔位置偏移公差 ±0.5 mm
HOLE_DIA_TOL_MM = 0.3       # 螺栓孔孔径偏差公差 ±0.3 mm
HOUGH_MIN_DIST  = 15        # 霍夫圆：圆心最小间距（像素）
HOUGH_PARAM1    = 60        # 霍夫圆：内部 Canny 高阈值
HOUGH_PARAM2    = 20        # 霍夫圆：累加器阈值
HOUGH_MIN_R     = 7         # 霍夫圆：最小半径（像素）
HOUGH_MAX_R     = 14        # 霍夫圆：最大半径（像素）

# —— NG 判定规则表 ——
DEFECT_AREA_NG_PX = 30      # 单个缺陷面积 > 30 px²（=0.3 mm²）→ NG
CONF_MIN_OK       = 0.75    # 置信度低于此值时结果标记"低置信"

# ================================================================
# 六、plc_link：Modbus/TCP 寄存器规划（保持寄存器，16bit）
# ================================================================
MODBUS_HOST = "127.0.0.1"
MODBUS_PORT = 502

REG_TRIGGER   = 0    # HR0 : 触发命令（上位机写 1 触发一次检测，视觉端处理后自动清零）
REG_BUSY      = 1    # HR1 : 忙闲状态 0=空闲 1=忙 2=完成（本次结果有效）
REG_RESULT    = 2    # HR2 : 检测结果码 0=无 1=OK 2=NG 999=看门狗故障
REG_DEFECT    = 3    # HR3 : 缺陷类型码（位组合，位号见下方 DEFECT_TYPES 注册表）
REG_DEV_X     = 4    # HR4 : X 定位偏差，0.01mm 单位，有符号
REG_DEV_Y     = 5    # HR5 : Y 定位偏差，0.01mm 单位，有符号
REG_ANGLE     = 6    # HR6 : 工件角度，0.1° 单位，有符号
REG_HEARTBEAT = 10   # HR10: 心跳计数（视觉端每处理一轮 +1）

N_REGS = 16          # 寄存器区总长（HR0~HR15，预留扩展）

BUSY_IDLE, BUSY_BUSY, BUSY_DONE = 0, 1, 2
RESULT_NONE, RESULT_OK, RESULT_NG, RESULT_FAULT = 0, 1, 2, 999

# ----------------------------------------------------------------
# 缺陷类型注册表（唯一出处）
# ----------------------------------------------------------------
# 新增一种缺陷只改这张表：位号自动进 HR3 位组合、中文名自动进测试报告
# 与看板图例、可注入类型自动进合成器与批量验收统计、面域类缺陷自动进
# 「面积超阈值」NG 规则。字段含义：
#   name       程序内类型名（记录/接口/寄存器编码都用它）
#   cn         中文名（报告与看板展示）
#   bit        HR3 缺陷码位号（0~15，不可与已有重复）
#   area_blob  是否属于"基准比对连通域"类缺陷（受面积阈值 NG 规则约束）
#   injectable 合成器是否可注入（locate_fail 是链路状态，不可注入）
DEFECT_TYPES = (
    {"name": "scratch",      "cn": "划痕",     "bit": 0,
     "area_blob": True,  "injectable": True},
    {"name": "chip",         "cn": "崩边",     "bit": 1,
     "area_blob": True,  "injectable": True},
    {"name": "stain",        "cn": "污渍",     "bit": 2,
     "area_blob": True,  "injectable": True},
    {"name": "bolt_shift",   "cn": "孔偏移",   "bit": 3,
     "area_blob": False, "injectable": True},
    {"name": "bolt_missing", "cn": "孔缺失",   "bit": 4,
     "area_blob": False, "injectable": True},
    {"name": "locate_fail",  "cn": "定位失败", "bit": 5,
     "area_blob": False, "injectable": False},
)

DEFECT_BIT = {d["name"]: 1 << d["bit"] for d in DEFECT_TYPES}
DEFECT_CN = {d["name"]: d["cn"] for d in DEFECT_TYPES}
INJECTABLE_TYPES = tuple(d["name"] for d in DEFECT_TYPES if d["injectable"])
AREA_BLOB_TYPES = tuple(d["name"] for d in DEFECT_TYPES if d["area_blob"])

assert len({d["bit"] for d in DEFECT_TYPES}) == len(DEFECT_TYPES), \
    "DEFECT_TYPES 的 bit 位号不可重复"

WATCHDOG_TIMEOUT_S = 2.0    # 看门狗：触发后超过该时长未完成 → 写故障码
HEARTBEAT_PERIOD_S = 1.0    # 空闲时心跳递增周期（秒）


def to_int16(value: int) -> int:
    """把有符号整数映射为 Modbus 16bit 寄存器值（两补码表示）"""
    return int(value) & 0xFFFF


def from_int16(raw: int) -> int:
    """把 Modbus 16bit 寄存器值还原为有符号整数（两补码解析）"""
    raw &= 0xFFFF
    return raw - 0x10000 if raw >= 0x8000 else raw


# ----------------------------------------------------------------
# 定点数编解码（HR4/5/6 的缩放系数与打包/解包唯一出处）
# ----------------------------------------------------------------
# 上位机与视觉端必须使用同一组系数；此前编码在 plc_server、解码在
# modbus_client_test 各写一遍字面量，改精度要两头同步。
DEV_SCALE_MM = 100      # HR4/HR5 定位偏差：0.01mm/LSB
ANGLE_SCALE_DEG = 10    # HR6 角度：0.1°/LSB


def pack_dev_mm(v_mm: float) -> int:
    """定位偏差(mm) → 寄存器原码（四舍五入到 0.01mm，两补码）"""
    return to_int16(round(v_mm * DEV_SCALE_MM))


def pack_angle_deg(v_deg: float) -> int:
    """角度(°) → 寄存器原码（四舍五入到 0.1°，两补码）"""
    return to_int16(round(v_deg * ANGLE_SCALE_DEG))


def unpack_dev_mm(raw: int) -> float:
    """寄存器原码 → 定位偏差(mm)"""
    return from_int16(raw) / DEV_SCALE_MM


def unpack_angle_deg(raw: int) -> float:
    """寄存器原码 → 角度(°)"""
    return from_int16(raw) / ANGLE_SCALE_DEG


# ================================================================
# 七、dashboard 看板服务参数
#     注：本机 5000 端口常被其他开发服务占用（Windows 允许重复绑定但
#     连接会进先绑定者，症状是"看板打开却是别的页面"），故默认 5001。
# ================================================================
DASH_HOST = "127.0.0.1"
DASH_PORT = 5001
