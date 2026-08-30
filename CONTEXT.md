# CONTEXT.md —— 领域术语表

> 本表是项目领域词汇的单一出处：写代码、写文档、做评审时对同一概念
> 使用同一名词。新增术语随决策当场登记；改名须全表同步。
>
> 约定：模块名用等宽字体；「生产期」= 装机运行时，「测试期」= 合成
> 数据与验收时。

## 工件与成像

| 术语 | 英文 | 定义 | 归属 |
|---|---|---|---|
| 工件基准模型 | part model | 法兰盘工件的名义定义的唯一出处：规范几何 + 标准成像环境 + 名义外观 + 黄金资产。生产算法只从这里取事实。 | `part_model.py` |
| 规范位姿（基准位姿） | canonical pose | 图像中心 `(400,300)`、旋转 0°、缩放 1.0 的参考位姿；一切几何期望值在此坐标系下定义。 | `config.CANON_CENTER` |
| 规范几何 | canonical geometry | 基准位姿下的工件几何事实：螺栓孔分布（数量/分度圆/方位角）、键槽几何与方位角、外圆半径等，全部由 config 参数经 `part_model` 推导。 | `part_model.py` |
| 键槽方位角 | keyway angle | `KEYWAY_ANGLE_DEG`，破坏旋转对称的定位特征方位。**单源**：合成避让、基准渲染、真值、检测保护扇区、定位精修全链路从它派生。 | `part_model.py` |
| 名义外观 | nominal appearance | 无缺陷工件的灰度呈现：环带纹理公式 + 固定种子拉丝噪声 + 外圆暗环 + 孔/键槽。确定性（逐次调用逐位一致）。 | `part_model.build_part` |
| 材料掩膜 | material mask | 「工件材料圆盘」的 0/255 掩膜。语义口径：孔是盘内特征（只挖灰度层）；键槽挖穿露皮带（从掩膜去除）。 | `part_model.build_part` |
| 标准成像环境 | standard imaging | 固定光照增益场 + 传送带背景纹理。产线相机光源固定 ⇒ 基准图与测试图共享同一环境，比对时可抵消。 | `part_model.py` |

## 检测链路

| 术语 | 英文 | 定义 | 归属 |
|---|---|---|---|
| 黄金资产 | golden assets | 基准图 `make_reference()`（标准工位成像的无缺陷画面）与匹配模板 `make_template()`（基准图裁剪，落盘缓存+指纹）。locate 的模板来源、inspect 的比对参照。 | `part_model.py` |
| 三级定位流水线 | locate pipeline | 模板匹配粗定位 → 凸包质心精修（中心/缩放）→ 键槽剖面精修角度。 | `locate/locate.py` |
| 三分支检测 | three-branch inspection | 基准比对分支（划痕/污渍）+ 外圆轮廓分支（崩边）+ 几何测量分支（孔偏移/孔缺失）。 | `detect/detect.py` |
| NG 判定 | NG judgement | 规则表判定：缺陷面积超限 / 崩边内凹 / 孔位超差 / 孔缺失 / 定位失败（安全策略按 NG）。 | `config.py §五` |
| 缺陷类型 | defect types | 五类：`scratch` 划痕 · `chip` 崩边 · `stain` 污渍 · `bolt_shift` 孔偏移 · `bolt_missing` 孔缺失；另有协议位 `locate_fail` 定位失败。 | 注入:`synth` 检出:`inspect` |
| 像素当量 | mm-per-pixel | 像素↔毫米换算系数（仿真设定 0.1mm/px）。⚠ 双口径现状见设计说明书 §9：标定复检值与 config 设定值并存。 | `calib` / `config` |

## 链路与看板

| 术语 | 英文 | 定义 | 归属 |
|---|---|---|---|
| 寄存器规划 | register map | HR0~HR10 协议：触发/忙闲/结果码/缺陷位/偏差定点数/心跳。地址与位编码单源于 config §六。 | `config.py §六` |
| 两段式同步 | two-phase sync | 上位机写触发后先等 HR1=1(受理) 再等 HR1=2(完成)，防止读到上一轮残留 DONE。 | `modbus_client_test.wait_done` |
| 看门狗 | watchdog | 触发后 `WATCHDOG_TIMEOUT_S` 未完成 → 写故障码 999 并作废迟到结果（正常结果与故障互斥，只赢一次）。 | `plc_server.run` |
| 检测记录流 | records stream | `data/records.jsonl` 追加式 JSON Lines；看板轮询消费。字段契约由 `vision_pipeline.build_record` 唯一构造（FAULT 与正常记录满键同构，单测锁定）；seq 跨会话续接、ts 带日期。 | 构造:`vision_pipeline.build_record` 写:`plc_server` 读:`dashboard` |
| 仿真验证值 | simulated metrics | 全部性能指标均为合成数据上的实测，不等价真实产线性能（项目免责声明的口径）。 | 文档全局 |

## 决策记录索引

- [ADR-0001](docs/adr/0001-production-no-simulator-dep.md)：生产算法不得依赖仿真器 —— 工件定义单源 `part_model`。（2026-08-24）
