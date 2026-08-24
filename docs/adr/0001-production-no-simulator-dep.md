# ADR-0001：生产算法不得依赖仿真器 —— 工件定义单源 `part_model`

- 状态：已接受（Accepted）
- 日期：2026-08-24
- 关联：架构体检报告候选 1 / `CONTEXT.md` 术语「工件基准模型」

## 背景（Context）

项目初始结构中，工件的事实分散在测试数据生成器里：

- `locate/locate.py` 在运行时延迟导入 `simulator.synth` 的
  `make_template`/`make_reference` 取匹配模板与基准小图；
- `inspect/inspect.py` 顶部导入 `simulator.synth` 的
  `make_reference`/`pose_matrix`/`apply_affine`/`bolt_centers_canonical`
  ——检测算法的比对基准图由**仿真器**渲染、孔位期望值由仿真器的几何
  函数提供；
- 叠加目录名 `inspect` 与标准库冲突，逼出 `run_batch.py` 与
  `plc_server.py` 各一份逐字相同的 importlib 加载胶水。

删除测试结论：删掉 `synth.py`，定位与检测当场瘫痪——它承载的是生产期
真实需求，却放在了测试期模块的 seam 上。后果：① 生产代码无法在无仿真器
环境部署；② 基准资产锁在各消费方的模块级全局缓存，测试无法注入替代
基准；③ 「迁移真实产线」（README 免责声明承诺的方向）缺少干净的换源缝。

## 决策（Decision）

1. 新增顶层模块 **`part_model.py`（工件基准模型）**，作为工件名义定义的
   **唯一出处**，包含四部分：
   - 规范几何（螺栓孔分布、键槽几何与方位角、位姿矩阵、仿射变换）；
   - 标准成像环境（固定光照场、传送带背景纹理）；
   - 名义外观（环带纹理、固定种子拉丝噪声、无缺陷工件图层 `build_part()`）；
   - 黄金资产（基准图 `make_reference()`、匹配模板 `make_template()`，
     后者落盘缓存 + 参数指纹校验）。
2. **依赖方向铁律：`locate` / `inspect` 只许 `import part_model` 获取
   工件事实；任何生产模块不得 `import simulator`。** `simulator/synth`
   是 `part_model` 的下游组合者（叠加缺陷注入与随机化合成测试帧）；
   `run_batch` / `plc_server` 作为模拟相机的消费者使用 `synth_frame`
   不受此限。
3. 工件几何参数（含键槽方位角等此前散落的隐式约定）一律从 `config`
   经 `part_model` 派生，禁止在使用点重写字面量。
4. 黄金资产的渲染逻辑变更时递增 `part_model._ASSET_VERSION` 使落盘
   缓存失效。

## 被否决的备选

- **维持 synth 同名 re-export 作永久层**：解耦只停在纸面，删除测试不过。
- **把黄金资产分别搬进 locate/inspect**：制造两份渲染知识，违背单源初衷。
- **仅改目录名解决 stdlib 重名**：消除胶水但不解决方向性依赖（已另行
  处理为独立改进项）。

## 后果（Consequences）

正向：

- 定位/检测可在无仿真器环境下运行与测试；「标准样件拍照建基准」成为
  `part_model` 黄金资产的天然第二个 adapter；
- 几何期望值单源后，改一个 config 参数（如键槽方位角）全链路一致生效
  （已在非零方位端到端验证）；
- 测试可注入替代基准/小尺寸资产，不再受 800×600 渲染链绑架。

代价与约束：

- 新增一个顶层模块；评审新代码时须对照第 2 条铁律执行；
- `data/template.png` + 指纹 sidecar 成为合法缓存资产（指纹不符自动重建，
  损坏自愈）；改动渲染代码必须递增 `_ASSET_VERSION`；
- 迁移真机时，「黄金资产来自程序渲染」需替换为「来自标准样件成像」——
  这正是本 ADR 预留的 seam，替换范围被限定在 `part_model` 内部。
