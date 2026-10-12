# ADR 0004:`mmtrain compose` 产出训练蓝图,而非可 launch 的 RunSpec

- 状态:已接受
- 日期:2026-10-12
- 版本:2.0.1

## 1. 背景与决策

主线二要求:用户说"我要训 X",系统应组装出训练方案,并主动提示**实验指标、
可视化需求、下一步晋级指示**。

**决策**:实现 `mmtrain compose` 交互向导(选模型 → 仅列兼容目标 → reward),
输出一份**训练蓝图**(组件清单 + 指标 + 可视化 + 下一步),而**不是**一份可直接
`phase run` 的 RunSpec。

**关键依据**:完整 RunSpec 需要 `AssetRef`(SHA-256 资产哈希)、`RuntimeSpec`
(注册主机/GPU 拓扑)、`LifecycleSpec` —— 这些是 **WebTrainer 控制面在运行时填充**的,
compose 向导不知道也无权编造。让 compose 产出 RunSpec 会诱导用户以为"填完就能跑",
绕过晋级阶梯。

## 2. 备选方案与后果

- **备选 A:compose 直接产出 RunSpec。** 放弃:资产哈希与运行时拓扑不可得;且
  `ComponentSpec` 的 validator 明确禁止在 config 里写 path/url/host,蓝图阶段
  根本填不出合法 spec。
- **备选 B:蓝图落盘为 json/md。** 暂缓:MVP 先只打印;落盘留 `--output` 作为后续。
- **后果(正面)**:蓝图如实标注"这是计划不是 spec",职责边界清晰;指标表对齐
  代码真实返回字段,不虚构。
- **后果(约束)**:面向 agent(全参数非交互)与面向人(问答+默认值+解释)必须
  同时成立,防止"人完全不参与而退化"。

## 3. 受影响的稳定契约

- 逻辑(`compose/compose.py`:兼容过滤、蓝图生成)是纯函数,交互层(CLI)只做
  input/print,核心可单测。
- 指标知识表(`compose/guidance.py`)与注册表由 `test_compose.py` 保持同步。
- 不引入第二个 argparse 边界(ADR 0002):compose 挂在现有 `cli.py` 的子命令下。

## 4. 同一迁移中移除/修正的旧物

无文件移除。修正一处代码/文档不一致:identity reward 只是 examples 里的函数,
未注册为插件,故从 reward 指标表移除并注释,待注册后补回。

## 5. 验证与回滚边界

- **验证**:`test_compose.py` 覆盖兼容过滤、reward 过滤、guidance 表与注册表同步、
  不兼容报错;CLI 交互/非交互两模式实测。
- **回滚**:compose 是独立子包 + 一个子命令,移除即删 `compose/` 与 cli 分支。
- **何时可推翻**:当 WebTrainer 与控制面就绪、能向 compose 提供资产/运行时上下文时,
  可开新 ADR 让 compose 升级为产出合法 RunSpec 草稿,但仍须保留晋级阶梯门禁。
