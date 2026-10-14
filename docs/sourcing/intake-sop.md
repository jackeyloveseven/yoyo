# 新方法导入 SOP(Intake Standard Operating Procedure)

把一篇论文、或上游项目里的一个好实现,变成 YoYo 的一个**可组合、可测试、有文档**
的零件。本流程对 RL 新方法、OPSD 变体、少步蒸馏新方法一视同仁。

## 总原则:吸收想法,重写实现

上游实现通常缠在自己的 trainer 里(例:AI-Toolkit 的 D-OPSD 写在 129KB 的
`SDTrainer.py` 里)。我们**不 vendor 这些代码**:提取数学、超参和实现技巧,用 YoYo
的零件风格重写成独立函数(参考范式:SimpleTuner 把"新训练方法"收敛为`一个新
loss/调度模块 + 一组单测 + 一页说明`,而非改 Trainer 主干),使其通过全部架构门。

## 什么时候值得导入(准入门槛)

同时满足:

- **有真实增益**:更快 / 更省显存 / 数值更稳 / 效果可测更好,能说出理由。
- **能解耦成零件**:核心价值能脱离原框架独立存在并单测。
- **有可信出处**:论文(arXiv 链接)、官方仓库、或上游可复现的实现。

不满足但仍有参考价值 → 只在 `docs/sourcing/README.md` 雷达表里记一行结论,不导入。

## 边界:什么算"一个零件"

| 类型 | 落点 | 边界(不能做什么) |
| :--- | :--- | :--- |
| **新目标/损失**(RL、OPSD、蒸馏) | `src/mmtrainer/algorithms/<name>/` | 只做数学,不 import 具体模型;不持有优化器状态 |
| **新求解器/采样步** | `src/mmtrainer/nn/flow.py` | 纯张量函数,不碰模型实例 |
| **新调度/正则** | `src/mmtrainer/nn/schedules.py` 等 | 无全局状态 |
| **新奖励** | `src/mmtrainer/rewards/` | 实现 `RewardBackend` 协议,不内嵌训练循环 |
| **新模型** | `src/mmtrainer/models/` + `BaseModelAdapter` | 不 import 其它模型包 |

**红线**:任何零件都不得引入"通用 optimizer step"抽象——那是 engine 刻意不提供的
(见 `ARCHITECTURE_V2.md`)。ObjectivePlugin 拥有自己的内部训练算法。

## 导入步骤(每一步都有验证)

以"导入某 RL 新方法 `foo_rl`"为例:

1. **技术摘要(人复核)** → 验证:3 句话说清它比普通 GRPO/NFT 好在哪、量化理由;
   写进当次 ADR 或雷达表,注明出处与上游 commit。
2. **零件骨架(agent)** → 验证:`algorithms/foo_rl/` 只含 `__init__.py`/`config.py`/
   `objectives.py`,纯函数;签名带类型注解与中文 docstring(细节见附录"产出结构")。
3. **数值正确性测试** → 验证:按"测试要求"写单测,放 `tests/v2/test_foo_rl.py` 并通过。
4. **注册(一行)** → 验证:`registry/builtins.py` 加懒加载条目,`mmtrain registry list`
   可见;RL 类声明 `reward_required=True` 与所需模型 capability。
5. **契约卡(文档)** → 验证:`docs/algorithms/foo_rl.md`(契约/数值约定/边界/出处)
   过 `check_docs.py`。
6. **可组合示例(可选但推荐)** → 验证:`examples/` 下能 50 行内拼出跑通。

完成 1–5 即"源码可用"。**不声明 GPU 就绪**——那需要按 `USAGE.md` 的晋级阶梯
(no-update → one-update → checkpoint-readback → 冻结评估)拿 receipt。

## 测试要求(新方法必须回答的问题)

不要只测"能跑",针对方法的本质写断言:

- **RL / GRPO 类**:零方差组优势归一化是否中性?ratio 裁剪正确?KL 在 FP16 输入下
  按 FP32 算?确定性 transition 被掩码而非赋 Dirac?(参照 `test_nft_numerics.py`。)
- **OPSD / 自蒸馏类**:教师目标 `detach`?信任域边界生效?权重和为正被校验?
- **少步蒸馏类**:1–4 步下学生与教师分布匹配误差随步数单调?DMD fake-score 用
  独立网络而非共享权重?
- **通用**:nan/inf 优化前报错?形状不匹配给可读错误?

## 三个人工把关点(防退化、防走歪)

1. **第 1 步**:人确认"这个增益是真的",防止把伪优化吸进来。
2. **第 3 步**:人 review 测试是否覆盖了方法的*失败模式*,而不只是 happy path。
3. **合入前**:人确认没有 vendor 上游耦合代码、没有破坏任何架构门。

## 完成后必须更新的地方

- `docs/algorithms/README.md` 的契约表加一行。
- 若是注册目标,更新根 `README.md` 零件清单。
- 若是重大吸收(>500 行),在 `docs/adr/` 留一篇 ADR,命名 `NNNN-<method>-intake.md`,
  写明:取代了什么、从哪个上游/论文学来、验证与回滚边界。

---

## 附录:agent 执行模板(import-method)

下面这段可原样保存为 Claude Code skill(`~/.claude/skills/import-method/SKILL.md`),
供 agent 执行第 2、3、5 步的骨架生成。**第 1 步(增益判断)与所有人工把关点由人
确认,agent 不替人拍板。** 它故意不放仓库内,以免触发 `check_docs.py` 的
`duplicate-training-code-skill-root` 规则。

### 输入(向用户问全,缺则停下)

1. 方法名与类型:RL / OPSD / 少步蒸馏 / 求解器 / 调度 / 奖励 / 模型适配器。
2. 出处:arXiv 链接,或上游仓库 + commit。
3. 增益一句话 + 量化理由(人工把关点 1)。
4. 落点:按本 SOP"边界"表映射(如 RL → `algorithms/<name>/`)。

### 产出结构(不擅自加文件)

以 RL 为例,零件 `src/mmtrainer/algorithms/<name>/` 只含(落点与边界按上文"边界"表):

- `__init__.py`(只导出公开符号,`__all__` 显式)、`config.py`(frozen dataclass)、
  `objectives.py`(纯函数损失:只做数学不 import 模型;FP16/BF16 上溢 FP32 再算
  概率/KL/指数;nan/inf 优化前 `raise`;返回 frozen Metrics;教师/参考目标
  `detach`;不引入"通用 optimizer step")。
- 测试 `tests/v2/test_<name>.py`(按"测试要求"写断言,`torch.manual_seed` 固定)。
- 契约卡 `docs/algorithms/<name>.md`(契约/数值约定/边界/出处,过 `check_docs.py`)。
- 注册(RL 类):`registry/builtins.py` 懒加载条目,声明 `reward_required` 与 capability。

### 交还给人前的自检清单

- [ ] `check_v2_quality.py`、`ruff check`、`ruff format --check` 通过
- [ ] 新测试通过且覆盖失败模式(人工把关点 2)
- [ ] `check_docs.py` 通过,无 vendor 上游耦合、未破坏架构门(人工把关点 3)
- [ ] 契约表加一行;重大吸收留 ADR
