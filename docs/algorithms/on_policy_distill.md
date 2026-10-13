# On-Policy Distillation 少步蒸馏(在线策略蒸馏)

`mmtrainer.algorithms.on_policy_distill` 提供模型中立的教师轨迹对齐蒸馏零件:只做
数学,不 import 任何具体模型,也不持有教师网络或调度状态。教师多步 Euler 轨迹由
调用方构造并 `detach` 后传入。

**命名警告**:本零件与仓库已有的 `image_opd` / `flow_opd_ode_loss`(DiffusionOPD,
一步确定性转移匹配,**需 reward**)是两个不同方法,仅缩写相同。本零件全名
On-Policy Distillation,**无需 reward**。

## 公开契约

```python
@dataclass(frozen=True)
class OnPolicyDistillInputs:
    student_velocity: Tensor   # v_θ(x_t, t),带梯度
    teacher_x0: Tensor         # 教师多步 Euler 终点,调用方已 detach
    x_t: Tensor                # 加噪样本
    sigma: Tensor              # t,shape (batch,)
    sample_weight: Tensor | None = None  # 可选逐样本权重(leading shape)

@dataclass(frozen=True)
class OnPolicyDistillConfig:
    num_distill_steps: int = 4   # 教师步数(调用方构造轨迹时用)
    min_distill_idx: int = 0     # 最小蒸馏索引(调用方过滤时间步时用)
    loss_weight: float = 1.0
    event_dims: EventDims = None

@dataclass(frozen=True)
class OnPolicyDistillMetrics:
    total_loss: Tensor        # 标量,= loss_weight × mean(per_sample_loss)
    per_sample_loss: Tensor   # (batch,),加权逐样本 MSE
    mse_loss: Tensor          # 标量,未加权逐样本 MSE 均值(诊断,detach)
    x0_rms: Tensor            # 标量,学生/教师干净样本差的 RMS(诊断,detach)

def on_policy_distill_loss(
    inputs: OnPolicyDistillInputs,
    config: OnPolicyDistillConfig | None = None,
) -> OnPolicyDistillMetrics
```

`student_velocity`、`teacher_x0`、`x_t` 三者必须同形状同设备(逐元素一致);
`sigma` 为 `(batch,)` 且落在 `[0, 1]`。调用方需先用
:func:`mmtrainer.nn.flow.euler_flow_step` 从 `x_t` 出发走 `num_distill_steps` 步
到 `t = 0`,把终点作为 `teacher_x0` 传入(参考 `examples/train_on_policy_distill.py`)。

## 数学定义

沿用仓库 rectified-flow 约定(`x_t = (1-t)x_0 + t·noise`,`v = noise - x_0`):

1. `x0_student = x_t - t · v_θ(x_t, t)`(保留梯度);
2. `x0_teacher`:教师从同一 `x_t` 走 `num_distill_steps` 个 Euler 步的终点,
   调用方构造并 `detach`(零件内部再 `detach` 一次);
3. `L = loss_weight · w(t) · mean((x0_student - x0_teacher)²)`,逐样本在事件维上
   求均值后乘以 `sample_weight`,再在 batch 上求均值。

`mse_loss` 与 `x0_rms` 只作诊断、不参与反传。

## 数值约定

- FP16/BF16 输入在计算前统一上溢到 FP32(`working_dtype`)。
- `num_distill_steps ≥ 1`、`min_distill_idx ≥ 0`、`loss_weight > 0`,在配置构造期
  校验(`__post_init__`)。
- `sample_weight` 必须有限、非负、shape 为事件维之外的 leading shape;缺省视为全 1。
- 任何非有限输入或结果都在返回前 `raise ValueError`,不会把 nan 带进优化器。

## detach 边界

`teacher_x0` 在零件内部无条件重 `detach`(防线,即使调用方漏了)。梯度只沿
`x0_student` 一条路径回传到 `student_velocity`,不存在经过教师轨迹的二阶路径。
教师网络的前向与调度都不属于本零件。

## 与 LCM / DiffusionOPD 的关系

| 方法 | 监督信号 | 需要教师多步? | 需要 reward? |
| :--- | :--- | :--- | :--- |
| LCM(`latent_consistency`) | 相邻 ODE 步自举一致性 | 否(EMA 目标单步) | 否 |
| **本方法** | 教师多步轨迹终点对齐 | **是**(每样本 `num_distill_steps` 次教师前向) | 否 |
| DiffusionOPD(`image_opd`) | 一步确定性转移匹配 + 奖励 | 否 | **是** |

LCM 自举信号噪声大但便宜;本方法信号更稳但每样本成本高 `num_distill_steps` 倍。
两者可叠加使用(不同 loss 项求和)。

## 边界与注册

- 注册目标 `on_policy_distill` 需要模型具备 `image_training` 能力,且不需要 reward
  (`reward_required=False`)。
- 零件不提供端到端 trainer、采样器或 checkpoint 语义;模型适配器负责 conditioning
  与教师网络的加载/冻结。
- 源码可用不等于 GPU 就绪,晋级仍按 `USAGE.md` 的阶梯取 receipt。

## 出处

- DiffSynth-Studio 的 OPD 实现(数学对照,按 ADR-0003 只参考数学不抄框架):
  `examples/qwen_image_21/model_training/special/opd/train_opd_lora.py`
  ([github.com/modelscope/DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)),
  2026-10-13 雷达核查记录见 `docs/sourcing/README.md`。
