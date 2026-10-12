# LCM 少步蒸馏（Latent Consistency）

`mmtrainer.algorithms.latent_consistency` 提供模型中立的一致性蒸馏零件:只做数学,
不 import 任何具体模型,也不持有优化器或 EMA 状态。目标网络由调用方以 EMA 维护,
零件只接收已 `detach` 的 `target_velocity`。

## 公开契约

```python
@dataclass(frozen=True)
class LCMConsistencyInputs:
    noisy: Tensor            # x_t
    online_velocity: Tensor  # v_θ(x_t, t),带梯度
    target_velocity: Tensor  # v_θ⁻(x_{t'}, t'),调用方已 detach
    sigma_curr: Tensor       # t,shape (batch,)
    sigma_prev: Tensor       # t' = t - Δt,shape (batch,)

@dataclass(frozen=True)
class LCMConsistencyConfig:
    huber_c: float = 1e-3
    snr_weight: Tensor | None = None   # 可选逐样本权重(leading shape)
    event_dims: EventDims = None

@dataclass(frozen=True)
class LCMConsistencyMetrics:
    total_loss: Tensor
    per_sample_loss: Tensor
    raw_huber: Tensor
    consistency_rms: Tensor

def lcm_consistency_loss(
    inputs: LCMConsistencyInputs,
    config: LCMConsistencyConfig | None = None,
) -> LCMConsistencyMetrics
```

`noisy`、`online_velocity`、`target_velocity` 三者必须同形状同设备(逐元素一致);
`sigma_curr`/`sigma_prev` 为 `(batch,)` 且与 latent 同设备。调用方需先用同一条
`euler_flow_step(noisy, online_velocity.detach(), sigma_curr, sigma_prev)` 构造
`x_{t'}` 并查询 EMA 目标网络,再把结果作为 `target_velocity` 传入。

## 数学定义

沿用仓库 rectified-flow 约定(`x_t = (1-t)x_0 + t·noise`,`v = noise - x_0`):

1. `x0_online = x_t - t · v_θ(x_t, t)`;
2. `x_{t'} = euler_flow_step(x_t, v_θ(x_t, t).detach(), t, t')`(自举状态,梯度不经过);
3. `x0_target = (x_{t'} - t' · v_θ⁻(x_{t'}, t')).detach()`;
4. `L = w(t) · (sqrt(Δx0² + c²) - c)`,`Δx0 = x0_online - x0_target`,`c = huber_c`。

`total_loss` 是逐样本损失在 batch 上的均值;`per_sample_loss` 已乘 `snr_weight`;
`raw_huber` 保持未加权的逐样本伪 Huber;`consistency_rms` 是 `Δx0` 的 RMS,与
`raw_huber` 一样只作诊断、不参与反传。`huber_c = 0` 时伪 Huber 退化为 `|Δx0|`,
且梯度在 `Δx0 = 0` 处无定义(实现返回 0),请保留正的 `c` 以获得平滑梯度。

## 数值约定

- FP16/BF16 输入在计算前统一上溢到 FP32,`float64` 输入保持 `float64`。
- `huber_c` 必须有限且非负;`snr_weight` 必须有限、非负、shape 为事件维之外的
  leading shape。
- `sigma_prev` 必须严格小于 `sigma_curr`(Euler 步朝干净样本方向),两者都须落在
  `[0, 1]`;违反即 `raise ValueError`。
- 任何非有限输入(noisy/velocity/sigma/weight)或非有限结果都在返回前
  `raise ValueError`,不会把 nan 带进优化器。
- 形状不匹配统一给出带张量名的可读错误(来自 `require_matching_tensors` 与逐项校验)。

## detach 边界

`target_velocity` 在零件内部无条件重 `detach`(防线,即使调用方漏了);`x_{t'}` 的
构造全程在 `torch.no_grad()` 下进行。因此梯度只沿 `x0_online` 一条路径回传到
`online_velocity`,不存在经过 Euler 步的二阶路径。EMA 参数更新与目标网络前向都不
属于本零件。

## 与 TDM / DMD 的关系

TDM 与 DMD 系需要教师多步 rollout(或独立的 fake-score 网络)来构造分布级目标;
LCM 只用**相邻 ODE 步的自举一致性**:监督信号完全由同一轨迹上两个相邻时刻的干净
预测差决定,单次前向即可,无需教师多步采样,也无需额外网络。代价是噪声更大、依赖
自举状态质量——`sigma_prev` 与 `sigma_curr` 的间隔(Δt)由调用方的调度决定。

## 边界与注册

- 注册目标 `latent_consistency` 需要模型具备 `image_training` 能力,且不需要 reward
  (`reward_required=False`)。
- 零件不提供端到端 trainer、采样器或 checkpoint 语义;模型适配器负责 conditioning、
  native 调度与 EMA 维护。
- 源码可用不等于 GPU 就绪,晋级仍按 `USAGE.md` 的阶梯取 receipt。

## 出处

- Latent Consistency Models, Luo et al. 2023([arXiv:2310.04378](https://arxiv.org/abs/2310.04378));
- LCM-LoRA 与扩散/流匹配一致性变体([arXiv:2311.05556](https://arxiv.org/abs/2311.05556))。
