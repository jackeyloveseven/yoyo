"""Static guidance knowledge for compose blueprints.

Every metric named here corresponds to a field actually returned by the objective
implementations (for example ``PPOMetrics`` in
:mod:`mmtrainer.algorithms.flow_grpo.objectives`) or produced by a reward backend,
so the blueprint never invents numbers the pipeline cannot emit. Keep this table
in sync with the algorithms; tests assert the named objectives and rewards exist
in the registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ObjectiveGuidance:
    """What to watch and produce when training one objective."""

    summary: str
    metrics: tuple[str, ...]
    visualizations: tuple[str, ...]
    notes: tuple[str, ...] = field(default=())


# 目标 → 指标 / 可视化。指标名对齐各目标实现返回的字段。
OBJECTIVE_GUIDANCE: dict[str, ObjectiveGuidance] = {
    "diffusionnft": ObjectiveGuidance(
        summary="DiffusionNFT 在线 RL:用有界 optimality 概率驱动的隐式正/负策略 + 固定基准 KL。",
        metrics=(
            "total_loss(总损失)",
            "optimality 分布(各组奖励映射概率的均值/方差)",
            "kl_loss(固定基准 KL 项)",
            "EMA-old 漂移(旧策略与当前策略的参数距离)",
            "reward 曲线(见所选 reward 指标)",
        ),
        visualizations=(
            "奖励随 step 曲线",
            "每个评估周期的采样网格(同一 prompt 多 seed)",
            "optimality 直方图(检查是否塌缩到 0.5 或饱和到 0/1)",
        ),
        notes=("零方差组应保持中性(概率 0.5);出现全 0/全 1 说明奖励区分度不足。",),
    ),
    "flow_grpo": ObjectiveGuidance(
        summary="Flow-GRPO 轨迹 RL:SDE 转移带对数概率,组相对优势 + 裁剪 PPO + 参考 KL。",
        metrics=(
            "loss(PPO 总损失)",
            "policy_loss(裁剪代理损失)",
            "kl_loss(参考 KL 项)",
            "clip_fraction(被裁剪的 ratio 比例,应 <~0.2)",
            "approx_kl(近似 KL,监控策略漂移)",
            "valid_fraction(有效 transition 比例)",
            "reward 曲线(见所选 reward 指标)",
        ),
        visualizations=(
            "ratio 分布直方图(看是否频繁触裁剪边界)",
            "奖励随 step 曲线 + 组内优势箱线图",
            "采样网格(评估周期)",
        ),
        notes=("clip_epsilon=None 时不裁剪;SenseNova 未公开其 epsilon,需显式提供。",),
    ),
    "diffusion_opsd": ObjectiveGuidance(
        summary="DiffusionOPSD 在线自蒸馏:有界信任域内的奖励目标拟合。",
        metrics=(
            "total_loss(总损失)",
            "分支拟合损失(正/负分支 MSE)",
            "优势尺度(advantage_scale 加权后的幅度)",
            "reward 曲线(见所选 reward 指标)",
        ),
        visualizations=(
            "师生(当前策略 vs 行为策略)输出对比网格",
            "奖励曲线 + 信任域边界命中频率",
        ),
        notes=("教师/行为目标必须 detach,不回传梯度;检查代码里是否误共享权重。",),
    ),
    "flowmimic_opsd": ObjectiveGuidance(
        summary="FlowMimic-OPSD:干净分布匹配的在线自蒸馏(无需 reward)。",
        metrics=(
            "total_loss(总损失)",
            "mimic 项(干净分布匹配损失)",
            "base 项(基础重建损失)",
        ),
        visualizations=(
            "朴素轨迹 vs 有利轨迹的输出对比网格",
            "mimic/base 两项随 step 的相对占比",
        ),
        notes=("无需 reward,适合无可靠评分器的场景。",),
    ),
    "edit_r1": ObjectiveGuidance(
        summary="Edit-R1:DiffusionNFT + 非 CoT MLLM 评分 token logits 的图像编辑 RL。",
        metrics=(
            "total_loss(DiffusionNFT 总损失)",
            "期望连续得分(softmax 加权的 score-token 期望)",
            "组均值/方差(高均值低方差组过滤)",
            "MLLM 多维分(见 edit_r1_mllm reward 指标)",
        ),
        visualizations=(
            "源图 / 编辑后 / 指令 三联网格",
            "各评分维度雷达图(Instruction_Following/Consistency/Visual_Quality)",
            "组过滤通过率随 step 曲线",
        ),
        notes=("过滤阈值 mean>0.9 && std<0.05;通过率过低说明奖励区分度或编辑质量不足。",),
    ),
    "image_sft": ObjectiveGuidance(
        summary="Flow SFT:监督流匹配,最小化预测速度与目标速度的 MSE。",
        metrics=(
            "loss(流匹配 MSE)",
            "per_sample_mse 分布(找出离群样本)",
            "lr(学习率,配合调度器检查)",
        ),
        visualizations=(
            "训练/验证 loss 曲线",
            "采样网格(同一 prompt,看拟合是否过/欠)",
        ),
    ),
    "image_dpo": ObjectiveGuidance(
        summary="耦合噪声 Flow-DPO:共享噪声的赢/输对偏好优化。",
        metrics=(
            "loss(DPO 损失)",
            "隐式奖励差(赢样本与输样本的对数概率差)",
            "偏好准确率(赢样本得分高于输样本的比例)",
        ),
        visualizations=(
            "赢/输样本对网格(并排对比)",
            "隐式奖励差随 step 曲线(应单调上升)",
            "偏好准确率曲线",
        ),
    ),
    "image_opd": ObjectiveGuidance(
        summary="OPD 在线偏好蒸馏:ODE 轨迹上的偏好对齐(需 reward)。",
        metrics=(
            "loss(OPD-ODE 损失)",
            "reward 曲线(见所选 reward 指标)",
            "轨迹端点误差(学生少步 vs 教师多步)",
        ),
        visualizations=(
            "学生少步采样 vs 教师多步采样对比网格",
            "奖励曲线 + 端点误差曲线",
        ),
    ),
    "latent_consistency": ObjectiveGuidance(
        summary="LCM 少步蒸馏:相邻 ODE 步的自举一致性,把任意 t 的加噪样本一步映射回 x0。",
        metrics=(
            "total_loss(一致性总损失)",
            "consistency_rms(学生/目标干净样本差的 RMS)",
            "raw_huber(未加权的逐样本一致性误差)",
        ),
        visualizations=(
            "学生一步采样 vs 教师多步采样对比网格",
            "consistency_rms 曲线(应随训练单调下降)",
        ),
        notes=("无需 reward,也不需要独立 fake-score 网络;目标网络由调用方以 EMA 维护并 detach。",),
    ),
}

# reward 插件 → 它实际产出的指标。
REWARD_METRICS: dict[str, tuple[str, ...]] = {
    "rational_rewards": (
        "Instruction_Following(0-1)",
        "Consistency(0-1)",
        "Visual_Quality(0-1)",
        "overall(加权综合,0-1)",
    ),
    "hpsv3pp": ("hps 美学分(HPSv3++ 原始分)",),
    "pickscore": ("pickscore 偏好分(对齐人类偏好的概率)",),
    "edit_r1_mllm": (
        "Instruction_Following(1-5 → 0-1)",
        "Consistency(1-5 → 0-1)",
        "Visual_Quality(1-5 → 0-1)",
        "overall(加权综合)",
    ),
    "professional_ocr": ("OCR 准确率(目标文本匹配度)",),
    "identity": ("cosine 身份相似度(ArcFace 5 点对齐,1 - cosine_identity_loss)",),
    "none": (),
}

# 晋级阶梯下一步建议(对齐 USAGE.md 的 qualification 顺序)。
NEXT_STEP_GUIDANCE: tuple[str, ...] = (
    "1. CPU preflight:`mmtrain phase run --phase preflight` 验证组件可构造、契约成立。",
    "2. GPU no-update:跑一次前向,确认无参数更新下数值有限、显存可控。",
    "3. one optimizer update:做一次完整更新,确认 loss 有限、梯度非 nan。",
    "4. checkpoint readback:写出再读回 checkpoint,核对 SHA-256 与参数一致。",
    "5. 冻结评估 + 质量 receipt:固定权重评估,登记 receipt 后才可 launch_enabled=true。",
)
