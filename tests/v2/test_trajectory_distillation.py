"""Unit tests for Trajectory Distribution Matching (TDM) objectives and flow adapter."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mmtrainer.algorithms.trajectory_distillation import (  # noqa: E402
    RectifiedFlowTDMAdapter,
    RectifiedFlowTDMConfig,
    TDMFakeScoreMetrics,
    TDMGeneratorMetrics,
    TDMScoreBatch,
    TDMStudentSample,
    TDMTeacherRollout,
    euler_flow_step,
    heun_midpoint_step,
    interpolate_descending_time_grid,
    tdm_cooperative_distribution_matching_loss,
    tdm_fake_score_loss,
    tdm_generator_loss,
)


def _batch_clean(shape: tuple[int, ...] = (2, 3, 4, 4)) -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(42)
    fake_clean = torch.randn(shape, generator=gen)
    student_clean = torch.randn(shape, generator=gen)
    return fake_clean, student_clean


# ==============================================================================
# 1. tdm_fake_score_loss Tests
# ==============================================================================


def test_tdm_fake_score_loss_exact_match_zero() -> None:
    """When fake matches student clean predictions, loss and RMS are strictly 0."""
    fake_clean, _ = _batch_clean()
    student_clean = fake_clean.clone()
    leading = fake_clean.shape[:1]  # default event_dims = (1, 2, 3)
    snr = torch.ones(leading)
    importance = torch.ones(leading)

    metrics = tdm_fake_score_loss(fake_clean, student_clean, snr, importance)
    assert isinstance(metrics, TDMFakeScoreMetrics)
    assert metrics.total_loss.ndim == 0
    assert metrics.total_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert metrics.raw_mse.abs().max().item() == pytest.approx(0.0, abs=1e-7)
    assert metrics.fake_student_rms.item() == pytest.approx(0.0, abs=1e-7)
    assert metrics.per_sample_loss.shape == leading


def test_tdm_fake_score_loss_shape_and_tensor_mismatch() -> None:
    """Shape, dtype, or device mismatch between fake and student raises ValueError."""
    fake_clean, student_clean = _batch_clean((2, 3, 4, 4))
    leading = (2,)
    snr = torch.ones(leading)
    importance = torch.ones(leading)

    # Shape mismatch
    with pytest.raises(ValueError, match="must match fake_clean shape and device"):
        tdm_fake_score_loss(fake_clean[:1], student_clean, snr[:1], importance[:1])

    with pytest.raises(ValueError, match="must match fake_clean shape and device"):
        tdm_fake_score_loss(fake_clean, student_clean[:, :2], snr, importance)


def test_tdm_fake_score_loss_event_dims_handling() -> None:
    """event_dims correctly defines the reduction dimensions and leading shape."""
    shape = (2, 3, 4, 5)
    fake_clean, student_clean = _batch_clean(shape)

    # event_dims=None defaults to range(1, ndim) -> leading shape is (2,)
    snr_1 = torch.ones(2)
    imp_1 = torch.ones(2)
    m_default = tdm_fake_score_loss(fake_clean, student_clean, snr_1, imp_1, event_dims=None)
    assert m_default.per_sample_loss.shape == (2,)

    # event_dims=2 -> trailing 2 dims (4, 5) are events, leading shape is (2, 3)
    snr_2 = torch.ones(2, 3)
    imp_2 = torch.ones(2, 3)
    m_2 = tdm_fake_score_loss(fake_clean, student_clean, snr_2, imp_2, event_dims=2)
    assert m_2.per_sample_loss.shape == (2, 3)

    # event_dims=(2, 3) sequence matches event_dims=2
    m_seq = tdm_fake_score_loss(fake_clean, student_clean, snr_2, imp_2, event_dims=(2, 3))
    assert torch.allclose(m_2.per_sample_loss, m_seq.per_sample_loss)

    # event_dims=0 -> all dimensions are leading
    snr_all = torch.ones(shape)
    imp_all = torch.ones(shape)
    m_0 = tdm_fake_score_loss(fake_clean, student_clean, snr_all, imp_all, event_dims=0)
    assert m_0.per_sample_loss.shape == shape

    # Weight shape mismatch with expected leading shape
    with pytest.raises(ValueError, match="must have leading shape"):
        tdm_fake_score_loss(fake_clean, student_clean, torch.ones(2), imp_2, event_dims=2)

    # Non-contiguous trailing block raises ValueError
    with pytest.raises(ValueError, match="contiguous trailing block"):
        tdm_fake_score_loss(fake_clean, student_clean, torch.ones(2, 4), imp_2, event_dims=(1, 3))

    # Invalid event_dims count
    with pytest.raises(ValueError, match="event_dims count is invalid"):
        tdm_fake_score_loss(fake_clean, student_clean, snr_1, imp_1, event_dims=10)


def test_tdm_fake_score_loss_snr_and_importance_weighting() -> None:
    """SNR weights and importance weights scale per_sample_loss multiplicatively."""
    fake_clean, student_clean = _batch_clean((2, 3, 4, 4))
    snr_base = torch.tensor([1.0, 1.0])
    imp_base = torch.tensor([1.0, 1.0])

    base_metrics = tdm_fake_score_loss(fake_clean, student_clean, snr_base, imp_base)

    # Double SNR: loss doubles, raw MSE and RMS unchanged
    snr_double = torch.tensor([2.0, 2.0])
    metrics_double = tdm_fake_score_loss(fake_clean, student_clean, snr_double, imp_base)
    assert metrics_double.total_loss.item() == pytest.approx(
        base_metrics.total_loss.item() * 2, rel=1e-6
    )
    assert torch.allclose(metrics_double.raw_mse, base_metrics.raw_mse)
    assert metrics_double.fake_student_rms.item() == pytest.approx(
        base_metrics.fake_student_rms.item(), rel=1e-6
    )

    # Non-uniform importance weights
    imp_custom = torch.tensor([0.5, 1.5])
    metrics_custom = tdm_fake_score_loss(fake_clean, student_clean, snr_base, imp_custom)
    expected_per_sample = base_metrics.raw_mse * snr_base * imp_custom
    assert torch.allclose(metrics_custom.per_sample_loss, expected_per_sample)

    # Negative weights rejected
    with pytest.raises(ValueError, match="must be non-negative"):
        tdm_fake_score_loss(fake_clean, student_clean, torch.tensor([-1.0, 1.0]), imp_base)
    with pytest.raises(ValueError, match="must be non-negative"):
        tdm_fake_score_loss(fake_clean, student_clean, snr_base, torch.tensor([1.0, -0.5]))


def test_tdm_fake_score_loss_rms_calculation() -> None:
    """fake_student_rms accurately computes root-mean-square error."""
    fake_clean = torch.full((2, 3, 4, 4), 2.5)
    offset = 1.5
    student_clean = torch.full((2, 3, 4, 4), 2.5 + offset)
    snr = torch.ones(2)
    importance = torch.ones(2)

    metrics = tdm_fake_score_loss(fake_clean, student_clean, snr, importance)
    assert metrics.fake_student_rms.item() == pytest.approx(offset, rel=1e-6)
    raw_mse_val = (
        metrics.raw_mse.item() if metrics.raw_mse.ndim == 0 else metrics.raw_mse.mean().item()
    )
    assert raw_mse_val == pytest.approx(offset**2, rel=1e-6)


def test_tdm_fake_score_loss_non_finite_rejection() -> None:
    """NaN or Inf in any tensor raises ValueError."""
    fake_clean, student_clean = _batch_clean((2, 3, 4, 4))
    snr = torch.ones(2)
    importance = torch.ones(2)

    nan_fake = fake_clean.clone()
    nan_fake[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="must contain only finite values"):
        tdm_fake_score_loss(nan_fake, student_clean, snr, importance)

    inf_student = student_clean.clone()
    inf_student[1, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="must contain only finite values"):
        tdm_fake_score_loss(fake_clean, inf_student, snr, importance)

    nan_snr = snr.clone()
    nan_snr[0] = float("nan")
    with pytest.raises(ValueError, match="must contain only finite values"):
        tdm_fake_score_loss(fake_clean, student_clean, nan_snr, importance)

    inf_imp = importance.clone()
    inf_imp[1] = float("inf")
    with pytest.raises(ValueError, match="must contain only finite values"):
        tdm_fake_score_loss(fake_clean, student_clean, snr, inf_imp)


# ==============================================================================
# 2. tdm_cooperative_distribution_matching_loss Tests
# ==============================================================================


def test_tdm_cooperative_alias_identity() -> None:
    """tdm_cooperative_distribution_matching_loss is identical to tdm_generator_loss."""
    assert tdm_cooperative_distribution_matching_loss is tdm_generator_loss


def test_tdm_cooperative_loss_closed_form_exact_match_zero() -> None:
    """When student == teacher and fake matches teacher, cooperative loss is strictly 0."""
    state = torch.randn(2, 3, 4, 4)
    # student == teacher and fake == teacher -> cooperative = student, delta = 0
    metrics = tdm_cooperative_distribution_matching_loss(state, state, state)

    assert isinstance(metrics, TDMGeneratorMetrics)
    assert metrics.total_loss.ndim == 0
    assert metrics.total_loss.item() == pytest.approx(0.0, abs=1e-7)
    assert metrics.teacher_fake_delta_rms.item() == pytest.approx(0.0, abs=1e-7)
    assert metrics.student_teacher_rms.item() == pytest.approx(0.0, abs=1e-7)
    assert torch.allclose(metrics.cooperative_target, state)


def test_tdm_cooperative_loss_analytic_offset() -> None:
    """Analytically verify cooperative target and loss value with huber_c=None."""
    # 1D single-element tensors for exact manual math
    student = torch.tensor([[1.0, 2.0]])
    teacher = torch.tensor([[3.0, 4.0]])
    fake = torch.tensor([[2.0, 3.0]])

    # cooperative = (student + teacher - fake) = [1+3-2, 2+4-3] = [2.0, 3.0]
    # normalizer_error = |student - teacher| = [2.0, 2.0]
    # normalizer = 2.0 (event_dims=(1,) over 2 elements)
    # delta = student - cooperative = [1-2, 2-3] = [-1.0, -1.0]
    # pointwise = delta^2 / normalizer = [1.0/2.0, 1.0/2.0] = [0.5, 0.5]
    # per_sample = mean([0.5, 0.5]) = 0.5
    metrics = tdm_cooperative_distribution_matching_loss(
        student, teacher, fake, huber_c=None, event_dims=1
    )
    expected_coop = torch.tensor([[2.0, 3.0]])
    assert torch.allclose(metrics.cooperative_target, expected_coop)
    assert metrics.normalizer.item() == pytest.approx(2.0, abs=1e-6)
    assert metrics.total_loss.item() == pytest.approx(0.5, abs=1e-6)
    assert metrics.teacher_fake_delta_rms.item() == pytest.approx(1.0, abs=1e-6)
    assert metrics.student_teacher_rms.item() == pytest.approx(2.0, abs=1e-6)


def test_tdm_cooperative_loss_guidance_scale_cfg() -> None:
    """CFG scales teacher clean prediction: unconditional + cfg_scale * (cond - uncond)."""
    student = torch.tensor([[1.0, 1.0]])
    teacher_cond = torch.tensor([[3.0, 5.0]])
    teacher_uncond = torch.tensor([[1.0, 1.0]])
    fake = torch.tensor([[2.0, 2.0]])

    cfg_scale = 2.0
    # teacher_cfg = uncond + cfg_scale * (cond - uncond) = [1+2*2, 1+2*4] = [5.0, 9.0]
    # cooperative = student + teacher_cfg - fake = [1+5-2, 1+9-2] = [4.0, 8.0]
    metrics = tdm_cooperative_distribution_matching_loss(
        student,
        teacher_cond,
        fake,
        teacher_unconditional_clean=teacher_uncond,
        cfg_scale=cfg_scale,
        huber_c=None,
        event_dims=1,
    )
    expected_coop = torch.tensor([[4.0, 8.0]])
    assert torch.allclose(metrics.cooperative_target, expected_coop)

    # teacher_fake_delta_rms = rms(teacher_cfg - fake) = rms([5-2, 9-2]) = sqrt((9+49)/2) = sqrt(29)
    expected_delta_rms = (torch.tensor([3.0, 7.0]).square().mean()).sqrt().item()
    assert metrics.teacher_fake_delta_rms.item() == pytest.approx(expected_delta_rms, rel=1e-6)

    # CFG scale validation errors
    with pytest.raises(ValueError, match="CFG requires teacher_unconditional_clean"):
        tdm_cooperative_distribution_matching_loss(student, teacher_cond, fake, cfg_scale=1.5)

    with pytest.raises(ValueError, match="cfg_scale must be finite and at least 1"):
        tdm_cooperative_distribution_matching_loss(
            student, teacher_cond, fake, teacher_unconditional_clean=teacher_uncond, cfg_scale=0.5
        )


def test_tdm_cooperative_loss_gradient_flow_and_target_detach() -> None:
    """Target is detached; only student receives gradient."""
    student = torch.randn(2, 3, 4, 4, requires_grad=True)
    teacher = torch.randn(2, 3, 4, 4, requires_grad=True)
    fake = torch.randn(2, 3, 4, 4, requires_grad=True)
    uncond = torch.randn(2, 3, 4, 4, requires_grad=True)

    metrics = tdm_cooperative_distribution_matching_loss(
        student, teacher, fake, teacher_unconditional_clean=uncond, cfg_scale=1.5
    )
    # Cooperative target must be detached
    assert not metrics.cooperative_target.requires_grad
    assert not metrics.normalizer.requires_grad

    metrics.total_loss.backward()

    # Student received non-zero gradients
    assert student.grad is not None
    assert student.grad.abs().sum().item() > 0

    # Teacher, fake, and unconditional teacher received no gradients
    assert teacher.grad is None
    assert fake.grad is None
    assert uncond.grad is None


def test_tdm_cooperative_loss_non_finite_and_parameter_rejection() -> None:
    """Non-finite inputs or invalid hyperparameters raise ValueError."""
    shape = (2, 3, 4, 4)
    student = torch.randn(shape)
    teacher = torch.randn(shape)
    fake = torch.randn(shape)

    # NaN student
    nan_student = student.clone()
    nan_student[0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="must contain only finite values"):
        tdm_cooperative_distribution_matching_loss(nan_student, teacher, fake)

    # Inf teacher
    inf_teacher = teacher.clone()
    inf_teacher[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="must contain only finite values"):
        tdm_cooperative_distribution_matching_loss(student, inf_teacher, fake)

    # Inf fake
    inf_fake = fake.clone()
    inf_fake[0, 0, 0, 0] = float("inf")
    with pytest.raises(ValueError, match="must contain only finite values"):
        tdm_cooperative_distribution_matching_loss(student, teacher, inf_fake)

    # Invalid huber_c
    with pytest.raises(ValueError, match="huber_c must be finite and positive"):
        tdm_cooperative_distribution_matching_loss(student, teacher, fake, huber_c=0.0)
    with pytest.raises(ValueError, match="huber_c must be finite and positive"):
        tdm_cooperative_distribution_matching_loss(student, teacher, fake, huber_c=-0.1)

    # Invalid normalizer_floor
    with pytest.raises(ValueError, match="normalizer_floor must be finite and positive"):
        tdm_cooperative_distribution_matching_loss(student, teacher, fake, normalizer_floor=0.0)


# ==============================================================================
# 3. TDMTeacherRollout and ODE Stepping in rectified_flow.py
# ==============================================================================


def test_ode_stepping_methods_euler_and_heun() -> None:
    """Euler flow step (1st order) and Heun midpoint step (2nd order)."""
    latents = torch.tensor([[2.0, 4.0]])
    # Constant velocity field v = [1.0, 3.0]
    velocity = torch.tensor([[1.0, 3.0]])
    sigma_curr = 1.0
    sigma_next = 0.5
    # dt = 0.5 - 1.0 = -0.5
    # Euler: latents + dt * velocity = [2.0 - 0.5, 4.0 - 1.5] = [1.5, 2.5]
    step_euler = euler_flow_step(latents, velocity, sigma_curr, sigma_next)
    assert torch.allclose(step_euler, torch.tensor([[1.5, 2.5]]))

    # Heun with differing v_curr and v_next
    v_curr = torch.tensor([[1.0, 2.0]])
    v_next = torch.tensor([[3.0, 4.0]])
    # blended_v = 0.5 * (v_curr + v_next) = [2.0, 3.0]
    # Heun: latents + (-0.5) * blended_v = [2.0 - 1.0, 4.0 - 1.5] = [1.0, 2.5]
    step_heun = heun_midpoint_step(latents, v_curr, v_next, sigma_curr, sigma_next)
    assert torch.allclose(step_heun, torch.tensor([[1.0, 2.5]]))


def test_ode_stepping_accuracy_on_linear_ode() -> None:
    """Heun achieves higher accuracy than Euler on dx/dt = -x."""
    # dx/dt = -x, analytic solution x(t) = x_0 * exp(-(t - t_0))
    # Step from t=1.0 to t=0.0 with dt = -1.0
    x0 = torch.tensor([[1.0]])
    t_curr, t_next = 1.0, 0.0
    v_curr = -x0  # -1.0

    # Euler step: x_next = x0 + dt * (-x0) = 1.0 + (-1.0)*(-1.0) = 2.0
    x_euler = euler_flow_step(x0, v_curr, t_curr, t_next)
    assert x_euler.item() == pytest.approx(2.0)

    # Heun step: predictor x_pred = 2.0, v_next = -2.0, blended_v = 0.5 * (-1 - 2) = -1.5
    # x_heun = 1.0 + (-1.0) * (-1.5) = 2.5
    v_next = -x_euler
    x_heun = heun_midpoint_step(x0, v_curr, v_next, t_curr, t_next)
    assert x_heun.item() == pytest.approx(2.5)

    # Exact solution x(0) for dx/dt = -x going backwards from t=1 to t=0:
    # x(0) = x(1) * exp(1) = 1 * e = 2.71828...
    exact = 2.718281828
    euler_err = abs(x_euler.item() - exact)
    heun_err = abs(x_heun.item() - exact)
    assert heun_err < euler_err


def test_tdm_teacher_rollout_container() -> None:
    """TDMTeacherRollout stores endpoints and validates finite tensors."""
    clean = torch.randn(2, 4, 8)
    timesteps = torch.tensor([1.0, 0.5, 0.0])
    traj = (torch.randn(2, 4, 8), torch.randn(2, 4, 8), clean)

    rollout = TDMTeacherRollout(clean=clean, timesteps=timesteps, trajectory=traj)
    assert torch.equal(rollout.clean, clean)
    assert torch.equal(rollout.timesteps, timesteps)
    assert len(rollout.trajectory) == 3

    # Non-finite rejection
    with pytest.raises(ValueError, match="must contain only finite values"):
        TDMTeacherRollout(clean=torch.tensor([float("nan")]))
    with pytest.raises(ValueError, match="must contain only finite values"):
        TDMTeacherRollout(clean=clean, timesteps=torch.tensor([float("inf")]))


def test_rectified_flow_tdm_adapter_role_queries() -> None:
    """Adapter strictly queries role 'student', 'teacher', 'teacher_unconditional', 'fake'."""
    queries: list[str] = []

    def mock_initial_noise(
        conditioning: tuple[int, ...], generator: torch.Generator | None
    ) -> torch.Tensor:
        del conditioning, generator
        return torch.ones(2, 3, 4)

    def mock_velocity(
        state: torch.Tensor, time: torch.Tensor, conditioning: tuple[int, ...], role: str
    ) -> torch.Tensor:
        del time, conditioning
        queries.append(role)
        return torch.zeros_like(state)

    def mock_grid(conditioning: tuple[int, ...], steps: int) -> torch.Tensor:
        del conditioning
        return torch.linspace(1.0, 0.0, steps + 1)

    adapter = RectifiedFlowTDMAdapter(
        initial_noise_fn=mock_initial_noise,
        velocity_fn=mock_velocity,
        time_grid_fn=mock_grid,
        config=RectifiedFlowTDMConfig(event_dims=2),
    )

    # 1. sample_student queries 'student'
    sample = adapter.sample_student((2, 3, 4), student_steps=3, requires_grad=False)
    assert isinstance(sample, TDMStudentSample)
    assert all(q == "student" for q in queries)
    assert len(queries) > 0
    queries.clear()

    # 2. prepare_score_batch produces valid TDMScoreBatch
    score_batch = adapter.prepare_score_batch(sample, (2, 3, 4))
    assert isinstance(score_batch, TDMScoreBatch)
    assert score_batch.noisy.shape == (2, 3, 4)
    assert score_batch.snr_weight.shape == (2,)

    # 3. predict_clean queries specified model roles
    adapter.predict_clean(score_batch, (2, 3, 4), model="fake")
    assert queries[-1] == "fake"
    adapter.predict_clean(score_batch, (2, 3, 4), model="teacher")
    assert queries[-1] == "teacher"
    adapter.predict_clean(score_batch, (2, 3, 4), model="teacher_unconditional")
    assert queries[-1] == "teacher_unconditional"
    queries.clear()

    # 4. rollout_teacher with CFG=1.0 queries only 'teacher'
    rollout_euler = adapter.rollout_teacher(
        (2, 3, 4), teacher_steps=2, method="euler", cfg_scale=1.0
    )
    assert isinstance(rollout_euler, TDMTeacherRollout)
    assert all(q == "teacher" for q in queries)
    assert len(queries) == 2
    queries.clear()

    # 5. rollout_teacher with CFG > 1.0 queries 'teacher' and 'teacher_unconditional'
    rollout_cfg = adapter.rollout_teacher(
        (2, 3, 4), teacher_steps=2, method="euler", cfg_scale=2.0, return_trajectory=True
    )
    assert isinstance(rollout_cfg, TDMTeacherRollout)
    assert "teacher" in queries and "teacher_unconditional" in queries
    assert rollout_cfg.trajectory is not None and len(rollout_cfg.trajectory) == 3
    queries.clear()

    # 6. rollout_teacher with Heun method
    rollout_heun = adapter.rollout_teacher((2, 3, 4), teacher_steps=2, method="heun", cfg_scale=1.0)
    assert isinstance(rollout_heun, TDMTeacherRollout)
    # Heun takes 2 evaluations per step -> 4 queries
    assert len(queries) == 4
    assert all(q == "teacher" for q in queries)
    queries.clear()

    # 7. Unsupported role raises ValueError
    with pytest.raises(ValueError, match="unsupported model role"):
        adapter._velocity(
            torch.ones(2, 3, 4),
            torch.tensor([0.5, 0.5]),
            (2, 3, 4),
            "unsupported_role",  # type: ignore[arg-type]
        )


def test_interpolate_descending_time_grid() -> None:
    """Descending time grid maps noise times strictly within bounds."""
    noise_grid = torch.tensor([1.0, 0.75, 0.5, 0.0])
    model_grid = torch.tensor([1000.0, 750.0, 500.0, 0.0])

    # Exact points
    noise_time = torch.tensor([1.0, 0.75, 0.0])
    model_time = interpolate_descending_time_grid(noise_time, noise_grid, model_grid)
    assert torch.allclose(model_time, torch.tensor([1000.0, 750.0, 0.0]))

    # Midpoints
    mid_noise = torch.tensor([0.875, 0.25])
    mid_model = interpolate_descending_time_grid(mid_noise, noise_grid, model_grid)
    assert torch.allclose(mid_model, torch.tensor([875.0, 250.0]))

    # Out of bounds raises ValueError
    with pytest.raises(ValueError, match="noise_time lies outside the paired grid"):
        interpolate_descending_time_grid(torch.tensor([1.1]), noise_grid, model_grid)
    with pytest.raises(ValueError, match="noise_time lies outside the paired grid"):
        interpolate_descending_time_grid(torch.tensor([-0.1]), noise_grid, model_grid)

    # Non-descending grid raises ValueError
    with pytest.raises(ValueError, match="noise_grid must be strictly descending"):
        interpolate_descending_time_grid(
            torch.tensor([0.5]), torch.tensor([0.0, 1.0]), torch.tensor([0.0, 1000.0])
        )
