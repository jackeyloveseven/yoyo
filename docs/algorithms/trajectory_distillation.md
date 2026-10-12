# Trajectory Distribution Matching

`mmtrainer.algorithms.trajectory_distillation` contains model-neutral TDM
objectives and rectified-flow adapter primitives. It does not expose an end-to-end
trainer or a registered v2 ObjectivePlugin.

- `tdm_fake_score_loss` trains the fake-score role.
- `tdm_generator_loss` evaluates the stopped cooperative student target.
- `RectifiedFlowTDMAdapter` owns generic anchor/noising orchestration through
  model-supplied velocity callbacks.
- `RectifiedFlowTDMConfig`, `TDMStudentSample`, and `TDMScoreBatch` validate the
  state passed between the two updates.

A production integration needs logically separate student, teacher, fake-score,
and optional unconditional-teacher roles. Student and fake-score optimizer state
must remain independent even when the roles share a heavyweight base through
adapter storage contexts.

Plugin registration, checkpoint encoding, model conditioning, and GPU qualification
remain future integration work.

Primary reference: [Trajectory Distribution Matching](https://arxiv.org/abs/2503.06674).
