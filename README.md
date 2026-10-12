# 🪀 YoYo — *You Only Yield Once*

> **Composable Post-Training, Distillation & Reinforcement Learning Scaffold for Flow-Matching and Diffusion Models**
>
> *Write the mathematical primitives and model adapter once — yield any SFT, DPO, distillation, RL, or inference pipeline by composing components.*

[![CI & Architecture Gates](https://github.com/jackeyloveseven/yoyo/actions/workflows/ci.yml/badge.svg)](https://github.com/jackeyloveseven/yoyo/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10–3.13](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB.svg)](pyproject.toml)

---

## Why YoYo?

Most multimodal diffusion and flow-matching repositories devolve into sprawling, single-use training scripts where model conditioning, ODE solvers, FSDP2 wrappers, and objective math are copy-pasted across dozens of directories.

**YoYo** takes the opposite approach:
- **Pure Composable Primitives**: Every flow-matching sampler, ODE step, learning-rate schedule, sparse LoRA-MoE router, distillation regularizer, and RL surrogate lives in [`src/mmtrainer/`](src/mmtrainer) as an independent, unit-tested building block.
- **25-Line Model Onboarding**: Any diffusion or rectified-flow architecture (FLUX.2-klein, Qwen-Image, Krea2 Single-Stream MMDiT, FireRed, SenseNova, HiDream, or your next custom backbone) plugs in by subclassing [`BaseModelAdapter`](src/mmtrainer/plugins/base.py).
- **Zero-Bloat & Strict AST Gates**: Enforced automatically on every commit via [`check_v2_quality.py`](check_v2_quality.py), [`check_structure.py`](check_structure.py), and [`check_docs.py`](check_docs.py) — zero cross-model imports, zero 50-token code duplication, cyclomatic complexity $\le 15$, and zero eager GPU imports at CLI startup.

---

## Architecture & Component Map

```text
yoyo/
├── src/mmtrainer/
│   ├── nn/                  # Flow interpolation, Euler/DPM2 ODE steps, WSD/Cosine schedules, FSDP2, Sparse LoRA-MoE
│   ├── models/              # Composable adapters (Klein, Qwen-Image, Krea2 MMDiT, FireRed, SenseNova, HiDream)
│   ├── algorithms/
│   │   ├── sft_dpo/         # Flow SFT, Coupled-Noise DPO, OPD, DMD2, GT-Reg, NormGuard, REPA
│   │   ├── diffusion_alignment/ # DiffusionNFT, Asymmetric DPO, EMA policy updates
│   │   ├── flow_grpo/       # SDE transitions with log-prob, Group Relative Advantages, Clipped PPO + Reference KL
│   │   ├── diffusion_opsd/  # On-Policy Self-Distillation (bounded trust-region reward targets)
│   │   ├── flowmimic_opsd/  # Clean-distribution matching OPSD
│   │   └── edit_r1/         # MLLM-guided Edit-R1 optimality shaping
│   ├── datasets/            # JSONL manifests for T2I, single/multi-reference editing, and preference pairs
│   ├── rewards/             # 5-point differentiable ArcFace identity loss & RationalRewards VLM parser
│   ├── engine/              # Phase lifecycle (preflight -> train -> evaluate -> publish)
│   ├── registry/            # Deterministic lazy plugin registry
│   └── runtime/             # Atomic SHA-256 checkpoint manifests & compact BF16 parameter state
├── examples/                # 6 self-contained scripts composing components for SFT, DPO, Distill, RL & Inference
├── docs/                    # Algorithm derivations, ADRs, and numerical characterizations
└── tests/v2/                # 50+ unit and integration tests covering all components and examples
```

---

## Quickstart

### 1. Installation

```bash
git clone https://github.com/jackeyloveseven/yoyo.git
cd yoyo
python3 -m pip install -e ".[test,checkpoint]"
```

### 2. Inspect Registered Plugins & Validate Recipes

```bash
mmtrain registry list
mmtrain plan validate --spec recipes/v2/firered_diffusionnft.json
```

### 3. Run Composable Training & RL Examples

Every script in [`examples/`](examples/README.md) runs out-of-the-box on CPU or GPU to verify the full forward/backward component pipeline:

```bash
# 1. Supervised Flow-Matching SFT (Krea2 SingleStreamDiT + Cosine LR)
PYTHONPATH=src:. python3 examples/train_sft_edit.py

# 2. Coupled-Noise Flow-DPO Preference Alignment
PYTHONPATH=src:. python3 examples/train_dpo_preference.py

# 3. Multi-Objective Distillation (OPD + DMD2 + NormGuard + REPA + Sparse LoRA-MoE)
PYTHONPATH=src:. python3 examples/train_distill_opd_dmd.py

# 4. Online RL with DiffusionNFT + VLM Multi-Dimensional & Identity Rewards
PYTHONPATH=src:. python3 examples/train_rl_diffusionnft.py

# 5. Trajectory RL with Flow-GRPO & DiffusionOPSD
PYTHONPATH=src:. python3 examples/train_rl_flow_grpo.py

# 6. Plugging in a Brand-New Model Architecture + Euler/DPM2 Rollout Inference
PYTHONPATH=src:. python3 examples/custom_model_and_infer.py
```

---

## Plugging In a New Model Architecture

To integrate a new flow or diffusion model, subclass [`BaseModelAdapter`](src/mmtrainer/plugins/base.py) and combine it with any objective in `mmtrainer.algorithms`:

```python
import torch
from mmtrainer.algorithms.sft_dpo import flow_sft_loss
from mmtrainer.nn import euler_flow_step, flow_sample, resolution_shifted_timesteps
from mmtrainer.plugins.base import BaseModelAdapter


class MyCustomAdapter(BaseModelAdapter):
    model_id = "my_custom_dit"

    def prepare_sft_batch(self, **kwargs):
        sample = flow_sample(kwargs["clean_latents"], sigmas=kwargs.get("sigmas"))
        return {
            "noisy": sample.noisy_latents,
            "target": sample.target_velocity,
            "sigmas": sample.sigmas,
            "cond": kwargs["cond"],
        }

    def predict_velocity(self, transformer, batch, **kwargs):
        return transformer(batch["noisy"], batch["sigmas"], batch["cond"]).float()
```

See [`examples/custom_model_and_infer.py`](examples/custom_model_and_infer.py) for a complete runnable demonstration including dynamic [`LazyRegistry`](src/mmtrainer/registry/core.py) registration and hybrid Euler/DPM2 rollout sampling.

---

## Documentation & Verification Gates

| Guide | Path |
| :--- | :--- |
| CLI & Lifecycle Usage | [`USAGE.md`](USAGE.md) |
| Layered Architecture & Dependency Rules | [`ARCHITECTURE_V2.md`](ARCHITECTURE_V2.md) |
| Contribution & Code Quality Standards | [`CONTRIBUTING_V2.md`](CONTRIBUTING_V2.md) |
| Composable Examples Index | [`examples/README.md`](examples/README.md) |
| Mathematical & Algorithm Specifications | [`docs/algorithms/README.md`](docs/algorithms/README.md) |
| Refactoring & Component Extraction Notes | [`REFACTORING_HANDOFF.md`](REFACTORING_HANDOFF.md) |

Run all verification gates locally before opening a pull request:

```bash
PYTHONPATH=src python3 check_v2_quality.py
python3 check_structure.py
python3 check_docs.py
PYTHONPATH=src:. pytest tests/v2 -q
```
