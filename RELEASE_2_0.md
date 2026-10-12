# Training Code 2.0

Training Code 2.0 建立新的 typed RunSpec、单一 CLI、懒加载插件注册表、模型中立 phase engine、v2
checkpoint 完整性合同和一组经 v1 数值对照的 diffusion 算法原语。2.x 是合并后的 `main`；开发分支
`release/training-code-v2.x` 只用于 PR，不作为长期平行主线。

## 发布状态

- 源码版本：`2.0.1`，标签 `training-code-v2.0.1`；2.0.1 从 main 移除误带入的 v1 冻结环境清单，
  该清单仍完整保存在 `training-code-v1.9.0` tag/release。
- v1 最终版本：`training-code-v1.9.0`，仅供旧 full-state exact-resume。
- FireRed、SenseNova、Klein、Qwen Image、HiDream 和 Krea2 已登记为 lazy model plugin。
- DiffusionNFT、Flow-GRPO、DiffusionOPSD、FlowMimic-OPSD、Image SFT/DPO/OPD 已登记为 objective plugin。
- 所有 v2 recipe 和四个 WebTrainer entrypoint template 默认禁发；本发布不声明任何 GPU 资格或模型晋级。

## 验证

```bash
PYTHONPATH=training_code/src python3 training_code/check_v2_quality.py
PYTHONPATH=training_code/src:training_code python3 -m pytest -q training_code/tests/v2
cd webtrainer/rust && ./scripts/cargo.sh test -p webtrainer-recipes
```

算法对照 receipt 位于
[`receipts/v2/v1_v2_algorithm_characterization.json`](receipts/v2/v1_v2_algorithm_characterization.json)。
v2 只有在对应 model plugin 完成 no-update、one-update 和 checkpoint readback 后，才能发布新的 recipe
版本把 `launch_enabled` 改为 true。
