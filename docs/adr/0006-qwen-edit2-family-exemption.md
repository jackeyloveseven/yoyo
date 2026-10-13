# ADR 0006:模型家族版本演化的显式豁免(白名单)

- 状态:已接受
- 日期:2026-10-13
- 版本:2.0.1

## 1. 背景与决策

导入 Qwen-Image-Edit 2.x(2509/2511)适配器时(`qwen_image_edit2.py`),同批零件的
架构门(ADR-0002)给出两条冲突的指令:

1. **禁止跨模型 import**:`models/<a>` 不得 import `models/<b>`,本意是防止模型适配器
   之间长出一张耦合网(改一个模型引发另一个模型爆炸)。照此,`qwen_image_edit2` 不得
   import `qwen_image`;
2. **禁止 50-token 重复窗口**:两个文件不得共享 50 个连续规范化 token,本意是防止
   同一段逻辑复制两遍后只改一处。照此,edit2 不得把原版的 VAE 编码/latent 打包/
   前向协议重写一遍。

**决策**:承认"模型家族版本演进"是合法场景,在
[`src/mmtrainer/quality/audit.py`](../../src/mmtrainer/quality/audit.py) 引入
`MODEL_FAMILY_EXEMPTIONS` 映射(键 = 子版本模块,值 = 它被允许依赖的家族基线模块),
**只放开 import 单向边**,重复窗口门不变。

**依据**:2.x 与原版事实同源——共享 Qwen2.5-VL 文本编码器、Wan2.1-VAE、MMDiT 前向
签名(差异仅在 diffusers 管线名与 1-3 张控制图)。这不是"两个模型互借代码",而是
"同一模型新版本复用旧版已验证零件"。

## 2. 备选方案与后果

- **备选 A(若两门都硬守)**:edit2 重写原版的 VAE 编码/pack/前向协议。放弃:既触发
  重复窗口门,又得实质差异(重写的代码与原版逐字相同才算功能相同)——门在要求
  "复制",与门的目的相反。
- **备选 B:把共享零件上移到公共层(如 `models/_qwen_common.py`)。** 放弃:原版函数
  (`encode_vae_image` / `pack_qwen_latents` / `predict_qwen`)已在其模块内稳定且被
  现有代码引用,上移 = 改原版公开面,为一个新版本付全局改动的代价;且新文件仍与
  原版文件共享这些函数的调用处 token,重复窗口门的风险没消失。
- **备选 C:什么也不做,让 `qwen_image_edit_2x` 保持不可用。** 放弃:多图编辑
  (1-3 张参考图)是 Qwen 家族 2509+ 的实际用法,放弃即放弃该能力。
- **后果(约束一)**:每条豁免必须在 `docs/adr/` 有对应 ADR 记录理由——防止白名单
  沦为"随手绕过架构门"的后门。新增豁免必须先写 ADR。
- **后果(约束二)**:豁免只单向解除 import 禁令,重复窗口门全量照常执行。edit2 因此
  必须**真复用**原版函数(如 `encode_qwen_prompt`),而不是引用一次再抄一遍。
- **后果(约束三)**:豁免是版本演进而非家族扩散——未来 Qwen 3.x 只应出现在
  `qwen_image_edit3` → {`qwen_image`, `qwen_image_edit2`},不允许 edit2 反向依赖
  子版本形成的链式豁免(门只查一层显式映射,但 ADR 约定不扩张)。

## 3. 受影响的稳定契约

- `audit.py` 新增模块级常量 `MODEL_FAMILY_EXEMPTIONS`(当前唯一条目:
  `qwen_image_edit2` → `qwen_image`);
- `qwen_image.py` 新增公共函数 `encode_qwen_prompt`(把 SFT/DPO 里重复的
  `pipeline.encode_prompt` 调用块抽成一份,edit2 复用;否则重复窗口门判违规);
- `qwen_image_edit2.py` 的 `control_pil_images` 注解改为 `list[list[Any]]`——
  Plus 管线契约是每个 prompt 一组控制图(嵌套列表),原版的扁平列表不对;
- 能力矩阵(`docs/models/README.md`)新增一列,由 `test_model_cards.py` 守同步。

## 4. 同一迁移中移除/修正的旧物

无文件移除。修正:`tests/v2/test_cli.py` 的模型清单硬编码列表改为从注册表动态取
(与 `test_model_cards.py` 同类问题——每加一个模型改一次测试)。

## 5. 验证与回滚边界

- **验证**:`check_v2_quality.py` 全绿(0 violations);`tests/v2` 全过(含 edit2 5 个
  新单测:多流形状、单控制退化、控制图上限/维度校验、DPO 成对共享、适配器表面);
  ruff / format / check_structure / check_docs 全绿。
- **回滚**:删 `qwen_image_edit2.py`、其单测、能力卡、注册表两行,并撤
  `MODEL_FAMILY_EXEMPTIONS` 与 `encode_qwen_prompt`(仅 edit2 使用),改动局部。
- **何时可推翻**:若未来出现第二个以上家族版本适配器、且共享面超出"函数复用"
  (如需要共享数据类或常量表),再评估备选 B(公共模块)是否变得更便宜。
- **遗留**:`QWEN_EDIT2_LORA_TARGET_MODULES` 继承原版 12 模块名,**未 GPU 验证**;
  晋级时若命名漂移以实际 `named_modules` 扫描为准。
