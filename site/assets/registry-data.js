/* YoYo 注册表离线快照 — 由 build_builtin_registries() 生成，请勿手改 */
window.YOYO_REGISTRY = {
 "source": "mmtrainer.registry build_builtin_registries()",
 "note": "离线内联快照；与 docs/models/README.md 同源，由 tests/v2/test_model_cards.py 强制一致。",
 "models": [
  {
   "id": "hidream",
   "name": "HiDream",
   "desc": "HiDream 编辑 + T2I，复用 SenseNova 适配器与 LoRA 靶点。",
   "capabilities": [
    "image_edit",
    "image_training",
    "t2i"
   ]
  },
  {
   "id": "klein",
   "name": "Klein",
   "desc": "FLUX.2-klein 编辑 + T2I，BF16 参数更新有实测数值记录。",
   "capabilities": [
    "image_edit",
    "image_training",
    "t2i"
   ]
  },
  {
   "id": "krea2",
   "name": "Krea2",
   "desc": "Krea2 单流 MMDiT，唯一同时跨 RL 系与 SFT/DPO 系的已注册模型。",
   "capabilities": [
    "diffusion_alignment",
    "image_edit",
    "image_training"
   ]
  },
  {
   "id": "qwen_image",
   "name": "Qwen-Image",
   "desc": "Qwen-Image 编辑 + 监督训练，成对预测拆分。",
   "capabilities": [
    "image_edit",
    "image_training"
   ]
  },
  {
   "id": "qwen_image_edit_2x",
   "name": "Qwen-Image-Edit 2.x",
   "desc": "Qwen-Image-Edit 2509/2511，1-3 张控制图多图编辑。",
   "capabilities": [
    "image_edit",
    "image_training"
   ]
  },
  {
   "id": "sensenova",
   "name": "SenseNova",
   "desc": "SenseNova 多流 latent，编辑 / T2I + 全 RL 系。",
   "capabilities": [
    "diffusion_alignment",
    "image_edit",
    "lora",
    "t2i"
   ]
  }
 ],
 "objectives": [
  {
   "id": "diffusionnft",
   "family": "RL",
   "desc": "DiffusionNFT 在线 RL：有界 optimality 概率驱动的隐式正/负策略 + 固定基准 KL。",
   "reward_required": true,
   "required": [
    "diffusion_alignment"
   ],
   "row": {
    "klein": false,
    "qwen_image": false,
    "qwen_image_edit_2x": false,
    "hidream": false,
    "sensenova": true,
    "krea2": true
   }
  },
  {
   "id": "flow_grpo",
   "family": "RL",
   "desc": "Flow-GRPO 轨迹 RL：SDE 转移带对数概率，组相对优势 + 裁剪 PPO + 参考 KL。",
   "reward_required": true,
   "required": [
    "diffusion_alignment"
   ],
   "row": {
    "klein": false,
    "qwen_image": false,
    "qwen_image_edit_2x": false,
    "hidream": false,
    "sensenova": true,
    "krea2": true
   }
  },
  {
   "id": "diffusion_opsd",
   "family": "RL",
   "desc": "DiffusionOPSD 在线自蒸馏：有界信任域内的奖励目标拟合。",
   "reward_required": true,
   "required": [
    "diffusion_alignment"
   ],
   "row": {
    "klein": false,
    "qwen_image": false,
    "qwen_image_edit_2x": false,
    "hidream": false,
    "sensenova": true,
    "krea2": true
   }
  },
  {
   "id": "edit_r1",
   "family": "RL",
   "desc": "Edit-R1：DiffusionNFT + 非 CoT MLLM 评分 token logits 的图像编辑 RL。",
   "reward_required": true,
   "required": [
    "diffusion_alignment",
    "image_edit"
   ],
   "row": {
    "klein": false,
    "qwen_image": false,
    "qwen_image_edit_2x": false,
    "hidream": false,
    "sensenova": true,
    "krea2": true
   }
  },
  {
   "id": "flowmimic_opsd",
   "family": "蒸馏",
   "desc": "FlowMimic-OPSD：干净分布匹配的在线自蒸馏，无需 reward。",
   "reward_required": false,
   "required": [
    "diffusion_alignment"
   ],
   "row": {
    "klein": false,
    "qwen_image": false,
    "qwen_image_edit_2x": false,
    "hidream": false,
    "sensenova": true,
    "krea2": true
   }
  },
  {
   "id": "image_sft",
   "family": "SFT",
   "desc": "Flow SFT：监督流匹配，最小化预测速度与目标速度的 MSE。",
   "reward_required": false,
   "required": [
    "image_training"
   ],
   "row": {
    "klein": true,
    "qwen_image": true,
    "qwen_image_edit_2x": true,
    "hidream": true,
    "sensenova": false,
    "krea2": true
   }
  },
  {
   "id": "image_dpo",
   "family": "DPO",
   "desc": "耦合噪声 Flow-DPO：共享噪声的赢/输对偏好优化。",
   "reward_required": false,
   "required": [
    "image_training"
   ],
   "row": {
    "klein": true,
    "qwen_image": true,
    "qwen_image_edit_2x": true,
    "hidream": true,
    "sensenova": false,
    "krea2": true
   }
  },
  {
   "id": "image_opd",
   "family": "OPD",
   "desc": "OPD 在线偏好蒸馏：ODE 轨迹上的偏好对齐（需 reward）。",
   "reward_required": true,
   "required": [
    "image_training"
   ],
   "row": {
    "klein": true,
    "qwen_image": true,
    "qwen_image_edit_2x": true,
    "hidream": true,
    "sensenova": false,
    "krea2": true
   }
  },
  {
   "id": "latent_consistency",
   "family": "LCM",
   "desc": "LCM 少步蒸馏：相邻 ODE 步的自举一致性。",
   "reward_required": false,
   "required": [
    "image_training"
   ],
   "row": {
    "klein": true,
    "qwen_image": true,
    "qwen_image_edit_2x": true,
    "hidream": true,
    "sensenova": false,
    "krea2": true
   }
  },
  {
   "id": "on_policy_distill",
   "family": "OPD",
   "desc": "On-Policy Distillation：学生一步预测对齐教师多步 Euler 轨迹终点。",
   "reward_required": false,
   "required": [
    "image_training"
   ],
   "row": {
    "klein": true,
    "qwen_image": true,
    "qwen_image_edit_2x": true,
    "hidream": true,
    "sensenova": false,
    "krea2": true
   }
  }
 ],
 "rewards": [
  "edit_r1_mllm",
  "hpsv3pp",
  "identity",
  "pickscore",
  "professional_ocr",
  "rational_rewards"
 ],
 "reward_metrics": {
  "rational_rewards": [
   "Instruction_Following(0-1)",
   "Consistency(0-1)",
   "Visual_Quality(0-1)",
   "overall(加权综合,0-1)"
  ],
  "hpsv3pp": [
   "hps 美学分(HPSv3++ 原始分)"
  ],
  "pickscore": [
   "pickscore 偏好分(对齐人类偏好的概率)"
  ],
  "edit_r1_mllm": [
   "Instruction_Following(1-5 → 0-1)",
   "Consistency(1-5 → 0-1)",
   "Visual_Quality(1-5 → 0-1)",
   "overall(加权综合)"
  ],
  "professional_ocr": [
   "OCR 准确率(目标文本匹配度)"
  ],
  "identity": [
   "cosine 身份相似度(ArcFace 5 点对齐,1 - cosine_identity_loss)"
  ]
 },
 "next_steps": [
  "1. CPU preflight:`mmtrain phase run --phase preflight` 验证组件可构造、契约成立。",
  "2. GPU no-update:跑一次前向,确认无参数更新下数值有限、显存可控。",
  "3. one optimizer update:做一次完整更新,确认 loss 有限、梯度非 nan。",
  "4. checkpoint readback:写出再读回 checkpoint,核对 SHA-256 与参数一致。",
  "5. 冻结评估 + 质量 receipt:固定权重评估,登记 receipt 后才可 launch_enabled=true。"
 ]
};
