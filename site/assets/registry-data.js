/* YoYo 注册表离线快照 — 手写维护；真值来自 build_builtin_registries()，由测试强制一致 */
window.YOYO_REGISTRY = {
 "source": "真值来源: mmtrainer.registry.build_builtin_registries()（本文件是手写快照,仓库里没有生成器）",
 "note": "离线内联快照；由 tests/v2/test_site_display_sync.py 强制与 build_builtin_registries() 一致（同文件还钉死 index.html 的手写矩阵与注册表派生的计数）。name/desc/family 是编辑性文案，注册表里没有对应来源，不在断言范围。firered 按 commit da715d1 的营销红线不出现在展示面（与宣传片'FireRed 不上镜'同源），reward 'none' 为空占位不展示。",
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
 ]
};
