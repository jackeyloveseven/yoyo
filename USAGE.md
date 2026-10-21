# YoYo (Training Code v2) 使用指南

## 1. 安装与查看注册零件

```bash
python3 -m pip install -e '.[test,checkpoint,promo]'
mmtrain registry list
```

> **设计哲学**：导入核心包绝不提前加载 Torch 或 CUDA，几毫秒就能启动。具体模型与检查点依赖一律按需懒加载。

## 2. 校验训练规格（RunSpec）

RunSpec 是严格不可变的训练蓝图。为杜绝恶意注入与路径逃逸，规格中严禁直接传递裸文件路径或 URL，所有资产必须通过注册 ID 与 SHA-256 哈希引用：

```bash
mmtrain plan validate --spec recipes/v2/firered_diffusionnft.json
```

仓库自带的内置模板配方默认带有占位哈希，并处于禁用状态（`launch_enabled=false`），必须在真实环境中由控制面解析后方可正式拉起。

## 3. 按阶段执行生命周期

为了保证“真正跑起来的”和“当初审核通过的”一模一样，每一步 `phase run` 都必须显式传入 `--expected-sha256`。CLI 启动时会在内存中重新计算哈希并严格对拍，只要有哪怕一个字节改动就立刻拒跑：

```bash
SHA=$(mmtrain plan validate --spec /path/to/run-spec.json | python3 -c "import sys,json;print(json.load(sys.stdin)['spec_sha256'])")
mmtrain phase run --spec /path/to/run-spec.json --expected-sha256 "$SHA" --phase preflight
mmtrain phase run --spec /path/to/run-spec.json --expected-sha256 "$SHA" --phase train
mmtrain phase run --spec /path/to/run-spec.json --expected-sha256 "$SHA" --phase evaluate
mmtrain phase run --spec /path/to/run-spec.json --expected-sha256 "$SHA" --phase publish
```

执行引擎严格遵循四阶段生命周期：`preflight`（静态体检与配置核验）、`train`（模型训练）、`evaluate`（冻结评估）、`publish`（产物发行）。引擎专注于状态流转与安全防线，具体的数学损失由各算法插件自理。集群调度、显卡分配与远程登录由外层平台负责。

## 4. 权重导入与晋级阶梯

* **v1 权重单向导入**：v2 只导入策略权重（`policy`），并坚决重置优化器、EMA、随机种子与游标状态：
  ```bash
  mmtrain checkpoint import-policy \
    --source /path/to/v1-checkpoint.pt --source-sha256 <sha256> \
    --source-release training-code-v1.9.0 --model firered --output /path/to/output/
  ```
* **晋级阶梯**：代码写完不代表能直接上集群大规模训练，必须一步一个脚印通过验证阶梯：
  1. CPU 纯函数与数据流契约测试；
  2. GPU 零更新前向基准对拍；
  3. 单步反向传播与参数真实更新；
  4. 检查点落盘与读回核验；
  5. 冻结评估并沉淀正式 Receipt 凭据。
  在未挂载真实通过凭证前，`launch_enabled=false` 默认保持安全关闭（Fail-Closed）。
