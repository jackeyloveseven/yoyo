# 重构交接:monorepo → 可组合零件库(v2)

一次性迁移的存档记录(2026-10)。零件现状**不在本页**——活清单在注册表
(`mmtrain registry list`)与各能力卡/契约卡里,本页不再维护手工符号清单
(它会腐烂;v2 重构初期的完整符号清单见 git 历史本文件首版)。

- **备份**:`../training_code_backup` 与重构前逐字节一致;活跃目录 `.` 从
  97MB 单体脚本(71MB vendored 第三方 + 24MB 遗留脚本树)减到 1.9MB 零件库。
- **决策叙述**:为什么抽零件、边界在哪,见 [`BUILDING.md`](BUILDING.md) 首条
  与 [`adr/`](adr/README.md)(ADR-0001/0002)。
- **验证门**(迁移时跑的,与 CI 相同):

```bash
PYTHONPATH=src python3 check_v2_quality.py
python3 check_structure.py
python3 check_docs.py
PYTHONPATH=src:. pytest tests/v2 -q
```
