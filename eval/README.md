# 测试说明（eval）

## 文件结构
- `README.md`：本说明文档。
- `checkpoints/newagent_config.json`：新 Agent 的初始化配置（由训练脚本生成）。

## 对应 Agent 与加载方式
- `agents/basic_agent.py` 中的 `NewAgent` 与 `MCTSAgent` 均支持从 `eval/checkpoints/newagent_config.json` 加载候选参数与策略权重：
  - `NewAgent(checkpoint=None)`：若存在上述文件则自动加载，否则使用内置默认参数。
  - `MCTSAgent(checkpoint=None)`：同样支持从该配置文件加载并覆盖默认参数。

## 环境
- Python 3.10+

## 复现实验
- 运行评估：`python evaluate.py`
  - 默认对战为 `BasicAgent` vs `MCTSProAgent`切换为其他组合）。
  - 评估脚本固定进行 40 局轮换对战，统计 `AGENT_A_WIN`、`AGENT_B_WIN` 与 `SAME`，并计算双方得分与胜率。
