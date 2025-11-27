# 测试说明

## 环境
- 安装依赖：`pip install -r requirements.txt`

## 使用方法
- 确保存在 `eval/checkpoints/newagent_config.json`（可通过 `python train/train_newagent.py` 生成）。
- 运行评估：`python evaluate.py`

## 说明
- 评估脚本固定 40 局轮换对战，统计 `AGENT_A_WIN`、`AGENT_B_WIN` 与 `SAME`，并计算双方得分与胜率。
- `NewAgent` 会在初始化时自动加载 `eval/checkpoints/newagent_config.json`（若存在），否则使用默认参数。