# 训练说明（train）

## 文件结构
- `README.md`：本说明文档。
- `train_newagent.py`：训练/搜索脚本，针对 `agents/basic_agent.py` 中的 `MCTSAgent` 进行超参数搜索并生成可供 `NewAgent/MCTSAgent` 使用的初始化配置。
- `runs/`：可选的训练日志目录（TensorBoard 事件文件），用于训练过程可视化。

## 训练目标
- 生成 `eval/checkpoints/newagent_config.json`，用于初始化 `NewAgent` 或 `MCTSAgent`。
- 同时保存每个候选配置的评估统计（平均回报、决策时长等），便于后续分析与复现实验。

## 环境
- Python 3.10+
- 可选依赖：
  - `tqdm`：显示训练进度条
  - `torch` 或 `tensorboardX`：写入 TensorBoard 日志

## 训练命令
- 基本运行：`python train/train_newagent.py`
- 可选参数：
  - `--episodes`：每个配置的试验次数（默认 20，越大越稳）
  - `--out`：输出 checkpoint 路径（默认 `eval/checkpoints/newagent_config.json`）
  - `--logdir`：TensorBoard 日志目录（默认 `train/runs/newagent`）
  - `--fast`：使用极小搜索空间快速生成样例配置
  - `--verbose`：输出更详细的 trial 信息

## 训练过程说明
- 搜索空间：脚本内置若干候选配置（力度、角度微调、杆法偏移、鲁棒采样数、未来走位权重等），逐一进行多次仿真试验并统计平均回报。
- 打分标准：基于环境事件与犯规的启发式奖励，并记录决策耗时与鲁棒仿真统计。
- 复现性：固定随机种子（42），保证相同参数下可复现结果。
- 输出产物：
  - `eval/checkpoints/newagent_config.json`：最佳配置（作为测试阶段的初始化 checkpoint）。

## 与 Agent 的关系
- `MCTSAgent`：训练时用于评估不同候选配置的策略效果与稳定性。
- `NewAgent`：测试时可直接加载生成的 `newagent_config.json`，在不改动 `poolenv.py` 前提下进行对战评估。
