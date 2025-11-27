# 训练说明

## 环境
- Python 3.10+
- 安装依赖：`pip install -r requirements.txt`

## 目标
- 生成 `eval/checkpoints/newagent_config.json`，用于初始化 `NewAgent`。

## 训练命令
- 运行：`python train/train_newagent.py`
- 可选参数：
  - `--episodes` 训练轮数，默认 5
  - `--out` 输出路径，默认 `eval/checkpoints/newagent_config.json`
  - `--logdir` TensorBoard 日志目录，默认 `train/runs/newagent`

## 说明
- 训练过程对 `NewAgent` 的候选动作参数进行小规模搜索，依据仿真评分挑选较优配置并写入 checkpoint。
- 该过程可复现，固定随机种子；可按需提升搜索规模以获得更优配置。
- 进度条：如已安装 `tqdm`，训练过程将显示进度；未安装时以普通输出替代。
- TensorBoard：如环境安装了 `torch` 并带有 `torch.utils.tensorboard`，或安装了 `tensorboardX`，将生成日志到 `--logdir`；可使用命令 `tensorboard --logdir train/runs/newagent` 进行可视化。