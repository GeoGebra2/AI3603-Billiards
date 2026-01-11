# AI3603-Billiards MCTSProAgent 介绍文档

## 概述
- 本项目在不修改 `poolenv.py` 的前提下实现并接入 `MCTSProAgent`，与课程提供的 `BasicAgent` 进行对战评估。
- `MCTSProAgent` 采用 **蒙特卡洛树搜索 (MCTS)** 结合 **物理仿真** 的策略，通过启发式候选生成、鲁棒性仿真评估以及前瞻性价值估计，在可复现的设置下追求稳定的进球与低犯规。

## 算法设计

### 1. 候选动作生成 (Candidate Generation)
- **目标选择**：优先选择当前合法的目标球（距离白球最近的若干个），若己方球清空则选择黑 8。
- **几何启发**：
  - 计算 **Ghost Ball** 位置：基于目标球和袋口位置，反推白球击打点。
  - **遮挡检测**：检测白球到 Ghost Ball 路径 (`_occluded_cg`) 以及 Ghost Ball 到袋口路径 (`_occluded_tp`) 是否存在障碍球。
- **参数扰动**：
  - 对每个可行角度，生成不同的力度 (`V0`)、击球角度微调 (`phi`) 和杆法偏移 (`a`, `b`)。
  - 包含基于距离的力度自适应 (`V_req`) 和角度搜索范围自适应。
- **兜底策略**：若无有效候选，生成指向目标球中心的直球或随机安全球。

### 2. MCTS 搜索策略
- **Selection (选择)**：使用 UCB (Upper Confidence Bound) 公式选择最有潜力的候选动作节点。
  - `c_puct`: 平衡探索 (Exploration) 与利用 (Exploitation)。
- **Expansion & Evaluation (扩展与评估)**：
  - **物理仿真**：使用 `pooltool` 对选定动作进行仿真。为提高鲁棒性，仿真时引入高斯噪声 (`sim_noise`) 模拟执行误差。
  - **多样本评估**：对同一动作进行多次带噪仿真 (`robust_samples`)，取平均回报，降低偶然性影响。
- **Reward Function (回报函数)**：
  - **基础回报**：调用 `analyze_shot_for_reward` 计算进球、犯规等基础分。
  - **未来收益 (`_future_reach`)**：评估击球后白球位置对剩余目标球的可达性，鼓励走位。
  - **风险惩罚 (`_risk_penalty`)**：惩罚白球进袋风险高或力度过大的危险击球。
  - **前瞻价值 (`_evaluate_next`)**：对仿真后的局面进行一步 Lookahead 搜索，评估下一杆的最佳期望得分（Gamma 衰减）。
- **Backpropagation (反向传播)**：更新节点的访问次数 `N` 和总回报 `Q`。

### 3. 动作先验 (Action Prior)
- 在 MCTS 初始化阶段，根据启发式规则（如力度偏差、角度偏差、杆法偏移量）计算动作的先验概率，引导搜索优先探索更合理的动作。

## 代码结构与改动
- `agents/new_agent.py`：
  - **`MCTSProAgent`**：核心类，继承自 `Agent`。
  - `decision(balls, my_targets, table)`：主入口，执行 MCTS 流程。
  - `generate_candidate_actions(...)`：生成候选动作列表。
  - `_simulate_with_timeout(...)`：带超时控制的并行物理仿真。
  - `_evaluate_next(...)`：下一状态的快速评估。
- `evaluate.py`：配置 `agent_b = MCTSProAgent()`，执行对战评估。

## 使用指南
- **环境依赖**：需安装 `pooltool` 及 `numpy`。
- **运行评估**：
  ```bash
  python evaluate.py
  ```
  脚本将默认进行 40 局对战（可修改 `n_games`），输出胜率统计。

## 参数与可调项
在 `MCTSProAgent.__init__` 中可调整关键参数：
- **搜索参数**：
  - `n_simulations`: MCTS 迭代次数 (默认 60)。
  - `c_puct`: UCB 探索系数 (默认 1.25)。
  - `robust_samples`: 单次评估的仿真采样数 (默认 4)。
- **物理参数**：
  - `sim_noise`: 仿真噪声标准差 (V0, phi, theta, a, b)。
  - `rollout_timeout`: 仿真超时时间。
- **策略参数**：
  - `lambda_future`: 走位奖励权重。
  - `gamma_next`: 下一步价值的折扣因子。

## 评估与结果
- 相比基础 Agent，MCTSProAgent 在进球稳定性、防守能力和解球成功率上有显著提升。
- 建议在评估时关注胜率以及平均单局得分。

## 后续改进建议
- **并行化优化**：利用 GPU 或多进程进一步加速 MCTS 仿真。
- **价值网络**：训练神经网络替代 `_evaluate_next` 进行更深层的价值评估。
- **对手建模**：在树搜索中加入对敌方策略的预测，进行 Minimax 风格的博弈规划。
