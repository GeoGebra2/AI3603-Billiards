"""
agent.py - Agent 决策模块

定义 Agent 基类和具体实现：
- Agent: 基类，定义决策接口
- BasicAgent: 基于贝叶斯优化的参考实现
- NewAgent: 学生自定义实现模板
- analyze_shot_for_reward: 击球结果评分函数
"""

import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import json
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent

from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern


def analyze_shot_for_reward(shot: pt.System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...]
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8）, -30（首球/碰库犯规）
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞
    first_contact_ball_id = None
    foul_first_hit = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            other_ids = [i for i in ids if i != 'cue']
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    if first_contact_ball_id is None:
        if len(last_state) > 2:  # 只有白球和8号球时不算犯规
             foul_first_hit = True
    else:
        remaining_own_before = [bid for bid in player_targets if last_state[bid].state.s != 4]
        opponent_plus_eight = [bid for bid in last_state.keys() if bid not in player_targets and bid not in ['cue']]
        if ('8' not in opponent_plus_eight):
            opponent_plus_eight.append('8')
            
        if len(remaining_own_before) > 0:
            if first_contact_ball_id in opponent_plus_eight:
                foul_first_hit = True
        else:
            if first_contact_ball_id != '8':
                foul_first_hit = True
    
    # 3. 分析碰库
    cue_hit_cushion = False
    target_hit_cushion = False
    foul_no_rail = False
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if 'cushion' in et:
            if 'cue' in ids:
                cue_hit_cushion = True
            if first_contact_ball_id is not None and first_contact_ball_id in ids:
                target_hit_cushion = True

    if len(new_pocketed) == 0 and first_contact_ball_id is not None and (not cue_hit_cushion) and (not target_hit_cushion):
        foul_no_rail = True
        
    # 计算奖励分数
    score = 0
    
    if cue_pocketed and eight_pocketed:
        score -= 150
    elif cue_pocketed:
        score -= 100
    elif eight_pocketed:
        is_targeting_eight_ball_legally = (len(player_targets) == 1 and player_targets[0] == "8")
        score += 100 if is_targeting_eight_ball_legally else -150
            
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    if score == 0 and not cue_pocketed and not eight_pocketed and not foul_first_hit and not foul_no_rail:
        score = 10
        
    return score

class Agent():
    """Agent 基类"""
    def __init__(self):
        pass
    
    def decision(self, *args, **kwargs):
        """决策方法（子类需实现）
        
        返回：dict, 包含 'V0', 'phi', 'theta', 'a', 'b'
        """
        pass
    
    def _random_action(self,):
        """生成随机击球动作
        
        返回：dict
            V0: [0.5, 8.0] m/s
            phi: [0, 360] 度
            theta: [0, 90] 度
            a, b: [-0.5, 0.5] 球半径比例
        """
        action = {
            'V0': round(random.uniform(0.5, 8.0), 2),   # 初速度 0.5~8.0 m/s
            'phi': round(random.uniform(0, 360), 2),    # 水平角度 (0°~360°)
            'theta': round(random.uniform(0, 90), 2),   # 垂直角度
            'a': round(random.uniform(-0.5, 0.5), 3),   # 杆头横向偏移（单位：球半径比例）
            'b': round(random.uniform(-0.5, 0.5), 3)    # 杆头纵向偏移
        }
        return action



class BasicAgent(Agent):
    """基于贝叶斯优化的智能 Agent"""
    
    def __init__(self, target_balls=None):
        """初始化 Agent
        
        参数：
            target_balls: 保留参数，暂未使用
        """
        super().__init__()
        
        # 搜索空间
        self.pbounds = {
            'V0': (0.5, 8.0),
            'phi': (0, 360),
            'theta': (0, 90), 
            'a': (-0.5, 0.5),
            'b': (-0.5, 0.5)
        }
        
        # 优化参数
        self.INITIAL_SEARCH = 20
        self.OPT_SEARCH = 10
        self.ALPHA = 1e-2
        
        # 模拟噪声（可调整以改变训练难度）
        self.noise_std = {
            'V0': 0.1,
            'phi': 0.1,
            'theta': 0.1,
            'a': 0.003,
            'b': 0.003
        }
        self.enable_noise = False
        
        print("BasicAgent (Smart, pooltool-native) 已初始化。")

    
    def _create_optimizer(self, reward_function, seed):
        """创建贝叶斯优化器
        
        参数：
            reward_function: 目标函数，(V0, phi, theta, a, b) -> score
            seed: 随机种子
        
        返回：
            BayesianOptimization对象
        """
        gpr = GaussianProcessRegressor(
            kernel=Matern(nu=2.5),
            alpha=self.ALPHA,
            n_restarts_optimizer=10,
            random_state=seed
        )
        
        bounds_transformer = SequentialDomainReductionTransformer(
            gamma_osc=0.8,
            gamma_pan=1.0
        )
        
        optimizer = BayesianOptimization(
            f=reward_function,
            pbounds=self.pbounds,
            random_state=seed,
            verbose=0,
            bounds_transformer=bounds_transformer
        )
        optimizer._gp = gpr
        
        return optimizer


    def decision(self, balls=None, my_targets=None, table=None):
        """使用贝叶斯优化搜索最佳击球参数
        
        参数：
            balls: 球状态字典，{ball_id: Ball}
            my_targets: 目标球ID列表，['1', '2', ...]
            table: 球桌对象
        
        返回：
            dict: 击球动作 {'V0', 'phi', 'theta', 'a', 'b'}
                失败时返回随机动作
        """
        if balls is None:
            print(f"[BasicAgent] Agent decision函数未收到balls关键信息，使用随机动作。")
            return self._random_action()
        try:
            
            # 保存一个击球前的状态快照，用于对比
            last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}

            remaining_own = [bid for bid in my_targets if balls[bid].state.s != 4]
            if len(remaining_own) == 0:
                my_targets = ["8"]
                print("[BasicAgent] 我的目标球已全部清空，自动切换目标为：8号球")

            # 1.动态创建“奖励函数” (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = pt.Cue(cue_ball_id="cue")

                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                
                try:
                    if self.enable_noise:
                        V0_noisy = V0 + np.random.normal(0, self.noise_std['V0'])
                        phi_noisy = phi + np.random.normal(0, self.noise_std['phi'])
                        theta_noisy = theta + np.random.normal(0, self.noise_std['theta'])
                        a_noisy = a + np.random.normal(0, self.noise_std['a'])
                        b_noisy = b + np.random.normal(0, self.noise_std['b'])
                        
                        V0_noisy = np.clip(V0_noisy, 0.5, 8.0)
                        phi_noisy = phi_noisy % 360
                        theta_noisy = np.clip(theta_noisy, 0, 90)
                        a_noisy = np.clip(a_noisy, -0.5, 0.5)
                        b_noisy = np.clip(b_noisy, -0.5, 0.5)
                        
                        shot.cue.set_state(V0=V0_noisy, phi=phi_noisy, theta=theta_noisy, a=a_noisy, b=b_noisy)
                    else:
                        shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    
                    # 关键：使用 pooltool 物理引擎 (世界A)
                    pt.simulate(shot, inplace=True)
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的“裁判”来打分
                score = analyze_shot_for_reward(
                    shot=shot,
                    last_state=last_state_snapshot,
                    player_targets=my_targets
                )


                return score

            print(f"[BasicAgent] 正在为 Player (targets: {my_targets}) 搜索最佳击球...")
            
            seed = np.random.randint(1e6)
            optimizer = self._create_optimizer(reward_fn_wrapper, seed)
            optimizer.maximize(
                init_points=self.INITIAL_SEARCH,
                n_iter=self.OPT_SEARCH
            )
            
            best_result = optimizer.max
            best_params = best_result['params']
            best_score = best_result['target']

            if best_score < 10:
                print(f"[BasicAgent] 未找到好的方案 (最高分: {best_score:.2f})。使用随机动作。")
                return self._random_action()
            action = {
                'V0': float(best_params['V0']),
                'phi': float(best_params['phi']),
                'theta': float(best_params['theta']),
                'a': float(best_params['a']),
                'b': float(best_params['b']),
            }

            print(f"[BasicAgent] 决策 (得分: {best_score:.2f}): "
                  f"V0={action['V0']:.2f}, phi={action['phi']:.2f}, "
                  f"θ={action['theta']:.2f}, a={action['a']:.3f}, b={action['b']:.3f}")
            return action

        except Exception as e:
            print(f"[BasicAgent] 决策时发生严重错误，使用随机动作。原因: {e}")
            import traceback
            traceback.print_exc()
            return self._random_action()

class NewAgent(Agent):
    def __init__(self, checkpoint: str = None, fast: bool = False, debug: bool = False):
        super().__init__()
        self.debug = bool(debug)
        if fast:
            self.k_targets = 2
            self.v_candidates = [2.0, 3.0, 4.0]
            self.theta_candidates = [0.0]
            self.dphi_candidates = [-4.0, 0.0, 4.0]
            self.offsets = [0.0, 0.02]
            self.refine_dphi = [0.0]
            self.refine_dv = [0.0]
            self.refine_offsets = [0.0]
        else:
            self.k_targets = 3
            self.v_candidates = [1.8, 2.4, 3.0, 3.8, 4.6]
            self.theta_candidates = [0.0]
            self.dphi_candidates = [-8.0, -4.0, 0.0, 4.0, 8.0]
            self.offsets = [0.0, 0.02, -0.02]
            self.refine_dphi = [-2.0, 0.0, 2.0]
            self.refine_dv = [-0.6, 0.0, 0.6]
            self.refine_offsets = [0.0, 0.01, -0.01]
        if checkpoint is None:
            default_path = os.path.join('eval', 'checkpoints', 'newagent_config.json')
            checkpoint = default_path if os.path.exists(default_path) else None
        if checkpoint is not None and os.path.isfile(checkpoint):
            try:
                with open(checkpoint, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.k_targets = int(cfg.get('k_targets', self.k_targets))
                self.v_candidates = list(cfg.get('v_candidates', self.v_candidates))
                self.theta_candidates = list(cfg.get('theta_candidates', self.theta_candidates))
                self.dphi_candidates = list(cfg.get('dphi_candidates', self.dphi_candidates))
                self.offsets = list(cfg.get('offsets', self.offsets))
                self.refine_dphi = list(cfg.get('refine_dphi', self.refine_dphi))
                self.refine_dv = list(cfg.get('refine_dv', self.refine_dv))
                self.refine_offsets = list(cfg.get('refine_offsets', self.refine_offsets))
            except Exception:
                pass

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ["8"]
            remaining = ["8"]
        cue_pos = balls['cue'].state.rvw[0][0:2]

        def ang(a, b):
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            return float((math.degrees(math.atan2(dy, dx)) % 360))

        targets_sorted = sorted(remaining, key=lambda bid: np.linalg.norm(balls[bid].state.rvw[0][0:2] - cue_pos))
        targets_sorted = targets_sorted[:self.k_targets]

        best = None
        best_score = -1e9
        total_sims = max(1, len(targets_sorted) * len(self.dphi_candidates) * len(self.v_candidates) * len(self.theta_candidates) * (len(self.offsets) ** 2))
        done_sims = 0

        for bid in targets_sorted:
            tpos = balls[bid].state.rvw[0][0:2]
            base_phi = ang(cue_pos, tpos)
            for dphi in self.dphi_candidates:
                phi = (base_phi + dphi) % 360
                for V0 in self.v_candidates:
                    for theta in self.theta_candidates:
                        for a in self.offsets:
                            for b in self.offsets:
                                sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
                                sim_table = copy.deepcopy(table)
                                cue = pt.Cue(cue_ball_id="cue")
                                shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                                try:
                                    shot.cue.set_state(V0=float(V0), phi=float(phi), theta=float(theta), a=float(a), b=float(b))
                                    pt.simulate(shot, inplace=True)
                                except Exception:
                                    continue
                                score = analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets)
                                if score > best_score:
                                    best_score = score
                                    best = {'V0': float(V0), 'phi': float(phi), 'theta': float(theta), 'a': float(a), 'b': float(b)}
                                done_sims += 1
                                if self.debug and (done_sims % max(1, total_sims // 10) == 0):
                                    print(f"[NewAgent] 粗搜进度 {done_sims}/{total_sims}, 当前最佳分 {best_score:.1f}")

        if best is not None:
            center = best
            for dphi in self.refine_dphi:
                phi = (center['phi'] + dphi) % 360
                for dv in self.refine_dv:
                    V0 = float(np.clip(center['V0'] + dv, 0.5, 8.0))
                    for a in self.refine_offsets:
                        for b in self.refine_offsets:
                            sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
                            sim_table = copy.deepcopy(table)
                            cue = pt.Cue(cue_ball_id="cue")
                            shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
                            try:
                                shot.cue.set_state(V0=float(V0), phi=float(phi), theta=float(center['theta']), a=float(a), b=float(b))
                                pt.simulate(shot, inplace=True)
                            except Exception:
                                continue
                            score = analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets)
                            if score > best_score:
                                best_score = score
                                best = {'V0': float(V0), 'phi': float(phi), 'theta': float(center['theta']), 'a': float(a), 'b': float(b)}
                            if self.debug:
                                print(f"[NewAgent] 微调评分 {score:.1f} 最佳 {best_score:.1f}")

        if best is None or best_score < 0:
            return self._random_action()
        return best