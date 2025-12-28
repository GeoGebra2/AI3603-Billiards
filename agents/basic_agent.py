import math
import pooltool as pt
from pooltool.system.datatypes import System
from pooltool.objects import Cue
from pooltool.evolution import simulate
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
import copy
import os
from datetime import datetime
import random
import signal
# from poolagent.pool import Pool as CuetipEnv, State as CuetipState
# from poolagent import FunctionAgent
import json
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from .agent import Agent


# ============ 超时安全模拟机制 ============
class SimulationTimeoutError(Exception):
    """物理模拟超时异常"""
    pass

def _timeout_handler(signum, frame):
    """超时信号处理器"""
    raise SimulationTimeoutError("物理模拟超时")

def simulate_with_timeout(shot, timeout=3):
    """带超时保护的物理模拟
    
    参数：
        shot: pt.System 对象
        timeout: 超时时间（秒），默认3秒
    
    返回：
        bool: True 表示模拟成功，False 表示超时或失败
    
    说明：
        使用 signal.SIGALRM 实现超时机制（仅支持 Unix/Linux）
        超时后自动恢复，不会导致程序卡死
    """
    # 设置超时信号处理器
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout)  # 设置超时时间
    
    try:
        pt.simulate(shot, inplace=True)
        signal.alarm(0)  # 取消超时
        return True
    except SimulationTimeoutError:
        print(f"[WARNING] 物理模拟超时（>{timeout}秒），跳过此次模拟")
        return False
    except Exception as e:
        signal.alarm(0)  # 取消超时
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)  # 恢复原处理器

# ============================================



def analyze_shot_for_reward(shot: System, last_state: dict, player_targets: list):
    """
    分析击球结果并计算奖励分数（完全对齐台球规则）
    
    参数：
        shot: 已完成物理模拟的 System 对象
        last_state: 击球前的球状态，{ball_id: Ball}
        player_targets: 当前玩家目标球ID，['1', '2', ...] 或 ['8']
    
    返回：
        float: 奖励分数
            +50/球（己方进球）, +100（合法黑8）, +10（合法无进球）
            -100（白球进袋）, -150（非法黑8/白球+黑8）, -30（首球/碰库犯规）
    
    规则核心：
        - 清台前：player_targets = ['1'-'7'] 或 ['9'-'15']，黑8不属于任何人
        - 清台后：player_targets = ['8']，黑8成为唯一目标球
    """
    
    # 1. 基本分析
    new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state[bid].state.s != 4]
    
    # 根据 player_targets 判断进球归属（黑8只有在清台后才算己方球）
    own_pocketed = [bid for bid in new_pocketed if bid in player_targets]
    enemy_pocketed = [bid for bid in new_pocketed if bid not in player_targets and bid not in ["cue", "8"]]
    
    cue_pocketed = "cue" in new_pocketed
    eight_pocketed = "8" in new_pocketed

    # 2. 分析首球碰撞（定义合法的球ID集合）
    first_contact_ball_id = None
    foul_first_hit = False
    valid_ball_ids = {'1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15'}
    
    for e in shot.events:
        et = str(e.event_type).lower()
        ids = list(e.ids) if hasattr(e, 'ids') else []
        if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
            # 过滤掉 'cue' 和非球对象（如 'cue stick'），只保留合法的球ID
            other_ids = [i for i in ids if i != 'cue' and i in valid_ball_ids]
            if other_ids:
                first_contact_ball_id = other_ids[0]
                break
    
    # 首球犯规判定：完全对齐 player_targets
    if first_contact_ball_id is None:
        # 未击中任何球（但若只剩白球和黑8且已清台，则不算犯规）
        if len(last_state) > 2 or player_targets != ['8']:
            foul_first_hit = True
    else:
        # 首次击打的球必须是 player_targets 中的球
        if first_contact_ball_id not in player_targets:
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
        
    # 4. 计算奖励分数
    score = 0
    
    # 白球进袋处理
    if cue_pocketed and eight_pocketed:
        score -= 150  # 白球+黑8同时进袋，严重犯规
    elif cue_pocketed:
        score -= 100  # 白球进袋
    elif eight_pocketed:
        # 黑8进袋：只有清台后（player_targets == ['8']）才合法
        if player_targets == ['8']:
            score += 100  # 合法打进黑8
        else:
            score -= 150  # 清台前误打黑8，判负
            
    # 首球犯规和碰库犯规
    if foul_first_hit:
        score -= 30
    if foul_no_rail:
        score -= 30
        
    # 进球得分（own_pocketed 已根据 player_targets 正确分类）
    score += len(own_pocketed) * 50
    score -= len(enemy_pocketed) * 20
    
    # 合法无进球小奖励
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
        
        print("BasicAgent (贝叶斯优化版) 已初始化。")

    
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

            # 1.动态创建"奖励函数" (Wrapper)
            # 贝叶斯优化器会调用此函数，并传入参数
            def reward_fn_wrapper(V0, phi, theta, a, b):
                # 创建一个用于模拟的沙盒系统
                sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = Cue(cue_ball_id="cue")

                shot = System(table=sim_table, balls=sim_balls, cue=cue)
                
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
                    
                    # 关键：使用带超时保护的物理模拟（3秒上限）
                    if not simulate_with_timeout(shot, timeout=3):
                        return 0  # 超时是物理引擎问题，不惩罚agent
                except Exception as e:
                    # 模拟失败，给予极大惩罚
                    return -500
                
                # 使用我们的"裁判"来打分
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
            self.k_targets = 1
            self.v_candidates = [2.6, 3.2]
            self.theta_candidates = [0.0]
            self.dphi_candidates = [-2.0, 0.0, 2.0]
            self.offsets = [0.0]
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
        self.use_pocket_aiming = False
        self.occlusion_filter = False
        self.robust_samples = 0
        self.early_stop_score = None
        self.robust_noise_std = {'V0': 0.08, 'phi': 0.05, 'theta': 0.05, 'a': 0.002, 'b': 0.002}
        self.stats_occluded_target_count = 0
        self.stats_occluded_pocket_count = 0
        self.stats_total_sims = 0
        self.stats_best_next_reach = 0
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
                self.use_pocket_aiming = bool(cfg.get('use_pocket_aiming', self.use_pocket_aiming))
                self.occlusion_filter = bool(cfg.get('occlusion_filter', self.occlusion_filter))
                self.robust_samples = int(cfg.get('robust_samples', self.robust_samples))
                self.early_stop_score = cfg.get('early_stop_score', self.early_stop_score)
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

        def dist_point_to_segment(p, a, b):
            ap = p - a
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 1e-9:
                return float(np.linalg.norm(ap))
            t = float(np.dot(ap, ab) / denom)
            t = max(0.0, min(1.0, t))
            closest = a + t * ab
            return float(np.linalg.norm(p - closest))

        def dist_and_t(p, a, b):
            ap = p - a
            ab = b - a
            denom = float(np.dot(ab, ab))
            if denom <= 1e-9:
                return float(np.linalg.norm(ap)), 0.0
            t = float(np.dot(ap, ab) / denom)
            t = max(0.0, min(1.0, t))
            closest = a + t * ab
            return float(np.linalg.norm(p - closest)), t

        def occluded(a, b):
            thresh = 0.015
            for k, ball in balls.items():
                if k in ['cue'] or balls[k].state.s == 4:
                    continue
                pos = ball.state.rvw[0][0:2]
                d, t = dist_and_t(pos, a, b)
                if d < thresh and 0.1 < t < 0.9:
                    return True
            return False

        def occluded_tp(t, p, ignore_ids=None):
            thresh = 0.015
            for k, ball in balls.items():
                if ignore_ids and k in ignore_ids:
                    continue
                if balls[k].state.s == 4:
                    continue
                pos = ball.state.rvw[0][0:2]
                d, tt = dist_and_t(pos, t, p)
                if d < thresh and 0.1 < tt < 0.9:
                    return True
            return False

        def ghost_phi(cue, target, pocket):
            v = pocket - target
            n = np.linalg.norm(v)
            if n < 1e-9:
                return ang(cue, target)
            u = v / n
            d = 0.057
            ghost = target - u * d
            return ang(cue, ghost)

        def pocketability(bid):
            tpos = balls[bid].state.rvw[0][0:2]
            score = 0.0
            if hasattr(table, 'pockets') and isinstance(table.pockets, dict):
                unblocked = 0
                min_dist = 1e9
                for pk in table.pockets:
                    ppos = table.pockets[pk].center[0:2]
                    dpk = float(np.linalg.norm(ppos - tpos))
                    if dpk < min_dist:
                        min_dist = dpk
                    if not (self.occlusion_filter and occluded_tp(tpos, ppos, ignore_ids=[bid, 'cue'])):
                        unblocked += 1
                score = float(unblocked * 100.0 - min_dist)
            else:
                score = -float(np.linalg.norm(tpos - cue_pos))
            return score

        targets_sorted = sorted(remaining, key=lambda bid: ( -pocketability(bid), np.linalg.norm(balls[bid].state.rvw[0][0:2] - cue_pos) ))
        targets_sorted = targets_sorted[:self.k_targets]

        best = None
        best_score = -1e9
        total_sims = max(1, len(targets_sorted) * len(self.dphi_candidates) * len(self.v_candidates) * len(self.theta_candidates) * (len(self.offsets) ** 2))
        done_sims = 0

        for bid in targets_sorted:
            tpos = balls[bid].state.rvw[0][0:2]
            phi_candidates = []
            if self.use_pocket_aiming and hasattr(table, 'pockets') and isinstance(table.pockets, dict):
                blocked_all = True
                nearest_pk = None
                nearest_dist = 1e9
                for pk in table.pockets:
                    ppos = table.pockets[pk].center[0:2]
                    dpk = float(np.linalg.norm(ppos - tpos))
                    if dpk < nearest_dist:
                        nearest_dist = dpk
                        nearest_pk = pk
                    if self.occlusion_filter and occluded_tp(tpos, ppos, ignore_ids=[bid, 'cue']):
                        self.stats_occluded_pocket_count += 1
                        continue
                    blocked_all = False
                    phi_p = ghost_phi(cue_pos, tpos, ppos)
                    for dphi in self.dphi_candidates:
                        phi_candidates.append((phi_p + dphi) % 360)
                if blocked_all and nearest_pk is not None:
                    ppos = table.pockets[nearest_pk].center[0:2]
                    phi_p = ghost_phi(cue_pos, tpos, ppos)
                    for dphi in self.dphi_candidates:
                        phi_candidates.append((phi_p + dphi) % 360)
            else:
                base_phi = ang(cue_pos, tpos)
                for dphi in self.dphi_candidates:
                    phi_candidates.append((base_phi + dphi) % 360)
            if self.occlusion_filter:
                if occluded(cue_pos, tpos):
                    self.stats_occluded_target_count += 1
                    continue
            for phi in phi_candidates:
                for V0 in self.v_candidates:
                    for theta in self.theta_candidates:
                        for a in self.offsets:
                            for b in self.offsets:
                                sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
                                sim_table = copy.deepcopy(table)
                                cue = Cue(cue_ball_id="cue")
                                shot = System(table=sim_table, balls=sim_balls, cue=cue)
                                try:
                                    if self.robust_samples and self.robust_samples > 0:
                                        rs = []
                                        for _ in range(self.robust_samples):
                                            V0n = float(np.clip(V0 + np.random.normal(0, self.robust_noise_std['V0']), 0.5, 8.0))
                                            phin = float((phi + np.random.normal(0, self.robust_noise_std['phi'])) % 360)
                                            thetan = float(np.clip(theta + np.random.normal(0, self.robust_noise_std['theta']), 0, 90))
                                            an = float(np.clip(a + np.random.normal(0, self.robust_noise_std['a']), -0.5, 0.5))
                                            bn = float(np.clip(b + np.random.normal(0, self.robust_noise_std['b']), -0.5, 0.5))
                                            shot.cue.set_state(V0=V0n, phi=phin, theta=thetan, a=an, b=bn)
                                            simulate(shot, inplace=True)
                                            sc = analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets)
                                            cue_new = shot.balls['cue'].state.rvw[0][0:2]
                                            remain = [x for x in my_targets if shot.balls[x].state.s != 4]
                                            reach = 0
                                            for x in remain:
                                                t2 = shot.balls[x].state.rvw[0][0:2]
                                                if not occluded(cue_new, t2):
                                                    reach += 1
                                            sc += float(min(3, reach) * 8.0)
                                            rs.append(sc)
                                        score = float(np.mean(rs))
                                    else:
                                        shot.cue.set_state(V0=float(V0), phi=float(phi), theta=float(theta), a=float(a), b=float(b))
                                        simulate(shot, inplace=True)
                                        score = analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets)
                                        cue_new = shot.balls['cue'].state.rvw[0][0:2]
                                        remain = [x for x in my_targets if shot.balls[x].state.s != 4]
                                        reach = 0
                                        for x in remain:
                                            t2 = shot.balls[x].state.rvw[0][0:2]
                                            if not occluded(cue_new, t2):
                                                reach += 1
                                        score += float(min(3, reach) * 8.0)
                                        hit_cushion = False
                                        for e in shot.events:
                                            et = str(e.event_type).lower()
                                            if 'cushion' in et:
                                                hit_cushion = True
                                                break
                                        if hit_cushion and score <= 10:
                                            score += 12.0
                                except Exception:
                                    continue
                                if score > best_score:
                                    best_score = score
                                    best = {'V0': float(V0), 'phi': float(phi), 'theta': float(theta), 'a': float(a), 'b': float(b)}
                                done_sims += 1
                                if self.debug and (done_sims % max(1, total_sims // 10) == 0):
                                    print(f"[NewAgent] 粗搜进度 {done_sims}/{total_sims}, 当前最佳分 {best_score:.1f}")
                                if self.early_stop_score is not None and best_score >= float(self.early_stop_score):
                                    break
                        if self.early_stop_score is not None and best_score >= float(self.early_stop_score):
                            break
                    if self.early_stop_score is not None and best_score >= float(self.early_stop_score):
                        break
                if self.early_stop_score is not None and best_score >= float(self.early_stop_score):
                    break

        if best is None:
            safe_phis = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
            for phi in safe_phis:
                V0 = 3.0
                theta = 0.0
                a = 0.0
                b = 0.0
                sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
                sim_table = copy.deepcopy(table)
                cue = Cue(cue_ball_id="cue")
                shot = System(table=sim_table, balls=sim_balls, cue=cue)
                try:
                    shot.cue.set_state(V0=V0, phi=phi, theta=theta, a=a, b=b)
                    simulate(shot, inplace=True)
                except Exception:
                    continue
                score = analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets)
                cue_new = shot.balls['cue'].state.rvw[0][0:2]
                remain = [x for x in my_targets if shot.balls[x].state.s != 4]
                reach = 0
                for x in remain:
                    t2 = shot.balls[x].state.rvw[0][0:2]
                    if not occluded(cue_new, t2):
                        reach += 1
                score += float(min(3, reach) * 8.0)
                hit_cushion = False
                for e in shot.events:
                    et = str(e.event_type).lower()
                    if 'cushion' in et:
                        hit_cushion = True
                        break
                if hit_cushion and score <= 10:
                    score += 12.0
                if score > best_score:
                    best_score = score
                    best = {'V0': float(V0), 'phi': float(phi), 'theta': float(theta), 'a': float(a), 'b': float(b)}
                done_sims += 1

        if best is not None:
            top_candidates = [best]
            center_list = top_candidates
            for center in center_list:
                for dphi in self.refine_dphi:
                    phi = (center['phi'] + dphi) % 360
                    for dv in self.refine_dv:
                        V0 = float(np.clip(center['V0'] + dv, 0.5, 8.0))
                        for a in self.refine_offsets:
                            for b in self.refine_offsets:
                                sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
                                sim_table = copy.deepcopy(table)
                                cue = Cue(cue_ball_id="cue")
                                shot = System(table=sim_table, balls=sim_balls, cue=cue)
                                try:
                                    shot.cue.set_state(V0=float(V0), phi=float(phi), theta=float(center['theta']), a=float(a), b=float(b))
                                    simulate(shot, inplace=True)
                                except Exception:
                                    continue
                                score = analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets)
                                cue_new = shot.balls['cue'].state.rvw[0][0:2]
                                remain = [x for x in my_targets if shot.balls[x].state.s != 4]
                                reach = 0
                                for x in remain:
                                    t2 = shot.balls[x].state.rvw[0][0:2]
                                    if not occluded(cue_new, t2):
                                        reach += 1
                                score += float(min(3, reach) * 8.0)
                                if score > best_score:
                                    best_score = score
                                    best = {'V0': float(V0), 'phi': float(phi), 'theta': float(center['theta']), 'a': float(a), 'b': float(b)}
                                if self.debug:
                                    print(f"[NewAgent] 微调评分 {score:.1f} 最佳 {best_score:.1f}")
                                if self.early_stop_score is not None and best_score >= float(self.early_stop_score):
                                    break
                            if self.early_stop_score is not None and best_score >= float(self.early_stop_score):
                                break
                        if self.early_stop_score is not None and best_score >= float(self.early_stop_score):
                            break
                    if self.early_stop_score is not None and best_score >= float(self.early_stop_score):
                        break

        self.stats_total_sims = int(done_sims)

        if best is None or best_score < 0:
            return self._random_action()
        try:
            sim_balls = {k: copy.deepcopy(v) for k, v in balls.items()}
            sim_table = copy.deepcopy(table)
            cue = Cue(cue_ball_id="cue")
            shot = System(table=sim_table, balls=sim_balls, cue=cue)
            shot.cue.set_state(V0=float(best['V0']), phi=float(best['phi']), theta=float(best['theta']), a=float(best['a']), b=float(best['b']))
            simulate(shot, inplace=True)
            new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state_snapshot[bid].state.s != 4]
            first_contact_ball_id = None
            for e in shot.events:
                et = str(e.event_type).lower()
                ids = list(e.ids) if hasattr(e, 'ids') else []
                if ('cushion' not in et) and ('pocket' not in et) and ('cue' in ids):
                    other_ids = [i for i in ids if i != 'cue']
                    if other_ids:
                        first_contact_ball_id = other_ids[0]
                        break
            cue_hit_cushion = False
            target_hit_cushion = False
            for e in shot.events:
                et = str(e.event_type).lower()
                ids = list(e.ids) if hasattr(e, 'ids') else []
                if 'cushion' in et:
                    if 'cue' in ids:
                        cue_hit_cushion = True
                    if first_contact_ball_id is not None and first_contact_ball_id in ids:
                        target_hit_cushion = True
            if (len(new_pocketed) == 0) and (first_contact_ball_id is not None) and (not cue_hit_cushion) and (not target_hit_cushion):
                safe_phis = [30.0, 90.0, 150.0, 210.0, 270.0, 330.0]
                replaced = False
                for phi_s in safe_phis:
                    sim_balls2 = {k: copy.deepcopy(v) for k, v in balls.items()}
                    sim_table2 = copy.deepcopy(table)
                    cue2 = Cue(cue_ball_id="cue")
                    shot2 = System(table=sim_table2, balls=sim_balls2, cue=cue2)
                    try:
                        shot2.cue.set_state(V0=3.0, phi=phi_s, theta=0.0, a=0.0, b=0.0)
                        simulate(shot2, inplace=True)
                    except Exception:
                        continue
                    cue_hit_cushion2 = False
                    for e in shot2.events:
                        et = str(e.event_type).lower()
                        ids = list(e.ids) if hasattr(e, 'ids') else []
                        if 'cushion' in et and ('cue' in ids):
                            cue_hit_cushion2 = True
                            break
                    new_pocketed2 = [bid for bid, b in shot2.balls.items() if b.state.s == 4 and last_state_snapshot[bid].state.s != 4]
                    if cue_hit_cushion2 or len(new_pocketed2) > 0:
                        best = {'V0': 3.0, 'phi': phi_s, 'theta': 0.0, 'a': 0.0, 'b': 0.0}
                        shot = shot2
                        break
            cue_new = shot.balls['cue'].state.rvw[0][0:2]
            remain = [x for x in my_targets if shot.balls[x].state.s != 4]
            reach = 0
            for x in remain:
                t2 = shot.balls[x].state.rvw[0][0:2]
                if not occluded(cue_new, t2):
                    reach += 1
            self.stats_best_next_reach = int(reach)
        except Exception:
            self.stats_best_next_reach = 0
        return best

class MCTSAgent(Agent):
    def __init__(self, checkpoint: str = None):
        super().__init__()
        # default search/config values (can be overridden by config file or dict)
        self.k_targets = 2
        self.v_candidates = [2.6, 3.4]
        self.theta_candidates = [0.0]
        self.dphi_candidates = [-3.0, 0.0, 3.0]
        self.offsets = [-0.01, 0.0, 0.01]
        self.robust_samples = 2
        self.robust_noise_std = {'V0': 0.08, 'phi': 0.06, 'theta': 0.04, 'a': 0.002, 'b': 0.002}
        self.lambda_future = 0.35
        self.stats_total_sims = 0

        # MCTS / runtime parameters (defaults)
        self.mcts_iterations = 30
        self.c_puct = 1.25
        self.max_candidates = 12
        self.rollout_sample_size = 6
        self.expansion_per_node = 4
        # per-simulation timeouts (seconds)
        self.simulation_timeout = 2.5
        self.rollout_timeout = 1.5
        # overall decision timeout (guard in training wrapper is separate)
        self.decision_timeout = 180

        # candidate generation tuning defaults
        self.cand_k_v = 1.8
        self.cand_c_v = 0.8
        self.cand_ref_d = 1.0
        self.cand_alpha_dphi = 6.0
        self.cand_max_dphi = 12.0
        self.cand_v_factors = [0.88, 1.0, 1.12]
        self.cand_base_offsets = [0.0, 0.01, -0.01]

        # load optional checkpoint (path or dict)
        cfg = None
        if checkpoint is None:
            default_path = os.path.join('eval', 'checkpoints', 'newagent_config.json')
            checkpoint = default_path if os.path.exists(default_path) else None

        try:
            if isinstance(checkpoint, dict):
                cfg = checkpoint
            elif isinstance(checkpoint, str) and os.path.isfile(checkpoint):
                with open(checkpoint, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
        except Exception:
            cfg = None

        if cfg:
            # top-level simple mappings
            self.k_targets = int(cfg.get('k_targets', self.k_targets))
            if 'v_candidates' in cfg:
                self.v_candidates = list(cfg.get('v_candidates', self.v_candidates))
            if 'theta_candidates' in cfg:
                self.theta_candidates = list(cfg.get('theta_candidates', self.theta_candidates))
            if 'dphi_candidates' in cfg:
                self.dphi_candidates = list(cfg.get('dphi_candidates', self.dphi_candidates))
            if 'offsets' in cfg:
                self.offsets = list(cfg.get('offsets', self.offsets))
            self.robust_samples = int(cfg.get('robust_samples', self.robust_samples))
            self.lambda_future = float(cfg.get('lambda_future', self.lambda_future))

            # mcts / runtime mappings
            self.mcts_iterations = int(cfg.get('mcts_iterations', self.mcts_iterations))
            self.c_puct = float(cfg.get('c_puct', self.c_puct))
            self.max_candidates = int(cfg.get('max_candidates', self.max_candidates))
            self.rollout_sample_size = int(cfg.get('rollout_sample_size', self.rollout_sample_size))
            self.expansion_per_node = int(cfg.get('expansion_per_node', self.expansion_per_node))
            self.decision_timeout = float(cfg.get('decision_timeout', self.decision_timeout))

            # timeouts for internal simulate calls (optional)
            self.simulation_timeout = float(cfg.get('simulation_timeout', self.simulation_timeout))
            self.rollout_timeout = float(cfg.get('rollout_timeout', self.rollout_timeout))

            # nested candidate_generation
            cg = cfg.get('candidate_generation', {}) if isinstance(cfg, dict) else {}
            self.cand_k_v = float(cg.get('k_v', self.cand_k_v))
            self.cand_c_v = float(cg.get('c_v', self.cand_c_v))
            self.cand_ref_d = float(cg.get('ref_d', self.cand_ref_d))
            self.cand_alpha_dphi = float(cg.get('alpha_dphi', self.cand_alpha_dphi))
            self.cand_max_dphi = float(cg.get('max_dphi', self.cand_max_dphi))
            self.cand_v_factors = list(cg.get('v_factors', self.cand_v_factors))
            self.cand_base_offsets = list(cg.get('base_offsets', self.cand_base_offsets))
            self.max_candidates = int(cg.get('max_candidates', self.max_candidates))

            # robustness
            rob = cfg.get('robustness', {}) if isinstance(cfg, dict) else {}
            self.robust_samples = int(rob.get('robust_samples', self.robust_samples))
            self.robust_noise_std = dict(rob.get('robust_noise_std', self.robust_noise_std))

            # future and risk
            fr = cfg.get('future_and_risk', {}) if isinstance(cfg, dict) else {}
            self.lambda_future = float(fr.get('lambda_future', self.lambda_future))
            self.risk_penalty_coeff = float(fr.get('risk_penalty_coeff', 200.0))

    def decision(self, balls=None, my_targets=None, table=None):
        """UCT-style MCTS decision. Modifies only MCTSAgent behaviour.

        Characteristics:
        - Small fixed-iteration MCTS (keeps runtime predictable)
        - Fast rollout policy sampling a few candidate shots
        - Per-simulation timeout via ThreadPoolExecutor.result(timeout=...)
        - Small transposition cache (in-memory) to reuse previously simulated states
        """
        # local imports to avoid changing module-level imports
        import concurrent.futures
        import time

        if balls is None or my_targets is None or table is None:
            return self._random_action()

        # basic snapshots
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ["8"]
            remaining = ["8"]
        cue_pos = balls['cue'].state.rvw[0][0:2]

        # simple helpers (reused logic from existing agent implementations)
        def ang(a, b):
            dx = b[0] - a[0]
            dy = b[1] - a[1]
            return float((math.degrees(math.atan2(dy, dx)) % 360))

        def ghost_phi(cue, target, pocket):
            v = pocket - target
            n = np.linalg.norm(v)
            if n < 1e-9:
                return ang(cue, target)
            u = v / n
            d = 0.057
            ghost = target - u * d
            return ang(cue, ghost)

        def occluded_tp(t, p, ignore_ids=None):
            thresh = 0.015
            for k, ball in balls.items():
                if ignore_ids and k in ignore_ids:
                    continue
                if balls[k].state.s == 4:
                    continue
                pos = ball.state.rvw[0][0:2]
                ap = pos - t
                ab = p - t
                denom = float(np.dot(ab, ab))
                if denom <= 1e-9:
                    continue
                tt = float(np.dot(ap, ab) / denom)
                tt = max(0.0, min(1.0, tt))
                closest = t + tt * ab
                d = float(np.linalg.norm(pos - closest))
                if d < thresh and 0.1 < tt < 0.9:
                    return True
            return False

        # candidate generation: adaptive per-target generator using configured template params
        targets_sorted = sorted(remaining, key=lambda bid: np.linalg.norm(balls[bid].state.rvw[0][0:2] - cue_pos))
        targets_sorted = targets_sorted[:self.k_targets]
        candidates = []

        def generate_for_target(bid):
            tpos = balls[bid].state.rvw[0][0:2]
            d = float(np.linalg.norm(tpos - cue_pos))
            # estimate required V0 using linear heuristic
            V_req = max(0.5, float(self.cand_k_v * d + self.cand_c_v))
            # dynamic dphi range
            dphi_max = min(float(self.cand_max_dphi), float(self.cand_alpha_dphi) * (d / max(1e-6, float(self.cand_ref_d))))
            # offsets scaled by distance (closer -> smaller offsets)
            offset_scale = min(1.0, d / max(1e-6, float(self.cand_ref_d)))
            offsets = [o * offset_scale for o in list(self.cand_base_offsets)]

            pocket_phis = []
            if hasattr(table, 'pockets') and isinstance(table.pockets, dict):
                for pk in table.pockets:
                    ppos = table.pockets[pk].center[0:2]
                    if not occluded_tp(tpos, ppos, ignore_ids=[bid, 'cue']):
                        pocket_phis.append(ghost_phi(cue_pos, tpos, ppos))
            if not pocket_phis:
                pocket_phis = [ang(cue_pos, tpos)]

            v_candidates = [float(np.clip(V_req * f, 0.5, 8.0)) for f in list(self.cand_v_factors)]
            # dphi candidates derived from dynamic max
            if dphi_max <= 0.0:
                dphi_list = [0.0]
            else:
                # choose three points: -max, 0, +max (can be extended)
                dphi_list = [-dphi_max, 0.0, dphi_max]

            out = []
            for phi0 in pocket_phis:
                for dphi in dphi_list:
                    phi = (phi0 + dphi) % 360
                    for V0 in v_candidates:
                        for a in offsets:
                            for b in offsets:
                                out.append((V0, phi, 0.0, a, b))
            return out

        # aggregate and limit candidates
        for bid in targets_sorted:
            cand_for = generate_for_target(bid)
            candidates.extend(cand_for)

        # deduplicate approximately by rounding V0 and phi
        uniq = {}
        for V0, phi, theta, a, b in candidates:
            key = (int(round(V0*100)), int(round(phi)))
            if key not in uniq:
                uniq[key] = (V0, phi, theta, a, b)
        candidates = list(uniq.values())
        # sort candidates by proximity to ideal (prefer lower V0 and nearer phi to ghost)
        candidates = sorted(candidates, key=lambda x: (x[0], abs(x[1] - ang(cue_pos, balls[targets_sorted[0]].state.rvw[0][0:2]))))
        # limit overall candidates to configured max
        candidates = candidates[:max(1, int(getattr(self, 'max_candidates', 12)))]

        if not candidates:
            return self._random_action()

        # small transposition cache keyed by a lightweight state-hash
        transposition = {}

        def state_hash(balls_state):
            # create a compact fingerprint of pocketed flags + coarse positions
            items = []
            for k, v in sorted(balls_state.items()):
                pos = v.state.rvw[0][0:2]
                items.append((k, int(round(pos[0]*100)), int(round(pos[1]*100)), int(v.state.s)))
            return tuple(items)

        # run a single simulation with timeout; returns (score, shot) or (None, None) on failure
        def run_simulation_with_timeout(V0, phi, theta, a, b, balls_src, table_src, timeout_s=None):
            def _worker():
                sim_balls = {k: copy.deepcopy(v) for k, v in balls_src.items()}
                sim_table = copy.deepcopy(table_src)
                cue = Cue(cue_ball_id="cue")
                shot = System(table=sim_table, balls=sim_balls, cue=cue)
                shot.cue.set_state(V0=float(V0), phi=float(phi), theta=float(theta), a=float(a), b=float(b))
                simulate(shot, inplace=True)
                return shot

            if timeout_s is None:
                timeout_s = float(self.simulation_timeout)

            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_worker)
                try:
                    shot = fut.result(timeout=timeout_s)
                except concurrent.futures.TimeoutError:
                    return None, None
                except Exception:
                    return None, None
                return shot, shot

        def rollout_policy(balls_src, table_src, my_targets_now):
            # quick policy: evaluate small random subset of candidates and return best score
            best_local = -1e9
            best_action = None
            # sample up to rollout_sample_size random candidates to keep rollout cheap
            sample = random.sample(candidates, min(len(candidates), max(1, int(self.rollout_sample_size))))
            for V0, phi, theta, a, b in sample:
                # check cache
                h = state_hash(balls_src)
                key = (h, int(round(V0*100)), int(round(phi)))
                if key in transposition:
                    sc = transposition[key]
                else:
                    shot, shot_ret = run_simulation_with_timeout(V0, phi, theta, a, b, balls_src, table_src, timeout_s=self.rollout_timeout)
                    if shot is None:
                        sc = -1e9
                    else:
                        sc = analyze_shot_for_reward(shot=shot, last_state=last_state_snapshot, player_targets=my_targets_now)
                    transposition[key] = sc
                if sc > best_local:
                    best_local = sc
                    best_action = (V0, phi, theta, a, b)
            return best_local, best_action

        # simple UCT node
        class Node:
            __slots__ = ('parent', 'action', 'visits', 'value', 'children', 'state')
            def __init__(self, parent, action, state):
                self.parent = parent
                self.action = action
                self.visits = 0
                self.value = 0.0
                self.children = []
                self.state = state

        root_state = {k: copy.deepcopy(v) for k, v in balls.items()}
        root = Node(parent=None, action=None, state=root_state)

        ITERATIONS = int(self.mcts_iterations)
        C_PUCT = float(self.c_puct)

        for it in range(ITERATIONS):
            # selection
            node = root
            path = [node]
            while node.children:
                # UCT: value/visits + c * sqrt(ln(N)/n)
                total_visits = sum(ch.visits for ch in node.children) + 1
                best_score = -1e9
                best_child = None
                for ch in node.children:
                    if ch.visits == 0:
                        score = 1e9 + random.random()
                    else:
                        score = (ch.value / ch.visits) + C_PUCT * math.sqrt(math.log(total_visits) / ch.visits)
                    if score > best_score:
                        best_score = score
                        best_child = ch
                node = best_child
                path.append(node)

            # expansion
            # expand with a few random candidate actions
            state_for_expand = node.state
            # pick up to expansion_per_node actions not yet in children
            tried = set(ch.action for ch in node.children)
            actions_pool = [a for a in candidates if a not in tried]
            random.shuffle(actions_pool)
            expand_actions = actions_pool[:max(1, int(self.expansion_per_node))]
            if not expand_actions:
                # leaf node; perform rollout from here
                score_rollout, act = rollout_policy(state_for_expand, table, my_targets)
                reward = score_rollout
                # backprop
                for p in reversed(path):
                    p.visits += 1
                    p.value += reward
                continue

            for act in expand_actions:
                # simulate the action briefly to get next state
                V0, phi, theta, a, b = act
                shot, shot_ret = run_simulation_with_timeout(V0, phi, theta, a, b, state_for_expand, table, timeout_s=self.simulation_timeout)
                if shot is None:
                    continue
                next_state = {k: copy.deepcopy(v) for k, v in shot.balls.items()}
                child = Node(parent=node, action=act, state=next_state)
                node.children.append(child)

            # rollout from a newly added child if any
            if node.children:
                child = random.choice(node.children)
                score_rollout, act = rollout_policy(child.state, table, my_targets)
                reward = score_rollout
                # backprop
                for p in reversed(path):
                    p.visits += 1
                    p.value += reward

        # choose best child of root by visit count / value
        if not root.children:
            return self._random_action()

        best_child = max(root.children, key=lambda ch: (ch.visits, ch.value))
        best_act = best_child.action
        # record stats roughly
        self.stats_total_sims = int(sum(ch.visits for ch in root.children))
        if best_act is None:
            return self._random_action()
        V0, phi, theta, a, b = best_act
        return {'V0': float(V0), 'phi': float(phi), 'theta': float(theta), 'a': float(a), 'b': float(b)}
