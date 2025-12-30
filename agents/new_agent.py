import math
import pooltool as pt
import numpy as np
from pooltool.objects import PocketTableSpecs, Table, TableType
from datetime import datetime
import copy
import random
import concurrent.futures

from .agent import Agent
from .basic_agent_pro import analyze_shot_for_reward

class NewAgent(Agent):
    """自定义 Agent 模板（待学生实现）"""
    
    def __init__(self):
        pass
    
    def decision(self, balls=None, my_targets=None, table=None):
        """决策方法
        
        参数：
            observation: (balls, my_targets, table)
        
        返回：
            dict: {'V0', 'phi', 'theta', 'a', 'b'}
        """
        return self._random_action()

class MCTSProAgent(Agent):
    def __init__(self, n_simulations=60, c_puct=1.25, max_candidates=36):
        super().__init__()
        self.n_simulations = n_simulations
        self.c_puct = c_puct
        self.max_candidates = max_candidates
        self.ball_radius = 0.028575
        self.sim_noise = {'V0': 0.1, 'phi': 0.15, 'theta': 0.1, 'a': 0.005, 'b': 0.005}
        self.robust_samples = 3
        self.simulation_timeout = 2.0
        self.rollout_timeout = 1.0
        self.lambda_future = 0.3
        self.gamma_next = 0.2
        self.rollout_top_k = 6
        self.prior_weight = 0.2
        self.robust_samples_next = 1
        self.cand_k_v = 1.8
        self.cand_c_v = 0.8
        self.cand_ref_d = 1.0
        self.cand_alpha_dphi = 6.0
        self.cand_max_dphi = 12.0
        self.cand_v_factors = [0.9, 1.0, 1.1]
        self.cand_base_offsets = [0.0, 0.01, -0.01]

    def _calc_angle_degrees(self, v):
        angle = math.degrees(math.atan2(v[1], v[0]))
        return angle % 360

    def _get_ghost_ball_target(self, cue_pos, obj_pos, pocket_pos):
        vec_obj_to_pocket = np.array(pocket_pos) - np.array(obj_pos)
        dist_obj_to_pocket = np.linalg.norm(vec_obj_to_pocket)
        if dist_obj_to_pocket == 0:
            return 0.0, 0.0
        unit_vec = vec_obj_to_pocket / dist_obj_to_pocket
        ghost_pos = np.array(obj_pos) - unit_vec * (2 * self.ball_radius)
        vec_cue_to_ghost = ghost_pos - np.array(cue_pos)
        dist_cue_to_ghost = np.linalg.norm(vec_cue_to_ghost)
        phi = self._calc_angle_degrees(vec_cue_to_ghost)
        return phi, dist_cue_to_ghost

    def _occluded_tp(self, balls, t, p, ignore_ids=None):
        thresh = 0.015
        for k, ball in balls.items():
            if ignore_ids and k in ignore_ids:
                continue
            if ball.state.s == 4:
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

    def generate_candidate_actions(self, balls, my_targets, table):
        actions = []
        cue_ball = balls.get('cue')
        if not cue_ball:
            return [self._random_action()]
        cue_pos = cue_ball.state.rvw[0][0:2]
        target_ids = [bid for bid in my_targets if balls[bid].state.s != 4]
        if not target_ids:
            target_ids = ['8']
        for tid in target_ids:
            obj_pos = balls[tid].state.rvw[0][0:2]
            d = float(np.linalg.norm(obj_pos - cue_pos))
            V_req = max(0.5, float(self.cand_k_v * d + self.cand_c_v))
            dphi_max = min(float(self.cand_max_dphi), float(self.cand_alpha_dphi) * (d / max(1e-6, float(self.cand_ref_d))))
            offset_scale = min(1.0, d / max(1e-6, float(self.cand_ref_d)))
            offsets = [o * offset_scale for o in list(self.cand_base_offsets)]
            pocket_phis = []
            for pk, pocket in table.pockets.items():
                ppos = pocket.center[0:2]
                if not self._occluded_tp(balls, obj_pos, ppos, ignore_ids=[tid, 'cue']):
                    phi0, _ = self._get_ghost_ball_target(cue_pos, obj_pos, ppos)
                    pocket_phis.append(phi0)
            if not pocket_phis:
                phi0 = self._calc_angle_degrees(np.array(obj_pos) - np.array(cue_pos))
                pocket_phis = [phi0]
            v_candidates = [float(np.clip(V_req * f, 0.5, 8.0)) for f in list(self.cand_v_factors)]
            if dphi_max <= 0.0:
                dphi_list = [0.0]
            else:
                dphi_list = [-dphi_max, 0.0, dphi_max]
            for phi0 in pocket_phis:
                for dphi in dphi_list:
                    phi = (phi0 + dphi) % 360
                    for V0 in v_candidates:
                        for a in offsets:
                            for b in offsets:
                                actions.append({'V0': V0, 'phi': phi, 'theta': 0.0, 'a': a, 'b': b, 'V_req': V_req, 'dphi': float(dphi), 'dphi_max': float(dphi_max)})
        uniq = {}
        for a in actions:
            k = (int(round(a['V0'] * 100)), int(round(a['phi'])))
            if k not in uniq:
                uniq[k] = a
        actions = list(uniq.values())
        actions = actions[:max(1, int(self.max_candidates))]
        if len(actions) == 0:
            actions.append(self._random_action())
        random.shuffle(actions)
        return actions

    def _simulate_once(self, balls, table, action):
        sim_balls = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        sim_table = copy.deepcopy(table)
        cue = pt.Cue(cue_ball_id="cue")
        shot = pt.System(table=sim_table, balls=sim_balls, cue=cue)
        noisy_V0 = np.clip(action['V0'] + np.random.normal(0, self.sim_noise['V0']), 0.5, 8.0)
        noisy_phi = (action['phi'] + np.random.normal(0, self.sim_noise['phi'])) % 360
        noisy_theta = np.clip(action['theta'] + np.random.normal(0, self.sim_noise['theta']), 0, 90)
        noisy_a = np.clip(action['a'] + np.random.normal(0, self.sim_noise['a']), -0.5, 0.5)
        noisy_b = np.clip(action['b'] + np.random.normal(0, self.sim_noise['b']), -0.5, 0.5)
        cue.set_state(V0=noisy_V0, phi=noisy_phi, theta=noisy_theta, a=noisy_a, b=noisy_b)
        pt.simulate(shot, inplace=True)
        return shot

    def _simulate_with_timeout(self, balls, table, action, timeout_s):
        if timeout_s is None:
            timeout_s = float(self.simulation_timeout)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            fut = ex.submit(self._simulate_once, balls, table, action)
            try:
                shot = fut.result(timeout=timeout_s)
            except concurrent.futures.TimeoutError:
                return None
            except Exception:
                return None
            return shot

    def _state_hash(self, balls_state):
        items = []
        for k, v in sorted(balls_state.items()):
            pos = v.state.rvw[0][0:2]
            items.append((k, int(round(pos[0] * 100)), int(round(pos[1] * 100)), int(v.state.s)))
        return tuple(items)

    def _future_reach(self, shot, my_targets):
        cue_new = shot.balls['cue'].state.rvw[0][0:2]
        remain = [x for x in my_targets if shot.balls[x].state.s != 4]
        reach = 0
        for x in remain:
            t2 = shot.balls[x].state.rvw[0][0:2]
            if not self._occluded_tp(shot.balls, cue_new, t2, ignore_ids=[x, 'cue']):
                reach += 1
        return reach

    def _action_prior(self, act):
        dv = abs(float(act.get('V0', 0.0)) - float(act.get('V_req', float(act.get('V0', 0.0)))))
        v_term = 1.0 - float(np.clip(dv / 2.0, 0.0, 1.0))
        dphi_max = float(max(1e-6, act.get('dphi_max', 1.0)))
        dphi = abs(float(act.get('dphi', 0.0)))
        ang_term = 1.0 - float(np.clip(dphi / dphi_max, 0.0, 1.0))
        off_term = 1.0 - float(np.clip((abs(float(act.get('a', 0.0))) + abs(float(act.get('b', 0.0)))) / 0.02, 0.0, 1.0))
        prior = float(np.clip(0.4 * v_term + 0.4 * ang_term + 0.2 * off_term, 0.0, 1.0))
        return prior

    def _evaluate_next(self, shot, last_state_snapshot, my_targets, table):
        new_pocketed = [bid for bid, b in shot.balls.items() if b.state.s == 4 and last_state_snapshot[bid].state.s != 4]
        own_pocketed = [bid for bid in new_pocketed if bid in my_targets]
        cue_pocketed = 'cue' in new_pocketed
        if len(own_pocketed) == 0 or cue_pocketed:
            return 0.0
        my_targets_next = [bid for bid in my_targets if shot.balls[bid].state.s != 4]
        if len(my_targets_next) == 0:
            my_targets_next = ['8']
        cand_next = self.generate_candidate_actions(shot.balls, my_targets_next, table)
        sample = random.sample(cand_next, min(len(cand_next), max(1, int(self.rollout_top_k))))
        best_norm = 0.0
        for act in sample:
            shot2 = self._simulate_with_timeout(shot.balls, table, act, timeout_s=self.rollout_timeout)
            if shot2 is None:
                continue
            raw2 = analyze_shot_for_reward(shot2, {k: copy.deepcopy(v) for k, v in shot.balls.items()}, my_targets_next)
            norm2 = (raw2 - (-500.0)) / 650.0
            norm2 = float(np.clip(norm2, 0.0, 1.0))
            if norm2 > best_norm:
                best_norm = norm2
        return best_norm

    def decision(self, balls=None, my_targets=None, table=None):
        if balls is None or my_targets is None or table is None:
            return self._random_action()
        remaining = [bid for bid in my_targets if balls[bid].state.s != 4]
        if len(remaining) == 0:
            my_targets = ["8"]
        last_state_snapshot = {bid: copy.deepcopy(ball) for bid, ball in balls.items()}
        candidate_actions = self.generate_candidate_actions(balls, my_targets, table)
        n_candidates = len(candidate_actions)
        N = np.zeros(n_candidates)
        Q = np.zeros(n_candidates)
        transposition = {}
        h0 = self._state_hash(balls)
        for i in range(self.n_simulations):
            if i < n_candidates:
                idx = i
            else:
                total_n = np.sum(N)
                ucb_values = (Q / (N + 1e-6)) + self.c_puct * np.sqrt(np.log(total_n + 1) / (N + 1e-6))
                idx = int(np.argmax(ucb_values))
            act = candidate_actions[idx]
            key = (h0, int(round(act['V0'] * 100)), int(round(act['phi'])))
            if key in transposition:
                normalized_reward = transposition[key]
            else:
                rewards = []
                for _ in range(max(1, int(self.robust_samples))):
                    shot = self._simulate_with_timeout(balls, table, act, timeout_s=self.rollout_timeout)
                    if shot is None:
                        raw_reward = -500.0
                        rewards.append((raw_reward, 0))
                        continue
                    raw_reward = analyze_shot_for_reward(shot, last_state_snapshot, my_targets)
                    reach = self._future_reach(shot, my_targets)
                    raw_reward += float(self.lambda_future) * float(reach) * 15.0
                    rewards.append((raw_reward, reach))
                raw_mean = float(np.mean([r for r, _ in rewards]))
                normalized_reward = (raw_mean - (-500.0)) / 650.0
                normalized_reward = float(np.clip(normalized_reward, 0.0, 1.0))
                next_norm = self._evaluate_next(shot if len(rewards) > 0 and shot is not None else None or self._simulate_with_timeout(balls, table, act, timeout_s=self.rollout_timeout), last_state_snapshot, my_targets, table) if (len(rewards) > 0 and shot is not None) else 0.0
                normalized_reward = float((1.0 - self.gamma_next) * normalized_reward + self.gamma_next * next_norm)
                prior = self._action_prior(act)
                normalized_reward = float((1.0 - self.prior_weight) * normalized_reward + self.prior_weight * prior)
                transposition[key] = normalized_reward
            N[idx] += 1
            Q[idx] += normalized_reward
        avg_rewards = Q / (N + 1e-6)
        best_idx = int(np.argmax(avg_rewards))
        best_action = candidate_actions[best_idx]
        return best_action
