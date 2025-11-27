import argparse
import json
import os
import random
import numpy as np
import sys

# 确保项目根目录在 sys.path 中，便于从 train/ 下导入 agent 与 poolenv
_THIS_DIR = os.path.dirname(__file__)
_ROOT_DIR = os.path.abspath(os.path.join(_THIS_DIR, os.pardir))
if _ROOT_DIR not in sys.path:
    sys.path.insert(0, _ROOT_DIR)

from agent import NewAgent
from poolenv import PoolEnv
from datetime import datetime
import time
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

def eval_config(env, cfg, n_trials, writer=None, config_idx=0, verbose=False):
    agent = NewAgent(checkpoint=None, fast=True, debug=verbose)
    agent.k_targets = int(cfg['k_targets'])
    agent.v_candidates = list(cfg['v_candidates'])
    agent.theta_candidates = list(cfg['theta_candidates'])
    agent.dphi_candidates = list(cfg['dphi_candidates'])
    agent.offsets = list(cfg['offsets'])
    agent.refine_dphi = list(cfg['refine_dphi'])
    agent.refine_dv = list(cfg['refine_dv'])
    agent.refine_offsets = list(cfg['refine_offsets'])
    total = 0.0
    trial_iter = range(n_trials)
    trial_iter = tqdm(trial_iter, desc=f"config {config_idx} trials", leave=False)
    for i in trial_iter:
        t0 = time.time()
        target_ball = ['solid', 'stripe'][i % 2]
        if verbose:
            tqdm.write(f"[config {config_idx} trial {i}] reset target_ball={target_ball}")
        env.reset(target_ball=target_ball)
        player = env.get_curr_player()
        balls, my_targets, table = env.get_observation(player)
        t_decision0 = time.time()
        action = agent.decision(balls, my_targets, table)
        t_decision = time.time() - t_decision0
        if verbose:
            tqdm.write(f"[config {config_idx} trial {i}] decision_time={t_decision:.2f}s action=V0:{action['V0']:.2f},phi:{action['phi']:.2f},theta:{action['theta']:.2f},a:{action['a']:.3f},b:{action['b']:.3f}")
        info = env.take_shot(action)
        reward = 0.0
        if info.get('WHITE_BALL_INTO_POCKET') and info.get('BLACK_BALL_INTO_POCKET'):
            reward -= 150.0
        elif info.get('WHITE_BALL_INTO_POCKET'):
            reward -= 100.0
        elif info.get('BLACK_BALL_INTO_POCKET'):
            remaining = [bid for bid in my_targets if env.balls[bid].state.s != 4]
            reward += 100.0 if len(remaining) == 0 else -150.0
        reward += len(info.get('ME_INTO_POCKET', [])) * 50.0
        reward -= len(info.get('ENEMY_INTO_POCKET', [])) * 20.0
        total += reward
        t_trial = time.time() - t0
        if verbose:
            tqdm.write(f"[config {config_idx} trial {i}] reward={reward:.1f} trial_time={t_trial:.2f}s me_pocket={len(info.get('ME_INTO_POCKET', []))} enemy_pocket={len(info.get('ENEMY_INTO_POCKET', []))}")
        if hasattr(trial_iter, 'set_postfix'):
            try:
                trial_iter.set_postfix(reward=f"{reward:.1f}", t=f"{t_trial:.1f}s")
            except Exception:
                pass
        if writer is not None:
            writer.add_scalar(f'config_{config_idx}/trial_reward', reward, i)
            writer.add_scalar(f'config_{config_idx}/decision_time_s', t_decision, i)
            writer.add_scalar(f'config_{config_idx}/trial_time_s', t_trial, i)
    return total / float(max(1, n_trials))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=5)
    parser.add_argument('--out', type=str, default=os.path.join('eval', 'checkpoints', 'newagent_config.json'))
    parser.add_argument('--logdir', type=str, default=os.path.join('train', 'runs', 'newagent'))
    parser.add_argument('--fast', action='store_true', default=True)
    parser.add_argument('--verbose', action='store_true', default=False)
    args = parser.parse_args()
    random.seed(42)
    np.random.seed(42)
    env = PoolEnv()
    run_dir = os.path.join(args.logdir, datetime.now().strftime('%Y%m%d-%H%M%S'))
    os.makedirs(run_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=run_dir)
    search_space = [
        {
            'k_targets': 3,
            'v_candidates': [1.8, 2.4, 3.0, 3.8, 4.6],
            'theta_candidates': [0.0],
            'dphi_candidates': [-8.0, -4.0, 0.0, 4.0, 8.0],
            'offsets': [0.0, 0.02, -0.02],
            'refine_dphi': [-2.0, 0.0, 2.0],
            'refine_dv': [-0.6, 0.0, 0.6],
            'refine_offsets': [0.0, 0.01, -0.01]
        },
        {
            'k_targets': 2,
            'v_candidates': [2.0, 2.8, 3.6, 4.4],
            'theta_candidates': [0.0],
            'dphi_candidates': [-6.0, -3.0, 0.0, 3.0, 6.0],
            'offsets': [0.0, 0.015, -0.015],
            'refine_dphi': [-1.5, 0.0, 1.5],
            'refine_dv': [-0.5, 0.0, 0.5],
            'refine_offsets': [0.0, 0.008, -0.008]
        }
    ]
    best_cfg = None
    best_score = -1e9
    outer_iter = enumerate(search_space)
    outer_iter = tqdm(list(outer_iter), desc='configs', leave=True)
    for idx_cfg, cfg in outer_iter:
        score = eval_config(env, cfg, n_trials=args.episodes, writer=writer, config_idx=idx_cfg, verbose=args.verbose)
        if score > best_score:
            best_score = score
            best_cfg = cfg
        if writer is not None:
            writer.add_scalar('config/avg_reward', score, idx_cfg)
    if writer is not None:
        writer.add_text('meta', json.dumps({'episodes': args.episodes, 'out': args.out}))
        writer.flush()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(best_cfg, f, ensure_ascii=False, indent=2)
    print(json.dumps({'out': args.out, 'avg_reward': best_score}, ensure_ascii=False))
    if writer is not None:
        writer.add_text('best', json.dumps({'avg_reward': best_score, 'checkpoint': args.out}, ensure_ascii=False))
        writer.flush()
        writer.close()

if __name__ == '__main__':
    main()