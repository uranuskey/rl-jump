"""Render the completed fixed-assistance training evidence, without physics."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, required=True)
    args = parser.parse_args()
    audit = read(HERE / 'TRAINING_RESULT.json')
    assert audit['status'] == 'AUDITED'
    initial = read(args.training / 'baseline.json')
    entries = [dict(update=0, evaluation=initial)]
    entries += [read(path) for path in sorted(args.training.glob('evaluation_*.json'))]
    updates = [entry['update'] for entry in entries]
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for key, ax, label, floor in (
        ('mean_wheel_cm', axes[0, 0], 'Mean simultaneous wheel clearance (cm)', 9.065),
        ('mean_com_cm', axes[0, 1], 'Mean airborne whole-COM rise (cm)', 11.74),
    ):
        ax.plot(updates, [entry['evaluation'][key] for entry in entries], 'o-', lw=1.5)
        ax.axhline(floor, color='#b65d20', ls='--', label='Selection mean floor')
        ax.set(ylabel=label, xlabel='PPO update')
        ax.legend()
    ax = axes[1, 0]
    ax.plot(updates, [entry['evaluation']['mean_landing_metrics']['peak_force_n'] for entry in entries], 'o-', color='#17654d')
    ax.set(ylabel='Mean landing force peak (N)', xlabel='PPO update')
    ax = axes[1, 1]
    ax.plot(updates, [100 * entry['evaluation']['passed'] / entry['evaluation']['worlds'] for entry in entries], 'o-', label='Deterministic evaluation')
    sampled = [read(path) for path in sorted(args.training.glob('update_*.json'))]
    ax.plot([row['update'] for row in sampled], [100 * row['trial']['passed'] / row['trial']['worlds'] for row in sampled], alpha=.55, lw=1, label='Exploration samples')
    ax.set(ylabel='Height + stable landing passed (%)', xlabel='PPO update', ylim=(-3, 103))
    ax.legend()
    for ax in axes.flat:
        ax.grid(alpha=.2)
        ax.axvline(audit['selected_update'], color='#585858', ls=':', alpha=.7)
    fig.suptitle('24 V / 62.5% attitude assistance / flat landing\n256 worlds, 128 updates; original 45 known conditions', fontsize=14)
    fig.savefig(HERE / 'TRAINING_CURVES.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, constrained_layout=True)
    evaluation = read(args.evaluation / 'result.json')
    names = ['baseline'] if evaluation['selected_is_initial'] else ['baseline', 'selected']
    for name in names:
        with np.load(args.evaluation / (name + '_traces.npz')) as archive:
            phase = archive['phase'][:, 0]
            force = archive['sensor_wheel_force_n'][:, 0].sum(-1)
            legs = archive['sensor_leg_height_m'][:, 0].mean(-1)
            ticks = archive['ticks'][:, 0]
        touch = np.flatnonzero(phase == 3)[0]
        time = (ticks - ticks[touch]) * .0025
        mask = (time >= -.05) & (time <= .8)
        axes[0].plot(time[mask], force[mask], label=name)
        axes[1].plot(time[mask], legs[mask] * 1000, label=name)
    axes[0].set(ylabel='Both-wheel support force (N)')
    axes[1].set(ylabel='Mean hip-to-wheel height (mm)', xlabel='Time from touchdown (s)')
    for ax in axes:
        ax.axvline(0, ls='--', color='gray', alpha=.5)
        ax.grid(alpha=.2)
        ax.legend()
    fig.suptitle('Nominal 0 ms-delay case: landing force and actual leg compression\nSimulation comparison at the same 62.5% assistance', fontsize=13)
    fig.savefig(HERE / 'LANDING_COMPARISON.png', dpi=160)
    plt.close(fig)
    print(json.dumps({'curves': str(HERE / 'TRAINING_CURVES.png'), 'landing': str(HERE / 'LANDING_COMPARISON.png')}))


if __name__ == '__main__':
    main()
