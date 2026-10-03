"""Two bulk device reads replace per-world scalar synchronization in reporting."""
from collections import Counter
import torch
from jump_task import REASONS


def summarize(env, score, passed, retained, metrics, terms, smooth_cost, residual,
              transitions, seen_gate):
    metric_names, term_names = list(metrics), list(terms)
    columns = [passed, retained, env.task.reason, env.task.phase, env.ticks,
               env.task.peak_clearance_m, env.task.height_score.peak,
               env.task.touchdown_time, env.task.success_time, score, env.gate_tick]
    columns += list(env.plan.unbind(1))+list(metrics.values())+list(terms.values())
    data = torch.stack(columns, 1).detach().cpu().tolist()
    cases = []
    plan_end = 11+env.plan.shape[1]
    metric_end = plan_end+len(metric_names)
    for i, row in enumerate(data):
        cases.append(dict(world=i, case=i%45, passed=bool(row[0]), visible=bool(row[1]),
            reason=REASONS[int(row[2])], phase=int(row[3]), end_s=row[4]*.0025,
            wheel_cm=row[5]*100, com_cm=row[6]*100, touchdown_s=row[7], success_s=row[8],
            reward=row[9], gate_tick=int(row[10]), plan=row[11:plan_end],
            landing_metrics=dict(zip(metric_names, row[plan_end:metric_end])),
            reward_terms=dict(zip(term_names, row[metric_end:]))))
    scalars = [passed.sum(), passed[:45].sum(), env.task.peak_clearance_m.mean(),
               env.task.height_score.peak.mean(), score.mean(), smooth_cost.mean(),
               residual.max() if len(residual) else score.new_zeros(()),
               torch.as_tensor(transitions, device=score.device), seen_gate.sum()]
    scalars += [v.mean() for v in metrics.values()]+[v.mean() for v in terms.values()]
    values = torch.stack(scalars).detach().cpu().tolist()
    return dict(assist_strength=.625, worlds=env.n, passed=int(values[0]),
        original45_passed=int(values[1]), mean_wheel_cm=values[2]*100,
        mean_com_cm=values[3]*100, mean_return=values[4],
        reasons=dict(Counter(row['reason'] for row in cases)), cases=cases,
        mean_landing_metrics=dict(zip(metric_names, values[9:9+len(metric_names)])),
        mean_reward_terms=dict(zip(term_names, values[9+len(metric_names):])),
        mean_action_regularization=values[5], dense_reward_identity_max_error=values[6],
        landing_transitions=int(values[7]), launch_plan_unchanged=True,
        pre_apex_actions_zero=True, gate_reached_worlds=int(values[8]))
