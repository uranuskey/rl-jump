"""Bounded physical proposal search; search samples NEVER enter on-policy PPO."""
import guided_paths
import torch
from torch.distributions import Normal
from guided_rollout import trial
from param_runtime import write

# Offsets in the existing bounded curve logits, followed by unchanged 8 gains.
PROPOSALS=[
 ('baseline',(0,0,0,0,0)),
 ('quicker_approach',(0,-.8,0,0,0)), ('slower_approach',(0,.8,0,0,0)),
 ('longer_approach_leg',(.8,0,0,0,0)), ('shorter_approach_leg',(-.8,0,0,0,0)),
 ('deeper_cushion',(0,0,-1.5,0,.2)), ('deep_cushion',(0,0,-2.5,0,.3)),
 ('quicker_cushion',(0,0,-1.,-.8,.2)), ('longer_cushion',(0,0,-1.5,.8,.3)),
 ('quick_deep',(0,-.7,-1.8,-.6,.3)), ('long_leg_deep',(.6,0,-1.8,-.4,.3)),
 ('short_quick_deep',(-.6,-.7,-1.8,-.6,.3)),
 ('long_slow_deep',(.6,.6,-1.8,.6,.3)),
 ('early_fast_cushion',(.6,-1.4,-2.,-1.2,.4)),
 ('late_fast_cushion',(-.6,.6,-2.,-1.2,.4)),
 ('short_slow_deep',(-1.,.8,-2.,.6,.4)),
 ('early_mild',(0,-1.4,-.8,-.4,.1)),
 ('late_mild',(0,1.2,-.8,-.4,.1)),
 ('long_early',(.9,-.9,-1.2,-.9,.3)),
 ('short_early',(-1.2,-.9,-1.2,-.9,.3)),
 ('deep_slow_recovery',(0,-.4,-2.5,.2,.9)),
]


class Proposals:
    def __init__(self,policy,profiles,n):
        self.policy=policy
        # Each full group contains the same 45 delay/initial-state conditions.
        ids=(torch.arange(n,device=policy.std.device)//45).clamp(max=len(profiles)-1)
        offsets=policy.std.new_tensor([list(p[1])+[0.]*8 for p in profiles])
        self.offset=offsets[ids]
    def distribution(self,obs):
        return Normal(self.policy.actor(obs)+self.offset,self.policy.std.expand(len(obs),-1))


def search(env,standing,launch,policy,limit,out):
    assert env.n==512
    candidates=[]
    for batch,start in enumerate((1,11)):
        profiles=[PROPOSALS[0]]+PROPOSALS[start:start+10]
        *_,summary=trial(env,standing,launch,Proposals(policy,profiles,env.n),limit)
        write(out/f'guide_batch_{batch:02d}.json',summary)
        group=[]
        for i,(name,offset) in enumerate(profiles):
            rows=summary['cases'][i*45:(i+1)*45]
            assert len(rows)==45 and [r['case'] for r in rows]==list(range(45))
            avg=lambda key:sum(r['landing_metrics'][key] for r in rows)/45
            qualified=(all(r['passed'] for r in rows)
                and sum(r['wheel_cm'] for r in rows)/45>=9.065
                and sum(r['com_cm'] for r in rows)/45>=11.74)
            group.append(dict(name=name,offset=list(offset)+[0.]*8,batch=batch,
                passed=sum(r['passed'] for r in rows),qualified=qualified,
                force_n=avg('peak_force_n'),max_force_n=max(r['landing_metrics']['peak_force_n'] for r in rows),
                stroke_m=avg('first_stop_com_drop_m'),motion=avg('mean_settling_motion')))
        baseline=group[0]['force_n']
        for row in group:
            row['matched_baseline_force_n']=baseline
            row['improvement_n']=baseline-row['force_n']
        candidates.extend(group)
        write(out/'guide_search.json',dict(status='SEARCHING',candidates=candidates))
        print({'event':'guide_batch','batch':batch,'qualified':sum(r['qualified'] for r in group)},flush=True)
    acceptable=[r for r in candidates if r['qualified'] and r['improvement_n']>=1.]
    chosen=min(acceptable,key=lambda r:(r['force_n'],r['motion'])) if acceptable else candidates[0]
    result=dict(status='SEARCHED',candidates=candidates,chosen=chosen,
                proposals_are_ppo_samples=False,worlds_per_candidate=45)
    write(out/'guide_search.json',result)
    return result
