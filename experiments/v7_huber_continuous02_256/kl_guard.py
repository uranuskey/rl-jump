"""Transactional Adam step safeguard on the entire current on-policy rollout.

The trial Adam moments do not depend on its learning rate. Interpolating all
parameters is equivalent to scaling that step's LR; accepted moments persist.
A rejected step restores parameters AND optimizer moments/step counters.
This bounds empirical old-to-new Gaussian mean KL, not closed-loop behavior.
"""
import copy
import torch

class RolloutKLGuard:
 def __init__(self,policy,optimizer,obs,mean,std,limit=.02):
  self.policy=policy;self.optimizer=optimizer;self.obs=obs.detach();self.mean=mean.detach();self.std=std.detach();self.limit=limit
  self.rows=[];self.pre=optimizer.register_step_pre_hook(self.before);self.post=optimizer.register_step_post_hook(self.after)
 @torch.no_grad()
 def kl(self):
  mu=self.policy.actor(self.obs);sd=self.policy.std.expand_as(mu)
  if not bool(torch.isfinite(mu).all() and torch.isfinite(sd).all() and (sd>0).all()):return float('inf')
  return float((torch.log(sd/self.std)+(self.std.square()+(self.mean-mu).square())/(2*sd.square())-.5).sum(-1).mean())
 def before(self,opt,args,kwargs):
  self.old=[p.detach().clone() for p in self.policy.parameters()];self.old_state=copy.deepcopy(opt.state_dict())
 @torch.no_grad()
 def after(self,opt,args,kwargs):
  params=list(self.policy.parameters());new=[p.detach().clone() for p in params]
  if not all(bool(torch.isfinite(p).all()) for p in new):
   for p,old in zip(params,self.old):p.copy_(old)
   opt.load_state_dict(self.old_state)
   raise FloatingPointError('Nonfinite candidate Adam parameters; restored and stopped')
  trials=[];accepted=False
  for alpha in (1.,.5,.25,.125,.0625,.03125,.015625):
   if alpha!=1.:
    for p,old,candidate in zip(params,self.old,new):p.copy_(old+alpha*(candidate-old))
   kl=self.kl();trials.append(dict(alpha=alpha,kl=kl))
   if kl<=self.limit:
    accepted=True;break
  if not accepted:
   for p,old in zip(params,self.old):p.copy_(old)
   opt.load_state_dict(self.old_state);alpha=0.;kl=self.kl()
  if kl>self.limit+1e-7:raise AssertionError('Rollback failed to restore empirical KL bound')
  self.rows.append(dict(accepted=accepted,alpha=alpha,accepted_kl=kl,trial_full_kl=trials[0]['kl'],trials=trials))
  del self.old,self.old_state
 def close(self):self.pre.remove();self.post.remove()
