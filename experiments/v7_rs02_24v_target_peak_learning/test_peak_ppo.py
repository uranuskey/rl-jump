"""Numerical tests of accepted steps, KL bounds and Adam rollback on rejection."""
import copy
import peak_runtime
import unittest
import torch
from slot_learning import Policy, exploration
from peak_ppo import PPO


class OptimizerTests(unittest.TestCase):
    def data(self):
        torch.manual_seed(99)
        policy = Policy()
        exploration(policy,0)
        obs = torch.randn(64,17)*.1
        with torch.no_grad():
            action = policy.distribution(obs).sample()
            reward = 100*(action[:,2]-policy.actor(obs)[:,2])/policy.std[2]
        return policy,obs,action,reward,torch.ones(64,dtype=torch.bool)

    def test_backtracking_learns_within_kl_bound(self):
        policy,obs,action,reward,eligible = self.data()
        ppo = PPO(policy,actor_lr=.001,attempts=12)
        before = [v.clone() for v in policy.actor.parameters()]
        old_mean = policy.actor(obs).detach().clone()
        stats = ppo.update(obs,action,reward,eligible)
        self.assertGreater(stats['actor_steps'],0)
        self.assertGreater(stats['rejected_actor_steps'],0)
        self.assertTrue(any(not torch.equal(a,b) for a,b in zip(before,policy.actor.parameters())))
        self.assertLessEqual(stats['max_accepted_kl'],.005)
        final_kl = float((((policy.actor(obs).detach()-old_mean)/policy.std).square().sum(-1)*.5).mean())
        self.assertLessEqual(final_kl,.005)
        self.assertEqual(stats['actor_lr_next'],ppo.actor_optimizer.param_groups[0]['lr'])
        self.assertGreater(len(ppo.actor_optimizer.state_dict()['state']),0)

    def test_total_rejection_preserves_parameters_and_adam_moments(self):
        policy,obs,action,reward,eligible = self.data()
        ppo = PPO(policy,actor_lr=5e-5,attempts=7)
        self.assertGreater(ppo.update(obs,action,reward,eligible)['actor_steps'],0)
        ppo.attempts=1
        for group in ppo.actor_optimizer.param_groups:
            group['lr']=1.
        with torch.no_grad():
            action=policy.distribution(obs).sample()
            reward=100*(action[:,2]-policy.actor(obs)[:,2])/policy.std[2]
        before = [v.clone() for v in policy.actor.parameters()]
        old_moments = copy.deepcopy(ppo.actor_optimizer.state_dict()['state'])
        stats = ppo.update(obs,action,reward,eligible)
        self.assertEqual(stats['actor_steps'],0)
        self.assertEqual(stats['rejected_actor_steps'],8)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,policy.actor.parameters())))
        new_moments=ppo.actor_optimizer.state_dict()['state']
        self.assertEqual(old_moments.keys(),new_moments.keys())
        for parameter,values in old_moments.items():
            for key,value in values.items():
                self.assertTrue(torch.equal(value,new_moments[parameter][key]))


if __name__=='__main__':
    torch.set_num_threads(1)
    unittest.main()
