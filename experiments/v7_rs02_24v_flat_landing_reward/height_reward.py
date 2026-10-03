"""Reward actual whole-system COM flight, never standing height or wheel tuck."""
import torch

SCALE = 120.0


class FlightHeight:
    def __init__(self, n, device, dtype=torch.float32):
        self.float_names = ('release_z', 'release_time', 'release_vz', 'candidate_peak',
                            'peak', 'credited', 'total_reward')
        self.bool_names = ('candidate', 'confirmed', 'apex')
        for name in self.float_names:
            setattr(self, name, torch.zeros(n, device=device, dtype=dtype))
        for name in self.bool_names:
            setattr(self, name, torch.zeros(n, device=device, dtype=torch.bool))

    def reset(self, mask):
        for name in self.float_names + self.bool_names:
            getattr(self, name)[mask] = 0

    def update(self, x, before, after, active, time, *, compress=1, flight=2):
        free = x['wheel_force_n'].max(1).values <= .5
        searching = active & (before == compress) & ~self.confirmed
        interrupted = searching & ~free
        self.candidate[interrupted] = False
        beginning = searching & free & ~self.candidate
        self.release_z[beginning] = x['com_z_m'][beginning]
        self.release_time[beginning] = time[beginning]
        self.release_vz[beginning] = x['com_vz_mps'][beginning]
        self.candidate_peak[beginning] = 0
        self.candidate |= beginning
        accumulating = active & free & self.candidate & ((before == compress) | (before == flight))
        rise = (x['com_z_m'] - self.release_z).clamp_min(0)
        self.candidate_peak = torch.where(accumulating, torch.maximum(self.candidate_peak, rise), self.candidate_peak)
        self.confirmed |= active & (after == flight) & self.candidate & (self.release_vz >= .15)
        scoring = active & free & self.confirmed & (after == flight)
        self.peak = torch.where(scoring, torch.maximum(self.peak, self.candidate_peak), self.peak)
        reward = SCALE * (self.peak - self.credited).clamp_min(0)
        self.credited = self.peak.clone()
        self.total_reward += reward
        self.apex |= scoring & (x['com_vz_mps'] <= 0)
        return reward
