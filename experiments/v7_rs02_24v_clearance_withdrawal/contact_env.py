"""Observe exact GPU contact pairs at the first terminal event; no physics edits."""
import diagnostic_runtime
import torch
import warp as wp
from fix_env import ConstraintEnv

@wp.kernel
def capture_pairs(ncon:wp.array[int], geom:wp.array[wp.vec2i], world:wp.array[int],
                  body:wp.array[int], dist:wp.array[float], force:wp.array[wp.spatial_vector],
                  take:wp.array[wp.bool], count:wp.array[int],
                  pairs:wp.array3d[int], values:wp.array3d[float]):
    c=wp.tid()
    if c<ncon[0]:
        w=world[c]
        if take[w]:
            ga=geom[c][0];gb=geom[c][1]
            f=wp.spatial_top(force[c])
            if body[ga]>0 and body[gb]>0 and f[0]>1.e-4 and dist[c]<-1.e-5:
                j=wp.atomic_add(count,w,1)
                if j<16:
                    pairs[w,j,0]=ga;pairs[w,j,1]=gb
                    values[w,j,0]=dist[c];values[w,j,1]=f[0]

class ContactEnv(ConstraintEnv):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.contact_count=wp.zeros(self.n,dtype=wp.int32,device=self.device)
        self.contact_pairs=wp.zeros((self.n,16,2),dtype=wp.int32,device=self.device)
        self.contact_values=wp.zeros((self.n,16,2),dtype=wp.float32,device=self.device)
        self.contact_q=torch.zeros_like(self.q)
        self.contact_v=torch.zeros_like(self.v)
        self.contact_seen=torch.zeros_like(self.paused)
    def reset(self,mask,**kwargs):
        result=super().reset(mask,**kwargs)
        if hasattr(self,'contact_count'):
            for name in ('contact_count','contact_pairs','contact_values'):
                wp.to_torch(getattr(self,name))[mask]=0
            self.contact_q[mask]=0;self.contact_v[mask]=0;self.contact_seen[mask]=False
        return result
    def capture_terminal(self,x,mask,phase_before):
        if hasattr(self,'contact_seen'):
            take=mask & ~self.contact_seen & x['self_contact']
            self.contact_q[take]=self.q[take];self.contact_v[take]=self.v[take]
            with wp.ScopedStream(self.stream):
                wp.launch(capture_pairs,dim=self.gd.naconmax,inputs=[
                    self.gd.nacon,self.gd.contact.geom,self.gd.contact.worldid,self.gm.geom_bodyid,
                    self.gd.contact.dist,self.forces_wp,wp.from_torch(take,dtype=wp.bool),
                    self.contact_count,self.contact_pairs,self.contact_values])
            self.contact_seen |= take
        return super().capture_terminal(x,mask,phase_before)
