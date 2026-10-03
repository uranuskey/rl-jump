"""CPU-only portability, frozen-file, model, XML, and reward validation."""
from pathlib import Path
import sys,json,torch,mujoco
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'experiments/v7_rs02_24v_flat_landing_reward'))
import bootstrap, runtime, landing, learning, standing_policy, test_reward
print('frozen_sha256',runtime.verify())
checks=test_reward.main()
standing=standing_policy.JumpPolicy('cpu').eval();policy=learning.Policy('cpu').eval()
with torch.no_grad():
    assert torch.isfinite(standing.actor(torch.zeros(3,145))).all()
    assert torch.isfinite(policy.actor(torch.zeros(3,17))).all()
model=mujoco.MjModel.from_xml_path(str(ROOT/'experiments/v7_mujoco_training_gate/v7_full_collision.xml'))
project_files=[]
for name,module in list(sys.modules.items()):
    file=getattr(module,'__file__',None)
    if file:
        path=Path(file).resolve()
        if path.is_relative_to(ROOT):project_files.append(str(path.relative_to(ROOT)))
        if 'serial_wheel_leg_rl' in str(path) and '.venv' not in str(path):raise RuntimeError('Borrowed original source: '+str(path))
print(json.dumps(dict(status='PASS_OFFLINE_ONLY',reward_checks=len(checks),model_nq=model.nq,model_nv=model.nv,project_modules=sorted(project_files)),indent=2))
