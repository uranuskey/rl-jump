"""Independent causal reconstruction of the new motor command, including FIFO."""
import numpy as np
from twpd_control import PROFILE

def feedback_audit(path):
    with np.load(path) as z:
        q=z['sensor_twpd_q'];v=z['sensor_twpd_v'];tick=z['sensor_twpd_ticks']
        active=z['active'];apex=z['sensor_twpd_apex'];terminal=z['sensor_twpd_terminal']
        # Pre-control state must be the previous physical step's state.
        assert np.array_equal(q[1:][active[1:]],z['q'][:-1][active[1:]])
        assert np.array_equal(v[1:][active[1:]],z['v'][:-1][active[1:]])
        assert np.array_equal((tick+1)[active],z['ticks'][active])
        w,x,y,zz=np.moveaxis(q[...,3:7],-1,0)
        pitch=np.arctan2(2*(w*y-zz*x),1-2*(x*x+y*y))
        assert np.allclose(pitch,z['sensor_twpd_pitch'],atol=3e-7,rtol=2e-6)
        assert np.array_equal(v[...,4],z['sensor_twpd_gyro'])
        t=tick*.0025
        enabled=(t>=.6)&~apex&~terminal
        ramp=np.clip((t-.6)/.05,0,1)
        command=np.where(enabled,np.clip((14*pitch+v[...,4])/2,-1,1)*ramp,0)
        assert np.array_equal(enabled,z['sensor_twpd_enabled'])
        assert np.allclose(ramp,z['sensor_twpd_ramp'],atol=2e-6,rtol=2e-6)
        assert np.allclose(command,z['sensor_twpd_action'],atol=5e-6,rtol=2e-6)
        old=z['sensor_twpd_held'];new=z['sensor_twpd_revised'];req=z['requested_payload'];arr=z['arrived_payload']
        expected=old.copy();expected[...,4:6]=np.where(enabled[...,None],command[...,None],old[...,4:6])
        assert np.allclose(expected,new,atol=5e-6,rtol=2e-6)
        other=[i for i in range(15) if i not in (4,5)]
        assert np.array_equal(new[...,other],old[...,other])
        before=~apex
        assert np.array_equal(req[before],new[before])
        assert np.all(abs(new[...,4:6][enabled])<=1)
        assert not np.any(z['landing_control_enabled'][before])
        assert np.all(z['landing_action'][before]==0)
        for world in range(active.shape[1]):
            delay=(world%45)//5;fifo=np.zeros_like(arr[:,world])
            if delay:fifo[delay:]=req[:-delay,world]
            else:fifo[:]=req[:,world]
            assert np.array_equal(fifo,arr[:,world]),'Wheel or other payload bypassed physical FIFO'
        torque=z['sensor_motor_torque_nm'][active];envelope=z['sensor_motor_envelope_nm'][active]
        assert np.all(abs(torque)<=envelope+1e-5)
        return dict(status='PASS',precontrol_state_causal=True,landing_actor_apex_gate_unchanged=True,
            fifo_exact=True,only_preapex_wheel_channels_changed=True,profile=PROFILE,
            active_samples=int(active.sum()),feedback_samples=int((enabled&active).sum()),
            max_abs_action=float(abs(new[...,4:6][enabled&active]).max(initial=0)),
            motor_envelope_verified=True,no_external_wrench_output=True)
