"""Moving-reference PD in output-motor coordinates; one FIFO for all command channels."""
import torch
from height_contract import servo as prior_servo,payload as prior_payload,to_motor,UPPER,LOWER,X


def reference_motor_velocity(height,height_velocity):
    """Analytic derivative of the positive-knee IK,not a difference of held50Hz commands."""
    cosine=(X*X+height.square()-UPPER*UPPER-LOWER*LOWER)/(2*UPPER*LOWER)
    knee=torch.acos(cosine)
    knee_velocity=-height*height_velocity/(UPPER*LOWER*torch.sin(knee))
    hip_velocity=(-X*height_velocity-(UPPER*LOWER*cosine+LOWER*LOWER)*knee_velocity)/(X*X+height.square())
    return torch.stack((hip_velocity,hip_velocity+knee_velocity,hip_velocity,hip_velocity+knee_velocity),-1)


def payload(bounded,height,motor_velocity):
    if motor_velocity.shape!=(bounded.shape[0],4):
        raise ValueError('Four scaled reference motor velocities required')
    return torch.cat((prior_payload(bounded,height),motor_velocity),1)


def servo(arrived,q,v):
    if arrived.ndim!=2 or arrived.shape[1]!=11:
        raise ValueError('Actions,reference height,and reference velocity must share11channelFIFO')
    out=prior_servo(arrived[:,:7],q,v)
    base=out['motor_request']
    addition=2*arrived[:,7:11]
    request=base+addition
    pairs=request.reshape(-1,2,2)
    out.update(motor_base_request_nm=base,motor_velocity_feedforward_nm=addition,
               motor_velocity_reference_rad_s=arrived[:,7:11],motor_position_pre_rad=to_motor(q[:,:4]),
               motor_request=request,joint_request=torch.stack((pairs.sum(-1),pairs[...,1]),-1).reshape(-1,4))
    return out
