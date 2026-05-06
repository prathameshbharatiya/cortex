import time
from dataclasses import dataclass
from typing import Any

@dataclass
class ActuatorCommand:
    vx: float
    wz: float
    active_correction: bool

class SentinelPostprocessor:
    def __init__(self, platform_type: str):
        self.platform_type = platform_type
        self.last_cmd = (0.0, 0.0)
        self.max_accel = 0.5 # m/s^2
        self.last_time = time.time()

    def process(self, governed_msg: Any, state: Any) -> ActuatorCommand:
        now = time.time()
        dt = now - self.last_time
        if dt <= 0: dt = 0.01
        self.last_time = now
        
        # Handle both ROS2 messages and simple objects
        vx = governed_msg.linear.x if hasattr(governed_msg, 'linear') else getattr(governed_msg, 'vx', 0.0)
        wz = governed_msg.angular.z if hasattr(governed_msg, 'angular') else getattr(governed_msg, 'wz', 0.0)
        
        # 1. Jerk Reduction (Rate Limiting)
        max_dv = self.max_accel * dt
        dv = vx - self.last_cmd[0]
        if abs(dv) > max_dv:
            vx = self.last_cmd[0] + (max_dv if dv > 0 else -max_dv)
            
        # 2. Active Correction
        # If we are within 200ms of a limit, we actively counter-steer
        correction_active = False
        if state.time_to_violation < 0.2:
            vx *= 0.5 # Aggressive damping
            wz *= 0.5
            correction_active = True
            
        self.last_cmd = (vx, wz)
        
        return ActuatorCommand(
            vx=vx,
            wz=wz,
            active_correction=correction_active
        )
