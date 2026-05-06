import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional

@dataclass
class FilteredCommand:
    vx: float
    wz: float
    timestamp: float

@dataclass
class StateEstimate:
    joint_positions: list
    joint_velocities: list
    predicted_violation: bool
    time_to_violation: float # seconds

class SentinelPreprocessor:
    def __init__(self, platform_type: str):
        self.platform_type = platform_type
        self.last_ts = time.time()

    def process(self, raw_msg: Any, sensor_data: Dict[str, Any]) -> Tuple[FilteredCommand, StateEstimate]:
        # 1. Normalize Input
        # Handle both ROS2 messages and simple objects
        vx = getattr(raw_msg.linear, 'x', 0.0) if hasattr(raw_msg, 'linear') else getattr(raw_msg, 'vx', 0.0)
        wz = getattr(raw_msg.angular, 'z', 0.0) if hasattr(raw_msg, 'angular') else getattr(raw_msg, 'wz', 0.0)
        
        # 2. State Estimation & Prediction
        # We predict 500ms into the future
        dt = 0.5
        
        # Extract joint data (mocked for MVK, would come from sensors)
        j_pos = sensor_data.get('joint_positions', [0.0] * 6)
        j_vel = sensor_data.get('joint_velocities', [0.0] * 6)
        
        # Simple linear prediction: pos_next = pos + vel * dt
        predicted_pos = [p + v * dt for p, v in zip(j_pos, j_vel)]
        
        # Check for limit breaches in prediction
        # (Limits would be passed in config, using hardcoded for demo)
        violation = any(abs(p) > 2.8 for p in predicted_pos) # 2.8 rad limit
        
        # Calculate time to violation (very simple linear)
        ttv = 10.0 # Default safe
        for p, v in zip(j_pos, j_vel):
            if abs(v) > 0.001:
                dist = 2.8 - abs(p)
                t = dist / abs(v)
                ttv = min(ttv, t)

        filtered = FilteredCommand(vx=vx, wz=wz, timestamp=time.time())
        state = StateEstimate(
            joint_positions=j_pos,
            joint_velocities=j_vel,
            predicted_violation=violation,
            time_to_violation=ttv
        )
        
        return filtered, state
