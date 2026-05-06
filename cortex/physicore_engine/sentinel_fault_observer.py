import numpy as np
from collections import deque

class FaultObserver:
    """
    L5: Hardware Fault Observer
    Detects failure modes from parameter drift
    patterns in the RLS estimator output.
    """
    
    FAULT_SIGNATURES = {
        'MOTOR_BEARING_WEAR': {
            'friction_trend': 'increasing',
            'friction_rate_threshold': 0.005,
            'description': 'Friction increasing steadily — bearing wear'
        },
        'UNEXPECTED_PAYLOAD': {
            'mass_jump': True,
            'mass_delta_threshold': 0.5,
            'description': 'Sudden mass increase — unexpected payload'
        },
        'ACTUATOR_DEGRADATION': {
            'efficiency_trend': 'decreasing',
            'efficiency_rate_threshold': 0.01,
            'description': 'Efficiency dropping — actuator degradation'
        },
        'SENSOR_DRIFT': {
            'residual_trend': 'increasing',
            'residual_rate_threshold': 0.02,
            'description': 'Residual growing — possible sensor drift'
        }
    }
    
    def __init__(self, window_size=100):
        self.mass_history = deque(maxlen=window_size)
        self.friction_history = deque(maxlen=window_size)
        self.residual_history = deque(maxlen=window_size)
        self.detected_faults = []
    
    def update(self, mass, friction, residual):
        self.mass_history.append(mass)
        self.friction_history.append(friction)
        self.residual_history.append(residual)
        
        if len(self.mass_history) < 20:
            return None
        
        return self._classify_faults()
    
    def _classify_faults(self):
        faults = []
        
        # Check friction trend
        friction_arr = np.array(self.friction_history)
        friction_rate = np.polyfit(range(len(friction_arr)), friction_arr, 1)[0]
        
        if (friction_rate > self.FAULT_SIGNATURES['MOTOR_BEARING_WEAR']['friction_rate_threshold']):
            faults.append({
                'type': 'MOTOR_BEARING_WEAR',
                'confidence': min(1.0, friction_rate / 0.01),
                'description': self.FAULT_SIGNATURES['MOTOR_BEARING_WEAR']['description']
            })
        
        # Check mass jump
        if len(self.mass_history) >= 10:
            mass_delta = abs(self.mass_history[-1] - self.mass_history[-10])
            if (mass_delta > self.FAULT_SIGNATURES['UNEXPECTED_PAYLOAD']['mass_delta_threshold']):
                faults.append({
                    'type': 'UNEXPECTED_PAYLOAD',
                    'confidence': min(1.0, mass_delta / 1.0),
                    'description': f'Mass changed by {mass_delta:.2f}kg'
                })
        
        # Check residual trend
        residual_arr = np.array(self.residual_history)
        residual_rate = np.polyfit(range(len(residual_arr)), residual_arr, 1)[0]
        
        if (residual_rate > self.FAULT_SIGNATURES['SENSOR_DRIFT']['residual_rate_threshold']):
            faults.append({
                'type': 'SENSOR_DRIFT',
                'confidence': min(1.0, residual_rate / 0.05),
                'description': self.FAULT_SIGNATURES['SENSOR_DRIFT']['description']
            })
        
        # Unknown anomaly — doesn't match any pattern
        if (not faults and 
            residual_arr[-1] > 0.8 and
            abs(friction_rate) < 0.001):
            
            mass_delta = 0
            if len(self.mass_history) >= 10:
                mass_delta = abs(self.mass_history[-1] - self.mass_history[-10])
                
            if abs(mass_delta) < 0.1:
                faults.append({
                    'type': 'UNKNOWN_ANOMALY',
                    'confidence': 0.5,
                    'description': 'Anomalous behavior not matching known fault signatures'
                })
        
        return faults if faults else None
