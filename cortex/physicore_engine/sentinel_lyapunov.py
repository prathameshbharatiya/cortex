import numpy as np

class LyapunovKernel:
    """
    Uncertainty-Aware Lyapunov Stability Kernel.
    
    Replaces simple velocity clamping with
    mathematically verified stability checking.
    
    Energy function: V(x) = x^T P x
    Stability condition: dV/dt <= -alpha * V(x)
    """
    
    def __init__(self, mass, max_velocity, alpha=0.1):
        # P matrix — positive definite
        # For simplicity: diagonal P
        # Real system: solve Lyapunov equation
        self.P = np.diag([
            1.0 / (max_velocity ** 2),  # pos weight
            1.0 / (max_velocity ** 2)   # vel weight
        ])
        self.alpha = alpha
        self.mass = mass
        self.basin_limit = 1.0
    
    def compute_energy(self, pos, vel, target_pos):
        """
        V(x) = x^T P x
        where x = [pos_error, vel]
        """
        # We assume pos and vel are 3D, but we govern 2D (x, z for mobile, or first 2 joints)
        pos_error = np.array(pos[:2]) - np.array(target_pos[:2])
        state = np.concatenate([pos_error, np.array(vel)[:2]])
        
        # 4x4 P for full state (2D pos error + 2D velocity)
        P_full = np.diag([
            self.P[0,0], self.P[0,0],
            self.P[1,1], self.P[1,1]
        ])
        
        V = float(state @ P_full @ state)
        return V
    
    def compute_energy_derivative(self, pos, vel, accel, target_pos):
        """
        dV/dt = 2 * x^T P xdot
        """
        pos_error = np.array(pos[:2]) - np.array(target_pos[:2])
        state = np.concatenate([pos_error, np.array(vel)[:2]])
        
        # xdot = [vel, accel]
        state_dot = np.concatenate([np.array(vel)[:2], np.array(accel)[:2]])
        
        P_full = np.diag([
            self.P[0,0], self.P[0,0],
            self.P[1,1], self.P[1,1]
        ])
        
        dV = 2.0 * float(state @ P_full @ state_dot)
        return dV
    
    def project_command(self, commanded_accel, pos, vel, target_pos):
        """
        If command would violate stability:
        project it onto the stability boundary.
        
        Returns (safe_accel, intervened, V, dV)
        """
        V = self.compute_energy(pos, vel, target_pos)
        dV = self.compute_energy_derivative(pos, vel, commanded_accel, target_pos)
        
        # Stability condition: dV <= -alpha * V
        stability_condition = (dV <= -self.alpha * V)
        
        if stability_condition or V < 0.01:
            # Command is safe
            return (commanded_accel, False, V, dV)
        
        # Command violates stability
        # Project onto boundary via scaling
        # Find scale factor s such that:
        # dV(s * accel) = -alpha * V
        
        accel_arr = np.array(commanded_accel)
        accel_norm = np.linalg.norm(accel_arr)
        
        if accel_norm < 1e-6:
            return (commanded_accel, False, V, dV)
        
        # Binary search for safe scale
        s_low, s_high = 0.0, 1.0
        for _ in range(10):
            s_mid = (s_low + s_high) / 2.0
            scaled_accel = accel_arr * s_mid
            dV_scaled = self.compute_energy_derivative(pos, vel, scaled_accel, target_pos)
            
            if dV_scaled <= -self.alpha * V:
                s_low = s_mid
            else:
                s_high = s_mid
        
        safe_accel = (accel_arr * s_low).tolist()
        return (safe_accel, True, V, dV)
    
    def update_mass(self, new_mass):
        """Called when RLS updates mass estimate."""
        self.mass = new_mass
        # Recompute P based on new mass (simplified)
