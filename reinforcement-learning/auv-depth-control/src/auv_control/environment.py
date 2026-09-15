"""Simplified deterministic AUV dynamics used by the original experiment."""

import numpy as np


class AUVConstantDepthEnv:
    """Simplified deterministic vertical-plane AUV simulation.

    This is a lightweight project-specific environment, not a Gym environment.
    The five-element state is ``[z - z_r, cos(theta), sin(theta), w, q]`` and
    the two-element action is ``[tau_1, tau_2]``. Its equations and mixed,
    semi-implicit update order intentionally match the saved implementation.

    ``step`` always returns ``done=False``: experiment episodes are fixed-horizon
    truncations of a continuing depth-regulation task.
    """

    def __init__(
        self,
        dt=0.1,
        u0=1.0,
        rho=(1.2, 1.0, 0.1, 0.1),
        R=(0.1, 0.1),
        max_tau=100.0,
        zr=8,
        z0=2,
    ):
        # Inertial and hydrodynamic coefficients retained from the original
        # simplified simulation; they do not constitute a full REMUS model.
        self.m = 30.51
        self.Iyy = 3.45
        self.Zq_dot = -1.93
        self.Zw_dot = -35.5
        self.Mq_dot = -4.88
        self.Mw_dot = -1.93
        self.Zuq = -28.6
        self.Zww = -131.0
        self.Zqq = -0.632
        self.Muq = -2.0
        self.Muw = 24.0
        self.Mww = 3.18
        self.Mqq = -188.0

        self.dt = dt
        self.u0 = u0
        self.rho1, self.rho2, self.rho3, self.rho4 = rho
        self.R = np.diag(R)
        self.max_tau = max_tau
        self.zr = zr
        self.z0 = z0

    def reset(self, zr=None, z0=None):
        """Reset to the stored depths, optionally replacing those defaults."""
        if zr is not None:
            self.zr = zr
        if z0 is not None:
            self.z0 = z0
        self.z = self.z0
        self.w = 0.0
        self.theta = 0.0
        self.q = 0.0
        return self._get_state()

    def _get_state(self):
        dz = self.z - self.zr
        return np.array(
            [dz, np.cos(self.theta), np.sin(self.theta), self.w, self.q],
            dtype=np.float32,
        )

    def step(self, action):
        """Clip and apply one action, then return the post-transition reward.

        Velocity and pitch-rate updates precede the position and pitch updates,
        so this is not an ordinary all-states-at-once forward Euler step.
        """
        tau = np.clip(action, -self.max_tau, self.max_tau)
        tau1, tau2 = tau

        mass_matrix = np.array(
            [
                [self.m - self.Zw_dot, -self.Zq_dot],
                [-self.Mw_dot, self.Iyy - self.Mq_dot],
            ]
        )
        forces = np.array(
            [
                self.m * self.u0 * self.q
                + self.Zuq * self.u0 * self.q
                + self.Zww * abs(self.w) * self.w
                + self.Zqq * abs(self.q) * self.q
                + tau1,
                self.Muq * self.u0 * self.q
                + self.Muw * self.u0 * self.w
                + self.Mww * abs(self.w) * self.w
                + self.Mqq * abs(self.q) * self.q
                + tau2,
            ]
        )

        w_dot, q_dot = np.linalg.solve(mass_matrix, forces)
        self.w += w_dot * self.dt
        self.q += q_dot * self.dt
        self.z += (
            self.w * np.cos(self.theta) - self.u0 * np.sin(self.theta)
        ) * self.dt
        self.theta += self.q * self.dt

        dz = self.z - self.zr
        cost = (
            self.rho1 * dz**2
            + self.rho2 * self.theta**2
            + self.rho3 * self.w**2
            + self.rho4 * self.q**2
            + tau.dot(self.R.dot(tau))
        )
        reward = -cost
        return self._get_state(), reward, False, {}
