"""Integral-augmented LQR (LQI) controller from the original experiment.

The controller uses its own linear continuous-time plant rather than the RL
environment's plant equations, so its trajectory is illustrative rather than
a strict same-plant benchmark against DDPG or TD3.
"""

import numpy as np
from scipy import linalg
from scipy.integrate import odeint


A = np.array(
    [
        [-1.0421, 0.7856, 0.0, 0.0207],
        [6.0038, -0.6624, 0.0, -0.7083],
        [1.0, 0.0, 0.0, -2.0],
        [0.0, 1.0, 0.0, 0.0],
    ]
)
B = np.array(
    [
        [0.0153, 0.0035],
        [-0.0035, 0.1209],
        [0.0, 0.0],
        [0.0, 0.0],
    ]
)
C = np.array(
    [
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

A_AUG = np.block(
    [
        [A, np.zeros((4, 2))],
        [-C, np.zeros((2, 2))],
    ]
)
B_AUG = np.vstack([B, np.zeros((2, 2))])

RHO1, RHO2, RHO3, RHO4 = 1.0, 1.0, 0.1, 0.1
Q = np.diag([RHO3, RHO4, 0.001, 0.5 * RHO2, RHO1, 0.5 * RHO2])
R = 0.05 * np.eye(2)


def lqr(system_matrix, input_matrix, state_cost, input_cost):
    """Solve the continuous-time algebraic Riccati equation."""
    solution = linalg.solve_continuous_are(
        system_matrix, input_matrix, state_cost, input_cost
    )
    gain = np.linalg.inv(input_cost).dot(input_matrix.T.dot(solution))
    eigenvalues, _ = linalg.eig(system_matrix - input_matrix.dot(gain))
    return gain, solution, eigenvalues


K_FULL, RICCATI_SOLUTION, CLOSED_LOOP_EIGENVALUES = lqr(A_AUG, B_AUG, Q, R)
KX = -K_FULL[:, :4]
KE = -K_FULL[:, 4:]

A_CLOSED_LOOP = np.block(
    [
        [A + B.dot(KX), B.dot(KE)],
        [-C, np.zeros((2, 2))],
    ]
)
B_CLOSED_LOOP = np.block(
    [
        [np.zeros((4, 2))],
        [np.eye(2)],
    ]
)


def dynamics(state, _time, reference):
    """Closed-loop LQI dynamics in the original linear plant model."""
    return A_CLOSED_LOOP.dot(state) + B_CLOSED_LOOP.dot(reference)


def simulate_lqi(
    start_depth=2.0,
    target_depth=8.0,
    duration=100.0,
    num_points=1000,
):
    """Simulate the separate linear plant with continuous-time integration."""
    time = np.linspace(0.0, duration, num_points)
    initial_state = np.array([0.0, 0.0, start_depth, 0.0, 0.0, 0.0])
    reference = np.array([target_depth, 0.0])
    solution = odeint(dynamics, initial_state, time, args=(reference,))

    physical_states = solution[:, :4].T
    integral_errors = solution[:, 4:].T
    controls = (KX.dot(physical_states) + KE.dot(integral_errors)).T

    return {
        "time": time,
        "solution": solution,
        "depths": solution[:, 2],
        "thetas": solution[:, 3],
        "ws": solution[:, 0],
        "qs": solution[:, 1],
        "actions": controls,
    }
