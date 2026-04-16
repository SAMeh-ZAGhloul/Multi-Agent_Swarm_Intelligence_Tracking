"""Extended Kalman Filter with CTRA (Constant Turn Rate and Acceleration) model."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from ..constants import (
    MEASUREMENT_DIM,
    MEASUREMENT_NOISE_POS,
    PROCESS_NOISE_OMEGA,
    PROCESS_NOISE_POS,
    PROCESS_NOISE_VEL,
    STATE_DIM,
)

Vec7 = npt.NDArray[np.float64]
Vec3 = npt.NDArray[np.float64]
Mat7x7 = npt.NDArray[np.float64]
Mat3x3 = npt.NDArray[np.float64]
Mat3x7 = npt.NDArray[np.float64]


class ExtendedKalmanFilter:
    """
    EKF for 3D drone tracking with CTRA motion model.

    State vector (7,):
        x = [px, py, pz, vx, vy, vz, omega]
            pos(m), pos(m), pos(m), vel(m/s) x3, turn_rate(rad/s)

    Measurement vector (3,):
        z = [px, py, pz]

    Attributes
    ----------
    x : Vec7
        Current state estimate
    P : Mat7x7
        Current covariance estimate
    Q : Mat7x7
        Process noise covariance
    R : Mat3x3
        Measurement noise covariance
    """

    def __init__(
        self,
        x0: Vec7 | None = None,
        P0: Mat7x7 | None = None,
    ) -> None:
        """
        Initialize the EKF.

        Parameters
        ----------
        x0 : Vec7 | None
            Initial state estimate. If None, zeros are used.
        P0 : Mat7x7 | None
            Initial covariance. If None, identity is used.
        """
        self.x = x0 if x0 is not None else np.zeros(STATE_DIM, dtype=np.float64)
        self.P = P0 if P0 is not None else np.eye(STATE_DIM, dtype=np.float64)

        # Process noise covariance
        self.Q = self._build_process_noise()

        # Measurement noise covariance
        self.R = MEASUREMENT_NOISE_POS**2 * np.eye(
            MEASUREMENT_DIM, dtype=np.float64
        )

    def _build_process_noise(self) -> Mat7x7:
        """Build process noise covariance matrix Q."""
        q_pos = PROCESS_NOISE_POS**2
        q_vel = PROCESS_NOISE_VEL**2
        q_omega = PROCESS_NOISE_OMEGA**2

        Q = np.diag(
            np.array([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel, q_omega], dtype=np.float64)
        )
        return Q

    def _state_transition(
        self, x: Vec7, dt: float
    ) -> Vec7:
        """
        Compute state transition function f(x, dt).

        CTRA model:
        - Position integrates velocity
        - Velocity rotates by omega * dt
        - Omega is constant
        """
        px, py, pz = x[0], x[1], x[2]
        vx, vy, vz = x[3], x[4], x[5]
        omega = x[6]

        if abs(omega) < 1e-6:
            # Straight motion approximation
            px_new = px + vx * dt
            py_new = py + vy * dt
            pz_new = pz + vz * dt
            vx_new, vy_new, vz_new = vx, vy, vz
        else:
            # Constant turn rate
            theta = omega * dt
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Rotated velocity
            vx_new = vx * cos_t - vy * sin_t
            vy_new = vx * sin_t + vy * cos_t
            vz_new = vz

            # Position update (integrate rotated velocity)
            if abs(omega) > 1e-10:
                px_new = px + (vx * sin_t + vy * (1 - cos_t)) / omega
                py_new = py + (vy * sin_t - vx * (1 - cos_t)) / omega
            else:
                px_new = px + vx * dt
                py_new = py + vy * dt
            pz_new = pz + vz * dt

        x_new = np.array(
            [px_new, py_new, pz_new, vx_new, vy_new, vz_new, omega],
            dtype=np.float64,
        )
        return x_new

    def _state_jacobian(self, x: Vec7, dt: float) -> Mat7x7:
        """
        Compute Jacobian F = df/dx of state transition.

        This is the linearization of the CTRA model around the current state.
        """
        vx, vy = x[3], x[4]
        omega = x[6]

        F = np.eye(STATE_DIM, dtype=np.float64)

        if abs(omega) < 1e-6:
            # Straight motion - simple Jacobian
            F[0, 3] = dt  # dpx/dvx
            F[1, 4] = dt  # dpy/dvy
            F[2, 5] = dt  # dpz/dvz
        else:
            theta = omega * dt
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)

            # Position derivatives w.r.t. velocity
            F[0, 3] = sin_t / omega  # dpx/dvx
            F[0, 4] = (1 - cos_t) / omega  # dpx/dvy
            F[1, 3] = -(1 - cos_t) / omega  # dpy/dvx
            F[1, 4] = sin_t / omega  # dpy/dvy
            F[2, 5] = dt  # dpz/dvz

            # Velocity derivatives w.r.t. omega and velocity
            F[3, 4] = -sin_t  # dvx/dvy (via omega effect)
            F[3, 6] = -vx * sin_t - vy * (1 - cos_t)  # dvx/domega
            F[4, 3] = sin_t  # dvy/dvx
            F[4, 6] = vx * (1 - cos_t) - vy * sin_t  # dvy/domega

        return F

    def _measurement_function(self, x: Vec7) -> Vec3:
        """
        Compute measurement function h(x).

        For direct position measurement: h(x) = [px, py, pz]
        """
        return x[:MEASUREMENT_DIM].copy()

    def _measurement_jacobian(self, x: Vec7) -> Mat3x7:
        """
        Compute Jacobian H = dh/dx of measurement function.

        For direct position measurement: H = [I_3 | 0]
        """
        H = np.zeros((MEASUREMENT_DIM, STATE_DIM), dtype=np.float64)
        H[:MEASUREMENT_DIM, :MEASUREMENT_DIM] = np.eye(
            MEASUREMENT_DIM, dtype=np.float64
        )
        return H

    def predict(self, dt: float) -> tuple[Vec7, Mat7x7]:
        """
        Prediction step.

        Parameters
        ----------
        dt : float
            Time step in seconds.

        Returns
        -------
        x : Vec7
            Predicted state vector.
        P : Mat7x7
            Predicted covariance matrix.
        """
        # State prediction
        self.x = self._state_transition(self.x, dt)

        # Jacobian
        F = self._state_jacobian(self.x, dt)

        # Covariance prediction: P = F P F^T + Q
        self.P = F @ self.P @ F.T + self.Q

        return self.x, self.P

    def update(
        self, z: Vec3
    ) -> tuple[Vec7, Mat7x7, float]:
        """
        Update step with a single measurement.

        Parameters
        ----------
        z : Vec3
            Measurement vector [px, py, pz].

        Returns
        -------
        x : Vec7
            Updated state vector.
        P : Mat7x7
            Updated covariance matrix.
        innovation_norm : float
            Norm of the innovation (for gating).
        """
        # Predicted measurement
        z_pred = self._measurement_function(self.x)

        # Innovation (measurement residual)
        y = z - z_pred

        # Jacobian
        H = self._measurement_jacobian(self.x)

        # Innovation covariance: S = H P H^T + R
        S = H @ self.P @ H.T + self.R

        # Kalman gain: K = P H^T S^{-1}
        # Use solve for numerical stability instead of inv
        K = np.linalg.solve(S.T, (self.P @ H.T).T).T

        # State update: x = x + K y
        self.x = self.x + K @ y

        # Covariance update: P = (I - K H) P
        I = np.eye(STATE_DIM, dtype=np.float64)
        self.P = (I - K @ H) @ self.P

        # Compute innovation norm for gating
        innovation_norm = float(np.sqrt(y.T @ np.linalg.solve(S, y)))

        return self.x, self.P, innovation_norm

    def copy(self) -> ExtendedKalmanFilter:
        """Create a deep copy of this filter."""
        ekf = ExtendedKalmanFilter()
        ekf.x = self.x.copy()
        ekf.P = self.P.copy()
        ekf.Q = self.Q.copy()
        ekf.R = self.R.copy()
        return ekf
