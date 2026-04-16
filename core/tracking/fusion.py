"""Multi-sensor fusion using Covariance Intersection."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Vec = npt.NDArray[np.float64]
Mat = npt.NDArray[np.float64]


def covariance_intersection(
    x1: Vec, P1: Mat, x2: Vec, P2: Mat, omega: float = 0.5
) -> tuple[Vec, Mat]:
    """
    Fuse two state estimates using Covariance Intersection (CI).

    CI is used when the correlation between estimates is unknown,
    guaranteeing consistency (no double-counting of information).

    The fusion weight omega ∈ [0, 1] determines the relative trust:
    - omega = 0: trust estimate 1 completely
    - omega = 1: trust estimate 2 completely
    - omega = 0.5: equal trust (default)

    Optimal omega minimizes the trace or determinant of the fused covariance.

    Parameters
    ----------
    x1 : Vec
        State estimate from sensor 1.
    P1 : Mat
        Covariance of estimate 1.
    x2 : Vec
        State estimate from sensor 2.
    P2 : Mat
        Covariance of estimate 2.
    omega : float
        Fusion weight in [0, 1]. Default 0.5.

    Returns
    -------
    x_fused : Vec
        Fused state estimate.
    P_fused : Mat
        Fused covariance matrix.

    Notes
    -----
    CI update equations:

        P_fused^{-1} = omega * P1^{-1} + (1 - omega) * P2^{-1}

        x_fused = P_fused @ (omega * P1^{-1} @ x1 + (1 - omega) * P2^{-1} @ x2)

    References
    ----------
    Julier, S. J., & Uhlmann, J. K. (1997). "Non-divergent estimation
    algorithm in the presence of unknown correlations."
    """
    # Clamp omega to valid range
    omega = np.clip(omega, 0.0, 1.0)

    # Compute inverse covariances (information matrices)
    # Use solve for numerical stability
    try:
        P1_inv = np.linalg.inv(P1)
        P2_inv = np.linalg.inv(P2)
    except np.linalg.LinAlgError:
        # If either covariance is singular, fall back to the non-singular one
        if np.linalg.matrix_rank(P1) > np.linalg.matrix_rank(P2):
            return x1.copy(), P1.copy()
        return x2.copy(), P2.copy()

    # Fused information matrix
    P_fused_inv = omega * P1_inv + (1 - omega) * P2_inv

    # Fused covariance
    try:
        P_fused = np.linalg.inv(P_fused_inv)
    except np.linalg.LinAlgError:
        # Regularize if singular
        P_fused_inv += 1e-6 * np.eye(P1.shape[0])
        P_fused = np.linalg.inv(P_fused_inv)

    # Fused state
    x_fused = P_fused @ (omega * P1_inv @ x1 + (1 - omega) * P2_inv @ x2)

    return x_fused, P_fused


def find_optimal_omega(
    P1: Mat, P2: Mat, criterion: str = "trace"
) -> float:
    """
    Find the optimal fusion weight that minimizes fused covariance.

    Parameters
    ----------
    P1 : Mat
        Covariance of estimate 1.
    P2 : Mat
        Covariance of estimate 2.
    criterion : str
        Optimization criterion: "trace" or "det".

    Returns
    -------
    omega_opt : float
        Optimal fusion weight.
    """
    # Try omega values and find minimum
    omegas = np.linspace(0.0, 1.0, 21)
    best_omega = 0.5
    best_value = np.inf

    for omega in omegas:
        try:
            P_fused_inv = omega * np.linalg.inv(P1) + (1 - omega) * np.linalg.inv(
                P2
            )
            P_fused = np.linalg.inv(P_fused_inv)

            if criterion == "trace":
                value = np.trace(P_fused)
            else:  # det
                value = np.linalg.det(P_fused)

            if value < best_value:
                best_value = value
                best_omega = omega

        except np.linalg.LinAlgError:
            continue

    return best_omega


def multi_sensor_fusion(
    estimates: list[tuple[Vec, Mat]],
    weights: list[float] | None = None,
) -> tuple[Vec, Mat]:
    """
    Fuse multiple sensor estimates using generalized CI.

    Parameters
    ----------
    estimates : list[tuple[Vec, Mat]]
        List of (state, covariance) pairs from each sensor.
    weights : list[float] | None
        Relative weights for each estimate. If None, equal weights.

    Returns
    -------
    x_fused : Vec
        Fused state estimate.
    P_fused : Mat
        Fused covariance matrix.
    """
    n = len(estimates)
    if n == 0:
        raise ValueError("At least one estimate required")
    if n == 1:
        x, P = estimates[0]
        return x.copy(), P.copy()

    # Default to equal weights
    if weights is None:
        weights = [1.0 / n] * n

    # Normalize weights
    total_weight = sum(weights)
    weights = [w / total_weight for w in weights]

    # Compute inverse covariances
    inv_covs = []
    for _, P in estimates:
        try:
            inv_covs.append(np.linalg.inv(P))
        except np.linalg.LinAlgError:
            # Skip singular matrices
            inv_covs.append(np.zeros_like(P))

    # Fused information matrix
    P_fused_inv = np.zeros_like(estimates[0][1])
    for w, P_inv in zip(weights, inv_covs, strict=False):
        P_fused_inv += w * P_inv

    # Fused covariance
    try:
        P_fused = np.linalg.inv(P_fused_inv)
    except np.linalg.LinAlgError:
        P_fused_inv += 1e-6 * np.eye(P_fused_inv.shape[0])
        P_fused = np.linalg.inv(P_fused_inv)

    # Fused state
    x_fused = np.zeros_like(estimates[0][0])
    for (x, _), w, P_inv in zip(estimates, weights, inv_covs, strict=False):
        x_fused += w * P_inv @ x
    x_fused = P_fused @ x_fused

    return x_fused, P_fused
