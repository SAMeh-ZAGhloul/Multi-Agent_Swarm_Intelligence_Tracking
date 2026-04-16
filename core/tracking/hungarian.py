"""Hungarian algorithm for data association."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import linear_sum_assignment

Mat = npt.NDArray[np.float64]


def hungarian_assignment(
    cost_matrix: Mat,
    gate_threshold: float | None = None,
) -> tuple[list[int], list[int], list[int], list[int]]:
    """
    Solve the Linear Assignment Problem using the Hungarian algorithm.

    This function associates tracks to detections by minimizing the total
    assignment cost. Costs are typically Mahalanobis distances.

    Parameters
    ----------
    cost_matrix : Mat
        Cost matrix of shape (n_tracks, n_detections).
        cost_matrix[i, j] = cost of assigning track i to detection j.
    gate_threshold : float | None
        If provided, assignments with cost > threshold are rejected.
        Gated assignments go to the unmatched lists.

    Returns
    -------
    matched_tracks : list[int]
        Indices of tracks that were matched.
    matched_detections : list[int]
        Indices of detections that were matched.
    unmatched_tracks : list[int]
        Indices of tracks with no assignment.
    unmatched_detections : list[int]
        Indices of detections with no assignment.
    """
    n_tracks, n_detections = cost_matrix.shape

    # Replace NaN/Inf costs with large value
    cost_matrix = np.nan_to_num(
        cost_matrix, nan=1e6, posinf=1e6, neginf=1e6
    )

    # Solve assignment problem
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)

    matched_tracks: list[int] = []
    matched_detections: list[int] = []
    unmatched_tracks: list[int] = []
    unmatched_detections: list[int] = []

    # Track which tracks and detections were assigned
    assigned_tracks: set[int] = set()
    assigned_detections: set[int] = set()

    for track_idx, det_idx in zip(track_indices, detection_indices, strict=False):
        cost = cost_matrix[track_idx, det_idx]

        if gate_threshold is not None and cost > gate_threshold:
            # Gated out - both track and detection remain unmatched
            # Don't add to assigned sets, so they'll be caught below
            pass
        else:
            matched_tracks.append(track_idx)
            matched_detections.append(det_idx)
            assigned_tracks.add(track_idx)
            assigned_detections.add(det_idx)

    # Find unmatched tracks (not in matched_tracks)
    for i in range(n_tracks):
        if i not in assigned_tracks:
            unmatched_tracks.append(i)

    # Find unmatched detections
    for j in range(n_detections):
        if j not in assigned_detections:
            unmatched_detections.append(j)

    return matched_tracks, matched_detections, unmatched_tracks, unmatched_detections


def compute_cost_matrix(
    predicted_states: list[npt.NDArray[np.float64]],
    predicted_covariances: list[npt.NDArray[np.float64]],
    measurements: list[npt.NDArray[np.float64]],
    measurement_noise: npt.NDArray[np.float64],
) -> Mat:
    """
    Compute cost matrix using Mahalanobis distance.

    Parameters
    ----------
    predicted_states : list[Vec7]
        List of predicted state vectors from EKF.
    predicted_covariances : list[Mat7x7]
        List of predicted covariance matrices.
    measurements : list[Vec3]
        List of measurement vectors.
    measurement_noise : Mat3x3
        Measurement noise covariance matrix.

    Returns
    -------
    cost_matrix : Mat
        Cost matrix of shape (n_tracks, n_detections).
    """
    n_tracks = len(predicted_states)
    n_detections = len(measurements)

    cost_matrix = np.zeros((n_tracks, n_detections), dtype=np.float64)

    # Measurement matrix H (extracts position from state)
    H = np.zeros((3, 7), dtype=np.float64)
    H[:3, :3] = np.eye(3, dtype=np.float64)

    for i, (x_pred, P_pred) in enumerate(
        zip(predicted_states, predicted_covariances, strict=False)
    ):
        for j, z in enumerate(measurements):
            # Innovation: y = z - H x_pred
            y = z - H @ x_pred

            # Innovation covariance: S = H P H^T + R
            S = H @ P_pred @ H.T + measurement_noise

            # Mahalanobis distance: d = sqrt(y^T S^{-1} y)
            try:
                S_inv = np.linalg.inv(S)
                d2 = y.T @ S_inv @ y
                cost_matrix[i, j] = max(0.0, d2)  # Ensure non-negative
            except np.linalg.LinAlgError:
                # Singular matrix - use large cost
                cost_matrix[i, j] = 1e6

    return cost_matrix
