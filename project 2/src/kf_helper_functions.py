import numpy as np
from utils import *
from solver_utils import *

def enforce_psd(P, epsilon=1e-6):
    """Ensures that P is positive semi-definite (PSD)"""
    P = (P + P.T) / 2  # Ensure symmetry
    eigvals, eigvecs = np.linalg.eigh(P)  # Compute eigenvalues

    # Set negative eigenvalues to a small positive threshold
    eigvals[eigvals < epsilon] = epsilon
    
    # Reconstruct P using modified eigenvalues
    P_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return P_psd

def kalman_predict(prev_state, prev_cov, F, Q):
    """
    Performs the prediction step of the Kalman filter.

    Parameters:
        prev_state (numpy.ndarray): Previous state estimate (8x1).
        prev_cov (numpy.ndarray): Previous state covariance matrix (8x8).
        F (numpy.ndarray): State transition matrix (8x8).
        Q (numpy.ndarray): System noise covariance matrix (8x8).

    Returns:
        tuple: (Predicted state estimate (8x1), Predicted state covariance matrix (8x8))
    """

    # Compute predicted state estimate
    predicted_state = F @ prev_state  
    # Compute predicted state covariance
    predicted_cov = F @ prev_cov @ F.T + Q  

    # Explicitly enforce PSD to correct numerical errors
    predicted_cov = enforce_psd(predicted_cov)
    
    return predicted_state, predicted_cov

def kalman_correct(predicted_error_state, predicted_error_cov, nominal_state, raw_data, epoch):
    """
    Performs the correction step of the Kalman filter.

    Parameters:
        predicted_state (numpy.ndarray): Predicted state estimate (8x1).
        predicted_cov (numpy.ndarray): Predicted state covariance matrix (8x8).
        raw_data (dict): Dictionary containing raw satellite data.
        epoch (int): The current epoch index.

    Returns:
        tuple: (Corrected state estimate (8x1), Corrected state covariance matrix (8x8))
    """

    # Extract satellite data
    sat_positions, sat_velocities, rho, rho_std, rho_dot, _ = extract_satellite_data(raw_data, epoch)

    if sat_positions.shape[0] < 4:
        return nominal_state, predicted_error_state, predicted_error_cov, None
    
    # Compute measurement noise covariance matrix R (2M x 2M)
    rho_dot_std = 0.1 * rho_std
    R = np.diag(np.hstack(((rho_std**2).flatten(), (rho_dot_std**2).flatten())))

    # Extract receiver state components
    receiver_position = nominal_state[:3,0].reshape(1, 3)  # Reshape to (1,3)
    receiver_clock_bias = nominal_state[3,0]  # Scalar
    receiver_velocity = nominal_state[4:7,0].reshape(1, 3)  # Reshape to (1,3)
    receiver_clock_drift = nominal_state[7,0]  # Scalar

    # Compute line-of-sight unit vectors (M x 3)
    los_vectors = compute_los_unit_vectors(sat_positions, receiver_position)

    # Compute measurement sensitivity matrix H (2M x 8)
    G_u, H = compute_geometry_matrix(los_vectors)

    # Compute Dilution of Percision
    DOP = compute_dop_xyz(G_u)

    if DOP['PDOP'] > 15:  # Using DOP_z threshold
        print(f"High PDOP detected ({DOP['PDOP']}).")
        return nominal_state, predicted_error_state, predicted_error_cov, DOP

    # Compute measurement vector z (2M x 1)
    _, delta_rho = compute_pseudorange_delta(sat_positions, receiver_position, receiver_clock_bias, rho)
    _, delta_rho_dot = compute_pseudorange_rate_delta(los_vectors, sat_velocities, receiver_velocity, receiver_clock_drift, rho_dot)
    z = np.vstack((delta_rho, delta_rho_dot))  # 2M x 1

    # Compute Kalman gain: K = P_pred * H.T * inv(H*P_pred*H.T + R)
    S = (H @ predicted_error_cov @ H.T) + R  # Innovation covariance (2M x 2M)
    K = predicted_error_cov @ H.T @ np.linalg.pinv(S)  # Kalman gain (8 x 2M))

    # Compute corrected state: x_corr = x_pred + K * (z - H * x_pred)
    corrected_error_state = predicted_error_state + K @ (z - H @ predicted_error_state)

    # Compute corrected covariance: P_corr = P_pred - K * H * P_pred
    corrected_error_cov = predicted_error_cov - K @ H @ predicted_error_cov

    # Update nominal trajectory
    corrected_state = nominal_state - corrected_error_state

    # Enforce PSD on corrected covariance
    corrected_error_cov = enforce_psd(corrected_error_cov)

    return corrected_state, corrected_error_state, corrected_error_cov, DOP

def kalman_filter(raw_data, num_epochs, initial_state, initial_cov, Q):
    """
    Implements the Kalman filter for state estimation.

    Parameters:
        raw_data (dict): Dictionary containing raw satellite data.
        num_epochs (int): Number of time epochs.
        initial_state (numpy.ndarray): Initial state estimate (8x1).
        initial_cov (numpy.ndarray): Initial state covariance matrix estimate (8x8).
        Q (numpy.ndarray): System noise covariance matrix (8x8).

    Returns:
        tuple: (8 x num_epochs state estimate matrix, 8 x 8 x num_epochs state covariance matrix)
    """

    # Initialize state transition matrix
    F = np.eye(8) + np.diag(np.ones(4), k=4)

    # Preallocate state estimate storage (8 x num_epochs)
    state_estimates = np.zeros((8, num_epochs))
    error_state_estimate = np.zeros(8)
    DOP = np.zeros((4, num_epochs))

    # Preallocate state covariance storage (8 x 8 x num_epochs)
    error_state_covariances = np.zeros((8, 8, num_epochs))

    # Set initial values
    state_estimates[:, 0] = initial_state.flatten()
    error_state_covariances[:, :, 0] = initial_cov

    # Loop over each epoch
    for epoch in range(1, num_epochs):
        # Prediction step 
        predicted_error_state, predicted_error_cov = kalman_predict(error_state_estimate.reshape(-1,1), 
                                                        error_state_covariances[:, :, epoch - 1], 
                                                        F, 
                                                        Q)

        # Correction step 
        corrected_state, corrected_error_state, corrected_error_cov, DOP_info = kalman_correct(predicted_error_state, predicted_error_cov,
                                                                                      state_estimates[:, epoch - 1].reshape(-1,1),
                                                                                        raw_data, epoch)

        # Store results
        error_state_estimate = corrected_error_state.copy()
        state_estimates[:, epoch] = corrected_state.flatten()
        error_state_covariances[:, :, epoch] = corrected_error_cov

        if DOP_info is None:
            DOP[:, epoch] = -1  # Store -1 for all DOP values
        else:
            DOP[0,epoch] = DOP_info['DOP_x']
            DOP[1,epoch] = DOP_info['DOP_y']
            DOP[2,epoch] = DOP_info['DOP_z']
            DOP[3,epoch] = DOP_info['PDOP']

    return state_estimates, error_state_covariances, DOP
