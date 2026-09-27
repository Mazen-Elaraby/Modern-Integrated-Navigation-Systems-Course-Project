import numpy as np
from kf_helper_funcs import *

def kalman_predict(F, x_prev, P_prev, Q, G):
    """
    Perform the prediction step of the Kalman filter with noise coupling.

    Parameters:
    - F: State transition matrix (Phi) of shape (n, n)
    - x_prev: Previous error state estimate (x_k-1^+) of shape (n, 1)
    - P_prev: Previous error covariance matrix (P_k-1^+) of shape (n, n)
    - Q: Process noise covariance matrix of shape (m, m)
    - G: Noise coupling matrix (G_k-1), mapping process noise into the state space, shape (n, m)

    Returns:
    - x_pred: Predicted error state estimate (x_k^-)
    - P_pred: Predicted error covariance matrix (P_k^-)
    """
    # Predicted error state estimate: x_k^- = F * x_k-1^+
    x_pred = F @ x_prev

    # Predicted covariance matrix: P_k^- = F * P_k-1^+ * F^T + G * Q * G^T
    P_pred = F @ P_prev @ F.T + G @ Q @ G.T

    # Explicitly enforce PSD to correct numerical errors
    #P_pred = enforce_psd(P_pred)

    P_pred[6,:] = 0
    P_pred[7,:] = 0
    P_pred[8,:] = 0

    return x_pred, P_pred

def kalman_correction_step(x_nominal, x_error_pred, P_pred, H, z, R):
    """
    Perform the correction step of the Kalman filter.

    Parameters:
    - x_nominal: Nominal state vector (n, 1)
    - x_error_pred: Predicted error state estimate (x_k^-), shape (n, 1)
    - P_pred: Predicted error covariance matrix (P_k^-), shape (n, n)
    - H: Observation matrix (H_k), shape (m, n)
    - z: Measurement vector (z_k), shape (m, 1)
    - R: Measurement noise covariance matrix (R_k), shape (m, m)

    Returns:
    - x_error_corr: Corrected error state estimate (x_k^+), shape (n, 1)
    - x_nominal_updated: Updated nominal state (x_nominal + x_error_corr), shape (n, 1)
    - P_corr: Updated error state covariance matrix (P_k^+), shape (n, n)
    """
    # Compute Kalman gain: K_k = P_k^- H^T (H P_k^- H^T + R)^-1
    S = H @ P_pred @ H.T + R  # Innovation covariance
    K = P_pred @ H.T @ np.linalg.inv(S)  # Kalman gain
    
    # Compute measurement residual: z_k - H_k x_k^-
    residual = z - H @ x_error_pred
    
    # Corrected error state estimate: x_k^+ = x_k^- + K_k (z_k - H_k x_k^-)
    x_error_corr = x_error_pred + K @ residual

    # Corrected covariance: 
    P_corr = P_pred - K @ H @ P_pred
    P_corr[6,:] *= 0.99
    P_corr[7,:] *= 0.99
    P_corr[8,:] *= 0.99
    
    # Update nominal state: x_nominal_updated = x_nominal + x_error_corr
    x_nominal_updated = x_nominal + x_error_corr

    return x_error_corr, x_nominal_updated, P_corr

def kalman_filter(IMU_data, gps_data, odom_speed, num_epochs, init_state, init_err_cov, H, G, Q, a_od_corr_time, bwz_corr_time, outage_mask):
    
    # Preallocate storage 
    state_est = np.zeros((9, num_epochs))
    ins_ests = np.zeros((9, num_epochs))
    err_state_est = np.zeros(9)
    err_state_cov = np.zeros((9, 9, num_epochs))

    # Set initial values
    state_est[:, 0] = init_state.flatten()
    err_state_cov[:, :, 0] = init_err_cov

    # Define some constants
    a = 6378137 # Semi-major axis (meters)
    e = 0.08181919 # Eccentricity
    omega_e = 7.292115e-5 # Earth’s rotation rate (rad/s)

    p_prev = 0 # previous pitch estimate

    for epoch in range(1, num_epochs):

        # compute the normal radius (R_n) & the meridian radius of curvature (R_m)
        R_n = compute_RN(a, e, state_est[0, epoch - 1])
        R_m = compute_RM(a, e, state_est[0, epoch - 1])

        # perform 3D RISS Mechanization
        ins_est = riss_mechanization_3d(epoch, state_est[:, epoch - 1], IMU_data, odom_speed, R_m, R_n, omega_e)
        
        # Prediction step
        a_od = odom_speed[epoch - 1] - odom_speed[epoch - 2]
        F = compute_state_transition_matrix(R_n, R_m, state_est[0, epoch - 1], state_est[2, epoch - 1], 1,
                                            a_od, state_est[6, epoch - 1], p_prev,
                                              a_od_corr_time, bwz_corr_time) # compute State Transition Matrix
        
        p_prev = ins_est[7] # store pitch value for use in next epoch

        pred_err_state, pred_err_cov = kalman_predict(F, err_state_est.reshape(-1,1), err_state_cov[:,:, epoch - 1], Q, G)

        # correction step
        z = compute_measurement_residual(ins_est, gps_data, epoch)
        R = construct_R_matrix(gps_data, epoch, a, e) # measurment covariance matrix
        
        corr_err_state, corr_state, corr_err_cov = kalman_correction_step(state_est[:, epoch - 1].reshape(-1,1),
                                                               pred_err_state, pred_err_cov, H, z, R)

        # store results
        if (outage_mask[epoch]==1):
            err_state_est = pred_err_state.flatten()
            corr_state[6,0] = ins_est[6]
            state_est[:, epoch] = ins_est
            err_state_cov[:, :, epoch] = pred_err_cov
            ins_ests[:, epoch] = ins_est
        else:
            err_state_est = corr_err_state.flatten()
            corr_state[6,0] = ins_est[6]
            state_est[:, epoch] = corr_state.flatten()
            err_state_cov[:, :, epoch] = corr_err_cov
            ins_ests[:, epoch] = ins_est

    return state_est, err_state_cov, ins_ests
    