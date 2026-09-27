import numpy as np
from utils import *
from solver_utils import *

def compute_weight_matrix(elevation_angles, rho_dot_var):
    """
    Compute the 2M x 2M weight matrix for weighted least squares based on satellite elevation angles 
    and Doppler measurement variances.

    Parameters:
        elevation_angles (numpy.ndarray): Mx1 array containing the elevation angles (in degrees) of M visible satellites.
        rho_dot_var (numpy.ndarray): Mx1 array containing the variance of Doppler measurements.

    Returns:
        numpy.ndarray: 2M x 2M diagonal weight matrix.
    """
    M = len(elevation_angles)  # Number of satellites

    # Convert elevation angles from degrees to radians
    elevation_radians = np.radians(elevation_angles)

    # Compute sin²(elevation) for position weighting
    W_p = np.diag(np.sin(elevation_radians).flatten() ** 2)

    # Compute velocity weighting as the inverse of Doppler variance
    #W_v = np.diag(1 / rho_dot_var.flatten())
    W_v = W_p

    # Construct the 2M x 2M block diagonal weight matrix
    W = np.block([
        [W_p, np.zeros((M, M))],
        [np.zeros((M, M)), W_v]
    ])

    return W

def gauss_newton_svd_weighted(G, residuals, W, tol=5e-2):
    """Solve Gauss-Newton step using SVD with weights."""
    # Compute weighted Jacobian and residuals
    W_sqrt = np.sqrt(W)  # Square root of diagonal weight matrix
    G_weighted = W_sqrt @ G
    residuals_weighted = W_sqrt @ residuals

    # Compute SVD of weighted Jacobian
    U, S, Vt = np.linalg.svd(G_weighted, full_matrices=False)

    # Compute pseudo-inverse with truncation
    S_inv = np.zeros_like(S)
    S_inv[S > tol] = 1 / S[S > tol]
    Sigma_plus = np.diag(S_inv)

    # Compute update step using efficient element-wise multiplication
    delta_x = Vt.T @ Sigma_plus @ U.T @ residuals_weighted

    return delta_x

def gauss_newton_qr_weighted(G, residuals, W):
    """Solve Gauss-Newton step using QR decomposition with weights."""
    # Compute weighted Jacobian and residuals
    W_sqrt = np.sqrt(W)  # Square root of diagonal weight matrix
    G_weighted = W_sqrt @ G
    residuals_weighted = W_sqrt @ residuals

    # Perform QR decomposition on the weighted Jacobian
    Q, R = np.linalg.qr(G_weighted)

    # Solve the system R @ delta_x = Q.T @ residuals_weighted
    delta_x = np.linalg.solve(R, Q.T @ residuals_weighted)

    return delta_x

def nonlinear_least_squares(epoch, raw_data, initial_estimate, max_iterations=10, tol=1e-6):
    """
    Perform Nonlinear Least Squares (NLS) to estimate the receiver's position and velocity.

    Parameters:
        raw_data (dict): Dictionary containing satellite position, velocity, pseudorange, and Doppler measurements.
        initial_estimate (numpy.ndarray): Initial guess for the receiver state [x, y, z, b_r, v_x, v_y, v_z, d_r] (8x1).
        max_iterations (int): Maximum number of iterations for the NLS solver.
        tol (float): Convergence tolerance for stopping criterion.

    Returns:
        numpy.ndarray: Final estimated receiver state [x, y, z, b_r, v_x, v_y, v_z, d_r] (8x1).
    """
    estimate = initial_estimate.copy()

    # Extract satellite data for the current epoch
    sat_positions, sat_velocities, rho, rho_std, rho_dot, el_sat = extract_satellite_data(raw_data, epoch)

    M = sat_positions.shape[0]
    if M < 4:
        print(f"Insufficient satellites ({sat_positions.shape[0]} visible). Exiting NLS early.")
        return None, None, None

    for iteration in range(max_iterations):

        # Extract current estimates
        receiver_position = estimate[:3].reshape(1, 3)  # Reshape to (1,3)
        receiver_clock_bias = estimate[3]  # Scalar
        receiver_velocity = estimate[4:7].reshape(1, 3)  # Reshape to (1,3)
        receiver_clock_drift = estimate[7]  # Scalar

        # Compute LoS unit vectors (M x 3)
        los_vectors = compute_los_unit_vectors(sat_positions, receiver_position)

        # Compute pseudorange and rate-of-change residuals
        _, delta_rho = compute_pseudorange_delta(sat_positions, receiver_position, receiver_clock_bias, rho)
        _, delta_rho_dot = compute_pseudorange_rate_delta(los_vectors, sat_velocities, receiver_velocity, receiver_clock_drift, rho_dot)

        # Form residuals vector (2M x 1)
        residuals = np.vstack((delta_rho, delta_rho_dot))

        # Compute Jacobian matrix (Geometry matrix) (2M x 8)
        G_u, G = compute_geometry_matrix(los_vectors)

        # Compute Dilution of Percision
        DOP = compute_dop_xyz(G_u)

        # Compute (G^T G), Compute and log its condition number
        GTG = G.T @ G
        cond_num = np.linalg.cond(GTG)
        print(f"Condition number of (G^T G): {cond_num:.2e}")
        print("DOP: ", DOP['PDOP'])
        if cond_num > 1e12:
            print("Warning: High condition number detected! Numerical stability issues may occur.")

        # if DOP['PDOP'] > 15:  # Using DOP_z threshold
        #     print(f"High PDOP detected ({DOP['PDOP']}). Exiting NLS early.")
        #     return None, DOP, cond_num

        # Calculate Weight Matrix 
        rho_dot_std = 0.1 * rho_std
        rho_dot_var = rho_dot_std**2
        W = compute_weight_matrix(el_sat, rho_dot_var)

        # Solve the system using the Moore-Penrose Pseudoinverse
        #delta_x = np.linalg.pinv(GTG) @ G.T @ residuals  # Normal Equations solution
        #delta_x = np.linalg.pinv(G.T @ W @ G) @ (G.T @ W @ residuals)  # Weighted Least Squares Solution
        delta_x = gauss_newton_svd_weighted(G, residuals, W) # Weighted SVD Solution
        #delta_x = gauss_newton_qr_weighted(G, residuals, W)

        # Update the estimate
        estimate += delta_x.flatten()

        # Check for convergence
        update_norm = np.linalg.norm(delta_x)
        if update_norm < tol:
            break
    
    # Print how many iterations it took (whether it converged early or ran max_iterations)
    print(f"norm of the update vector {update_norm}")
    print(f"Epoch solved in {iteration + 1} iterations.")

    return estimate, DOP, cond_num

def solve_trajectory_nls(raw_data, num_epochs, max_iterations, tol):
    """
    Solve the Nonlinear Least Squares (NLS) problem for all epochs in the trajectory.

    Parameters:
        raw_data (dict): Dictionary containing satellite position, velocity, pseudorange, and Doppler measurements.
        num_epochs (int): Number of epochs to process (default: 2503).
        max_iterations (int): Maximum number of iterations for each epoch.
        tol (float): Convergence tolerance.

    Returns:
        numpy.ndarray: 8xNum_Epochs matrix of estimated receiver states.
    """
    # Initialize the estimate array (8 x num_epochs)
    estimates = np.zeros((8, num_epochs))
    DOP = np.zeros((4,num_epochs))
    COND = np.zeros((1,num_epochs))

    # Set initial estimate as an 8x1 zero vector
    initial_estimate = np.zeros(8)

    for epoch in range(num_epochs):
        print(f"Processing epoch {epoch+1}/{num_epochs}...")

        # Run NLS for the current epoch
        estimate, DOP_info, cond_num = nonlinear_least_squares(epoch, raw_data, initial_estimate, max_iterations, tol)

        # Handle cases where NLS failed (e.g., insufficient satellites)
        if estimate is None:
            print(f"Skipping epoch {epoch+1} due to insufficient satellites.")
            if epoch > 0:
                estimates[:, epoch] = estimates[:, epoch - 1]  # Propagate last known estimate
                
                DOP[0, epoch] = DOP_info['DOP_x'] if DOP_info is not None else DOP[0, epoch - 1]
                DOP[1, epoch] = DOP_info['DOP_y'] if DOP_info is not None else DOP[1, epoch - 1]
                DOP[2, epoch] = DOP_info['DOP_z'] if DOP_info is not None else DOP[2, epoch - 1]
                DOP[3, epoch] = DOP_info['PDOP'] if DOP_info is not None else DOP[3, epoch - 1]
                
                COND[0, epoch] = cond_num if cond_num is not None else COND[0, epoch - 1]

            continue

        # Store the estimate for this epoch
        estimates[:, epoch] = estimate

        DOP[0,epoch] = DOP_info['DOP_x']
        DOP[1,epoch] = DOP_info['DOP_y']
        DOP[2,epoch] = DOP_info['DOP_z']
        DOP[3,epoch] = DOP_info['PDOP']

        COND[0,epoch] = cond_num

        # Use the current estimate as the initial estimate for the next epoch
        initial_estimate = estimate.copy()

    return estimates, DOP, COND
