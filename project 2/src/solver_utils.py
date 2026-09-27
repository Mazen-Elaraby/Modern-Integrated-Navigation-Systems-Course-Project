import numpy as np

def compute_los_unit_vectors(sat_positions, receiver_position):
    """
    Compute the Line-of-Sight (LoS) unit vectors between the receiver and satellites.

    Parameters:
        sat_positions (numpy.ndarray): Mx3 matrix of satellite positions (each row is a satellite [x, y, z]).
        receiver_position (numpy.ndarray): 1x3 vector of estimated receiver position [x_EST, y_EST, z_EST].

    Returns:
        numpy.ndarray: Mx3 matrix of LoS unit vectors (each row is a unit vector [lx, ly, lz]).
    """
    # Compute the difference between satellite positions and receiver position (M x 3)
    diff = receiver_position - sat_positions  # Broadcasting: (1,3) - (M,3)

    # Compute Euclidean norm (M x 1), avoiding division by zero
    norm = np.linalg.norm(diff, axis=1, keepdims=True)  # M x 1 column vector

    # Compute unit vectors (M x 3)
    los_unit_vectors = diff / norm  # Element-wise division (broadcasting)

    return los_unit_vectors

def compute_pseudorange_delta(sat_positions, receiver_position, receiver_clock_bias, measured_rho):
    """
    Compute the estimated pseudorange and delta pseudorange for M satellites.

    Parameters:
        sat_positions (numpy.ndarray): Mx3 matrix of satellite positions (each row is [x, y, z]).
        receiver_position (numpy.ndarray): 1x3 vector of estimated receiver position [x_EST, y_EST, z_EST].
        receiver_clock_bias (float): Estimated receiver clock bias (b_r,EST).
        measured_rho (numpy.ndarray): Mx1 matrix of measured pseudorange values.

    Returns:
        tuple:
            estimated_rho (numpy.ndarray): Mx1 matrix of estimated pseudorange values.
            delta_rho (numpy.ndarray): Mx1 matrix of pseudorange residuals.
    """
    # Compute the Euclidean distance (M x 1) between receiver and satellites
    estimated_rho = np.linalg.norm(receiver_position - sat_positions, axis=1, keepdims=True)  # M x 1

    # Add receiver clock bias (scalar is broadcasted correctly)
    estimated_rho += receiver_clock_bias   # M x 1

    # Compute pseudorange residuals (delta rho) using measured rho
    delta_rho = measured_rho - estimated_rho  # M x 1

    return estimated_rho, delta_rho

def compute_pseudorange_rate_delta(los_vectors, sat_velocities, receiver_velocity, receiver_clock_drift, measured_rho_dot):
    """
    Compute the estimated rate of change of pseudorange and its delta for M satellites.

    Parameters:
        los_vectors (numpy.ndarray): Mx3 matrix of Line-of-Sight (LoS) unit vectors.
        sat_velocities (numpy.ndarray): Mx3 matrix of satellite velocities [vx, vy, vz].
        receiver_velocity (numpy.ndarray): 1x3 vector of estimated receiver velocity [vx_EST, vy_EST, vz_EST].
        receiver_clock_drift (float): Estimated receiver clock drift (d_r,EST).
        measured_rho_dot (numpy.ndarray): Mx1 matrix of measured rate of change of pseudorange values.

    Returns:
        tuple:
            estimated_rho_dot (numpy.ndarray): Mx1 matrix of estimated rate of change of pseudorange values.
            delta_rho_dot (numpy.ndarray): Mx1 matrix of residuals for rate of change of pseudorange.
    """
    # Compute the relative velocity (receiver velocity - satellite velocity) (M x 3)
    relative_velocity = receiver_velocity - sat_velocities  # Broadcasting (1,3) - (M,3) -> (M,3)

    # Compute the dot product of LoS vectors and relative velocity (M x 1)
    estimated_rho_dot = np.sum(los_vectors * relative_velocity, axis=1, keepdims=True)  # (M,1)

    # Add receiver clock drift (scalar, correctly broadcasted)
    estimated_rho_dot += receiver_clock_drift  # (M,1)

    # Compute delta pseudorange rate
    delta_rho_dot = measured_rho_dot - estimated_rho_dot  # (M,1)

    return estimated_rho_dot, delta_rho_dot

def compute_geometry_matrix(los_vectors):
    """
    Compute the geometry matrix (G) used in the least squares estimation for GPS positioning.

    Parameters:
        los_vectors (numpy.ndarray): Mx3 matrix of Line-of-Sight (LoS) unit vectors.

    Returns:
        numpy.ndarray: The 2M x 8 geometry matrix.
    """
    M = los_vectors.shape[0]  # Number of satellites

    # First M rows
    upper_half = np.hstack((los_vectors, np.ones((M, 1)), np.zeros((M, 4))))

    # Last M rows
    lower_half = np.hstack((np.zeros((M, 4)), los_vectors, np.ones((M, 1))))

    # Stack both halves vertically
    G = np.vstack((upper_half, lower_half))

    return upper_half, G

def compute_dop_xyz(G):
    """
    Compute PDOP and DOP in X, Y, Z directions directly from the given geometry matrix G.
    
    Parameters:
        G (np.array): Geometry matrix of shape (n, 4), where n is the number of satellites.
        
    Returns:
        dict: Dictionary containing PDOP, DOP_x, DOP_y, DOP_z.
    """

    # Compute DOP matrix H = (G^T G)^(-1) using pseudo-inverse for stability
    H = np.linalg.pinv(G.T @ G)  # Using pinv to handle singular cases

    # Extract X, Y, Z DOP components
    DOP_x = np.sqrt(H[0, 0])  # H_11
    DOP_y = np.sqrt(H[1, 1])  # H_22
    DOP_z = np.sqrt(H[2, 2])  # H_33

    # Compute PDOP
    PDOP = np.sqrt(DOP_x**2 + DOP_y**2 + DOP_z**2)

    return {"DOP_x": DOP_x, "DOP_y": DOP_y, "DOP_z": DOP_z, "PDOP": PDOP}
