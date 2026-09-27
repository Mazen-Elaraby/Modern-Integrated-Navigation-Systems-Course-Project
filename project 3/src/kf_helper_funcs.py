import numpy as np

def compute_RN(a, e, phi):
    """
    Compute the normal radius (R_n)
    """
    return a / np.sqrt(1 - e**2 * np.sin(phi)**2)

def compute_RM(a, e, phi):
    """
    Compute the meridian radius of curvature (R_m)
    """
    return (a * (1 - e**2)) / (1 - e**2 * np.sin(phi)**2)**(3/2)

def compute_F1(RN, RM, h, phi, delta_t):
    """
    Compute the F1 submatrix.
    """
    return np.array([
        [0, 1 / (RM + h) * delta_t, 0],
        [1 / ((RN + h) * np.cos(phi)) * delta_t, 0, 0],
        [0, 0, delta_t]
    ])

def compute_F2(v_od, A_k, p_k):
    """
    Compute the F2 submatrix.
    """
    return np.array([
        [v_od * np.cos(A_k) * np.cos(p_k), v_od * np.sin(-A_k) * np.cos(p_k), 0],
        [-v_od * np.sin(-A_k) * np.cos(p_k), v_od * np.cos(A_k) * np.cos(p_k), 0],
        [0, v_od * np.sin(p_k), 0]
    ])

def compute_F3(gamma_od, beta_z, delta_t):
    """
    Compute the F3 submatrix.
    """
    return np.array([
        [1, 0, -delta_t],
        [0, 1 - gamma_od * delta_t, 0],
        [0, 0, 1 - beta_z * delta_t]
    ])

def compute_state_transition_matrix(R_n, R_m, phi, h, delta_t, v_od, A_k, p_k, gamma_od, beta_z):
    """
    Compute the full state transition matrix based on submatrices.
    """
    
    F1 = compute_F1(R_n, R_m, h, phi, delta_t)
    F2 = compute_F2(v_od, A_k, p_k)
    F3 = compute_F3(gamma_od, beta_z, delta_t)

    I3 = np.eye(3)
    Z3 = np.zeros((3, 3))

    # Construct the full transition matrix
    F = np.block([
        [I3, F1, Z3],
        [Z3, I3, F2],
        [Z3, Z3, F3]
    ])

    return F

def riss_mechanization_3d(epoch, prev_state, IMU_data, odom_speed, R_m, R_n, omega_e):
    
    # Extract previous state
    lat, lon, alt, ve, _, _, yaw, SF_od, gyro_bias = prev_state
    #SF_od = 0
    #gyro_bias = 0
    # Extract data
    fx = IMU_data['accelerometer'][epoch, 0]
    fy = IMU_data['accelerometer'][epoch, 1]
    fz = IMU_data['accelerometer'][epoch, 2]
    omega_z = IMU_data['gyroscope'][0, epoch] - gyro_bias
    v_od = (1 + SF_od) * odom_speed[epoch]
    a_od = (1 + SF_od) * odom_speed[epoch] - (1 + SF_od) * odom_speed[epoch - 1]

    # compute attitude
    roll = np.arctan2(-(fx + v_od * omega_z), fz)
    pitch = np.arctan2(-(fy - a_od), np.sqrt(fx**2 + fz**2))
    new_yaw = yaw + (
        (np.cos(pitch) * np.cos(roll) * omega_z)
        - (omega_e * np.sin(lat))
        - (ve * np.tan(lat) / (R_n + alt))
    )
    new_yaw = new_yaw % (2 * np.pi) # Normalize yaw to [0, 2π]

    # compute velocity components
    ve = -np.sin(new_yaw) * np.cos(pitch) * v_od
    vn = np.cos(new_yaw) * np.cos(pitch) * v_od
    vu = np.sin(pitch) * v_od

    # compute positional components
    new_alt = alt + vu
    new_lat = lat + (vn) / (R_m + alt)
    new_long = lon + (ve) / ((R_n + alt) * np.cos(lat))

    return np.array([new_lat, new_long, new_alt, ve, vn, vu, new_yaw, pitch, roll])
    

def enforce_psd(P, epsilon=1e-6):
    """Ensures that P is positive semi-definite (PSD)"""
    P = (P + P.T) / 2  # Ensure symmetry
    # eigvals, eigvecs = np.linalg.eigh(P)  # Compute eigenvalues

    # # Set negative eigenvalues to a small positive threshold
    # eigvals[eigvals < epsilon] = epsilon
    
    # # Reconstruct P using modified eigenvalues
    # P_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return P

def construct_R_matrix(gps_data, index, a, e):
    """
    Construct the measurement noise covariance matrix (R).

    Parameters:
    - gps_data: Dictionary containing sensor data, including GPS data.
    - index: Index at which to extract the standard deviation values.
    - a: Earth's semi-major axis (meters).
    - e: Earth's eccentricity.

    Returns:
    - R: Measurement noise covariance matrix (6x6).
    """

    # Extract relevant GPS standard deviation values
    lat_std_meters = gps_data["latitude_std"][0, index]  # Latitude standard deviation in meters
    long_std_meters = gps_data["longitude_std"][0, index]  # Longitude standard deviation in meters
    alt_std_meters = gps_data["altitude_std"][0, index]  # Altitude standard deviation in meters

    # Extract latitude and altitude at the given index
    lat = gps_data["latitude"][0, index]
    alt = gps_data["altitude"][0, index]  

    # Compute radius of curvature values
    R_M = compute_RM(a, e, lat)  # Meridian radius of curvature (north-south direction)
    R_N = compute_RN(a, e, lat)  # Normal radius of curvature (east-west direction)

    # Convert latitude and longitude standard deviations from meters to radians
    lat_std_rad = lat_std_meters / (R_M + alt)
    long_std_rad = long_std_meters / ((R_N + alt) * np.cos(lat))

    # Construct the R matrix
    R = np.zeros((6, 6))
    R[0, 0] = lat_std_rad**2
    R[1, 1] = long_std_rad**2
    R[2, 2] = alt_std_meters**2
    R[3, 3] = (long_std_meters * 5)**2
    R[4, 4] = (lat_std_meters * 5)**2
    R[5, 5] = (alt_std_meters * 10)**2
    #0.00045
    return R * 0.00045

def compute_measurement_residual(ins_est, gps_data, epoch):
    """
    Computes the measurement residual δz_k between INS estimates and GPS measurements.

    Parameters:
        ins_est (np.array): INS estimated state 
            Format: [lat, lon, alt, Ve, Vn, Vu, yaw, pitch, roll]
        gps_data (dict): Dictionary containing GPS measurements
            Expected keys: ['latitude', 'longitude', 'altitude', 'velocity_east', 'velocity_north', 'velocity_up']
        epoch (int): The time index for retrieving corresponding GPS data

    Returns:
        np.array: Measurement residual δz_k = INS - GPS
    """

    # Extract INS estimated values
    lat_ins, lon_ins, alt_ins, ve_ins, vn_ins, vu_ins, _, _, _ = ins_est

    # Extract GPS values at the given epoch
    lat_gps = gps_data['latitude'][0, epoch]
    lon_gps = gps_data['longitude'][0, epoch]
    alt_gps = gps_data['altitude'][0,  epoch]
    ve_gps = gps_data['velocity_east'][0, epoch]
    vn_gps = gps_data['velocity_north'][0, epoch]
    vu_gps = gps_data['velocity_up'][0, epoch]

    # Compute measurement residual δz_k = INS - GPS
    dz_k = np.array([
        lat_ins - lat_gps,
        lon_ins - lon_gps,
        alt_ins - alt_gps,
        ve_ins - ve_gps,
        vn_ins - vn_gps,
        vu_ins - vu_gps
    ])
    dz_k = -1 * dz_k
    return dz_k.reshape(-1,1)
