import numpy as np

def load_satellite_data(filename: str):
    """
    Load a .npz file containing satellite data and extract raw and reference data.

    Parameters:
        filename (str): Path to the .npz file.

    Returns:
        tuple: (raw_data dictionary, reference_data dictionary)
    """
    # Load the .npz file
    loaded_data = np.load(filename, allow_pickle=True)

    # Extract dictionaries
    raw_data = loaded_data['raw_data'].item()
    reference_data = loaded_data['reference_data'].item()

    return raw_data, reference_data

def extract_satellite_data(raw_data, epoch_index, L=1575.42e6):
    """
    Extract satellite position, velocity, pseudorange, and rate of change of pseudorange for a given epoch.

    Parameters:
        raw_data (dict): Dictionary containing raw satellite data.
        epoch_index (int): The index of the current epoch in the estimation loop.
        L (float): The satellite transmitted frequency.

    Returns:
        tuple: (M x 3 satellite position matrix, 
                M x 3 satellite velocity matrix, 
                M x 1 pseudorange matrix, 
                M x 1 pseudorange standard deviation matrix,
                M x 1 rate of change of pseudorange matrix)
    """

    # Extract satellite position (M x 3)
    sat_x = raw_data['sat_x'][epoch_index]  # Mx1 array
    sat_y = raw_data['sat_y'][epoch_index]  # Mx1 array
    sat_z = raw_data['sat_z'][epoch_index]  # Mx1 array

    # Stack to form Mx3 matrix
    sat_positions = np.vstack((sat_x, sat_y, sat_z)).T  # M x 3 matrix

    M = sat_positions.shape[0]

    # Extract satellite velocity (M x 3)
    sat_vx = raw_data['sat_vx'][epoch_index]  # Mx1 array
    sat_vy = raw_data['sat_vy'][epoch_index]  # Mx1 array
    sat_vz = raw_data['sat_vz'][epoch_index]  # Mx1 array

    # Stack to form Mx3 matrix
    sat_velocities = np.vstack((sat_vx, sat_vy, sat_vz)).T  # M x 3 matrix

    # Extract pseudorange values (M x 1)
    rho = raw_data['rho'][epoch_index].reshape(-1, 1) if M > 1 else raw_data['rho'][epoch_index]

    # Extract satellite clock correction (M x 1)
    dt_sv = raw_data['dt_sv'][epoch_index].reshape(-1, 1) if M > 1 else raw_data['dt_sv'][epoch_index]

    # Define speed of light (c) in meters per second
    c = 299792458.0  # m/s

    # Apply clock correction to pseudorange
    rho_corrected = rho + (c * dt_sv)  # Apply correction element-wise

    # Extract pseudorange standard deviation (M x 1)
    rho_std = np.array(raw_data['rho_std'][epoch_index]).reshape(M, 1) if M > 1 else np.array(raw_data['rho_std'][epoch_index])

    # Extract Satellite Elevation Information
    el_sat = np.array(raw_data['el_sat'][epoch_index]).reshape(M, 1) if M > 1 else np.array(raw_data['el_sat'][epoch_index])

    # Extract Doppler values (M x 1)
    D = raw_data['doppler'][epoch_index].reshape(-1, 1) if M > 1 else raw_data['doppler'][epoch_index]

    # Compute rate of change of pseudorange (M x 1)
    rho_dot = - (D * c) / L  # Element-wise operation, returns M x 1 matrix

    return sat_positions, sat_velocities, rho_corrected, rho_std, rho_dot, el_sat

def ecef_to_geodetic_epochs(ecef_coords, max_iterations=10, tolerance=1e-9):
    """
    Converts multiple sets of ECEF coordinates (for several epochs) to geodetic coordinates
    (latitude, longitude, altitude) using an iterative algorithm.
    
    Parameters:
        ecef_coords (numpy array): A 3xN numpy array containing [x, y, z] coordinates for N epochs.
        tolerance (float): Convergence tolerance for the iterative algorithm.
    
    Returns:
        tuple: 
            - latitudes (numpy array): Latitude in radians for each epoch.
            - longitudes (numpy array): Longitude in radians for each epoch.
            - altitudes (numpy array): Altitude in meters for each epoch.
    """
    # WGS84 Ellipsoid constants
    a = 6378137.0  # Semi-major axis (m)
    e2 = 0.00669437999014  # Square of eccentricity
    
    num_epochs = ecef_coords.shape[1]  # Get number of epochs
    
    # Initialize arrays to store results
    latitudes = np.zeros(num_epochs)
    longitudes = np.zeros(num_epochs)
    altitudes = np.zeros(num_epochs)

    # Initial altitude and latitude estimate
    x, y, z = ecef_coords[:, 0]
    P = np.sqrt(x**2 + y**2)

    h = 0.0  # Initial altitude
    R_N = a / np.sqrt(1 - e2 * np.sin(0)**2)  # Estimate using lat = 0
    Pe = (R_N + h) * np.cos(0)
    lat = np.arctan2(z, Pe * (1 - e2))  # Initial approximation

    for epoch in range(num_epochs):
        # Extract x, y, z for the current epoch
        x, y, z = ecef_coords[:, epoch]

        # Compute longitude
        lon = np.arctan2(y, x)

        # Compute P = sqrt(x^2 + y^2) (planar projection)
        P = np.sqrt(x**2 + y**2)

        iteration_count = 0

        while iteration_count < max_iterations:

            iteration_count += 1
            
            # Compute radius of curvature in the prime vertical
            R_N = a / np.sqrt(1 - e2 * np.sin(lat)**2)

            # Compute updated altitude
            h_new = P / np.cos(lat) - R_N

            # Compute updated latitude using the correct formula
            numerator = z * (R_N + h_new)
            denominator = P * (R_N * (1 - e2) + h_new)
            lat_new = np.arctan2(numerator, denominator)

            # Check for convergence
            if abs(lat_new - lat) < tolerance and abs(h_new - h) < tolerance:
                break

            lat, h = lat_new, h_new

        # Store results
        latitudes[epoch] = lat
        longitudes[epoch] = lon
        altitudes[epoch] = h

        print(f"Epoch {epoch}: Converged in {iteration_count} iterations.")

    return latitudes, longitudes, altitudes

def geodetic_to_ecef(lat, lon, h, a=6378137.0, e=0.081819190842622):  
    """
    Convert geodetic coordinates (latitude, longitude, height) to ECEF coordinates.
    
    Parameters:
    - lat: Latitude in radians
    - lon: Longitude in radians
    - h: Ellipsoidal height in meters
    - a: Semi-major axis of the WGS84 ellipsoid (default: 6378137.0 meters)
    - e: Eccentricity of the WGS84 ellipsoid (default: 0.081819190842622)

    Returns:
    - (x, y, z): ECEF coordinates in meters
    """
    
    # Compute the prime vertical radius of curvature
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)

    # Compute ECEF coordinates
    x = (N + h) * np.cos(lat) * np.cos(lon)
    y = (N + h) * np.cos(lat) * np.sin(lon)
    z = (N * (1 - e**2) + h) * np.sin(lat)

    return x, y, z

def ecef_to_neu_velocity(velocity_matrix, longitude, latitude):
    """
    Converts velocity from ECEF coordinates (vx, vy, vz) to local NEU (vn, ve, vu).
    
    Parameters:
    velocity_matrix : np.ndarray
        A 3 x num_epochs matrix where rows correspond to [vx, vy, vz].
    longitude : np.ndarray
        A 1D array of shape (num_epochs,) containing longitudes in radians.
    latitude : np.ndarray
        A 1D array of shape (num_epochs,) containing latitudes in radians.
    
    Returns:
    vn : np.ndarray
        North velocity component of shape (num_epochs,).
    ve : np.ndarray
        East velocity component of shape (num_epochs,).
    vu : np.ndarray
        Up velocity component of shape (num_epochs,).
    """
    num_epochs = velocity_matrix.shape[1]
    
    # Initialize output arrays
    vn = np.zeros(num_epochs)
    ve = np.zeros(num_epochs)
    vu = np.zeros(num_epochs)
    
    # Iterate over each epoch to construct the transformation matrix and compute velocities
    for i in range(num_epochs):
        lon = longitude[i]
        lat = latitude[i]
        
        # Transformation matrix (transpose of Re_l from your reference)
        R_l_e = np.array([
            [-np.sin(lon), -np.sin(lat) * np.cos(lon),  np.cos(lat) * np.cos(lon)],
            [ np.cos(lon), -np.sin(lat) * np.sin(lon),  np.cos(lat) * np.sin(lon)],
            [ 0,           np.cos(lat),                np.sin(lat)]
        ])
        
        # Extract velocity vector for the current epoch
        v_ecef = velocity_matrix[:, i]
        
        # Transform velocity to local NEU frame
        v_enu = R_l_e.T @ v_ecef
        
        # Store results
        ve[i], vn[i], vu[i] = v_enu
    
    return ve, vn, vu  # Following the convention ve, vn, vu