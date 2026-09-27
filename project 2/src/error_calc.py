import numpy as np

def calculate_position_errors(ref_data, est_data, a, ecc):
    """
    Calculate position errors (East, North, 2D, Altitude, and 3D) at each time instant.

    Parameters:
        ref_data (tuple): Tuple containing reference data (altitude, latitude, longitude).
        est_data (tuple): Tuple containing estimated data (altitude, latitude, longitude).
        a (float): Earth's semi-major axis (equatorial radius, meters).
        ecc (float): Earth's eccentricity.

    Returns:
        position_errors (dict): Dictionary containing arrays for East, North, 2D, Altitude, and 3D position errors.
    """
    # Unpack reference data
    ref_latitude, ref_longitude, ref_altitude, _, _, _ = ref_data

    # Unpack estimated data
    est_latitude, est_longitude, est_altitude, _, _, _ = est_data

    num_epochs = len(ref_latitude)
    
    # Initialize arrays for errors
    delta_east_errors = np.zeros(num_epochs)
    delta_north_errors = np.zeros(num_epochs)
    delta_2d_errors = np.zeros(num_epochs)
    delta_altitude_errors = np.zeros(num_epochs)
    delta_3d_errors = np.zeros(num_epochs)

    for i in range(num_epochs):
        # Compute R_N and R_M using reference latitude at the current time step
        sin_lat = np.sin(ref_latitude[i])
        R_N = a / np.sqrt(1 - ecc**2 * sin_lat**2)  # Radius of curvature in the prime vertical
        R_M = a * (1 - ecc**2) / (1 - ecc**2 * sin_lat**2)**(3/2)  # Radius of curvature in the meridian

        # Calculate delta_lat and delta_long for position errors
        delta_lat = est_latitude[i] - ref_latitude[i]
        delta_long = est_longitude[i] - ref_longitude[i]

        # Convert latitude and longitude errors into North and East errors in meters
        delta_north_errors[i] = delta_lat * (R_M + ref_altitude[i])
        delta_east_errors[i] = delta_long * (R_N + ref_altitude[i]) * np.cos(ref_latitude[i])

        # Calculate 2D (horizontal) error
        delta_2d_errors[i] = np.sqrt(delta_north_errors[i]**2 + delta_east_errors[i]**2)

        # Calculate altitude error
        delta_altitude_errors[i] = est_altitude[i] - ref_altitude[i]

        # Calculate 3D position error using East, North, and Altitude components
        delta_3d_errors[i] = np.sqrt(delta_east_errors[i]**2 + delta_north_errors[i]**2 + delta_altitude_errors[i]**2)

    # Create dictionary for position errors
    position_errors = {
        "east": delta_east_errors,
        "north": delta_north_errors,
        "2d": delta_2d_errors,
        "altitude": delta_altitude_errors,
        "3d": delta_3d_errors
    }
    return position_errors

def calculate_max_position_error(position_errors):
    """
    Calculate the maximum position error for east, north, 2D, altitude, and 3D directions.

    Parameters:
        position_errors (dict): Dictionary containing arrays of position errors (east, north, 2d, altitude, 3d).
    
    Returns:
        max_position_error (dict): Dictionary containing the maximum position error for east, north, 2D, altitude, and 3D.
    """
    max_position_error = {
        "east": np.max(np.abs(position_errors["east"])),
        "north": np.max(np.abs(position_errors["north"])),
        "2d": np.max(np.abs(position_errors["2d"])),
        "altitude": np.max(np.abs(position_errors["altitude"])),
        "3d": np.max(np.abs(position_errors["3d"]))
    }
    return max_position_error

def calculate_rms_position_error(position_errors):
    """
    Calculate the RMS position error for east, north, 2D, altitude, and 3D directions.

    Parameters:
        position_errors (dict): Dictionary containing arrays of position errors (east, north, 2d, altitude, 3d).
    
    Returns:
        rms_position_error (dict): Dictionary containing the RMS position error for east, north, 2D, altitude, and 3D.
    """
    rms_position_error = {
        "east": np.sqrt(np.mean(position_errors["east"]**2)),
        "north": np.sqrt(np.mean(position_errors["north"]**2)),
        "2d": np.sqrt(np.mean(position_errors["2d"]**2)),
        "altitude": np.sqrt(np.mean(position_errors["altitude"]**2)),
        "3d": np.sqrt(np.mean(position_errors["3d"]**2))
    }
    return rms_position_error

def calculate_std_position_error(position_errors):
    """
    Calculate the standard deviation (STD) of position errors for east, north, 2D, altitude, and 3D directions.

    Parameters:
        position_errors (dict): Dictionary containing arrays of position errors (east, north, 2d, altitude, 3d).
    
    Returns:
        std_position_error (dict): Dictionary containing the standard deviation of position errors for east, north, 2D, altitude, and 3D.
    """
    std_position_error = {
        "east": np.std(position_errors["east"], ddof=0),
        "north": np.std(position_errors["north"], ddof=0),
        "2d": np.std(position_errors["2d"], ddof=0),
        "altitude": np.std(position_errors["altitude"], ddof=0),
        "3d": np.std(position_errors["3d"], ddof=0)
    }
    return std_position_error

def calculate_velocity_errors(ref_data, est_data):
    """
    Calculate velocity errors (East, North, and Up) at each time instant.

    Parameters:
        ref_data (tuple): Tuple containing reference velocity data (velocity_east, velocity_north, velocity_up).
        est_data (tuple): Tuple containing estimated velocity data (velocity_east, velocity_north, velocity_up).

    Returns:
        velocity_errors (dict): Dictionary containing arrays for East, North, and Up velocity errors.
    """
    # Unpack reference velocity data
    _, _, _, ref_velocity_east, ref_velocity_north, ref_velocity_up = ref_data

    # Unpack estimated velocity data
    _, _, _, est_velocity_east, est_velocity_north, est_velocity_up = est_data

    num_epochs = len(ref_velocity_east)

    # Initialize arrays for velocity errors
    delta_east_velocity_errors = np.zeros(num_epochs)
    delta_north_velocity_errors = np.zeros(num_epochs)
    delta_up_velocity_errors = np.zeros(num_epochs)

    for i in range(num_epochs):
        # Compute velocity errors in each direction
        delta_east_velocity_errors[i] = est_velocity_east[i] - ref_velocity_east[i]
        delta_north_velocity_errors[i] = est_velocity_north[i] - ref_velocity_north[i]
        delta_up_velocity_errors[i] = est_velocity_up[i] - ref_velocity_up[i]

    # Create dictionary for velocity errors
    velocity_errors = {
        "east": delta_east_velocity_errors,
        "north": delta_north_velocity_errors,
        "up": delta_up_velocity_errors
    }
    
    return velocity_errors

def calculate_max_velocity_error(velocity_errors):
    """
    Calculate the maximum velocity error for east, north, and up directions.

    Parameters:
        velocity_errors (dict): Dictionary containing arrays of velocity errors (east, north, up).
    
    Returns:
        max_velocity_error (dict): Dictionary containing the maximum velocity error for east, north, and up.
    """
    max_velocity_error = {
        "east": np.max(np.abs(velocity_errors["east"])),
        "north": np.max(np.abs(velocity_errors["north"])),
        "up": np.max(np.abs(velocity_errors["up"]))
    }
    return max_velocity_error

def calculate_rms_velocity_error(velocity_errors):
    """
    Calculate the RMS velocity error for east, north, and up directions.

    Parameters:
        velocity_errors (dict): Dictionary containing arrays of velocity errors (east, north, up).
    
    Returns:
        rms_velocity_error (dict): Dictionary containing the RMS velocity error for east, north, and up.
    """
    rms_velocity_error = {
        "east": np.sqrt(np.mean(velocity_errors["east"]**2)),
        "north": np.sqrt(np.mean(velocity_errors["north"]**2)),
        "up": np.sqrt(np.mean(velocity_errors["up"]**2))
    }
    return rms_velocity_error

def calculate_std_velocity_error(velocity_errors):
    """
    Calculate the standard deviation (STD) of velocity errors for east, north, and up directions.

    Parameters:
        velocity_errors (dict): Dictionary containing arrays of velocity errors (east, north, up).
    
    Returns:
        std_velocity_error (dict): Dictionary containing the standard deviation of velocity errors for east, north, and up.
    """
    std_velocity_error = {
        "east": np.std(velocity_errors["east"], ddof=0),
        "north": np.std(velocity_errors["north"], ddof=0),
        "up": np.std(velocity_errors["up"], ddof=0)
    }
    return std_velocity_error
