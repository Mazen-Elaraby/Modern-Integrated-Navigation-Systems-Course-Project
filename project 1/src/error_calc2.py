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
    ref_altitude, ref_latitude, ref_longitude, _, _, _, _, _, _ = ref_data

    # Unpack estimated data
    est_altitude, est_latitude, est_longitude, _, _, _, _, _, _ = est_data

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

def calculate_total_distance_traveled(ref_data, a, ecc):
    """
    Calculate the total distance traveled for East, North, 2D, Altitude, and 3D directions.

    Parameters:
        ref_data (tuple): Tuple containing reference data (altitude, latitude, longitude).
        a (float): Earth's semi-major axis (equatorial radius, meters).
        ecc (float): Earth's eccentricity.

    Returns:
        total_distance_traveled (dict): Dictionary containing scalar total distances for East, North, 2D, Altitude, and 3D.
    """
    # Unpack reference data
    ref_altitude, ref_latitude, ref_longitude, _, _, _, _, _, _ = ref_data

    num_epochs = len(ref_latitude)

    # Initialize arrays for distances traveled
    delta_distance_east = np.zeros(num_epochs)
    delta_distance_north = np.zeros(num_epochs)
    delta_distance_2d = np.zeros(num_epochs)
    delta_distance_altitude = np.zeros(num_epochs)
    delta_distance_3d = np.zeros(num_epochs)

    for i in range(1, num_epochs):
        # Compute R_N and R_M using reference latitude at the current time step
        sin_lat = np.sin(ref_latitude[i])
        R_N = a / np.sqrt(1 - ecc**2 * sin_lat**2)
        R_M = a * (1 - ecc**2) / (1 - ecc**2 * sin_lat**2)**(3/2)

        # Calculate delta distances traveled between successive steps
        delta_lat_travel = ref_latitude[i] - ref_latitude[i-1]
        delta_long_travel = ref_longitude[i] - ref_longitude[i-1]
        delta_altitude_travel = ref_altitude[i] - ref_altitude[i-1]

        # Distances traveled in East and North directions
        delta_distance_north[i] = delta_lat_travel * (R_M + ref_altitude[i])
        delta_distance_east[i] = delta_long_travel * (R_N + ref_altitude[i]) * np.cos(ref_latitude[i])

        # Altitude distance traveled
        delta_distance_altitude[i] = delta_altitude_travel

        # 2D distance traveled in meters
        delta_distance_2d[i] = np.sqrt(delta_distance_north[i]**2 + delta_distance_east[i]**2)

        # 3D distance traveled using East, North, and Altitude components
        delta_distance_3d[i] = np.sqrt(delta_distance_east[i]**2 + delta_distance_north[i]**2 + delta_distance_altitude[i]**2)

    # Sum the deltas to get the total distance traveled
    total_distance_traveled = {
        "east": np.sum(np.abs(delta_distance_east)),
        "north": np.sum(np.abs(delta_distance_north)),
        "2d": np.sum(np.abs(delta_distance_2d)),
        "altitude": np.sum(np.abs(delta_distance_altitude)),
        "3d": np.sum(np.abs(delta_distance_3d))
    }

    return total_distance_traveled

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

def calculate_error_percentage(rms_position_error, total_distance_traveled):
    """
    Calculate the error percentage of total distance traveled for east, north, 2D, altitude, and 3D directions.

    Parameters:
        rms_position_error (dict): Dictionary containing the RMS position errors for east, north, 2D, altitude, and 3D.
        total_distance_traveled (dict): Dictionary containing the scalar total distances traveled for east, north, 2D, altitude, and 3D.

    Returns:
        error_percentage (dict): Dictionary containing the error as a percentage of total distance traveled.
    """
    error_percentage = {}

    for key in rms_position_error.keys():
        if total_distance_traveled[key] != 0:  # Avoid division by zero
            error_percentage[key] = (rms_position_error[key] / total_distance_traveled[key]) * 100
        else:
            error_percentage[key] = 0.0  # Set to 0% if the total distance is zero

    return error_percentage
