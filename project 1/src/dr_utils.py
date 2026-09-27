import numpy as np
from collections import deque
from scipy.signal import butter

def initialize_arrays(num_epochs):
    """Pre-allocate arrays for position, velocity, and attitude."""
    return (
        np.zeros(num_epochs), np.zeros(num_epochs), np.zeros(num_epochs), # Position
        np.zeros(num_epochs), np.zeros(num_epochs), np.zeros(num_epochs), # Velocity
        np.zeros(num_epochs), np.zeros(num_epochs), np.zeros(num_epochs)  # Attitude
    )

def initialize_values(alt_est, lat_est, long_est, ve_est, vn_est, vu_est, yaw_est, pitch_est, roll_est, ref_data):
    """Initialize position, velocity, and attitude values from reference data."""
    ref_altitude, ref_latitude, ref_longitude, ref_ve, ref_vn, ref_vu, ref_az, ref_pitch, ref_roll = ref_data
    
    # Position
    alt_est[0] = ref_altitude[0]
    lat_est[0] = ref_latitude[0]
    long_est[0] = ref_longitude[0]
    
    # Velocity
    ve_est[0] = ref_ve[0]
    vn_est[0] = ref_vn[0]
    vu_est[0] = ref_vu[0]
    
    # Attitude
    yaw_est[0] = (-1 * ref_az[0]) % (2 * np.pi)
    pitch_est[0] = ref_pitch[0]
    roll_est[0] = ref_roll[0]

def extract_sensor_data(e, acc_data, gyro_data, odom_speed, biases, c):
    """Extract sensor data for current epoch."""
    return (
        acc_data[e, 0] - biases["bx"], 
        c * acc_data[e, 1] - biases["by"], 
        acc_data[e, 2] - biases["bz"], 
        gyro_data[e] - biases["bw"], 
        odom_speed[e]
    )

def compute_yaw(e, yaw_est, pitch_est, roll_est, lat_est, ve_est, alt_est, omega_z, omega_e, a, ecc):
    """Calculate yaw angle."""
    R_n = a / np.sqrt(1 - (ecc**2) * (np.sin(lat_est[e-1])**2))
    yaw_est[e] = yaw_est[e-1] + (
        (np.cos(pitch_est[e-1]) * np.cos(roll_est[e-1]) * omega_z)
        - (omega_e * np.sin(lat_est[e-1]))
        - (ve_est[e-1] * np.tan(lat_est[e-1]) / (R_n + alt_est[e-1]))
    )
    yaw_est[e] = yaw_est[e] % (2 * np.pi) # Normalize yaw to [0, 2π]

def compute_velocity(e, yaw_est, pitch_est, v_od, ve_est, vn_est, vu_est):
    """Calculate velocity components."""
    ve_est[e] = -np.sin(yaw_est[e]) * np.cos(pitch_est[e]) * v_od
    vn_est[e] = np.cos(yaw_est[e]) * np.cos(pitch_est[e]) * v_od
    vu_est[e] = np.sin(pitch_est[e]) * v_od

def compute_position(e, lat_est, long_est, alt_est, ve_est, vn_est, vu_est, a, ecc):
    """Calculate position components."""
    R_m = (a * (1 - ecc**2)) / ((1 - (ecc**2) * (np.sin(lat_est[e-1])**2)) ** (3/2))
    R_n = a / np.sqrt(1 - (ecc**2) * (np.sin(lat_est[e-1])**2))
    
    alt_est[e] = alt_est[e-1] + vu_est[e]
    
    lat_est[e] = lat_est[e-1] + (vn_est[e]) / (R_m + alt_est[e])
    #lat_est[e] = (lat_est[e] + np.pi/2) % np.pi - np.pi/2 # Normalize latitude to [-π/2, π/2]
    
    long_est[e] = long_est[e-1] + (ve_est[e]) / ((R_n + alt_est[e]) * np.cos(lat_est[e]))
    #long_est[e] = (long_est[e] + np.pi) % (2 * np.pi) - np.pi # Normalize longitude to [-π, π]

def compute_attitude(e, fx, fy, fz, omega_z, v_od, odom_speed, roll_est, pitch_est):
    """Calculate pitch and roll angles."""
    a_od = odom_speed[e] - odom_speed[e-1] # Acceleration from odometer data
    roll_est[e] = np.arctan2(-(fx + v_od * omega_z), fz)
    pitch_est[e] = np.arctan2(-(fy - a_od), np.sqrt(fx**2 + fz**2))

def compute_attitude_ema(e, fx, fy, fz, omega_z, v_od, odom_speed, filtered_speed, roll_est, pitch_est, alpha=0.1, beta=1):
    """Calculate pitch and roll angles using filtered acceleration from EMA."""
    
    filtered_speed[e] = alpha * odom_speed[e] + (1 - alpha) * filtered_speed[e-1]
    
    a_od = 2 * ((filtered_speed[e] - filtered_speed[e-1]) / (1 + beta))
    
    roll_est[e] = np.arctan2(-(fx + v_od * omega_z), fz)
    pitch_est[e] = np.arctan2(-(fy - a_od), np.sqrt(fx**2 + fz**2))

def compute_attitude_ma(e, fx, fy, fz, omega_z, v_od, odom_speed, filtered_speed, roll_est, pitch_est, window_size=3, beta=1):
    """Calculate pitch and roll angles using filtered acceleration from Moving Average + Tustin's Approximation."""
    
    # Initialize sliding window for moving average (only do this for the first epoch)
    if e == 1:
        global speed_window  # Use a global variable to maintain the window state
        speed_window = deque([odom_speed[0]], maxlen=window_size)

    speed_window.append(odom_speed[e])
    
    filtered_speed[e] = sum(speed_window) / len(speed_window)
    
    a_od = 2 * ((filtered_speed[e] - filtered_speed[e-1]) / (1 + beta))
    
    roll_est[e] = np.arctan2(-(fx + v_od * omega_z), fz)
    pitch_est[e] = np.arctan2(-(fy - a_od), np.sqrt(fx**2 + fz**2))

# Function to compute Butterworth filter coefficients (once before running epochs)
def initialize_butterworth_filter(cutoff=0.2, fs=1.0, order=1):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def compute_attitude_butterworth(e, fx, fy, fz, omega_z, v_od, odom_speed, filtered_speed, roll_est, pitch_est, b, a):
    """Calculate pitch and roll angles using filtered acceleration from the Butterworth filter."""
    
    filtered_speed[e] = b[0] * odom_speed[e] + b[1] * odom_speed[e-1] - a[1] * filtered_speed[e-1]

    a_od = filtered_speed[e] - filtered_speed[e-1]

    roll_est[e] = np.arctan2(-(fx + v_od * omega_z), fz)
    pitch_est[e] = np.arctan2(-(fy - a_od), np.sqrt(fx**2 + fz**2))

def compute_attitude_bw(e, fx, fy, fz, omega_z, v_od, odom_speed, filtered_speed, roll_est, pitch_est, b, a, beta=1):
    """Calculate pitch and roll angles using filtered acceleration from the Butterworth filter. and Tustin's Approximation"""
    
    # Generalize the filtered speed equation for Nth-order Butterworth
    filtered_speed[e] = sum(b[k] * odom_speed[e - k] for k in range(len(b)) if e - k >= 0) - \
                        sum(a[k] * filtered_speed[e - k] for k in range(1, len(a)) if e - k >= 0)
    
    # Compute acceleration difference
    a_od = 2 * ((filtered_speed[e] - filtered_speed[e-1]) / (1 + beta))

    # Calculate roll and pitch estimates
    roll_est[e] = np.arctan2(-(fx + v_od * omega_z), fz)
    pitch_est[e] = np.arctan2(-(fy - a_od), np.sqrt(fx**2 + fz**2))
