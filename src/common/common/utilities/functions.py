import numpy as np	
import math
from scipy.stats import laplace

def enu_to_lla(east, north, ref_lat, ref_lon, ref_alt):
    """
    Convert ENU coordinates to Latitude, Longitude, and Altitude (LLA).

    Parameters:
    - east: East coordinate in meters
    - north: North coordinate in meters
    - up: Up coordinate in meters
    - ref_lat: Reference latitude in degrees
    - ref_lon: Reference longitude in degrees
    - ref_alt: Reference altitude in meters

    Returns:
    - lat: Latitude in degrees
    - lon: Longitude in degrees
    - alt: Altitude in meters
    """
    R = 6378137.0  # Earth's radius in meters (approximate)
    
    # Convert reference lat/lon to radians
    ref_lat_rad = np.radians(ref_lat)
    ref_lon_rad = np.radians(ref_lon)
    
    # Calculate delta lat, delta lon
    dlat = north / R
    dlon = east / (R * np.cos(ref_lat_rad))
    
    # Convert back to degrees
    lat = np.degrees(ref_lat_rad + dlat)
    lon = np.degrees(ref_lon_rad + dlon)

    llastates = [lat, lon]
    
    return llastates

def E2N(angle, unit):
    """
    Convert East-referenced angle (clockwise from East) back to 
    North-referenced angle (clockwise from North).

    Parameters:
    - angle: The angle to convert (in radians if unit is 1, in degrees if unit is 0).
    - unit: 0 for degrees, 1 for radians.

    Returns:
    - The angle in the desired unit (degrees or radians).
    """
    if unit == 0:  # Degrees
        degrees = 90 - math.degrees(angle)
        return degrees % 360  # Normalize to [0, 360)

    if unit == 1:  # Radians
        radians = (np.pi / 2) - angle
        return radians % (2 * np.pi)  # Normalize to [0, 2π)


def lla_to_enu(lat, lon, alt, ref_lat, ref_lon, ref_alt):
    lat = np.radians(lat)
    lon = np.radians(lon)
    ref_lat = np.radians(ref_lat)
    ref_lon = np.radians(ref_lon)
    R = 6378137.0  # meters
    dlat = lat - ref_lat
    dlon = lon - ref_lon
    east = R * dlon * np.cos(ref_lat)
    north = R * dlat
    up = alt - ref_alt
    return east, north

def N2E(angle, unit):
    # unit: 0, degree, 1, radians
    # output: radians

    if unit == 0:
        radians = math.radians(90-angle)

    if unit == 1:
        radians = (np.pi/2)-angle

    return radians

#  angle is radians
def pi_2_pi(angle):
    
    new_angle = (angle+np.pi)%(2*np.pi)-np.pi

    return new_angle

def calc_pps_steering(x, y, yaw, targetx, targety):
    ""
    ""
    delta_x = targetx-x
    delta_y = targety-y
    desired_yaw = np.arctan2(delta_y, delta_x)
    # calculate the diffence with desired yaw and current yaw
    delta_yaw = desired_yaw - yaw

    return delta_yaw

def pos_dis(cur_pos, num_samples, mean, covar):

    # mean = [mean_x, mean_y]
    pos_samples = np.zeros((num_samples, 4))
    mean_posx = cur_pos[0] + mean[0]
    mean_posy = cur_pos[1] + mean[1]
    mean_pos = np.array([mean_posx, mean_posy])  # shape (2,)

    xy_samples = np.random.multivariate_normal(mean=mean_pos, cov=covar, size=num_samples)  # shape [Z, 2]

    # Repeat [psi, v] as fixed values
    psi_v = cur_pos[2:]                    # shape [2]
    psi_v_repeat = np.tile(psi_v, (num_samples, 1))  # shape [Z, 2]

    # Concatenate to get full [x, y, psi, v]
    pos_samples = np.hstack((xy_samples, psi_v_repeat))  # shape [Z, 4]
    pos_mean = np.mean(pos_samples, axis=0)

    return pos_mean, pos_samples


def gps_laplace_samples(true_pos, num_samples=10, scale=2.0):

    pos_samples = np.zeros((num_samples, 4))

    noise_x = laplace.rvs(loc=0, scale=scale, size=num_samples)
    noise_y = laplace.rvs(loc=0, scale=scale, size=num_samples)
    samples =true_pos[:2] + np.stack([noise_x, noise_y], axis=1)
        # Repeat [psi, v] as fixed values
    psi_v = true_pos[2:]                    # shape [2]
    psi_v_repeat = np.tile(psi_v, (num_samples, 1))  # shape [Z, 2]

    # Concatenate to get full [x, y, psi, v]
    pos_samples = np.hstack((samples, psi_v_repeat))  # shape [Z, 4]

    # Compute sample mean
    mean_sample = np.mean(pos_samples, axis=0)  # shape: [2]

    return mean_sample,  pos_samples

def pos_idd_dis(cur_pos, num_samples, std_dev):
    # cur_pos: [x, y, psi, v]
    x, y, psi, v = cur_pos

    # Generate independent Gaussian noise for x and y
    noise_x = np.random.normal(0, std_dev, num_samples)
    noise_y = np.random.normal(0, std_dev, num_samples)

    noisy_x = x + noise_x
    noisy_y = y + noise_y
    # print('x, y', x, y, noise_x, noise_y, noisy_x, noisy_y)

    # Repeat psi and v as fixed values
    psi_array = np.full(num_samples, psi)
    v_array = np.full(num_samples, v)

    # Stack all into [num_samples, 4]
    pos_samples = np.stack((noisy_x, noisy_y, psi_array, v_array), axis=1)
    pos_mean = np.mean(pos_samples, axis=0)
    # print('pos_mean, pos_samples', pos_mean, pos_samples)

    return pos_mean, pos_samples

def _adjust_target_position(self, position_x: float, position_y: float, pre_iter):
    """
    Override the target position if the vehicle is "close enough" (threshold check passes).
    """

    dis = np.sqrt((self._waypointsx-position_x)**2+(self._waypointsy-position_y)**2)
    target_index = np.argmin(dis)
    if pre_iter < 0:
        pre_iter = target_index
    for i in range(target_index, len(self._waypointsx)):
        dis_to_point = np.sqrt((self._waypointsx[i]-position_x)**2+(self._waypointsy[i]-position_y)**2)
        if dis_to_point >= C.LHD:
            target_index =i
            break

    # Get look ahead point coordinate
    look_ahead_x = self._waypointsx[self._waypoints_iterator ]
    look_ahead_y = self._waypointsy[self._waypoints_iterator ]

    return target_index

def _adjust_target_position_obs(self, position_x: float, position_y: float, obs_x: float = None, obs_y: float=None, pre_iter=None):
    """
    Override the target position if the vehicle is "close enough" (threshold check passes).
    """
    dis = np.sqrt((self._waypointsx-position_x)**2+(self._waypointsy-position_y)**2)
    skip_dis = C.r+C.obs_r+C.SAFETY_DIS
    target_index = np.argmin(dis)
    for i in range(target_index, len(self._waypointsx)):
        dis_to_point = np.sqrt((self._waypointsx[i]-position_x)**2+(self._waypointsy[i]-position_y)**2)
        if dis_to_point >= C.LHD:
            target_index =i
            print('origin:', target_index)

            break

    if C.static_obstacles_on and obs_x is not None and obs_y is not None:
        # Get look ahead point coordinate
        look_ahead_x = self._waypointsx[target_index]
        look_ahead_y = self._waypointsy[target_index]
        obs_dis = np.sqrt((obs_x-look_ahead_x)**2+(obs_y-look_ahead_y)**2)
        print('obs_x, obs_y', obs_x, obs_y)
        print('obs_dis < skip_dis', obs_dis, skip_dis)
        # if the target point is less than the safe distance, we need to skip that area
        if obs_dis < skip_dis:
            for i in range(target_index, len(self._waypointsx)):
                dis_to_point = np.sqrt((self._waypointsx[i]-position_x)**2+(self._waypointsy[i]-position_y)**2)
                if dis_to_point >= C.LHD+2*skip_dis:
                    target_index =i
                    print('skip:', target_index)
                    break

    return target_index