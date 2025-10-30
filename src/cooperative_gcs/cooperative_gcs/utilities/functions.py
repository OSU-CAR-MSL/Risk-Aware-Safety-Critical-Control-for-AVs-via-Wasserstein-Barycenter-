import numpy as np	
import math

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