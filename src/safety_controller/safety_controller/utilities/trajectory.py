import pandas as pd
from safety_controller.CurvesGenerator import cubic_spline as cs
import numpy as np
from safety_controller.utilities import functions
import os
import sys
from ament_index_python.packages import get_package_share_directory


class Trajectory:

    def __init__(self, para):

        path = para.get("path")
        dt = para.get("sample")
        v_ref = para.get("V_REF", [])

        # Process V_REF
        for speed in v_ref:
            V_REF = speed

        self.traj_file_path = os.path.join(get_package_share_directory("common"), path)

        # read trajectory and finalized local coordinate waypoints
        self.waypoints_lla = pd.read_csv(
            self.traj_file_path, header=None
        )  # read the reference lat and lon here

        wps_lat = self.waypoints_lla[0]
        wps_lon = self.waypoints_lla[1]
        wps_alt = self.waypoints_lla[2]

        AX, AY = functions.lla_to_enu(
            wps_lat, wps_lon, wps_alt, wps_lat[0], wps_lon[0], wps_alt[0]
        )
        RefX, RefY, RefYaw, _, _ = cs.calc_spline_course(AX, AY, ds=dt)
        ref_velocity = V_REF  # unit: m/s
        RefVel = [ref_velocity] * len(RefX)

        self.origin = [wps_lat[0], wps_lon[0], wps_alt[0]]
        self.wpt_ref = [RefX, RefY, RefYaw, RefVel]

        wps_lat = self.waypoints_lla[0]
        wps_lon = self.waypoints_lla[1]
        wps_alt = self.waypoints_lla[2]

        AX, AY = functions.lla_to_enu(
            wps_lat, wps_lon, wps_alt, wps_lat[0], wps_lon[0], wps_alt[0]
        )
        RefX, RefY, RefYaw, _, _ = cs.calc_spline_course(AX, AY, ds=dt)
        ref_velocity = V_REF  # unit: m/s
        RefVel = [ref_velocity] * len(RefX)

        self.origin = [wps_lat[0], wps_lon[0], wps_alt[0]]
        self.wpt_ref = [RefX, RefY, RefYaw, RefVel]
