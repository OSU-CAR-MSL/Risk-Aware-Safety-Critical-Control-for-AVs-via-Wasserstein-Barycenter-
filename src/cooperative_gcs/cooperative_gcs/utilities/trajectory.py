import pandas as pd
import CurvesGenerator.cubic_spline as cs
import numpy as np
from utilities import functions
import os
import sys
from ament_index_python.packages import get_package_share_directory

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, ".."))
sys.path.append(parent_dir)


class Trajectory:

    def __init__(self, para):

        self.path = para.get("path")
        self.dt = para.get("sample")
        self.v_ref = para.get("V_REF", [])

        # Process V_REF
        for speed in self.v_ref:
            self.V_REF = speed
        self.traj_file_path = os.path.join(
            get_package_share_directory("cooperative_gcs"), self.path
        )

    def waypoint_gen(self):
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
        RefX, RefY, RefYaw, _, _ = cs.calc_spline_course(AX, AY, ds=self.dt)
        ref_velocity = self.V_REF  # unit: m/s
        RefVel = [ref_velocity] * len(RefX)

        self.origin = [wps_lat[0], wps_lon[0], wps_alt[0]]
        wpt_ref = [RefX, RefY, RefYaw, RefVel]
        return wpt_ref
