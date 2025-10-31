from std_msgs.msg import Float64MultiArray
from std_msgs.msg import String
import numpy as np
import math
from safety_controller.utilities import functions
from safety_controller.utilities import trajectory as tr
import time


class VehicleControl:

    def _parameter_assign(self, all_para):

        self.para = all_para

        self.N = all_para.mpc_params.get("N")
        self.LHD = all_para.mpc_params.get("LHD")
        gps_setting = all_para.vehicle_params.get("GPS", {})
        self.sample_nums = gps_setting["samples"]

    def __init__(
        self,
        all_para,
        mpc,
        sce,
        gps_x,
        gps_y,
        curr_yawd,
        curr_speed,
        Csteer,
        Cacc,
        Cbrake,
        run_time,
    ):

        self._parameter_assign(all_para)

        self.ctrl = mpc
        self.obs = sce
        self.numobs = sce.numofobs
        # get current gps and transform coordination and heading
        self.current_lat = gps_x
        self.current_lon = gps_y
        self.current_speed = curr_speed
        self.Csteer = Csteer
        self.Cacc = Cacc
        self.Cbrake = Cbrake
        self.current_yaw = curr_yawd

        # trajectory initial
        self.traj = tr.Trajectory(self.para.trajectory_params)
        self._waypoints = self.traj.wpt_ref
        self._waypointsx = self.traj.wpt_ref[0]
        self._waypointsy = self.traj.wpt_ref[1]
        self._waypointsyaw = self.traj.wpt_ref[2]
        self._waypointsv = self.traj.wpt_ref[3]

        # debugging plot
        self._waypoints_iterator = -1

        # run_time count
        self.run_time = run_time

    def _update_publisher(self, publisher):
        self.infopub = publisher
        time.sleep(0.1)
        self.pub_config()

    def pub_config(self):

        # msg = Float64MultiArray()
        # length = len(self._waypointsx)

        # # Create the matrix
        # matrix = [
        #     [float(self._waypointsx[i]), float(self._waypointsy[i]), float(self._waypointsyaw[i]), float(self._waypointsv[i])]
        #     for i in range(length)
        # ]
        # flattened_data = [item for row in matrix for item in row]

        # msg.data = flattened_data

        config = {}
        config["run_time"] = self.run_time
        config["vehicle para"] = self.para.vehicle_params
        config["trajectory para"] = self.para.trajectory_params
        config["controller para"] = self.para.controller_params
        config["dynamics para"] = self.para.dynamics_params
        config["mpc para"] = self.para.mpc_params
        config["cbf para"] = self.para.cbf_params
        config["scenario para"] = self.para.scenarios
        config["sensors para"] = self.para.sensors

        config_msg = self.infopub.dict_msg(config)

        if self.infopub:
            time.sleep(0.1)
            self.infopub.config_publisher.publish(config_msg)
            # time.sleep(0.1)
            # self.infopub.trajectory_publisher.publish(msg)

    def _update_sensors(self, sensors_list):

        self.sensors_list = sensors_list  # Reset or init the dict

    def _update_wb(self, obj):

        self.wb_obj = obj  # Reset or init the dict

    def _adjust_target_position(self, position_x: float, position_y: float, pre_iter):
        """
        Override the target position if the vehicle is "close enough" (threshold check passes).
        """

        dis = np.sqrt(
            (self._waypointsx - position_x) ** 2 + (self._waypointsy - position_y) ** 2
        )
        target_index = np.argmin(dis)
        if pre_iter < 0:
            pre_iter = target_index
        for i in range(target_index, len(self._waypointsx)):
            dis_to_point = np.sqrt(
                (self._waypointsx[i] - position_x) ** 2
                + (self._waypointsy[i] - position_y) ** 2
            )
            # dis_to_end = np.sqrt((self._waypointsx[i]-self._waypointsx[-1])**2+(self._waypointsy[i]-self._waypointsy[-1])**2)
            # if dis_to_end < self.LHD:
            #     target_index = -1
            # else:
            #     if dis_to_point >= self.LHD:
            #         target_index =i
            #         break
            if dis_to_point >= self.LHD:
                target_index = i
                break

        # Get look ahead point coordinate
        look_ahead_x = self._waypointsx[self._waypoints_iterator]
        look_ahead_y = self._waypointsy[self._waypoints_iterator]

        return target_index

    # def _adjust_target_position_obs(self, position_x: float, position_y: float, pre_iter):
    #     """
    #     Override the target position if the vehicle is "close enough" (threshold check passes).
    #     """
    #     dis = np.sqrt((self._waypointsx-position_x)**2+(self._waypointsy-position_y)**2)
    #     skip_dis = 2+1.8+3
    #     target_index = np.argmin(dis)
    #     for i in range(target_index, len(self._waypointsx)):
    #         dis_to_point = np.sqrt((self._waypointsx[i]-position_x)**2+(self._waypointsy[i]-position_y)**2)
    #         if dis_to_point >=10:
    #             target_index =i
    #             break

    #     if True:
    #         # Get look ahead point coordinate
    #         look_ahead_x = self._waypointsx[target_index]
    #         look_ahead_y = self._waypointsy[target_index]
    #         obs_dis = np.sqrt(((-25)-look_ahead_x)**2+(43-look_ahead_y)**2)
    #         # if the target point is less than the safe distance, we need to skip that area
    #         if obs_dis < skip_dis:
    #             for i in range(target_index, len(self._waypointsx)):
    #                 dis_to_point = np.sqrt((self._waypointsx[i]-position_x)**2+(self._waypointsy[i]-position_y)**2)
    #                 if dis_to_point >= 10+2*skip_dis:
    #                     target_index =i
    #                     break

    #     return target_index

    def run_controller(
        self,
        acc_prev,
        steer_prev,
        cur_lat,
        cur_lon,
        cur_yaw,
        etime,
        true_vehicle_states,
        vehicle_noisy_samples,
    ):

        # current_yaw
        tmp = functions.N2E(cur_yaw, 0)
        tmp = functions.pi_2_pi(tmp)

        # coordinate transform - lla2enu
        self._position_x, self._position_y = functions.lla_to_enu(
            cur_lat,
            cur_lon,
            self.traj.origin[2],
            self.traj.origin[0],
            self.traj.origin[1],
            self.traj.origin[2],
        )

        x = self._position_x
        y = self._position_y
        yaw = tmp
        v = self.current_speed.value
        vehicle_current_states = [x, y, yaw, v]

        true_obs_pos = self.obs.update(vehicle_current_states, etime)
        obs_state_msg = Float64MultiArray()
        obs_state_msg.data = true_obs_pos.flatten().tolist()
        if self.infopub:
            self.infopub.obstacle_publisher.publish(obs_state_msg)

        """
        """
        print("True obstacle position: ", true_obs_pos)
        obs_newstates = true_obs_pos.copy()

        lidar_data = self.sensors_list["lidar"].samples_from_ricedis(
            self.numobs, obs_newstates, self.sample_nums
        )
        lidar_mean = np.mean(lidar_data[0, 0, :, :], axis=0)

        camera_data = self.sensors_list["camera"].samples_from_laplacedis(
            self.numobs, obs_newstates, self.sample_nums
        )
        camera_mean = np.mean(camera_data[0, 0, :, :], axis=0)

        v2x_data = self.sensors_list["v2x"].samples_from_gammadis(
            self.numobs, obs_newstates, self.sample_nums
        )
        v2x_mean = np.mean(v2x_data[0, 0, :, :], axis=0)

        x_sensor = [lidar_data, camera_data, v2x_data]
        b_sensor = [
            self.sensors_list["lidar"].weight,
            self.sensors_list["camera"].weight,
            self.sensors_list["v2x"].weight,
        ]

        # WB obstacle position for CVaR
        opti_meas = self.wb_obj.get_freesupport_wb(x_sensor, b_sensor)
        opti_meas = np.squeeze(opti_meas, axis=0)  # shape = (10, 2)

        # comment out this when CVaR implement
        # opti_meas = np.stack([lidar_mean, camera_mean, v2x_mean], axis=0)
        ##############################
        # mean obstacle position for QP
        obs_stacked = np.stack([lidar_mean, camera_mean, v2x_mean], axis=0)
        mean_obs_pos = np.mean(obs_stacked, axis=0)
        obs_newstates[:, :2] = mean_obs_pos[:2]
        """
        """

        self._waypoints_iterator = self._adjust_target_position(
            x, y, self._waypoints_iterator
        )

        target_x = self._waypointsx[
            self._waypoints_iterator : self._waypoints_iterator + self.N
        ]
        target_y = self._waypointsy[
            self._waypoints_iterator : self._waypoints_iterator + self.N
        ]
        target_yaw = functions.calc_pps_steering(x, y, yaw, target_x, target_y)
        target_v = self._waypointsv[
            self._waypoints_iterator : self._waypoints_iterator + self.N
        ]

        # update target x, y first and then update initial condition,
        # because the yaw error is calculated based on current pos and target x, y
        target_states = [target_x, target_y, target_yaw, target_v]
        prev_input = [acc_prev, steer_prev]

        if len(target_x) < self.N:
            padding_x = np.full((self.N - len(target_x),), target_x[-1])
            target_x = np.concatenate((target_x, padding_x))
            padding_y = np.full((self.N - len(target_y),), target_y[-1])
            target_y = np.concatenate((target_y, padding_y))
            padding_yaw = np.full((self.N - len(target_yaw),), target_yaw[-1])
            target_yaw = np.concatenate((target_yaw, padding_yaw))
            padding_v = np.full((self.N - len(target_v),), target_v[-1])
            target_v = np.concatenate((target_v, padding_v))
            target_states = [target_x, target_y, target_yaw, target_v]

        # update_n_solve(mpc_target, vehicle_noisy_states, obs_mean_qp, prev_control_input_mpc, opti_sensor_samples, vehicle_noisy_samples)
        sol_dict = self.ctrl.update_n_solve(
            target_states,
            vehicle_current_states,
            obs_newstates,
            prev_input,
            opti_meas,
            vehicle_noisy_samples,
        )

        # solution from mpc
        acc = sol_dict["u_control"][0]
        steer = sol_dict["u_control"][1]
        # prediction point
        z_predx = sol_dict["z_mpc"][self.N - 1][0]
        z_predy = sol_dict["z_mpc"][self.N - 1][1]

        rt_dict = {}
        rt_dict["current_lla"] = [cur_lat, cur_lon, cur_yaw, self.current_speed.value]
        rt_dict["true_vehicle"] = true_vehicle_states
        rt_dict["current_enu"] = vehicle_current_states
        rt_dict["target"] = [target_x[0], target_y[0], target_yaw[0], target_v[0]]
        rt_dict["output"] = np.array([acc_prev, steer_prev]).tolist()
        rt_dict["prediction_pos"] = [z_predx, z_predy]
        rt_dict["obs_true_states"] = true_obs_pos[0, :]
        rt_dict["obs_states"] = obs_newstates[0, :]
        rt_dict["obs_wb_states"] = opti_meas
        rt_dict["lidar_mean"] = lidar_mean
        rt_dict["camera_mean"] = camera_mean
        rt_dict["v2x_mean"] = v2x_mean

        if self.infopub:
            rtmsg = self.infopub.dict_msg(rt_dict)
            self.infopub.rtdict_publisher.publish(rtmsg)
            msg = self.infopub.dict_msg(sol_dict)
            self.infopub.mpc_publisher.publish(msg)

        # print('MPC output acc and steer(radius), degree: ',acc, steer, sol_dict['optimal'])

        throttle_scaling_factor = 900
        brake_scaling_factor = 1

        if acc >= 0:
            throttle_command = acc * throttle_scaling_factor  # Send throttle
            brake_command = 0  # No brake needed
        else:
            throttle_command = 0  # No throttle
            brake_command = acc * brake_scaling_factor

        # steering_command = math.degrees(delta)*16
        steering_command = math.degrees(steer) * 14

        if steering_command > 540:
            steering_command = min(540, steering_command)
        if steering_command < -540:
            steering_command = max(-540, steering_command)

        # indoor testing
        # self.Csteer.value = 0
        # self.Cacc.value = 0
        # self.Cbrake.value = 0
        self.Csteer.value = int(steering_command)
        self.Cbrake.value = brake_command
        self.Cacc.value = throttle_command

        return self.Csteer.value, self.Cacc.value, self.Cbrake.value, acc, steer
