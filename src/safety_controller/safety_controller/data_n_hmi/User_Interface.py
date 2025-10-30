import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import matplotlib.pyplot as plt
from std_msgs.msg import String
import json
from datetime import datetime
import pandas as pd
import matplotlib.patches as patches
import numpy as np	
import os
import math
import threading
import yaml

import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(parent_dir)

from utilities.configprocessor import ConfigProcessor
import utilities.trajectory as trajectory

first_run = 0

class MPCSubscriber(Node):

    def _parameter_assign(self, all_para):

        self.para  = all_para

        self.N = all_para.mpc_params.get("N")
        self.DT = all_para.mpc_params.get("DT")
        self.LHD = all_para.mpc_params.get("LHD")

        self.LF = all_para.vehicle_params.get("L_F")
        self.LR = all_para.vehicle_params.get("L_R")

        cbfpara = all_para.mpc_params.get("cbf", {})
        self.SAFETY_DIS = cbfpara.get("SAFETY_DIS", {})
        self.VEHICLE_R = cbfpara.get("VEHICLE_R", {})

        current_dir = os.path.dirname(os.path.abspath(__file__))
        path_name = all_para.trajectory_params.get("path", {})
        path_name = os.path.join(parent_dir, path_name)
        all_para.trajectory_params["path"] = path_name
        self.traj = trajectory.Trajectory(all_para.trajectory_params)


        scenarios = all_para.scenarios
        for scenario in scenarios:
            self.numofobs = scenario.get("numofobs", 0)
            self.obstacles = scenario.get("params", {})  # Get the parameters of the scenario
            self.SAFETY_DIS = scenario.get("SAFETY_DIS", None)

    def __init__(self):

        config_path = os.path.join(parent_dir, 'utilities', 'config_IFAC.yaml')
        with open(config_path, "r") as file:
            configfile = yaml.safe_load(file)
            config = ConfigProcessor(configfile)

        self._parameter_assign(config)


        self.lat = []
        self.lon = []
        self.num_peds = 0
        # self.timestamp = []

        self.first_run = 0

        lane = True

        print("Once%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")


        super().__init__('waypoints_mpc_follower_sub')

        self.config_subscription = self.create_subscription(
            String,
            'config_data',
            self.config_callback,
            10
        )

        self.rtdict_subscription = self.create_subscription(
            String,
            'rtdict_data',
            self.rtdict_callback,
            10
        )

        # Subscribe to control data
        self.mpc_subscription = self.create_subscription(
            String,
            'mpc_data',
            self.mpcdata_callback,
            10
        )



    def config_callback(self, msg):

        data = json.loads(msg.data)

        if len(data) > 1:

            self.controller_type =data.get('controller para', [])
            self.scename = [scenario["name"] for scenario in data.get("scenario para", []) if "name" in scenario]
            self.v_ref = data.get("trajectory para", {}).get("V_REF", [])
            self.run_time = data.get('run_time')
            self.test_name = f"{self.scename}_{self.run_time}_{self.controller_type}_{self.v_ref}"   

            self.trajectory_data = self.traj.wpt_ref

            self.init_plot()
            self.plot_traj_data()
            self.save_config_data(data)
            print('config save n replot!')

    def rtdict_callback(self, msg):
        data = json.loads(msg.data)
        if len(data) > 1:
            self.save_rtdict_data(data)
            print('rtdict save!')

            current_enu =data.get('current_enu', [])

            target = data.get('target', []) 
            prediction_pos = data.get('prediction_pos', [])
            obs_pos =  data.get('obs_states', [])
            obs_pos = np.array(obs_pos) if isinstance(obs_pos, list) else obs_pos

            x = current_enu[0]
            y = current_enu[1]

            vel = current_enu[3]
            vel = round(vel, 2)

            tar_x =  target[0]
            tar_y =  target[1]

            pred_x = prediction_pos[0]
            pred_y = prediction_pos[1]

            # Update x,y position based on the input from the controller 
            if (self.first_run == 0):
                self.x_i =  current_enu[0]
                self.y_i =  current_enu[1]
                self.yaw_i =  current_enu[2]
                self.vel_i =  current_enu[3]
                self.first_run = 1
                print("Initial state established")

            if obs_pos.ndim > 0 and obs_pos.size > 0:
                obs_x = obs_pos[::4]  # Extract every 4th column starting from index 0 (x positions)
                obs_y = obs_pos[1::4]  # Extract every 4th column starting from index 1 (y positions)
                obs_r = obs_pos[3::4]
            else:
                obs_x = np.array([])  # Return an empty NumPy array for consistency
                obs_y = np.array([])
                obs_r = np.array([])

            self.update_curr_plot(x, y, vel, tar_x, tar_y, pred_x, pred_y, obs_x, obs_y, obs_r)


    def mpcdata_callback(self, msg):
        data = json.loads(msg.data)
        if len(data) > 1:
            self.save_mpc_data(data)
            print('mpcdata save!')


    def init_plot(self):
        # Set up the plots

        plt.close('all')

        self.trajectory_data = []
        self.rtinfo_data = []
        self.mpc_data = []
  
        self.fig, (self.ax, self.zoom_ax) = plt.subplots(1, 2, figsize=(18, 6))

        num_sides = 6
        angles = np.linspace(0, 2 * np.pi, num_sides, endpoint=False)
        hexagon_x_template_small = np.append(np.cos(angles), np.cos(angles[0]))
        hexagon_y_template_small = np.append(np.sin(angles), np.sin(angles[0]))
        hexagon_x_template_big = 20.7*np.append(np.cos(angles), np.cos(angles[0]))
        hexagon_y_template_big = 1.7*np.append(np.sin(angles), np.sin(angles[0]))
        
        
        # stop_sign_positions = [(-5, -5), (-200, 6), (-275, 1)]        
        # yield_sign_positions = [(-948.525, 50.798), (-1307.752, 115.475), (-1310.864, 101.037)]

        # Main Plot layers

        self.ax_veh_circle = patches.Circle((0, 0), self.VEHICLE_R, color='orange', alpha=0.9, fill=True, label='Ego vehicle')
        self.trajectory_line, = self.ax.plot([], [], 'b-', label='Trajectory')  # Trajectory
        self.left_lane_line, = self.ax.plot([], [], color='Yellow', linestyle='--', label='Lane Line', linewidth=2)
        self.right_lane_line, = self.ax.plot([], [], color='yellow', linestyle='--', linewidth=2)  
        self.left_left_lane_line, = self.ax.plot([], [], color='black', linestyle='-', label='Lane Line', linewidth=2)
        self.right_right_lane_line, = self.ax.plot([], [], color='black', linestyle='-', linewidth=2)  
        self.current_position, = self.ax.plot([], [], 'r^', markersize=7, label='Current Position') 
        self.target_position, = self.ax.plot([], [], 'go', label='Target Position') 
        self.pred_position, = self.ax.plot([], [], 'o',  color='purple', label='MPC Prediction Position')  
        self.lane_center, = self.ax.plot([], [], 'o',  color='orange', markersize=2) 
        self.legend_proxy_circle = plt.Circle((0, 0), 0, color='grey', alpha=0.9, fill=True, label="Obstacle radius")
        self.ctrl_text_obj = self.ax.text(0.01, 0.98, "Controller: ", transform=self.ax.transAxes,
                                fontsize=12, verticalalignment='top', 
                                horizontalalignment='left', color='red', fontweight='bold')
        self.ctrl_text_obj.set_text('Controller: '+ str(self.controller_type))
        
        self.ax.add_patch(self.legend_proxy_circle)
        # for center in stop_sign_positions:
        #     hexagon_x = hexagon_x_template_big + center[0]
        #     hexagon_y = hexagon_y_template_big + center[1]
        #     self.ax.fill(hexagon_x, hexagon_y, color='red', linewidth=2)
        #     self.ax.text(center[0], center[1], 'STOP', color='black', fontsize=5, ha='center', va='center', fontweight='bold')

        # for center in yield_sign_positions:
        #     self.ax.plot(center[0], center[1], 'y^', markersize=7)
        #     self.ax.text(center[0], center[1], 'YIELD', color='black', fontsize=5, ha='center', va='center', fontweight='bold')

        # Initialize obstacle markers (for dynamic obstacles)
        self.obs_position, = self.ax.plot([], [], 'ko', label='Obstacle Position')  

        # Initialize lists for dynamic obstacle circles and paths
        self.ax_obs_circles = []  # List of obstacle circles
        self.ax_obs_lines = []  # List of obstacle trajectory lines

        self.ax.add_patch(self.ax_veh_circle)
        for circle in self.ax_obs_circles:
            self.ax.add_patch(circle)          
        self.ax.set_xlabel('X - East (m)')
        self.ax.set_ylabel('Y - North (m)')
        self.ax.set_title('Real-Time Vehicle Position')
        self.ax.legend(loc='upper right')
        self.ax.grid()

        # Zoomed Plot layers
        self.zax_veh_circle = patches.Circle((0, 0), self.VEHICLE_R, color='orange', alpha=0.9, fill=True, label='Ego vehicle')
        self.zax_obs_circle = patches.Circle((0, 1), 1, color='grey', alpha=0.9, fill=True, label='Obstacle radius')
        self.zoom_trajectory_line, = self.zoom_ax.plot([], [], 'b-', label='Trajectory')
        self.zoom_left_lane_line, = self.zoom_ax.plot([], [], color='Yellow', linestyle='--', label='Lane Line', linewidth=2)
        self.zoom_right_lane_line, = self.zoom_ax.plot([], [], color='yellow', linestyle='--', linewidth=2)
        self.zoom_left_left_lane_line, = self.zoom_ax.plot([], [], color='black', linestyle='-', label='Lane Line', linewidth=2)
        self.zoom_right_right_lane_line, = self.zoom_ax.plot([], [], color='black', linestyle='-', linewidth=2)
        self.zoom_current_position, = self.zoom_ax.plot([], [], 'r^', markersize=7, label='Current Position')
        self.zoom_target_position, = self.zoom_ax.plot([], [], 'go', label='Target Position')
        self.zoom_pred_position, = self.zoom_ax.plot([], [], 'o', color='purple', label='MPC Prediction Position')
        self.zoom_obs_position, = self.zoom_ax.plot([], [], 'ko', label='Obstacle')
        self.text_obj = self.zoom_ax.text(0.01, 0.98, "Vehicle speed in m/s", transform=self.zoom_ax.transAxes,
                                     fontsize=12, verticalalignment='top', 
                                     horizontalalignment='left', color='red', fontweight='bold')
        
        self.legend_proxy_circle = plt.Circle((0, 0), 0, color='grey', alpha=0.9, fill=True, label="Obstacle radius")
        self.zoom_ax.add_patch(self.legend_proxy_circle)
        # Initialize obstacle markers in zoomed plot
        self.zoom_obs_position, = self.zoom_ax.plot([], [], 'ko', label='Obstacle')

        # Initialize lists for dynamic obstacles in zoomed plot
        self.zax_obs_circles = []
        self.zax_obs_lines = []

        self.zoom_ax.add_patch(self.zax_veh_circle)
        for circle in self.zax_obs_circles:
            self.zoom_ax.add_patch(circle)   
        #

        # for center in stop_sign_positions:
        #     hexagon_x = hexagon_x_template_small + center[0]
        #     hexagon_y = hexagon_y_template_small + center[1]
        #     self.zoom_ax.fill(hexagon_x, hexagon_y, color='red', linewidth=2)
        #     self.zoom_ax.text(center[0], center[1], 'STOP', color='white', fontsize=7, ha='center', va='center', fontweight='bold')

        # for center in yield_sign_positions:
        #     self.zoom_ax.plot(center[0], center[1], 'y^', markersize=7)
        #     self.zoom_ax.text(center[0], center[1], 'YIELD', color='white', fontsize=5, ha='center', va='center', fontweight='bold')

        self.zoom_ax.set_xlim(-15, 15)
        self.zoom_ax.set_ylim(-15, 15)
        self.zoom_ax.set_xlabel('X - East (m)')
        self.zoom_ax.set_ylabel('Y - North (m)')
        self.zoom_ax.set_title('15-Meter Zoomed View Around Ego Vehicle')
        self.zoom_ax.legend(loc='upper right')
        self.zoom_ax.grid()

        self.plotonce = 0

        print("Plot Initialized!")


    def plot_traj_data(self):

        plt.ion()  # Ensures the figure is displayed until closed manually  # Enable interactive mode
        if len(self.trajectory_data) < 1:
            self.trajectory_data = self.traj.wpt_ref
        if len(self.trajectory_data) > 1:

            x = self.trajectory_data[0]  # Extract first column
            y = self.trajectory_data[1]  # Extract first column

            # Update trajectory line
            self.trajectory_line.set_data(x, y)
            self.zoom_trajectory_line.set_data(x, y)

            dx = np.diff(x)
            dy = np.diff(y)
            lane_width = 1.75

            length = np.sqrt(dx**2 + dy**2)
            dx = dx / length
            dy = dy / length

            offset_xl = -dy * lane_width
            offset_yl = dx * lane_width

            offset_xll = -dy * 5.25
            offset_yll = dx * 5.25

            x_left = x[:-1] + offset_xl
            y_left = y[:-1] + offset_yl
            x_right = x[:-1] - offset_xl
            y_right = y[:-1] - offset_yl

            x_left_left = x[:-1] + offset_xll
            y_left_left = y[:-1] + offset_yll
            x_right_right = x[:-1] - offset_xll
            y_right_right = y[:-1] - offset_yll


            self.left_lane_line.set_data(x_left, y_left)
            self.right_lane_line.set_data(x_right, y_right)
            self.zoom_left_lane_line.set_data(x_left, y_left)
            self.zoom_right_lane_line.set_data(x_right, y_right)

            self.left_left_lane_line.set_data(x_left_left, y_left_left)
            self.right_right_lane_line.set_data(x_right_right, y_right_right)
            self.zoom_left_left_lane_line.set_data(x_left_left, y_left_left)
            self.zoom_right_right_lane_line.set_data(x_right_right, y_right_right)
            

            self.ax.relim()  # Recalculate limits
            self.ax.autoscale_view()

            plt.draw()
            plt.pause(0.01)

    def update_curr_plot(self, x, y, vel, target_x, target_y, pred_x, pred_y, obs_x, obs_y, obs_r):
        # Update main plot

        self.current_position.set_data([x], [y])
        self.ax_veh_circle.set_center((x, y))
        self.target_position.set_data([target_x], [target_y])
        self.pred_position.set_data([pred_x], [pred_y])
        # self.obs_position.set_data([obs_x], [obs_y])
        # self.ax_obs_circle.set_center((obs_x, obs_y))
        
            
        # Update zoomed plot
        self.zoom_ax.set_xlim(x - 15, x + 15)
        self.zoom_ax.set_ylim(y - 15, y + 15)
        self.zoom_current_position.set_data([x], [y])
        self.zax_veh_circle.set_center((x, y))
        self.zoom_target_position.set_data([target_x], [target_y])
        self.zoom_pred_position.set_data([pred_x], [pred_y])
        self.text_obj.set_text('Vehicle speed: '+ str(vel)+'m/s')
        # self.zoom_obs_position.set_data([obs_x], [obs_y])
        # self.zax_obs_circle.set_center((obs_x, obs_y))
    

        # Update obstacle positions
        self.obs_position.set_data(obs_x, obs_y)
        self.zoom_obs_position.set_data(obs_x, obs_y)
        # Ensure enough circles exist
        if len(self.ax_obs_circles) != len(obs_x):
            # Remove old circles
            for circle in self.ax_obs_circles:
                circle.remove()
            self.ax_obs_circles = [plt.Circle((x, y), r, color='grey', alpha=0.9, fill=True) for x, y, r in zip(obs_x, obs_y, obs_r)]
            for circle in self.ax_obs_circles:
                self.ax.add_patch(circle)
        else:
            # Update existing circles
            for i, circle in enumerate(self.ax_obs_circles):
                circle.set_center((obs_x[i], obs_y[i]))
                circle.set_radius(obs_r[i])
        
        if len(self.zax_obs_circles) != len(obs_x):
            # Remove old circles
            for circle in self.zax_obs_circles:
                circle.remove()
            self.zax_obs_circles = [plt.Circle((x, y), r, color='grey', alpha=0.9, fill=True) for x, y, r in zip(obs_x, obs_y, obs_r)]
            for circle in self.zax_obs_circles:
                self.zoom_ax.add_patch(circle)
        else:
            # Update existing circles
            for i, circle in enumerate(self.zax_obs_circles):
                circle.set_center((obs_x[i], obs_y[i]))
                circle.set_radius(obs_r[i])

        plt.draw()
        plt.pause(0.01)


    def save_config_data(self, data):
        folder_path = f'output data/{self.scename}'
        self.ensure_directory_exists(folder_path)  # Create folder if not exist
        file_name = os.path.join(folder_path, f'config_{self.test_name}.json')
        if os.path.exists(file_name):
            with open(file_name, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        if isinstance(existing_data, list):
            existing_data.append(data)

        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def save_rtdict_data(self, data):
        folder_path = f'output data/{self.scename}'
        self.ensure_directory_exists(folder_path)  # Create folder if not exist
        file_name = os.path.join(folder_path, f'rt_data_{self.test_name}.json')
        if os.path.exists(file_name):
            with open(file_name, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        if isinstance(existing_data, list):
            existing_data.append(data)

        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def save_mpc_data(self, data):
        folder_path = f'output data/{self.scename}'
        self.ensure_directory_exists(folder_path)  # Create folder if not exist
        file_name = os.path.join(folder_path, f'mpc_data_{self.test_name}.json')
        if os.path.exists(file_name):
            with open(file_name, 'r') as file:
                existing_data = json.load(file)
        else:
            existing_data = []

        if isinstance(existing_data, list):
            existing_data.append(data)

        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def ensure_directory_exists(self, directory):
        """Ensure the target directory exists, if not create it."""
        if not os.path.exists(directory):
            os.makedirs(directory)


def main(args=None):
    rclpy.init(args=args)
    subscriber = MPCSubscriber()
    rclpy.spin(subscriber)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
