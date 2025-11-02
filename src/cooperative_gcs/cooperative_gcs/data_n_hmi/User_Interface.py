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

VEHICLE_COLORS = ['green', 'red', 'teal', 'cyan', 'magenta', 'lime', 'pink', 'brown']

class DynamicMultiVehicleMPCSubscriber(Node):

    def __init__(self):
        super().__init__('simulation_node')

        self.known_vehicles = set()  # Store detected vehicle IDs
        self.vehicle_subscriptions = {}  # Store dynamic subscriptions per vehicle
        self.vehicle_colors = {}  # Store color assignments for each vehicle
        self.vehicle_plots = {}  # Store plot elements for each vehicle
        self.vehicle_traj = {}  # Store plot elements for each vehicle
        self.zoom_fig = None  # Add new figure for zoomed views
        self.first_run = {}  # Store first run flag per vehicle
        self.color_index = 0  # To cycle through VEHICLE_COLORS
        # Subscribe to a common config topic for all vehicles
        self.start_publishers = {}
        self.expected_vehicles = 2  # Configurable: number of vehicles to wait for
        
        self.config_subscription = self.create_subscription(
            String,
            'config_data',
            self.config_callback,
            10
        )

        config_path = os.path.join(parent_dir, 'utilities', 'config.yaml')
        with open(config_path, "r") as file:
            configfile = yaml.safe_load(file)
            config = ConfigProcessor(configfile)

        self._parameter_assign(config)

        self.get_logger().info("Listening for vehicle configurations...")


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
        # path_name = all_para.trajectory_params.get("path", {})
        # path_name = os.path.join(parent_dir, path_name)
        # all_para.trajectory_params["path"] = path_name
        # print("path")
        # self.traj = trajectory.Trajectory(all_para.trajectory_params)


        scenarios = all_para.scenarios
        for scenario in scenarios:
            self.numofobs = scenario.get("numofobs", 0)
            self.obstacles = scenario.get("params", {})  # Get the parameters of the scenario
            self.SAFETY_DIS = scenario.get("SAFETY_DIS", None)


    def config_callback(self, msg):

        data = json.loads(msg.data)
        # print("config callback")
        if len(data) > 1:

            vehicle_id = data.get("vehicle para", {}).get("VEHICLE_ID", [])
            # data.get('vehicle_id')  # Extract vehicle ID

            if vehicle_id is None:
                self.get_logger().warn("Received config message without a vehicle_id.")
                return

            if vehicle_id in self.known_vehicles:
                # self.get_logger().info(f"Config received for known vehicle {vehicle_id}.")
                return

            self.get_logger().info(f"New vehicle detected: {vehicle_id}. Creating topic subscriptions...")

            self.known_vehicles.add(vehicle_id)  # Add to known vehicles
            namespace = f"/vehicle_{vehicle_id}"

            self.vehicle_colors[vehicle_id] = VEHICLE_COLORS[self.color_index % len(VEHICLE_COLORS)]
            self.color_index += 1
            self.first_run[vehicle_id] = 0  # Initialize first_run flag for this vehicle
            
            # Create subscriptions for this vehicle
            self.vehicle_subscriptions[vehicle_id] = {
                "rtdict": self.create_subscription(
                    String,
                    f'{namespace}/rtdict_data',
                    lambda msg, vid=vehicle_id: self.rtdict_callback(msg, vid),
                    10
                ),
                "mpc": self.create_subscription(
                    String,
                    f'{namespace}/mpc_data',
                    lambda msg, vid=vehicle_id: self.mpcdata_callback(msg, vid),
                    10
                )
            }

            # Create publisher for start command
            self.start_publishers[vehicle_id] = self.create_publisher(
                String,
                f'{namespace}/start_command',
                10
            )

            self.controller_type =data.get('controller para', [])
            self.scename = [scenario["name"] for scenario in data.get("scenario para", []) if "name" in scenario]
            self.v_ref = data.get("trajectory para", {}).get("V_REF", [])
            self.run_time = data.get('run_time',1)
            self.test_name = f"{self.scename}_{self.run_time}_{self.controller_type}_{self.v_ref}"

            trajectory_para = data.get("trajectory para", {})
            self.init_plot(vehicle_id)

            self.plot_traj(vehicle_id, trajectory_para)
            # print("trajectory_name", trajectory_name)
            # self.trajectory_data = self.traj.wpt_ref

            # Check if all expected vehicles are detected
            if len(self.known_vehicles) == self.expected_vehicles:
                self.get_logger().info("All expected vehicles detected, triggering prompt...")
                self.prompt_user_to_start()

            # self.plot_traj_data()
            self.save_config_data(data, vehicle_id)

    def plot_traj(self, vehicle_id, trajectory_para):

        plt.ion()  # Ensures the figure is displayed until closed manually  # Enable interactive mode
        print('trajectory_para: ', trajectory_para)
        traj = trajectory.Trajectory(trajectory_para)
        print('traj', traj)
        self.trajectory_data = traj.waypoint_gen()

        x = self.trajectory_data[0]  # Extract first column
        y = self.trajectory_data[1]  # Extract first column

        plots = self.vehicle_traj[vehicle_id]
        # Main plot
        plots['trajectory_line'].set_data([x], [y])


        # Update trajectory line
        # self.trajectory_line.set_data(x, y)
        # self.zoom_trajectory_line.set_data(x, y)

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

        plots['left_lane_line'].set_data(x_left, y_left)
        plots['right_lane_line'].set_data(x_right, y_right)
        plots['left_left_lane_line'].set_data(x_left_left, y_left_left)
        plots['right_right_lane_line'].set_data(x_right_right, y_right_right)
        # self.left_lane_line.set_data(x_left, y_left)
        # self.right_lane_line.set_data(x_right, y_right)
        # self.zoom_left_lane_line.set_data(x_left, y_left)
        # self.zoom_right_lane_line.set_data(x_right, y_right)

        # self.left_left_lane_line.set_data(x_left_left, y_left_left)
        # self.right_right_lane_line.set_data(x_right_right, y_right_right)
        # self.zoom_left_left_lane_line.set_data(x_left_left, y_left_left)
        # self.zoom_right_right_lane_line.set_data(x_right_right, y_right_right)
        

        self.ax.relim()  # Recalculate limits
        self.ax.autoscale_view()

        plt.draw()
        plt.pause(0.01)
        


    def rtdict_callback(self, msg, vehicle_id):

        data = json.loads(msg.data)

        if len(data) > 1:
            self.get_logger().info(f"[Vehicle {vehicle_id}] Real-time data received.")
            self.save_rtdict_data(data, vehicle_id)
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

            if self.first_run[vehicle_id] == 0:
                self.first_run[vehicle_id] = 1
                print(f"Initial state established for Vehicle {vehicle_id}")

            if obs_pos.ndim > 0 and obs_pos.size > 0:
                obs_x = obs_pos[::4]
                obs_y = obs_pos[1::4]
                obs_r = obs_pos[3::4]
            else:
                obs_x = np.array([])
                obs_y = np.array([])
                obs_r = np.array([])
            self.update_curr_plot(vehicle_id, x, y, vel, tar_x, tar_y, pred_x, pred_y, obs_x, obs_y, obs_r)

    def mpcdata_callback(self, msg, vehicle_id):
        data = json.loads(msg.data)
        if len(data) > 1:
            self.save_mpc_data(data, vehicle_id)
            print('mpcdata save!')


    def init_plot(self, vehicle_id):
        # Set up the plots
        if not hasattr(self, 'fig'):
            plt.close('all')
            self.fig = plt.figure(figsize=(18, 6 + 3 * len(self.known_vehicles)))

            self.ax = self.fig.add_subplot(1, 1, 1)  # Main plot takes full width initially
            self.trajectory_data = []
            self.rtinfo_data = []
            self.mpc_data = []

            # Main plot static elements

            self.obs_position, = self.ax.plot([], [], 'ko', label='Obstacle Position')
            self.ax_obs_circles = []
            self.ax.set_xlim(-120, 20)
            self.ax.set_ylim(-20, 140)
            self.ax.set_xlabel('X - East (m)')
            self.ax.set_ylabel('Y - North (m)')
            self.ax.set_title('Real-Time Vehicle Position')
            self.ax.legend(loc='upper right')
            self.ax.grid()
        # Set up the separate zoom figure
        # if not hasattr(self, 'zoom_fig'):
            self.zoom_fig = plt.figure(figsize=(12, 4 * len(self.known_vehicles)))
            self.zoom_axes = {}  # Dictionary to store zoom axes for each vehicle
  
        # Add a new zoomed subplot for this vehicle
        num_vehicles = len(self.known_vehicles)
        self.fig.set_size_inches(18, 6 + 3 * num_vehicles)  # Adjust height dynamically
        for ax in self.fig.axes[1:]:  # Remove old zoomed axes
            self.fig.delaxes(ax)

        # Initialize vehicle plot elements *before* assigning zoomed axes
        color = self.vehicle_colors[vehicle_id]

        self.vehicle_traj[vehicle_id] = {
            "trajectory_line": self.ax.plot([], [], 'b-', label='Trajectory')[0],
            "left_lane_line" : self.ax.plot([], [], color='yellow', linestyle='--', label='Lane Line', linewidth=2)[0],
            "right_lane_line" : self.ax.plot([], [], color='yellow', linestyle='--', label='Lane Line', linewidth=2)[0],
            "left_left_lane_line" : self.ax.plot([], [], color='black', linestyle='-', linewidth=2)[0],
            "right_right_lane_line" : self.ax.plot([], [], color='black', linestyle='-', linewidth=2)[0],
        }

        self.vehicle_plots[vehicle_id] = {
            # Main plot elements
            'current': self.ax.plot([], [], '^', color=color, markersize=7, label=f'Vehicle {vehicle_id} Current')[0],
            'target': self.ax.plot([], [], 'o', color=color, label=f'Vehicle {vehicle_id} Target')[0],
            'pred': self.ax.plot([], [], 's', color=color, label=f'Vehicle {vehicle_id} Prediction')[0],
            'circle': patches.Circle((0, 0), self.VEHICLE_R, color=color, alpha=0.5, fill=True),
        }

        self.ax.add_patch(self.vehicle_plots[vehicle_id]['circle'])
        self.zoom_fig.set_size_inches(12, 4 * num_vehicles)
        self.zoom_fig.clear()
        # Recreate zoomed subplots for all vehicles
        for i, vid in enumerate(sorted(self.known_vehicles)):  # Sort for consistent order
            zoom_ax = self.zoom_fig.add_subplot(num_vehicles, 1, i + 1)
            zoom_ax.set_xlim(-15, 15)
            zoom_ax.set_ylim(-15, 15)
            zoom_ax.set_xlabel('X - East (m)')
            zoom_ax.set_ylabel('Y - North (m)')
            zoom_ax.set_title(f'Zoomed View: Vehicle {vid}')
            zoom_ax.grid()

            # # Add zoomed plot elements to existing vehicle_plots entry
            # self.vehicle_plots[vid]['zoom_ax'] = zoom_ax
            # self.vehicle_plots[vid]['zoom_trajectory'] = zoom_ax.plot([], [], 'b-', label='Trajectory')[0]
            # self.vehicle_plots[vid]['zoom_current'] = zoom_ax.plot([], [], '^', color=self.vehicle_colors[vid], markersize=7)[0]
            # self.vehicle_plots[vid]['zoom_target'] = zoom_ax.plot([], [], 'o', color=self.vehicle_colors[vid])[0]
            # self.vehicle_plots[vid]['zoom_pred'] = zoom_ax.plot([], [], 's', color=self.vehicle_colors[vid])[0]
            # self.vehicle_plots[vid]['zoom_circle'] = patches.Circle((0, 0), self.VEHICLE_R, color=self.vehicle_colors[vid], alpha=0.5, fill=True)
            # self.vehicle_plots[vid]['zoom_obs'] = zoom_ax.plot([], [], 'ko', label='Obstacle')[0]
            # self.vehicle_plots[vid]['zoom_obs_circles'] = []
            # self.vehicle_plots[vid]['speed_text'] = zoom_ax.text(0.01, 0.98, f"Vehicle {vid} speed: ",
            #                                                      transform=zoom_ax.transAxes, fontsize=12,
            #                                                      verticalalignment='top', horizontalalignment='left',
            #                                                      color=self.vehicle_colors[vid], fontweight='bold')
            # zoom_ax.add_patch(self.vehicle_plots[vid]['zoom_circle'])
            # Add zoomed plot elements
            color = self.vehicle_colors[vid]
            self.vehicle_plots[vid].update({
                'zoom_ax': zoom_ax,
                'zoom_trajectory': zoom_ax.plot([], [], 'b-', label='Trajectory')[0],
                'zoom_current': zoom_ax.plot([], [], '^', color=color, markersize=7)[0],
                'zoom_target': zoom_ax.plot([], [], 'o', color=color)[0],
                'zoom_pred': zoom_ax.plot([], [], 's', color=color)[0],
                'zoom_circle': patches.Circle((0, 0), self.VEHICLE_R, color=color, alpha=0.5, fill=True),
                'zoom_obs': zoom_ax.plot([], [], 'ko', label='Obstacle')[0],
                'zoom_obs_circles': [],
                'speed_text': zoom_ax.text(0.01, 0.98, f"Vehicle {vid} speed: ",
                                        transform=zoom_ax.transAxes, fontsize=12,
                                        verticalalignment='top', horizontalalignment='left',
                                        color=color, fontweight='bold')
            })
            zoom_ax.add_patch(self.vehicle_plots[vid]['zoom_circle'])

        self.fig.tight_layout()
        self.zoom_fig.tight_layout()

        # Adjust main plot position
        self.ax.set_position([0.05, 0.05, 0.45, 0.9])  # Left half of figure
        self.ax.legend(loc='upper right')
        self.fig.tight_layout()
        print(f"Plot initialized for Vehicle {vehicle_id}!")


    def plot_traj_data(self):

        plt.ion()  # Ensures the figure is displayed until closed manually  # Enable interactive mode
        if len(self.trajectory_data) < 1:
            self.trajectory_data = self.traj.wpt_ref
        if len(self.trajectory_data) > 1:

            x = self.trajectory_data[0]  # Extract first column
            y = self.trajectory_data[1]  # Extract first column

            # Update trajectory line
            self.trajectory_line.set_data(x, y)
            # self.zoom_trajectory_line.set_data(x, y)

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
            # self.zoom_left_lane_line.set_data(x_left, y_left)
            # self.zoom_right_lane_line.set_data(x_right, y_right)

            self.left_left_lane_line.set_data(x_left_left, y_left_left)
            self.right_right_lane_line.set_data(x_right_right, y_right_right)
            # self.zoom_left_left_lane_line.set_data(x_left_left, y_left_left)
            # self.zoom_right_right_lane_line.set_data(x_right_right, y_right_right)
            

            # self.ax.relim()  # Recalculate limits
            # self.ax.autoscale_view()

            plt.draw()
            plt.pause(0.01)

    def update_curr_plot(self, vehicle_id, x, y, vel, tar_x, tar_y, pred_x, pred_y, obs_x, obs_y, obs_r):        # Update main plot

        plots = self.vehicle_plots[vehicle_id]
        # Main plot
        plots['current'].set_data([x], [y])
        plots['circle'].set_center((x, y))
        plots['target'].set_data([tar_x], [tar_y])
        plots['pred'].set_data([pred_x], [pred_y])

        # Zoomed plot
        zoom_ax = plots['zoom_ax']
        zoom_ax.set_xlim(x - 15, x + 15)
        zoom_ax.set_ylim(y - 15, y + 15)
        plots['zoom_current'].set_data([x], [y])
        plots['zoom_circle'].set_center((x, y))
        plots['zoom_target'].set_data([tar_x], [tar_y])
        plots['zoom_pred'].set_data([pred_x], [pred_y])
        plots['speed_text'].set_text(f"Vehicle {vehicle_id} speed: {vel} m/s")

        # Update obstacles in main and zoomed plot
        self.obs_position.set_data(obs_x, obs_y)
        plots['zoom_obs'].set_data(obs_x, obs_y)
        if len(self.ax_obs_circles) != len(obs_x):
            for circle in self.ax_obs_circles:
                circle.remove()
            self.ax_obs_circles = [plt.Circle((x, y), r, color='grey', alpha=0.9, fill=True) 
                                   for x, y, r in zip(obs_x, obs_y, obs_r)]
            print()
            for circle in self.ax_obs_circles:
                self.ax.add_patch(circle)
        else:
            for i, circle in enumerate(self.ax_obs_circles):
                circle.set_center((obs_x[i], obs_y[i]))
                circle.set_radius(obs_r[i])

        if len(plots['zoom_obs_circles']) != len(obs_x):
            for circle in plots['zoom_obs_circles']:
                circle.remove()
            plots['zoom_obs_circles'] = [plt.Circle((x, y), r, color='grey', alpha=0.9, fill=True) 
                                         for x, y, r in zip(obs_x, obs_y, obs_r)]
            for circle in plots['zoom_obs_circles']:
                zoom_ax.add_patch(circle)
        else:
            for i, circle in enumerate(plots['zoom_obs_circles']):
                circle.set_center((obs_x[i], obs_y[i]))
                circle.set_radius(obs_r[i])

        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        self.zoom_fig.canvas.draw()
        plt.pause(0.01)

    def start_all_vehicles(self):
        """Publish a start command to all known vehicles."""
        start_msg = String()
        start_msg.data = "start"
        for vehicle_id in self.known_vehicles:
            publisher = self.start_publishers.get(vehicle_id)
            if publisher:
                publisher.publish(start_msg)
                self.get_logger().info(f"Published start command to Vehicle {vehicle_id}")
            else:
                self.get_logger().warn(f"No publisher found for Vehicle {vehicle_id}")

    def prompt_user_to_start(self):
        """Prompt the user to start all vehicles in a separate thread."""
        def get_user_input():
            response = input(f"All {self.expected_vehicles} vehicles detected. Would you like to start all vehicles? (yes/no): ").strip().lower()
            if response in ('yes', 'y'):
                self.start_all_vehicles()
            else:
                self.get_logger().info("User chose not to start vehicles.")

        # Run input prompt in a separate thread to avoid blocking rclpy.spin
        thread = threading.Thread(target=get_user_input)
        thread.start()

    def save_config_data(self, data, vehicle_id):
        folder_path = os.path.expanduser(f'~/Downloads/cooperative-MPC-CBF-main/cooperative GCS/output data/{self.scename}')
        self.ensure_directory_exists(folder_path)  # Create folder if not exist
        file_name = os.path.join(folder_path, f'config_{self.test_name}_vehicle_{vehicle_id}.json')
        existing_data = []
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r') as file:
                    existing_data = json.load(file)
            except json.JSONDecodeError:
                # Backup corrupted file and start fresh
                bak_name = file_name + '.corrupt.' + datetime.now().strftime('%Y%m%d_%H%M%S')
                os.rename(file_name, bak_name)
                self.get_logger().warning(f"Corrupted JSON backed up to {bak_name}; starting new file.")
                existing_data = []

        if isinstance(existing_data, list):
            existing_data.append(data)

        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def save_rtdict_data(self, data, vehicle_id):
        folder_path = os.path.expanduser(f'~/Downloads/cooperative-MPC-CBF-main/cooperative GCS/output data/{self.scename}')
        self.ensure_directory_exists(folder_path)  # Create folder if not exist
        file_name = os.path.join(folder_path, f'rt_data_{self.test_name}_vehicle_{vehicle_id}.json')
        existing_data = []
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r') as file:
                    existing_data = json.load(file)
            except json.JSONDecodeError:
                bak_name = file_name + '.corrupt.' + datetime.now().strftime('%Y%m%d_%H%M%S')
                os.rename(file_name, bak_name)
                self.get_logger().warning(f"Corrupted JSON backed up to {bak_name}; starting new file.")
                existing_data = []

        if isinstance(existing_data, list):
            existing_data.append(data)

        with open(file_name, 'w') as file:
            json.dump(existing_data, file, indent=4)

    def save_mpc_data(self, data, vehicle_id):
        folder_path = os.path.expanduser(f'~/Downloads/cooperative-MPC-CBF-main/cooperative GCS/output data/{self.scename}')
        self.ensure_directory_exists(folder_path)  # Create folder if not exist
        file_name = os.path.join(folder_path, f'mpc_data_{self.test_name}_vehicle_{vehicle_id}.json')
        print('file_name: ', file_name)
        existing_data = []
        if os.path.exists(file_name):
            try:
                with open(file_name, 'r') as file:
                    existing_data = json.load(file)
            except json.JSONDecodeError:
                bak_name = file_name + '.corrupt.' + datetime.now().strftime('%Y%m%d_%H%M%S')
                os.rename(file_name, bak_name)
                self.get_logger().warning(f"Corrupted JSON backed up to {bak_name}; starting new file.")
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
    node = DynamicMultiVehicleMPCSubscriber()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == '__main__':
    main()
