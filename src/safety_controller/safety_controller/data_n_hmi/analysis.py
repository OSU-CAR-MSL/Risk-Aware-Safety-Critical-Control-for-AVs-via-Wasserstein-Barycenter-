import os
import json
import pandas as pd
import glob
import re
import matplotlib.pyplot as plt
import sys
import os
from collections import defaultdict
import numpy as np
from matplotlib import patches
import matplotlib.cm as cm  
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter

class Analysis():

    def __init__(self):


        cbf_cmap = LinearSegmentedColormap.from_list("cbf_cmap", ["#990000", "#ff9999"])  # Red gradient
        qp_cmap = LinearSegmentedColormap.from_list("qp_cmap", ["#006600", "#99ff99"])  # Green gradient
        cbf_lie_cmap = LinearSegmentedColormap.from_list("rcbf_qp_cmap", ["#0000b3", "#ccccff"])  # blue gradient
        cbf_nlie_cmap = LinearSegmentedColormap.from_list("mpc_cmp", ["#800080", "#ff99ff"])  # Magenta gradient

        #Store colormaps in a dictionary for easy lookup
        self.controller_cmaps = {
            "'MPC-DCBF'": cbf_cmap,
            "'MPC-DCBF-LIE'": cbf_lie_cmap,
            "MPC-WB-QP": cbf_nlie_cmap,

            "MPC-CBF-QP": qp_cmap,

        }
        print('self.controller_color', self.controller_cmaps)

        self.controller_color = {
            "'MPC-DCBF'": 'red',
            "'MPC-DCBF-LIE'": 'blue',
            "MPC-WB-QP": 'magenta',
            "MPC-CBF-QP": 'green',
        }

        # hardcode for cmap

        # for IFAC
        # mpc_cmp = LinearSegmentedColormap.from_list("mpc_cmp", ["#800080", "#ff99ff"])  # Magenta gradient
        # dc_cmap = LinearSegmentedColormap.from_list("dc_cmap", ["#990000", "#ff9999"])  # Red gradient
        # cbf_cmap = LinearSegmentedColormap.from_list("cbf_cmap", ["#000000", "#666666"])  # Gray gradient
        # qp_cmap = LinearSegmentedColormap.from_list("qp_cmap", ["#006600", "#99ff99"])  # Green gradient
        # hocbf_qp_cmap = LinearSegmentedColormap.from_list("hocbf_qp_cmap", ["#660066", "#cc99ff"])  # blue gradient

        # self.controller_cmaps = {
        #     "'MPC'": mpc_cmp,
        #     "'MPC-DC'": dc_cmap,
        #     "'MPC-DCBF'": cbf_cmap,
        #     "'MPC-CBF-QP'": qp_cmap,
        #     "'MPC-HOCBF-QP'": hocbf_qp_cmap
        # }
        # print('self.controller_color', self.controller_cmaps)

        # self.controller_color = {
        #     "'MPC'": 'magenta',
        #     "'MPC-DC'": 'red',
        #     "'MPC-DCBF'": 'black',
        #     "'MPC-CBF-QP'": 'green',
        #     "'MPC-HOCBF-QP'": 'blue',
        # }

        # mpc_cmp = LinearSegmentedColormap.from_list("mpc_cmp", ["#800080", "#ff99ff"])  # Magenta gradient
        # cbf_cmap = LinearSegmentedColormap.from_list("cbf_cmap", ["#990000", "#ff9999"])  # Red gradient
        # rcbf_cmap = LinearSegmentedColormap.from_list("rcbf_cmap", ["#000000", "#666666"])  # Gray gradient
        # ecbf_cmap = LinearSegmentedColormap.from_list("ecbf_cmap", ["#cc5500", "#ffcc99"])  # Orange gradient
        # dhocbf_cmap = LinearSegmentedColormap.from_list("cyan_cmap", ["#008b8b", "#66ffff"])  # Cyan gradient

        # qp_cmap = LinearSegmentedColormap.from_list("qp_cmap", ["#006600", "#99ff99"])  # Green gradient
        # rcbf_qp_cmap = LinearSegmentedColormap.from_list("rcbf_qp_cmap", ["#0000b3", "#ccccff"])  # blue gradient
        # ecbf_qp_cmap = LinearSegmentedColormap.from_list("ecbf_qp_cmap", ["#660066", "#cc99ff"])  # blue gradient

        # #Store colormaps in a dictionary for easy lookup
        # self.controller_cmaps = {
        #     "'MPC'": mpc_cmp,
        #     "'MPC-DCBF'": cbf_cmap,
        #     "'MPC-DRCBF'": rcbf_cmap,
        #     "'MPC-DECBF'": ecbf_cmap,
        #     "'MPC-DHOCBF'": dhocbf_cmap,

        #     "'MPC-CBF-QP'": qp_cmap,
        #     "'MPC-RCBF-QP'": rcbf_qp_cmap,
        #     "'MPC-ECBF-QP'": ecbf_qp_cmap
        # }
        # print('self.controller_color', self.controller_cmaps)

        # self.controller_color = {
        #     "'MPC'": 'magenta',
        #     "'MPC-DCBF'": 'red',
        #     "'MPC-DRCBF'": 'black',
        #     "'MPC-DECBF'": 'orange',
        #     "'MPC-DHOCBF'": 'cyan',

        #     "'MPC-CBF-QP'": 'green',
        #     "'MPC-RCBF-QP'": 'blue',
        #     "'MPC-ECBF-QP'": 'purple'
        # }

        self.obstacle_cmap = LinearSegmentedColormap.from_list("obstacle_cmap", ["#cccc00", "#ffff99"])  # Yellow gradient
        self.obs_color = ['yellow']

        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
        sys.path.append(parent_dir)
        self.directory = os.path.join(parent_dir, 'data_n_hmi', 'output data')
        self.all_files = glob.glob(os.path.join(self.directory, '*', "*.json"))

        grouped_all = self.grouped(self.all_files)

        self.plot_controller_info(grouped_all)



    def ensure_directory_exists(self, dir):
        """Ensure the target directory exists, if not create it."""
        if not os.path.exists(dir):
            os.makedirs(dir)
    
    def extract_info(self, filename):

        # Updated regex to match the correct format
        match = re.search(r"\['(\d+)'\]_(\d+)_\['(.*?)'\]_\[(.*?)\]", filename)

        if match:
            scenario, run_time, controller, velocity = match.groups()
            try:
                velocity = float(velocity)  # Convert velocity to float
            except ValueError:
                velocity = None  # Handle potential errors gracefully
            return scenario, run_time, controller, velocity
        return None, None, None
    
    def grouped(self, files):

        # Initialize a dictionary to store file paths grouped by scenario and velocity
        grouped_files = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))

        # Categorize files by scenario and velocity
        for file_path in files:
            filename = os.path.basename(file_path)
            scenario, run_time, controller, velocity = self.extract_info(filename)

            if scenario and run_time and velocity and controller is not None:
                grouped_files[scenario][run_time][velocity][controller].append(file_path)

        return grouped_files

    def plot_controller_info(self, grouped_files):

        # Extract only `mpc_data_**.json` files

        filtered_config_files = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        filtered_mpc_files    = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
        filtered_rt_files     = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))


        for scenario, run_time_dict in grouped_files.items():
            for run_time, velocity_dict in run_time_dict.items():
                for velocity, controller_dict in velocity_dict.items():
                    for controller, files in controller_dict.items():
                        config_files = [f for f in files if "config_" in os.path.basename(f)]
                        mpc_files = [f for f in files if "mpc_data" in os.path.basename(f)]
                        rt_files = [f for f in files if "rt_data" in os.path.basename(f)]
                        if mpc_files and rt_files:
                            filtered_config_files[scenario][run_time][velocity][controller] = config_files
                            filtered_mpc_files[scenario][run_time][velocity][controller] = mpc_files
                            filtered_rt_files[scenario][run_time][velocity][controller] = rt_files
                            
        self.plot_h_values(filtered_mpc_files, filtered_rt_files, filtered_config_files)
        # self.plot_h_values_gif(filtered_mpc_files, filtered_rt_files, filtered_config_files)
        self.plot_acc_steer_vel_values(filtered_mpc_files, filtered_rt_files, filtered_config_files)
        self.plot_trajectory(filtered_mpc_files, filtered_rt_files, filtered_config_files)
        self.generate_report(filtered_mpc_files, filtered_rt_files, filtered_config_files)
        # # self.generate_iter_report(filtered_mpc_files, filtered_rt_files, filtered_config_files)
        # self.optimality_ana(filtered_mpc_files, filtered_rt_files, filtered_config_files)
        # self.mpc_error_plot(filtered_mpc_files, filtered_rt_files, filtered_config_files)
        # self.mpc_error_plot_gif(filtered_mpc_files, filtered_rt_files, filtered_config_files)
        # self.plot_mpc_traj_gif(filtered_mpc_files, filtered_rt_files, filtered_config_files)
        # self.save_moving_traj_gif(filtered_mpc_files, filtered_rt_files, filtered_config_files)  
        # self.mpc_cost_plot_gif(filtered_mpc_files, filtered_rt_files, filtered_config_files)


    def plot_trajectory(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):

        # self.plot_static_traj(filtered_mpc_files, filtered_rt_files, filtered_config_files)
        self.plot_moving_traj(filtered_mpc_files, filtered_rt_files, filtered_config_files)


    
    def plot_mpc_traj_gif(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):

        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    for controller, mpc_files in sorted(controllers.items()):
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]

                        all_x, all_y = [], []
                        all_tar_x, all_tar_y = [], []

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):
                            self.get_parameters(config_file)
                            t = self.caltime(mpc_file)

                            with open(rt_file, 'r') as file:
                                data = json.load(file)  # Load JSON data

                            current_sts = [item["true_vehicle"] for item in data if "true_vehicle" in item]
                            current_sts = np.array(current_sts)
                            target = [item["target"] for item in data if "target" in item]
                            target = np.array(target)

                            min_length = min(len(t), current_sts.shape[0])
                            t = t[:min_length]
                            current_sts = current_sts[:min_length, :]
                            target = target[:min_length, :]

                            x_data = current_sts[:, 0]
                            y_data = current_sts[:, 1]
                            target_x = target[:, 0]
                            target_y = target[:, 1]

                            all_x.extend(x_data.tolist())
                            all_y.extend(y_data.tolist())
                            all_tar_x.extend(target_x.tolist())
                            all_tar_y.extend(target_y.tolist())

                            # Create figure
                            fig, axs = plt.subplots(figsize=(15, 10))
                            axs.plot(target_x, target_y, 'b', label='Reference Trajectory', linewidth= 2)  # Blue for reference trajectory
                            av_dot, = axs.plot([], [], 'ro', markersize=10, label="AV Current Position")  # Red dot for AV
                            tar_dot, = axs.plot([], [], 'go', markersize=10, label="Target Position")  # Red dot for AV

                            axs.set_xlabel('X (East) [m]', fontsize=26)
                            axs.set_ylabel('Y (North) [m]', fontsize=26)
                            axs.legend(fontsize=12)
                            axs.tick_params(axis='both', which='major', labelsize=22)
                            axs.grid(True)
                            axs.axis('equal')
                            fig.suptitle(f'{controller} Controlled AV Trajectory for Scenario {scenario}', fontsize=26)

                            # Define update function for animation
                            def update(frame):
                                av_dot.set_data([all_x[frame]], [all_y[frame]])  # Move the red dot to new position
                                tar_dot.set_data([all_tar_x[frame]], [all_tar_y[frame]])  # Move the red dot to new position
                                return av_dot,tar_dot,

                            # Create animation
                            ani = FuncAnimation(fig, update, frames=min_length, interval=100, blit=True)

                            # Save animation as GIF
                            folder_path = os.path.join(self.directory, f"['{scenario}_{run_time}']")
                            self.ensure_directory_exists(folder_path)
                            gif_path = os.path.join(folder_path, f'trajectory_{scenario}_{run_time}_{controller}_{velocity}.gif')
                            ani.save(gif_path, writer=PillowWriter(fps=10))

                            print(f"Saved animated GIF: {gif_path}")
                            plt.close(fig)


    def save_moving_traj_gif(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):

        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    for controller, mpc_files in sorted(controllers.items()):
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]

                        cmp_color = self.controller_cmaps[controller]  # Get the fixed color for this controller
                        color = self.controller_color[controller]

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):
                            self.get_parameters(config_file)
                            num_of_obstacles = int(self.numofobs[0])  # Get obstacle count
                            t = self.caltime(mpc_file)

                            with open(rt_file, 'r') as file:
                                data = json.load(file)  # Load JSON data

                            current_sts = [item["true_vehicle"] for item in data if "true_vehicle" in item]
                            obs_sts = [item["obs_true_states"] for item in data if "obs_true_states" in item]
                            current_sts = np.array(current_sts)
                            obs_sts = np.array(obs_sts)

                            min_length = min(len(t), current_sts.shape[0])
                            t = t[:min_length]
                            current_sts = current_sts[:min_length, :]
                            obs_sts = obs_sts[:min_length, :]

                            x_data = current_sts[:, 0]
                            y_data = current_sts[:, 1]

                            fig, axs = plt.subplots(figsize=(15, 10))

                            # Plot base trajectories
                            axs.plot(x_data, y_data, 'r-', label='AV Trajectory')
                            av_dot, = axs.plot([], [], 'ro', markersize=10, label="AV Current Position")
                            
                            obs_dots = []
                            for obs_idx in range(num_of_obstacles):
                                obs_dots.append(axs.plot([], [], 'bo', markersize=10, label=f'Pedestrian {obs_idx}')[0])

                            axs.set_xlabel('X (East) [m]', fontsize=26)
                            axs.set_ylabel('Y (North) [m]', fontsize=26)
                            axs.legend(fontsize=12)
                            axs.tick_params(axis='both', which='major', labelsize=22)
                            axs.grid(True)
                            axs.axis('equal')
                            fig.suptitle(f'{controller} Controlled AV Trajectory for Scenario {scenario}', fontsize=26)

                            # Define update function for animation
                            def update(frame):
                                av_dot.set_data([x_data[frame]], [y_data[frame]])  # Move AV position
                                for obs_idx in range(num_of_obstacles):
                                    obs_dots[obs_idx].set_data([obs_sts[frame, obs_idx*4]], [obs_sts[frame, obs_idx*4+1]])  # Move obstacle position
                                return [av_dot] + obs_dots

                            # Create animation
                            ani = FuncAnimation(fig, update, frames=min_length, interval=100, blit=True)

                            # Save as GIF
                            folder_path = os.path.join(self.directory, f"['{scenario}_{run_time}']")
                            self.ensure_directory_exists(folder_path)
                            gif_path = os.path.join(folder_path, f'trajectory_{scenario}_{run_time}_{controller}_{velocity}.gif')
                            ani.save(gif_path, writer=PillowWriter(fps=10))

                            print(f"Saved animated GIF: {gif_path}")
                            plt.close(fig)


    def plot_moving_traj(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):

        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    for i, (controller, mpc_files) in enumerate(sorted(controllers.items())):
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]

                        cmp_color = self.controller_cmaps[controller]  # Get the fixed color for this controller
                        color = self.controller_color[controller]
                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):
                            self.get_parameters(config_file)
                            num_of_obstacles = int(self.numofobs[0])  # Get obstacle count
                            t = self.caltime(mpc_file)
                            fig, axs = plt.subplots(figsize=(15, 10))

                            with open(rt_file, 'r') as file:
                                data = json.load(file)  # Load JSON data

                            current_sts = [item["true_vehicle"] for item in data if "true_vehicle" in item]
                            obs_sts = [item["obs_true_states"] for item in data if "obs_true_states" in item]
                            current_sts = np.array(current_sts)  # Take the first entry from the list
                            obs_sts = np.array(obs_sts)  # Take the first entry from the list

                            min_length = min(len(t), current_sts[:,0].shape[0])
                            t = t[:min_length]
                            current_sts = current_sts[:min_length, :]
                            obs_sts = obs_sts[:min_length, :]

                            x_data = current_sts[:,0]
                            y_data = current_sts[:,1]

                            axs.scatter(x_data, y_data, c=t, cmap=cmp_color, label='AV Trajectory', s=50)
                            axs.scatter(x_data[0], y_data[0], color=color, marker='o', s=250, label='AV Start')
                            axs.scatter(x_data[-1], y_data[-1], color=color, marker='*', s=250, label='AV End')
                            # Add zoomed-in inset
                            zoom_ax = inset_axes(axs, width="80%", height="80%", bbox_to_anchor=(0, 0, 0.5, 0.5), bbox_transform=axs.transAxes)
                            zoom_ax.scatter(x_data, y_data, c=t, cmap=cmp_color, s=50)
                            
                            for obs_idx in range(num_of_obstacles):
                                obs_color = self.obstacle_cmap

                                start_label = 'Pedestrian Start' if obs_idx == 0 else None
                                end_label = 'Pedestrian End' if obs_idx == 0 else None
                                closest_veh_label = 'Closest Point - AV' if obs_idx == 0 else None
                                closest_obs_label = 'Closest Point - Pedestrian' if obs_idx == 0 else None
                                radius_label = 'Obstacle radius' if obs_idx == 0 else None
                                av_radius_label = 'AV radius + Safety radius + Obstacle radius' if obs_idx == 0 else None

                                print('obs_sts', obs_sts.shape)
                                axs.scatter(obs_sts[:, obs_idx*4], obs_sts[:, obs_idx*4+1], c=t, cmap=obs_color, label=f'Pedestrian {obs_idx} Trajectory', s=50)
                                axs.scatter(obs_sts[0, obs_idx*4], obs_sts[0, obs_idx*4+1], color='Blue', marker='o', s=250, label=start_label)
                                axs.scatter(obs_sts[-1, obs_idx*4], obs_sts[-1, obs_idx*4+1], color='Blue', marker='*', s=250, label=end_label)
                                zoom_ax.scatter(obs_sts[:, obs_idx*4], obs_sts[:, obs_idx*4+1], c=t, cmap=obs_color, s=50)

                                distance = np.sqrt((x_data - obs_sts[:, obs_idx*4])**2 + (y_data - obs_sts[:, obs_idx*4+1])**2)
                                min_distance = (np.min(distance))
                                print('min_dis', min_distance)
                                closest_index = np.argmin(distance)
                                closest_vehicle_x = x_data[closest_index]
                                closest_vehicle_y = y_data[closest_index]
                                closest_obstacle_x = obs_sts[closest_index, obs_idx*4]
                                closest_obstacle_y = obs_sts[closest_index, obs_idx*4+1]

                                # Highlight closest point
                                axs.scatter(closest_vehicle_x, closest_vehicle_y, color='red', marker='x', s=250, label=closest_veh_label)
                                axs.scatter(closest_obstacle_x, closest_obstacle_y, color='blue', marker='x', s=250, label=closest_obs_label)
                                # Add a circle for the obstacle radius
                                circle = patches.Circle(
                                    (closest_obstacle_x, closest_obstacle_y),  # Corrected coordinates
                                    self.OBSTACLE_R[obs_idx],
                                    color='grey',
                                    alpha=0.9,
                                    fill=True,
                                    label=radius_label
                                )

                                # Add a circle for the vehicle radius + safety distance
                                circle2 = patches.Circle(
                                    (closest_obstacle_x, closest_obstacle_y),  # Corrected coordinates
                                    self.VEHICLE_R + self.SAFETY_DIS + self.OBSTACLE_R[obs_idx],
                                    color='grey',
                                    alpha=0.5,
                                    fill=True,
                                    label=av_radius_label
                                )
                                

                                zoom_ax.add_patch(circle)
                                zoom_ax.add_patch(circle2)

                            zoom_ax.scatter(closest_vehicle_x, closest_vehicle_y, color='red', marker='x', s=200)
                            zoom_ax.scatter(closest_obstacle_x, closest_obstacle_y, color='blue', marker='x', s=200)
                            
                            window_size = 15
                            zoom_ax.set_xlim(closest_vehicle_x - window_size, closest_vehicle_x + window_size)
                            zoom_ax.set_ylim(closest_vehicle_y - window_size, closest_vehicle_y + window_size)

                            # Customize inset plot
                            zoom_ax.set_title('Zoomed-In View', fontsize=18)
                            zoom_ax.tick_params(axis='both', which='major', labelsize=16)
                            zoom_ax.grid(True)
                            zoom_ax.legend(fontsize=14,loc='lower left')

                            # Add a grayscale color bar for time
                            sm = ScalarMappable(norm=Normalize(vmin=min(t), vmax=max(t)), cmap='Greys_r')
                            cbar = fig.colorbar(sm, ax=axs)
                            cbar.ax.tick_params(axis='y', which='major', labelsize=22)
                            cbar.set_label('Time [s]', color='black', fontsize=26)

                            axs.set_xlabel('X (East) [m]', fontsize=26)
                            axs.set_ylabel('Y (North) [m]', fontsize=26)
                            axs.legend(fontsize=12)
                            axs.tick_params(axis='both', which='major', labelsize=22)
                            axs.grid(True)
                            axs.axis('equal')
                            fig.suptitle(f'{controller} Controlled AV Trajectory for Scenario {scenario}', fontsize=26)

                            # Define folder path for saving results
                            folder_path = os.path.join(self.directory, f"['{scenario}_{run_time}']")
                            self.ensure_directory_exists(folder_path)  # Create folder if not exist
                            save_path = os.path.join(folder_path, f'trajectory_{scenario}_{run_time}_{controller}_{velocity}.jpg')
                            plt.savefig(save_path, dpi=300, format='jpg', bbox_inches='tight', transparent=False, pad_inches=0.2)
                            plt.close(fig)

                            # plt.show()

    def plot_static_traj(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):

        
        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    any_controller = next(iter(controllers.keys()))  # Get any controller key
                    config_files = filtered_config_files[scenario][run_time][velocity][any_controller]
                    self.get_parameters(config_files[0])  # Extract parameters once
                    num_of_obstacles = int(self.numofobs[0])  # Get obstacle count
                    fig, axs = plt.subplots()

                    for i, (controller, mpc_files) in enumerate(sorted(controllers.items())):
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]
                        color = self.controller_color[controller]  # Fixed color for this controller

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):
                            self.get_parameters(config_file)
                            num_of_obstacles = int(self.numofobs[0])  # Get obstacle count again if needed
                            t = self.caltime(mpc_file)

                            with open(rt_file, 'r') as file:
                                data = json.load(file)  # Load JSON data
                            current_sts = [item["true_vehicle"] for item in data if "true_vehicle" in item]
                            obs_sts = [item["obs_true_states"] for item in data if "obs_true_states" in item]
                            current_sts = np.array(current_sts)  # Take the first entry from the list
                            obs_sts = np.array(obs_sts)  # Take the first entry from the list
                            axs.plot(current_sts[:,0], current_sts[:,1],  label=f'{controller} controller AV trajectory', linewidth=2, color=color)

                            for obs_idx in range(num_of_obstacles):
                                # axs.plot(obs_sts[0,obs_idx*4], obs_sts[0,obs_idx*4+1], 'k.', label=f'Obstacle {obs_idx} center')
                                # Add label only for the first obstacle
                                
                                obstacle_label = f'Obstacle {obs_idx} center' if obs_idx == 0 else None
                                radius_label = 'Obstacle radius' if obs_idx == 0 else None
                                av_radius_label = 'AV radius + Safety radius + Obstacle radius' if obs_idx == 0 else None
                                
                                axs.plot(obs_sts[0, obs_idx*4], obs_sts[0, obs_idx*4+1], 'k.', label=obstacle_label)
                                
                                circle = patches.Circle(
                                    (obs_sts[0, obs_idx*4], obs_sts[0, obs_idx*4+1]), 
                                    self.OBSTACLE_R[obs_idx], 
                                    color='grey', alpha=0.9, fill=True, label=radius_label
                                )
                                
                                circle2 = patches.Circle(
                                    (obs_sts[0, obs_idx*4], obs_sts[0, obs_idx*4+1]), 
                                    self.VEHICLE_R + self.SAFETY_DIS + self.OBSTACLE_R[obs_idx], 
                                    color='grey', alpha=0.5, fill=True, label=av_radius_label
                                )

                                axs.add_patch(circle)
                                axs.add_patch(circle2)

                    axs.set_xlabel('X (East) [m]', fontsize=26)
                    axs.set_ylabel('Y (North) [m]', fontsize=26)
                    axs.legend(fontsize=10)
                    axs.tick_params(axis='both', which='major', labelsize=22)
                    axs.grid(True)
                    fig.suptitle(f'Controlled AV Trajectory for Scenario: {scenario}', fontsize=20)

                    # Define folder path for saving results
                    folder_path = os.path.join(self.directory, f"['{scenario}_{run_time}']")
                    self.ensure_directory_exists(folder_path)  # Create folder if not exist
                    save_path = os.path.join(folder_path, f'trajectory_{scenario}_{run_time}_{velocity}.jpg')
                    plt.savefig(save_path, dpi=300, format='jpg', bbox_inches='tight', transparent=False, pad_inches=0.2)
                    plt.close(fig)

                # plt.show()



    def plot_acc_steer_vel_values(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):

        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    fig1, axs1 = plt.subplots()  # acc figure
                    fig2, axs2 = plt.subplots()  # steering time figure
                    fig3, axs3 = plt.subplots()  # velocity figure

                    for i, (controller, mpc_files) in enumerate(sorted(controllers.items())):
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]
                        color = self.controller_color[controller]  # Get the fixed color for this controller

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):
                            # Extract parameters and solve time
                            self.get_parameters(config_file)
                            t = self.caltime(mpc_file)

                            if not t:
                                print(f"Skipping {mpc_file} due to missing solve_time data.")
                                continue

                            with open(rt_file, 'r') as file:
                                data = json.load(file)  # Load JSON data

                            output = [item["output"] for item in data if "output" in item]
                            current_sts = [item["true_vehicle"] for item in data if "true_vehicle" in item]
                            output = np.array(output)  # Take the first entry from the list
                            current_sts = np.array(current_sts)  # Take the first entry from the list
                            output_min_length = min(len(t), output[:,0].shape[0])
                            cur_min_length = min(len(t), current_sts[:,3].shape[0])
                            to = t[:output_min_length]
                            tc = t[:cur_min_length]
                            output = output[:output_min_length, :]
                            current_sts = current_sts[:cur_min_length, :]
                            axs1.plot(to, output[:,0], label=f"{controller}", linewidth=2, color=color)
                            axs2.plot(to, output[:,1], label=f"{controller}", linewidth=2, color=color)
                            axs3.plot(tc, current_sts[:,3], label=f"{controller}", linewidth=2, color=color)

                    axs1.set_xlabel('Time [s]', fontsize=20)
                    axs1.set_ylabel('Vehicle acceleration [m/s^2]', fontsize=20)
                    axs1.legend(fontsize=16)
                    axs1.tick_params(axis='both', which='major', labelsize=14)
                    axs1.grid(True)
                    fig1.suptitle(f'AV Acceleration Input from Controllers for Scenario: {scenario}', fontsize=20)


                    # Solve Time Figure
                    axs2.set_xlabel('Time [s]', fontsize=20)
                    axs2.set_ylabel('Steering [rad]', fontsize=20)
                    axs2.legend(fontsize=16)
                    axs2.tick_params(axis='both', which='major', labelsize=14)
                    axs2.grid(True)
                    fig2.suptitle(f'AV Steering Input from Controllers for Scenario: {scenario}', fontsize=20)

                    # Velocity Figure
                    axs3.set_xlabel('Time [s]', fontsize=20)
                    axs3.set_ylabel('Velocity [m/s]', fontsize=20)
                    axs3.legend(fontsize=16)
                    axs3.tick_params(axis='both', which='major', labelsize=14)
                    axs3.grid(True)
                    fig3.suptitle(f'AV Velocity for Scenario: {scenario}', fontsize=20)

                    # Step 6: Save all figures
                    folder_path = os.path.join(self.directory, f"['{scenario}_{run_time}']")
                    self.ensure_directory_exists(folder_path)

                    save_path1 = os.path.join(folder_path, f'Compare_acc_{scenario}_{run_time}_{velocity}.jpg')
                    save_path2 = os.path.join(folder_path, f'Compare_vel_{scenario}_{run_time}_{velocity}.jpg')
                    save_path3 = os.path.join(folder_path, f'Velocity_{scenario}_{run_time}_{velocity}.jpg')

                    fig1.savefig(save_path1, dpi=300, format='jpg', bbox_inches='tight', transparent=False, pad_inches=0.2)
                    fig2.savefig(save_path2, dpi=300, format='jpg', bbox_inches='tight', transparent=False, pad_inches=0.2)
                    fig3.savefig(save_path3, dpi=300, format='jpg', bbox_inches='tight', transparent=False, pad_inches=0.2)

                    print(f"Saved: {save_path1}")
                    print(f"Saved: {save_path2}")
                    print(f"Saved: {save_path3}")
                    plt.close(fig1)
                    plt.close(fig2)
                    plt.close(fig3)

                    # plt.show()


    def plot_h_values(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):

        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    any_controller = next(iter(controllers.keys()))  # 先拿一個 controller
                    config_files = filtered_config_files[scenario][run_time][velocity][any_controller]
                    self.get_parameters(config_files[0])  # 只抽一次參數
                    num_of_obstacles = int(self.numofobs[0])

                    fig, axs = plt.subplots(num_of_obstacles, 1, figsize=(10, 6 * num_of_obstacles), sharex=True)
                    if num_of_obstacles == 1:
                        axs = [axs]

                    for i, (controller, mpc_files) in enumerate(sorted(controllers.items())):
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]
                        print('controller', controller)
                        color = self.controller_color[controller]

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):
                            self.get_parameters(config_file)
                            t = self.caltime(mpc_file)

                            if not t:
                                print(f"Skipping {mpc_file} due to missing solve_time data.")
                                continue

                            # Extract h(x) values from rt_data
                            h_values = self.cal_h(rt_file)

                            # Convert `h_values` to a NumPy array for safe processing
                            if isinstance(h_values, list):
                                h_values = np.array(h_values, dtype=object)

                            min_length = min(len(t), len(h_values[0,:]))
                            t = t[:min_length]
                            h_values = h_values[:,:min_length]

                            # Plot h(x) for each obstacle in separate subplots
                            for obs_idx in range(num_of_obstacles):
                                if obs_idx < len(h_values):  # Ensure index is within bounds
                                    axs[obs_idx].plot(t, 
                                                    h_values[obs_idx], 
                                                    label=f"{controller} - Obs {obs_idx+1}", 
                                                    linewidth=2, 
                                                    color=color)  # Use assigned color
                    # Customize subplots
                    for obs_idx in range(num_of_obstacles):
                        axs[obs_idx].axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='h(x) = 0')
                        axs[obs_idx].set_ylabel(f'Obstacle {obs_idx+1} h(x)', fontsize=14)
                        axs[obs_idx].legend(fontsize=12)
                        axs[obs_idx].grid()

                    axs[-1].set_xlabel('Time [s]', fontsize=14)
                    fig.suptitle(f'Barrier Constraints for Scenario: {scenario}', fontsize=18)

                    # Define folder path for saving results
                    folder_path = os.path.join(self.directory, f"['{scenario}_{run_time}']")
                    # Ensure the directory exists
                    self.ensure_directory_exists(folder_path)  # Create folder if not exist

                    # Define the correct file path
                    save_path = os.path.join(folder_path, f'Compare_h_{scenario}_{run_time}_{velocity}.jpg')

                    # Save the plot
                    plt.savefig(save_path, dpi=300, format='jpg', bbox_inches='tight', transparent=False, pad_inches=0.2)
                    plt.close(fig)

                    # plt.show()

    def plot_h_values_gif(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):

        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    any_controller = next(iter(controllers.keys()))  # Get any controller key
                    config_files = filtered_config_files[scenario][run_time][velocity][any_controller]
                    self.get_parameters(config_files[0])  # Extract parameters once
                    num_of_obstacles = int(self.numofobs[0])  # Get obstacle count
                    
                    fig, axs = plt.subplots(num_of_obstacles, 1, figsize=(10, 6 * num_of_obstacles), sharex=True)
                    if num_of_obstacles == 1:
                        axs = [axs]  # Ensure axs is always a list for consistent indexing

                    # Store animation data
                    all_h_values = []
                    all_times = []
                    controller_labels = []
                    colors = []

                    for i, (controller, mpc_files) in enumerate(sorted(controllers.items())):
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]
                        color = self.controller_color[controller]

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):
                            self.get_parameters(config_file)
                            t = self.caltime(mpc_file)


                            if not t:
                                print(f"Skipping {mpc_file} due to missing solve_time data.")
                                continue

                            h_values = self.cal_h(rt_file)

                            # Convert `h_values` to a NumPy array for safe processing
                            if isinstance(h_values, list):
                                h_values = np.array(h_values, dtype=object)

                            min_length = min(len(t), len(h_values[0, :]))
                            t = t[:min_length]
                            h_values = h_values[:, :min_length]

                            all_h_values.append(h_values)
                            all_times.append(t)
                            controller_labels.append(f"{controller}")
                            colors.append(color)

                    # Define update function for animation
                    def update(frame):
                        for obs_idx in range(num_of_obstacles):
                            axs[obs_idx].clear()
                            axs[obs_idx].axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='h(x) = 0')

                            for idx, h_values in enumerate(all_h_values):
                                if obs_idx < len(h_values):
                                    axs[obs_idx].plot(all_times[idx][:frame], 
                                                    h_values[obs_idx][:frame], 
                                                    label=f"{controller_labels[idx]} - Obs {obs_idx+1}", 
                                                    linewidth=2, 
                                                    color=colors[idx])  # Use assigned color

                            axs[obs_idx].set_ylabel(f'Obstacle {obs_idx+1} h(x)', fontsize=14)
                            axs[obs_idx].legend(fontsize=12)
                            axs[obs_idx].grid()

                        axs[-1].set_xlabel('Time [s]', fontsize=14)
                        fig.suptitle(f'Barrier Constraints for Scenario: {scenario}', fontsize=18)

                    # Create animation
                    frames = max(len(t) for t in all_times)  # Maximum time steps among all controllers
                    ani = FuncAnimation(fig, update, frames=frames, interval=100)

                    # Define folder path for saving results
                    folder_path = os.path.join(self.directory, f"['{scenario}_{run_time}']")
                    self.ensure_directory_exists(folder_path)  # Create folder if not exist

                    # Save animation as GIF
                    gif_path = os.path.join(folder_path, f'Compare_h_{scenario}_{run_time}_{velocity}.gif')
                    ani.save(gif_path, writer=PillowWriter(fps=10))
                    plt.close(fig)

                    print(f"Saved animated GIF: {gif_path}")


    def get_parameters(self, config_file):

        with open(config_file, 'r') as file:

            data = json.load(file)  # Load JSON data
            mpc_para = [item["mpc para"] for item in data if "mpc para" in item]
            self.N = [item["N"] for item in mpc_para if "N" in item][0]

            cbf_para = [item["cbf para"] for item in data if "cbf para" in item]
            self.SAFETY_DIS = [item["SAFETY_DIS"] for item in cbf_para if "SAFETY_DIS" in item][0]       
            self.VEHICLE_R = [item["VEHICLE_R"] for item in cbf_para if "VEHICLE_R" in item][0]       

            sce_para = [item["scenario para"] for item in data if "scenario para" in item]
            self.numofobs = [item["numofobs"] for item in sce_para[0] if "numofobs" in item]
            # self.SAFETY_DIS = [item["SAFETY_DIS"] for item in sce_para[0] if "SAFETY_DIS" in item][0]
            # print('safty_dis', self.SAFETY_DIS)
            self.type = [item["type"] for item in sce_para[0] if "type" in item]
            para = [item["params"] for item in sce_para[0] if "params" in item]
            self.OBSTACLE_R = [item["OBSTACLE_R"] for item in para[0] if "OBSTACLE_R" in item]
            self.MPC_ITER = [item["mpc_max_iteration"] for item in sce_para[0] if "mpc_max_iteration" in item]
            self.QP_ITER = [item["qp_max_iteration"] for item in sce_para[0] if "qp_max_iteration" in item]


    def caltime(self, mpc_file):

        cal_time = []

        with open(mpc_file, 'r') as file:
            data = json.load(file)  # Load JSON data
        solve_times = [item["solve_time"] for item in data if "solve_time" in item]

        if not solve_times:
            print(f"No solve_time found in {mpc_file}")
            return []

        t = 0
        for time in solve_times:
            t += time
            cal_time.append(t)

        return cal_time
    
    def cal_h(self, rt_file):

        h = []
        h_values = [[] for _ in range(int(self.numofobs[0]))]
        with open(rt_file, 'r') as file:
            data = json.load(file)  # Load JSON data
        current_sts = [item["true_vehicle"] for item in data if "true_vehicle" in item]
        obs_sts = [item["obs_true_states"] for item in data if "obs_true_states" in item]
        current_sts = np.array(current_sts)  # Take the first entry from the list
        obs_sts = np.array(obs_sts)  # Take the first entry from the list

        for i in range(int(self.numofobs[0])):
            h = np.sqrt((current_sts[:,0] - obs_sts[:,i*4])**2 + (current_sts[:,1] - obs_sts[:,i*4+1])**2) - (self.VEHICLE_R + self.OBSTACLE_R[i] + self.SAFETY_DIS)
            h_values[i]= h
        return h_values
    
    def generate_report(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):
        
        report_data = []


        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    for controller, mpc_files in sorted(controllers.items()):
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):

                            self.get_parameters(config_file)
                            t = self.caltime(mpc_file)
                            num_of_obstacles = int(self.numofobs[0])  # Get obstacle count

                            with open(rt_file, 'r') as file:
                                rt_data = json.load(file)  # Load JSON data
                            with open(mpc_file, 'r') as file:
                                mpc_data = json.load(file)  # Load JSON data

                            solve_times = [item["solve_time"] for item in mpc_data if "solve_time" in item]
                            total_time = sum(solve_times)
                            mean_solve_time = np.mean(solve_times)
                            std_dev_solve_time = np.std(solve_times)

                            mpc_costs = [item["mpc_cost"] for item in mpc_data if "mpc_cost" in item]  # Summing up costs if available
                            mean_mpc_cost = np.mean(mpc_costs)
                            stddev_mpc_cost = np.std(mpc_costs)

                            ## calculate obstacle dis
                            current_sts = [item["true_vehicle"] for item in rt_data if "true_vehicle" in item]
                            obs_sts = [item["obs_true_states"] for item in rt_data if "obs_true_states" in item]
                            current_sts = np.array(current_sts)  # Take the first entry from the list
                            obs_sts = np.array(obs_sts)  # Take the first entry from the list
                            
                            min_length = min(len(t), current_sts[:,0].shape[0])
                            t = t[:min_length]
                            current_sts = current_sts[:min_length, :]
                            obs_sts = obs_sts[:min_length, :]

                            x_data = current_sts[:,0]
                            y_data = current_sts[:,1]

                            min_distance= np.zeros(num_of_obstacles)
                            safe_criteria= np.zeros(num_of_obstacles)
                            unsafe_criteria= np.zeros(num_of_obstacles)

                            safe_analysis = "safe"

                            for obs_idx in range(num_of_obstacles):
                                distance = np.sqrt((x_data - obs_sts[:, obs_idx*4])**2 + (y_data - obs_sts[:, obs_idx*4+1])**2)

                                tmp = (np.min(distance))
                                print('tmp', tmp)
                                min_distance[obs_idx] = round(tmp, 3)
                                print('min_distance', min_distance[obs_idx])

                                safe_criteria[obs_idx] = self.VEHICLE_R+self.OBSTACLE_R[obs_idx]+self.SAFETY_DIS
                                unsafe_criteria[obs_idx] = self.VEHICLE_R+self.OBSTACLE_R[obs_idx]
                                print('safe_criteria[obs_idx]', safe_criteria[obs_idx])                                
                                print('unsafe_criteria[obs_idx]', unsafe_criteria[obs_idx])

                            for d, unsafe, safe in zip(min_distance, unsafe_criteria, safe_criteria):
                                if d <= unsafe:
                                    safe_analysis = "collision"
                                    break  # Stop immediately if a collision is detected
                                elif unsafe < d < safe:
                                    safe_analysis = "unsafe"


                            ## calculate feasible solution num
                            optimal = [item["optimal"] for item in mpc_data if "optimal" in item]
                            total_solution = len(optimal)
                            infeasible_num = total_solution - sum(optimal)  # Summing feasible solutions

                            ## safe analysis

                            report_data.append({
                                "Scenario": scenario,
                                "Run_time": run_time,
                                "Velocity": velocity,
                                "Controller": controller,
                                "Total Time (s)": round(total_time, 5),
                                "Mean Solve Time (s)": round(mean_solve_time, 5),
                                "Stddev Solve Time (s)": round(std_dev_solve_time, 5),
                                "Mean Cost": round(mean_mpc_cost, 2),
                                "Stddev Cost": round(stddev_mpc_cost, 2),
                                "Infeasible Solution": f"{infeasible_num}/{total_solution}",
                                "Min Obstacle Distance (m)": min_distance,
                                "Safe Analysis": safe_analysis,
                                "Safe Criteria (Vr+Or+Ds)" : safe_criteria,
                                "Unafe Criteria (Vr+Or)" : unsafe_criteria,
                            })

        # Convert report data to DataFrame
        df = pd.DataFrame(report_data)

        # Define report file path
        report_path = os.path.join(self.directory, "scenario_report.csv")

        # Save report as CSV
        df.to_csv(report_path, index=False)
        print(f"Report saved: {report_path}")

        return df
    

    def generate_iter_report(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):
        
        report_data = []
        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    for controller, mpc_files in sorted(controllers.items()):
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):


                            self.get_parameters(config_file)
                            t = self.caltime(mpc_file)
                            num_of_obstacles = int(self.numofobs[0])  # Get obstacle count

                            with open(rt_file, 'r') as file:
                                rt_data = json.load(file)  # Load JSON data
                            with open(mpc_file, 'r') as file:
                                mpc_data = json.load(file)  # Load JSON data


                            mpc_solve_times = [item["mpc_solve_time"] for item in mpc_data if "mpc_solve_time" in item]
                            mpc_total_time = sum(mpc_solve_times)
                            mpc_mean_solve_time = np.mean(mpc_solve_times)
                            qp_solve_times = [item["qp_solve_time"] for item in mpc_data if "qp_solve_time" in item]
                            qp_total_time = sum(qp_solve_times)
                            qp_mean_solve_time = np.mean(qp_solve_times)
                            solve_times = [item["solve_time"] for item in mpc_data if "solve_time" in item]
                            total_time = sum(solve_times)
                            mean_solve_time = np.mean(solve_times)

                            mpc_iter = [item["mpc_iteration"] for item in mpc_data if "mpc_iteration" in item]
                            mpc_mean_iter = np.mean(mpc_iter)
                            mpc_mean_iter = round(mpc_mean_iter, 1)
                            qp_iter = [item["qp_iteration"] for item in mpc_data if "qp_iteration" in item]
                            qp_mean_iter = np.mean(qp_iter)
                            qp_mean_iter = round(qp_mean_iter, 1)

                            total_iter = [item["total_iteration"] for item in mpc_data if "total_iteration" in item]
                            toal_mean_iter = np.mean(total_iter)
                            toal_mean_iter = round(toal_mean_iter, 1)



                            ## calculate obstacle dis
                            current_sts = [item["true_vehicle"] for item in rt_data if "true_vehicle" in item]
                            obs_sts = [item["obs_true_states"] for item in rt_data if "obs_true_states" in item]
                            current_sts = np.array(current_sts)  # Take the first entry from the list
                            obs_sts = np.array(obs_sts)  # Take the first entry from the list
                            
                            min_length = min(len(t), current_sts[:,0].shape[0])
                            t = t[:min_length]
                            current_sts = current_sts[:min_length, :]
                            obs_sts = obs_sts[:min_length, :]

                            x_data = current_sts[:,0]
                            y_data = current_sts[:,1]

                            min_distance= np.zeros(num_of_obstacles)
                            safe_criteria= np.zeros(num_of_obstacles)
                            unsafe_criteria= np.zeros(num_of_obstacles)

                            safe_analysis = ""

                            for obs_idx in range(num_of_obstacles):
                                distance = np.sqrt((x_data - obs_sts[:, obs_idx*4])**2 + (y_data - obs_sts[:, obs_idx*4+1])**2)
                                tmp = (np.min(distance))
                                min_distance[obs_idx] = round(tmp, 3)
                                safe_criteria[obs_idx] = self.VEHICLE_R+self.OBSTACLE_R[obs_idx]+self.SAFETY_DIS
                                unsafe_criteria[obs_idx] = self.VEHICLE_R+self.OBSTACLE_R[obs_idx]

                            if np.any(min_distance <= unsafe_criteria):
                                safe_analysis = "collision"  # Immediate collision detected
                            elif np.any((min_distance > unsafe_criteria) & (min_distance < safe_criteria)):
                                safe_analysis = "unsafe"  # Unsafe zone detected
                            else:
                                safe_analysis = "safe"  # Everything is safe

                            ## calculate feasible solution num
                            optimal = [item["optimal"] for item in mpc_data if "optimal" in item]
                            total_solution = len(optimal)
                            infeasible_num = total_solution - sum(optimal)  # Summing feasible solutions

                            mpc_optimal = [item["mpc_qp_optimal"]["mpc"] for item in mpc_data if "mpc_qp_optimal" in item]
                            mpc_solution = len(mpc_optimal)
                            mpc_infeas_num = mpc_solution - sum(mpc_optimal)  # Summing feasible solutions 

                            qp_optimal = [item["mpc_qp_optimal"]["qp"] for item in mpc_data if "mpc_qp_optimal" in item]
                            qp_solution = len(qp_optimal)
                            qp_infeas_num = qp_solution - sum(qp_optimal)  # Summing feasible solutions


                            ## safe analysis

                            report_data.append({
                                "Scenario": scenario,
                                "Desined MPC/QP ITER": f"{self.MPC_ITER}/{self.QP_ITER}",
                                "Actual MPC/QP ITER": f"{mpc_mean_iter}/{qp_mean_iter}",
                                "Total ITER": toal_mean_iter,

                                "Total Time (s)": round(total_time, 5),
                                "Mean MPC Solve Time (s)": round(mpc_mean_solve_time, 5),
                                "Mean QP Solve Time (s)": round(qp_mean_solve_time, 5),
                                "Mean Solve Time (s)": round(mean_solve_time, 5),

                                "Infeasible Solution (MPC)": f"{mpc_infeas_num}/{total_solution}",
                                "Infeasible Solution (QP)": f"{qp_infeas_num}/{qp_solution}",
                                "Infeasible Solution": f"{infeasible_num}/{total_solution}",

                                "Min Obstacle Distance (m)": min_distance,
                                "Safe Analysis": safe_analysis,
                                "Safe Criteria (Vr+Or+Ds)" : safe_criteria,
                                "Unafe Criteria (Vr+Or)" : unsafe_criteria,
                            })

            # Convert report data to DataFrame
            df = pd.DataFrame(report_data)

            # Define report file path
            report_path = os.path.join(self.directory, "qp_iter_report.csv")

            # Save report as CSV
            df.to_csv(report_path, index=False)
            print(f"Report saved: {report_path}")

            return df
    


    def optimality_ana(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):

        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    for controller, mpc_files in sorted(controllers.items()):
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):

                            self.get_parameters(config_file)
                            t = self.caltime(mpc_file)
                            # num_of_obstacles = int(self.numofobs[0])  # Get obstacle count

                            # with open(rt_file, 'r') as file:
                            #     rt_data = json.load(file)  # Load JSON data
                            with open(mpc_file, 'r') as file:
                                mpc_data = json.load(file)  # Load JSON data

                            # solve_times = [item["solve_time"] for item in mpc_data if "solve_time" in item]
                            # optimalities = [item["optimal"] for item in mpc_data if "optimal" in item]
                            # control_input = [item["u_control"] for item in mpc_data if "u_control" in item]

                                                    # Convert to DataFrame
                            df = pd.DataFrame(mpc_data)
                            df["optimal_numeric"] = df["optimal"].astype(int)
                            df["u_control_1"] = df["u_control"].apply(lambda x: x[0] if isinstance(x, list) and len(x) > 0 else None)
                            df["u_control_2"] = df["u_control"].apply(lambda x: x[1] if isinstance(x, list) and len(x) > 1 else None)
                            

                            fig, axs = plt.subplots(2, 2, figsize=(12, 10))

                            # Plot optimal status (0,0)
                            axs[0, 0].plot(df.index, df["optimal_numeric"], label="Optimal Status", color="purple", linestyle='-')
                            axs[0, 0].set_xlabel("Iteration")
                            axs[0, 0].set_ylabel("Optimal (1: True, 0: False)")
                            axs[0, 0].set_title("Optimal Status over Iterations")
                            axs[0, 0].set_yticks([0, 1])
                            axs[0, 0].set_yticklabels(["False", "True"])
                            axs[0, 0].legend()
                            axs[0, 0].grid(True)

                            # Plot solve time (1,0)
                            axs[1, 0].plot(df.index, df["solve_time"], label="Solve Time", color="blue", linestyle='-')
                            axs[1, 0].scatter(df.index[df["optimal"] == False], df["solve_time"][df["optimal"] == False], color='red', label="Non-optimal")
                            axs[1, 0].set_xlabel("Iteration")
                            axs[1, 0].set_ylabel("Solve Time (s)")
                            axs[1, 0].set_title("Solve Time over Iterations")
                            axs[1, 0].legend()
                            axs[1, 0].grid(True)

                            # Plot first component of control input (1,0) - acceleration
                            axs[0, 1].plot(df.index, df["u_control_1"], label="Acceleration", color="green", linestyle='-')
                            axs[0, 1].scatter(df.index[df["optimal"] == False], df["u_control_1"][df["optimal"] == False], color='red', label="Non-optimal")
                            axs[0, 1].set_xlabel("Iteration")
                            axs[0, 1].set_ylabel("Acceleration")
                            axs[0, 1].set_title("Acceleration over Iterations")
                            axs[0, 1].legend()
                            axs[0, 1].grid(True)

                            # Plot second component of control input (1,1) - steering
                            axs[1, 1].plot(df.index, df["u_control_2"], label="Steering", color="blue", linestyle='-')
                            axs[1, 1].scatter(df.index[df["optimal"] == False], df["u_control_2"][df["optimal"] == False], color='red', label="Non-optimal")
                            axs[1, 1].set_xlabel("Iteration")
                            axs[1, 1].set_ylabel("Steering")
                            axs[1, 1].set_title("Steering over Iterations")
                            axs[1, 1].legend()
                            axs[1, 1].grid(True)

                            # Adjust layout and display
                            plt.tight_layout()

                            # Define folder path for saving results
                            folder_path = os.path.join(self.directory, f"['{scenario}_{run_time}']")
                            # Ensure the directory exists
                            self.ensure_directory_exists(folder_path)  # Create folder if not exist

                            # Define the correct file path
                            save_path = os.path.join(folder_path, f'optimal_{scenario}_{run_time}_{controller}_{velocity}.jpg')

                            # Save the plot
                            plt.savefig(save_path, dpi=300, format='jpg', bbox_inches='tight', transparent=False, pad_inches=0.2)
                            plt.close(fig)
                            # plt.show()

    def mpc_error_plot(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):


        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    colors = []

                    for controller, mpc_files in sorted(controllers.items()):
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]
                        color = self.controller_color[controller]

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):

                            self.get_parameters(config_file)
                            t = self.caltime(mpc_file)

                            with open(mpc_file, 'r') as file:
                                mpc_data = json.load(file)  # Load JSON data

                            states = [item["z_mpc"][0] for item in mpc_data if "z_mpc" in item]
                            reference = [item["z_ref"][0] for item in mpc_data if "z_ref" in item]

                            states = np.array(states)
                            reference = np.array(reference)
                            
                            fig1, axs1 = plt.subplots()  # acc figure
                            fig2, axs2 = plt.subplots()  # steering time figure
                            fig3, axs3 = plt.subplots()  # velocity figure

                            error_x = abs(states[:,0]-reference[:,0])
                            error_y = abs(states[:,1]-reference[:,1])
                            error_psi = abs(states[:,2] - reference[:,2])

                            # cur_min_length = min(len(t), error_x.shape[0])
                            tc = t
                            # error_x = error_x[:cur_min_length, :]
                            # error_y = error_x[:cur_min_length, :]
                            # error_psi = error_x[:cur_min_length, :]

                            axs1.plot(tc, error_x, label=f"{controller}", linewidth=2, color=color)
                            axs2.plot(tc, error_y, label=f"{controller}", linewidth=2, color=color)
                            axs3.plot(tc, error_psi, label=f"{controller}", linewidth=2, color=color)

                    axs1.set_xlabel('Time [s]', fontsize=20)
                    axs1.set_ylabel('x error [m]', fontsize=20)
                    axs1.legend(fontsize=16)
                    axs1.tick_params(axis='both', which='major', labelsize=14)
                    axs1.grid(True)
                    fig1.suptitle(f'MPC controller x-axis error', fontsize=20)


                    # Solve Time Figure
                    axs2.set_xlabel('Time [s]', fontsize=20)
                    axs2.set_ylabel('y error [m]', fontsize=20)
                    axs2.legend(fontsize=16)
                    axs2.tick_params(axis='both', which='major', labelsize=14)
                    axs2.grid(True)
                    fig2.suptitle(f'MPC controller y-axis error', fontsize=20)

                    # Velocity Figure
                    axs3.set_xlabel('Time [s]', fontsize=20)
                    axs3.set_ylabel('psi error [rad]', fontsize=20)
                    axs3.legend(fontsize=16)
                    axs3.tick_params(axis='both', which='major', labelsize=14)
                    axs3.grid(True)
                    fig3.suptitle(f'MPC controller heading error', fontsize=20)

                    # Step 6: Save all figures
                    folder_path = os.path.join(self.directory, f"['{scenario}_{run_time}']")
                    self.ensure_directory_exists(folder_path)

                    save_path1 = os.path.join(folder_path, f'MPC_errorx_{scenario}_{run_time}_{velocity}.jpg')
                    save_path2 = os.path.join(folder_path, f'MPC_errory_{scenario}_{run_time}_{velocity}.jpg')
                    save_path3 = os.path.join(folder_path, f'MPC_errorpsi_{scenario}_{run_time}_{velocity}.jpg')

                    fig1.savefig(save_path1, dpi=300, format='jpg', bbox_inches='tight', transparent=False, pad_inches=0.2)
                    fig2.savefig(save_path2, dpi=300, format='jpg', bbox_inches='tight', transparent=False, pad_inches=0.2)
                    fig3.savefig(save_path3, dpi=300, format='jpg', bbox_inches='tight', transparent=False, pad_inches=0.2)

                    print(f"Saved: {save_path1}")
                    print(f"Saved: {save_path2}")
                    print(f"Saved: {save_path3}")
                    plt.close(fig1)
                    plt.close(fig2)
                    plt.close(fig3)


    def mpc_error_plot_gif(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):
        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    all_x, all_y, all_psi = [], [], []
                    all_ref_x, all_ref_y, all_ref_psi = [], [], []
                    all_cost = []
                    all_times = []
                    controller_labels = []
                    colors = []

                    for controller, mpc_files in sorted(controllers.items()):
                        color = self.controller_color[controller]
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):
                            self.get_parameters(config_file)
                            t = self.caltime(mpc_file)

                            with open(mpc_file, 'r') as file:
                                mpc_data = json.load(file)  # Load JSON data

                            states = [item["z_mpc"][0] for item in mpc_data if "z_mpc" in item]
                            reference = [item["z_ref"][0] for item in mpc_data if "z_ref" in item]
                            cost = [item["mpc_cost"] for item in mpc_data if "mpc_cost" in item]

                            states = np.array(states)
                            reference = np.array(reference)
                            cost = np.array(cost)

                            cur_x = states[:, 0]
                            cur_y = states[:, 1]
                            cur_psi = states[:, 2]

                            ref_x = reference[:, 0]
                            ref_y = reference[:, 1]
                            ref_psi = reference[:, 2]

                            all_x.append(cur_x)
                            all_y.append(cur_y)
                            all_psi.append(cur_psi)
                            all_ref_x.append(ref_x)
                            all_ref_y.append(ref_y)
                            all_ref_psi.append(ref_psi)
                            all_cost.append(cost)

                            all_times.append(t)
                            controller_labels.append(controller)
                            colors.append(color)

                    # Create figures
                    fig1, axs1 = plt.subplots()
                    fig2, axs2 = plt.subplots()
                    fig3, axs3 = plt.subplots()
                    fig4, axs4 = plt.subplots()

                    def update(frame):
                        """Updates the animated plot at each frame."""
                        axs1.clear()
                        axs2.clear()
                        axs3.clear()
                        axs4.clear()

                        for idx, x in enumerate(all_x):
                            axs1.plot(all_times[idx][:frame], x[:frame], 
                                    label=f"{controller_labels[idx]}", linewidth=2, color=colors[idx])
                        for idx, x in enumerate(all_ref_x):
                            axs1.plot(all_times[idx][:frame], x[:frame], 
                                    label=f"x_ref", linewidth=2, color='black')

                        for idx, y in enumerate(all_y):
                            axs2.plot(all_times[idx][:frame], y[:frame], 
                                    label=f"{controller_labels[idx]}", linewidth=2, color=colors[idx])
                        for idx, y in enumerate(all_ref_y):
                            axs2.plot(all_times[idx][:frame], y[:frame], 
                                    label=f"y_ref", linewidth=2, color='black')

                        for idx, psi in enumerate(all_psi):
                            axs3.plot(all_times[idx][:frame], psi[:frame], 
                                    label=f"{controller_labels[idx]}", linewidth=2, color=colors[idx])
                        for idx, psi in enumerate(all_ref_psi):
                            axs3.plot(all_times[idx][:frame], psi[:frame], 
                                    label=f"psi_ref", linewidth=2, color='black')

                        for idx, c in enumerate(all_cost):
                            axs4.plot(all_times[idx][:frame], c[:frame], 
                                    label=f"cost", linewidth=2, color=colors[idx])

                        # Set labels and titles
                        axs1.set_xlabel('Time [s]', fontsize=14)
                        axs1.set_ylabel('x [m]', fontsize=14)
                        axs1.legend(fontsize=12)
                        axs1.grid(True)
                        fig1.suptitle(f'MPC controller x-axis tracking', fontsize=16)

                        axs2.set_xlabel('Time [s]', fontsize=14)
                        axs2.set_ylabel('y [m]', fontsize=14)
                        axs2.legend(fontsize=12)
                        axs2.grid(True)
                        fig2.suptitle(f'MPC controller y-axis tracking', fontsize=16)

                        axs3.set_xlabel('Time [s]', fontsize=14)
                        axs3.set_ylabel('psi [rad]', fontsize=14)
                        axs3.legend(fontsize=12)
                        axs3.grid(True)
                        fig3.suptitle(f'MPC controller heading tracking', fontsize=16)

                        axs4.set_xlabel('Time [s]', fontsize=14)
                        axs4.set_ylabel('cost', fontsize=14)
                        axs4.legend(fontsize=12)
                        axs4.grid(True)
                        fig4.suptitle(f'MPC controller cost', fontsize=16)

                    # Create animations
                    max_frames = max(len(t) for t in all_times)
                    ani1 = FuncAnimation(fig1, update, frames=max_frames, interval=100)
                    ani2 = FuncAnimation(fig2, update, frames=max_frames, interval=100)
                    ani3 = FuncAnimation(fig3, update, frames=max_frames, interval=100)
                    ani4 = FuncAnimation(fig4, update, frames=max_frames, interval=100)

                    # Save animations
                    folder_path = os.path.join(self.directory, f"['{scenario}_{run_time}']")
                    self.ensure_directory_exists(folder_path)

                    gif_path1 = os.path.join(folder_path, f'MPC_x_{scenario}_{run_time}_{velocity}.gif')
                    gif_path2 = os.path.join(folder_path, f'MPC_y_{scenario}_{run_time}_{velocity}.gif')
                    gif_path3 = os.path.join(folder_path, f'MPC_psi_{scenario}_{run_time}_{velocity}.gif')
                    gif_path4 = os.path.join(folder_path, f'MPC_cost_{scenario}_{run_time}_{velocity}.gif')

                    ani1.save(gif_path1, writer=PillowWriter(fps=10))
                    ani2.save(gif_path2, writer=PillowWriter(fps=10))
                    ani3.save(gif_path3, writer=PillowWriter(fps=10))
                    ani4.save(gif_path4, writer=PillowWriter(fps=10))

                    print(f"Saved animated GIFs: {gif_path1}, {gif_path2}, {gif_path3}, {gif_path4}")

                    # Save static plots as JPG
                    jpg_path1 = os.path.join(folder_path, f'MPC_x_{scenario}_{run_time}_{velocity}.jpg')
                    jpg_path2 = os.path.join(folder_path, f'MPC_y_{scenario}_{run_time}_{velocity}.jpg')
                    jpg_path3 = os.path.join(folder_path, f'MPC_psi_{scenario}_{run_time}_{velocity}.jpg')
                    jpg_path4 = os.path.join(folder_path, f'MPC_cost_{scenario}_{run_time}_{velocity}.jpg')

                    fig1.savefig(jpg_path1, dpi=300, bbox_inches='tight')
                    fig2.savefig(jpg_path2, dpi=300, bbox_inches='tight')
                    fig3.savefig(jpg_path3, dpi=300, bbox_inches='tight')
                    fig4.savefig(jpg_path4, dpi=300, bbox_inches='tight')

                    print(f"Saved static JPGs: {jpg_path1}, {jpg_path2}, {jpg_path3}, {jpg_path4}")

                    plt.close(fig1)
                    plt.close(fig2)
                    plt.close(fig3)
                    plt.close(fig4)

    def mpc_cost_plot_gif(self, filtered_mpc_files, filtered_rt_files, filtered_config_files):
        for scenario, run_times in filtered_mpc_files.items():
            for run_time, velocities in run_times.items():
                for velocity, controllers in sorted(velocities.items()):
                    all_cost = []
                    all_times = []
                    controller_labels = []
                    colors = []

                    for controller, mpc_files in sorted(controllers.items()):
                        color = self.controller_color[controller]
                        config_files = filtered_config_files[scenario][run_time][velocity][controller]
                        rt_files = filtered_rt_files[scenario][run_time][velocity][controller]

                        for config_file, mpc_file, rt_file in zip(config_files, mpc_files, rt_files):

                            self.get_parameters(config_file)
                            t = self.caltime(mpc_file)

                            with open(mpc_file, 'r') as file:
                                mpc_data = json.load(file)  # Load JSON data


                            cost = [item["mpc_cost"] for item in mpc_data if "mpc_cost" in item]

                            cost = np.array(cost)

                            all_cost.append(cost)

                            all_times.append(t)
                            controller_labels.append(controller)
                            colors.append(color)

                    # Create figures

                    fig4, axs4 = plt.subplots()

                    def update(frame):
                        """Updates the animated plot at each frame."""

                        axs4.clear()


                        for idx, c in enumerate(all_cost):
                            axs4.plot(all_times[idx][:frame], c[:frame], 
                                    label=f"{controller_labels[idx]}_cost", linewidth=2, color=colors[idx])


                        axs4.set_xlabel('Time [s]', fontsize=14)
                        axs4.set_ylabel('cost', fontsize=14)
                        axs4.legend(fontsize=12)
                        axs4.grid(True)
                        fig4.suptitle(f'MPC controller cost', fontsize=16)

                    # Create animations
                    max_frames = max(len(t) for t in all_times)
                    ani4 = FuncAnimation(fig4, update, frames=max_frames, interval=100)

                    # Save animations
                    folder_path = os.path.join(self.directory, f"['{scenario}_{run_time}']")
                    self.ensure_directory_exists(folder_path)

                    gif_path4 = os.path.join(folder_path, f'MPC_cost_{scenario}_{run_time}_{velocity}.gif')


                    ani4.save(gif_path4, writer=PillowWriter(fps=10))

                    print(f"Saved animated GIFs: {gif_path4}")

                    # Save static plots as JPG

                    jpg_path4 = os.path.join(folder_path, f'MPC_cost_{scenario}_{run_time}_{velocity}.jpg')


                    fig4.savefig(jpg_path4, dpi=300, bbox_inches='tight')

                    print(f"Saved static JPGs: {jpg_path4}")


                    plt.close(fig4)

                        

        
def main(args=None):
    ana = Analysis()

if __name__ == '__main__':
    main()
