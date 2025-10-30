from abc import ABC, abstractmethod
import numpy as np

class Scenario(ABC):
    def __init__(self):
        self.states = None

    @abstractmethod
    def update(self, *args, **kwargs):
        """Update the scenario states."""
        pass

    @abstractmethod
    def set_obs_init(self, *args, **kwargs):
        """Initialize obstacle position."""
        pass

    @staticmethod
    def create_scenarios(traj, all_para):

        scenario_map = {
            "static": StaticObstacleScenario,
            "sudden_ped": SuddenPedScenario,
            "overtaking": OvertakingScenario,
            "lkas": LaneScenario
        }
        scenarios = all_para.scenarios
        for scenario in scenarios:
            scenario_type = scenario.get("type", None)  # Get the type of the scenario (if any)
            numofobs = scenario.get("numofobs", None)
            scenario_params = scenario.get("params", {})  # Get the parameters of the scenario

        print('scenario_type', scenario_type)
        if scenario_type not in scenario_map:
            raise ValueError(f"Unknown dynamics type: {scenario_type}")

        return scenario_map[scenario_type](traj, scenario_params, numofobs, all_para.mpc_params)

class StaticObstacleScenario(Scenario):

    def __init__(self, trajectory, sce_para, numofobs, mpc_para):

        super().__init__()
        self.N = mpc_para.get("N")
        self.states = np.zeros((self.N + 1, numofobs*4))
        
        self.traj = trajectory
        self.numofobs = numofobs  # Store the number of obstacles
        self.obstacles = sce_para  # Store all obstacles (list of dictionaries)

    def set_obs_init(self):

        refx = self.traj.wpt_ref[0]
        refy = self.traj.wpt_ref[1]

        for obs_idx, obs in enumerate(self.obstacles):
            OBSTACLE_R = obs.get("OBSTACLE_R", None)
            PLACE_DIS = obs.get("PLACE_DIS", None)

            for i in range(len(refx)):
                dis_to_obs = np.sqrt(refx[i]**2 + refy[i]**2)
                if dis_to_obs >= PLACE_DIS:
                    self.states[:, obs_idx*4] = refx[i]
                    self.states[:, obs_idx*4+1] = refy[i]
                    self.states[:, obs_idx*4+2] = 0  # Static obstacle
                    self.states[:, obs_idx*4+3] = OBSTACLE_R
                    break  # Move to the next obstacle after assigning
        return self.states  # Return all states for all obstacles


    def update(self, cur_pos, time):  
        return self.states


class OvertakingScenario(Scenario):

    def __init__(self, trajectory, sce_para, numofobs, mpc_para):
        super().__init__()
        self.N = mpc_para.get("N")
        self.states = np.zeros((self.N+1,numofobs*4))
        
        self.traj = trajectory
        self.refx = self.traj.wpt_ref[0]
        self.refy = self.traj.wpt_ref[1]

        self.numofobs = numofobs  # Store the number of obstacles
        self.obstacles = sce_para  # Store all obstacles (list of dictionaries)

        print(f"Initialized OvertakingeScenario with {self.numofobs} obstacles.")


        self.pre_index = np.zeros(numofobs)


    def set_obs_pos(self, dis, pre_index):
        
        if pre_index.any()<0:
            obs_index = -1
        else:
            for i in range(0, len(self.refx)):
                dis_to_obs = np.sqrt((self.refx[i])**2+(self.refy[i])**2)
                if dis_to_obs >= dis:
                    obs_index = i
                    break
                else:
                    obs_index = -1
        pos = [self.refx[obs_index], self.refy[obs_index]]
        return pos, obs_index
    
    def set_obs_init(self):

        for obs_idx, obs in enumerate(self.obstacles):
            OBSTACLE_R = obs.get("OBSTACLE_R", None)
            PLACE_DIS = obs.get("PLACE_DIS", None)
            OBS_VEL = obs.get("OBS_VEL", None)
            for i in range(len(self.refx)):
                dis_to_obs = np.sqrt(self.refx[i]**2 + self.refy[i]**2)
                if dis_to_obs >= PLACE_DIS:
                    self.states[:,obs_idx*4] = self.refx[i]
                    self.states[:,obs_idx*4+1] = self.refy[i]
                    self.states[:,obs_idx*4+2] = OBS_VEL  
                    self.states[:,obs_idx*4+3] = OBSTACLE_R
                    return self.states

        return np.zeros((self.N+1,self.numofobs*4))  # Default if no valid position found


    def update(self, veh_pos, time):


        for obs_idx, obs in enumerate(self.obstacles):
            OBSTACLE_R = obs.get("OBSTACLE_R", None)
            PLACE_DIS = obs.get("PLACE_DIS", None)
            OBS_VEL = obs.get("OBS_VEL", None)
            moving_dis = OBS_VEL*time
            dis  = np.sqrt((0-self.states[0,obs_idx*4])**2+(0-self.states[0,obs_idx*4+1])**2)
            total_dis = dis+moving_dis
            update_pos, index = self.set_obs_pos(total_dis, self.pre_index[obs_idx])
            self.states[:,obs_idx*4] = update_pos[0]
            self.states[:,obs_idx*4+1] = update_pos[1]
            self.states[:,obs_idx*4+2] = OBS_VEL                  # moving obstacle
            self.states[:,obs_idx*4+3] = OBSTACLE_R
            self.pre_index[obs_idx] = index

        return self.states
    

class SuddenPedScenario(Scenario):

    def __init__(self, trajectory, sce_para, numofobs, mpc_para):

        self.N = mpc_para.get("N")
        self.states = np.zeros((self.N+1,4))
        self.sudden_obs_states = np.zeros((self.N+1,numofobs*4))

        self.numofobs = numofobs  # Store the number of obstacles
        self.obstacles = sce_para  # Store all obstacles (list of dictionaries)

        print(f"Initialized StaticObstacleScenario with {self.numofobs} obstacles.")

        self.start_x = np.zeros(numofobs)
        self.start_y = np.zeros(numofobs)
        self.end_x1 = np.zeros(numofobs)
        self.end_y1 = np.zeros(numofobs)
        self.end_x2 = np.zeros(numofobs)
        self.end_y2 = np.zeros(numofobs)
        self.obs_dis = np.zeros(numofobs)

        # Parse obstacle parameters
        for i, obs in enumerate(self.obstacles):
            OBSTACLE_R = obs.get("OBSTACLE_R", None)       
            OBS_VEL = obs.get("PED_VEL", None)     
            OBS_DIS = obs.get("SUDDEN_DIS", None)             
            self.obs_dis[i] = OBS_DIS
            strat_pos = obs.get("START_POS", [])
            if len(strat_pos) == 2:  # Ensure valid start position
                self.start_x[i], self.start_y[i] = strat_pos

            # Store first endpoint
            end_pos_1 = obs.get("END_POS_1", [])
            if len(end_pos_1) == 2:
                self.end_x1[i], self.end_y1[i] = end_pos_1

            # Store second endpoint
            end_pos_2 = obs.get("END_POS_2", [])
            if len(end_pos_2) == 2:
                self.end_x2[i], self.end_y2[i] = end_pos_2

        self.obs_xn = np.zeros((self.N+1,numofobs))
        self.obs_yn = np.zeros((self.N+1,numofobs))

        self.distance_to_obs = 1818
        self.states = self.sudden_obs_states

    def set_obs_init(self):
   
        for obs_idx, obs in enumerate(self.obstacles):
            OBSTACLE_R = obs.get("OBSTACLE_R", None)
            OBS_VEL = obs.get("PED_VEL", None)             
            strat_pos = obs.get("START_POS", [])
            start_x, start_y = strat_pos                                # Unpack into two variables

            self.states[:, obs_idx*4] = start_x
            self.states[:, obs_idx*4+1] = start_y
            self.states[:, obs_idx*4+2] = OBS_VEL  
            self.states[:, obs_idx*4+3] = OBSTACLE_R
        return self.states  # Return all states for all obstacles
    

    def update(self, veh_pos, time):

        vehicle_x = veh_pos[0]
        vehicle_y = veh_pos[1]

        for i in range(self.numofobs):  # Loop through all obstacles

            obs_x_current = self.sudden_obs_states[0, i*4]
            obs_y_current = self.sudden_obs_states[0, i*4+1]
            obs_vel = self.sudden_obs_states[0, i*4+2]  # Get velocity for each obstacle
            x_end = self.end_x1[i]
            y_end = self.end_y1[i]

            distance_to_end = np.sqrt((x_end - obs_x_current)**2 + (y_end - obs_y_current)**2)
            step_distance =  1 * time

            if distance_to_end <= step_distance:
                x_next = x_end
                y_next = y_end
                self.sudden_obs_states[:,i*4] = x_end
                self.sudden_obs_states[:,i*4+1] = y_end

            direction_x = (x_end - obs_x_current) / distance_to_end
            direction_y = (y_end - obs_y_current) / distance_to_end
            x_next = obs_x_current + direction_x * step_distance
            y_next = obs_y_current + direction_y * step_distance
            self.sudden_obs_states[:,i*4] = x_next
            self.sudden_obs_states[:,i*4+1] = y_next

            self.obs_xn[:,i] = x_next
            self.obs_yn[:,i] = y_next

            self.distance_to_obs = np.sqrt((vehicle_x - obs_x_current)**2 + (vehicle_y - obs_y_current)**2)
            if self.distance_to_obs <= self.obs_dis[i]:
                x_end = self.end_x2[i]
                y_end = self.end_y2[i]
                distance_to_end = np.sqrt((x_end - obs_x_current)**2 + (y_end - obs_y_current)**2)
                step_distance =  obs_vel * time
                if distance_to_end <= step_distance:
                    x_next = x_end
                    y_next = y_end
                    self.sudden_obs_states[:,i*4] = x_end
                    self.sudden_obs_states[:,i*4+1] = y_end
                direction_x = (x_end - obs_x_current) / distance_to_end
                direction_y = (y_end - obs_y_current) / distance_to_end
                x_next = obs_x_current + direction_x * step_distance
                y_next = obs_y_current + direction_y * step_distance
                self.sudden_obs_states[:,i*4] = x_next
                self.sudden_obs_states[:,i*4+1] = y_next
            
            self.states[:, i*4]   = self.sudden_obs_states[:, i*4]  
            self.states[:, i*4+1] = self.sudden_obs_states[:, i*4+1]  
            self.states[:, i*4+2] = self.sudden_obs_states[:, i*4+2]  
            self.states[:, i*4+3] = self.sudden_obs_states[:, i*4+3]  
        return self.states
        

    def nstep_update(self, veh_pos, time):
        vehicle_x = veh_pos[0]
        vehicle_y = veh_pos[1]
        obs_x_current = self.sudden_obs_states[0]
        obs_y_current = self.sudden_obs_states[1]
        x_end = self.end_x1
        y_end = self.end_y1
        distance_to_end = np.sqrt((x_end - obs_x_current)**2 + (y_end - obs_y_current)**2)
        step_distance =  1 * time
        if distance_to_end <= step_distance:
            x_next = x_end
            y_next = y_end
            self.sudden_obs_states[0] = x_end
            self.sudden_obs_states[1] = y_end
            print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Stopped after this")
        direction_x = (x_end - obs_x_current) / distance_to_end
        direction_y = (y_end - obs_y_current) / distance_to_end
        x_next = obs_x_current + direction_x * step_distance
        y_next = obs_y_current + direction_y * step_distance
        self.sudden_obs_states[0] = x_next
        self.sudden_obs_states[1] = y_next

        self.obs_xn[:] = x_next
        self.obs_yn[:] = y_next

        self.distance_to_obs = np.sqrt((vehicle_x - self.sudden_obs_states[0])**2 + (vehicle_y - self.sudden_obs_states[1])**2)

        if self.distance_to_obs <= self.OBS_DIS:
            obs_x_current = self.sudden_obs_states[0]
            obs_y_current = self.sudden_obs_states[1]
            x_end = self.end_x2
            y_end = self.end_y2
            distance_to_end = np.sqrt((x_end - obs_x_current)**2 + (y_end - obs_y_current)**2)
            step_distance =  self.sudden_obs_states[2] * time
            if distance_to_end <= step_distance:
                x_next = x_end
                y_next = y_end
                self.sudden_obs_states[0] = x_end
                self.sudden_obs_states[1] = y_end
                print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! Stopped after this")
            direction_x = (x_end - obs_x_current) / distance_to_end
            direction_y = (y_end - obs_y_current) / distance_to_end
            x_next = obs_x_current + direction_x * step_distance
            y_next = obs_y_current + direction_y * step_distance
            self.sudden_obs_states[0] = x_next
            self.sudden_obs_states[1] = y_next
            self.distance_to_vehicle = np.sqrt((vehicle_x - self.sudden_obs_states[0])**2 + (vehicle_y - self.sudden_obs_states[1])**2)  
            self.obs_xn[0] = obs_x_current
            self.obs_yn[0] = obs_y_current
            self.obs_xn[1] = x_next
            self.obs_yn[1] = y_next

            for i in range(2,self.N+1):
                distance_to_end = np.sqrt((x_end - self.obs_xn[i-1])**2 + (y_end - self.obs_yn[i-1])**2)
                direction_x = (x_end - self.obs_xn[i-1]) / distance_to_end
                direction_y = (y_end - self.obs_yn[i-1]) / distance_to_end
                self.obs_xn[i] = self.obs_xn[i-1] + direction_x * step_distance
                self.obs_yn[i] = self.obs_yn[i-1] + direction_y * step_distance
        print("NNNNNNNNNNNNNNNNNNN      N STEP:    ", self.obs_xn, self.obs_yn)
        # not sure
        self.states = self.sudden_obs_states
        return self.states




class LaneScenario(Scenario):

    def __init__(self, trajectory, sce_para, mpc_para):
        super().__init__()

        self.traj = trajectory  # Expecting a `Trajectory` instance
        scenario_params = sce_para.get("params", {})  # Get the parameters of the scenario
        self.OBSTACLE_R = scenario_params.get("OBSTACLE_R", None)  # Extract OBSTACLE_R
        self.LANE_WIDTH = scenario_params.get("LANE_WIDTH", None)  # Extract OBSTACLE_R

        self.states = np.empty((6))  # [x_left, y_left, x_right, y_right, center_x, center_y]

    def set_obs_init(self):
        # it should send only 4 states..
        pass

    def set_lanes(self, CX, CY):
        refx = self.traj.wpt_ref[0]
        refy = self.traj.wpt_ref[1]

        distances = np.sqrt((refx - CX) ** 2 + (refy - CY) ** 2)
        closest_index = np.argmin(distances) + 10

        dx = np.diff(refx)
        dy = np.diff(refy)
        length = np.sqrt(dx**2 + dy**2)
        dx /= length
        dy /= length

        # Compute perpendicular offsets
        offset_x = -dy * self.LANE_WIDTH
        offset_y = dx * self.LANE_WIDTH

        # Compute lane coordinates
        x_left = refx[:-1] + offset_x
        y_left = refy[:-1] + offset_y
        x_right = refx[:-1] - offset_x
        y_right = refy[:-1] - offset_y

        self.states[0] = x_left[closest_index]
        self.states[1] = y_left[closest_index]
        self.states[2] = x_right[closest_index]
        self.states[3] = y_right[closest_index]
        self.states[4] = refx[closest_index]
        self.states[5] = refy[closest_index]

    def update(self, CX, CY):
        """Update the lane states based on the current position."""
        return self.states