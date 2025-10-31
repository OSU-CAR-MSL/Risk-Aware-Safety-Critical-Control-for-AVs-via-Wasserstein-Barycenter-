
class ConfigProcessor:

    def __init__(self, config):
        self.config = config
        self.vehicle_params = {}
        self.trajectory_params = {}
        self.controller_params = {}
        self.dynamics_params = {}
        self.mpc_params = {}
        self.cbf_params = {}
        self.scenarios = []
        self.sensors = {}

        # Automatically process all sections
        self.process_vehicle()
        self.process_trajectory()
        self.process_controller()
        self.process_dynamics()
        self.process_mpc()
        self.process_scenarios()
        self.process_sensors()


    def process_vehicle(self):
        self.vehicle_params = self.config.get("vehicle", {})

    def process_trajectory(self):
        self.trajectory_params = self.config.get("trajectory", {})

    def process_controller(self):
        self.controller_params = self.config.get("controller",[])

    def process_dynamics(self):
        self.dynamics_params = self.config.get("dynamics", [])

    def process_mpc(self):
        self.mpc_params = self.config.get("mpc", {})
        self.cbf_params = self.mpc_params.get("cbf", {})

    def process_scenarios(self):
        self.scenarios = []  # Initialize the list

        for scenario in self.config["scenarios"]:
            
            if not isinstance(scenario, dict):
                raise TypeError(f"Expected dict, got {type(scenario)}: {scenario}")

            #  Extract scenario name and type
            scenario_name = scenario.get("name", "Unnamed Scenario")  # Default to "Unnamed Scenario" if missing
            scenario_type = scenario.get("type", None)
            scenario_safedis = scenario.get("SAFETY_DIS", None)

            #  Ensure scenario type exists
            if scenario_type is None:
                raise ValueError(f"Scenario '{scenario_name}' is missing a 'type' field.")

            #  Extract obstacles safely
            params = scenario.get("obstacles", [])
            if not isinstance(params, list):
                raise TypeError(f"Expected 'obstacles' to be a list in scenario '{scenario_name}', but got {type(params)}.")

            numofobs = len(params)  # Count the number of obstacles

            mpc_max_cpu_time = scenario.get("mpc_max_cpu_time", None)
            mpc_max_iteration = scenario.get("mpc_max_iteration", None)
            qp_max_cpu_time = scenario.get("qp_max_cpu_time", None)
            qp_max_iteration = scenario.get("qp_max_iteration", None)

            #  Structured scenario format
            structured_scenario = {
                "name": scenario_name,  # Store the scenario name
                "type": scenario_type,  # Store the type of scenario
                "numofobs": numofobs,  # Number of obstacles
                "params": params,       # List of obstacles
                "SAFETY_DIS": scenario_safedis,
                "mpc_max_cpu_time": mpc_max_cpu_time,
                "qp_max_cpu_time": qp_max_cpu_time,
                "mpc_max_iteration": mpc_max_iteration,
                "qp_max_iteration": qp_max_iteration,

            }

            for key, value in scenario.items():
                if key not in {"name", "type", "obstacles"}:
                    structured_scenario[key] = value  

            self.scenarios.append(structured_scenario)

    def process_sensors(self):

        for sensor in self.config["sensors"]:
            print(f"sensor: {sensor['type']}")
            sensor_type = sensor["type"].lower()  # Convert to lowercase for consistency
            self.sensors[sensor_type] = {
                "weight": sensor["weight"],
                "mean_x": sensor["mean_x"],
                "mean_y": sensor["mean_y"],
                "std_x": sensor["std_x"],
                "std_y": sensor["std_y"],
                "correlation": sensor["correlation"],
                "nu": sensor["nu"],
                "sigma": sensor["sigma"],
                "scale_laplace": sensor["scale_laplace"],
                "shape_gamma": sensor["shape_gamma"],
                "scale_gamma": sensor["scale_gamma"],
            }
