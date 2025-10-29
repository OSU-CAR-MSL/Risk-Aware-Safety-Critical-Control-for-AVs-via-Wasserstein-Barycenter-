
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

                # Automatically process all sections
        self.process_vehicle()
        self.process_trajectory()
        self.process_controller()
        self.process_dynamics()
        self.process_mpc()
        self.process_scenarios()


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

    # def process_scenarios(self):

    #     for scenario in self.config["scenarios"]:
    #         print("Raw Scenario:", scenario, type(scenario))  # Debugging output
            
    #         if not isinstance(scenario, dict):
    #             raise TypeError(f"Expected dict, got {type(scenario)}: {scenario}")

    #         # Extract scenario type (e.g., "static", "sudden_ped")
    #         scenario_type = list(scenario.keys())[0]
    #         scenario_data = scenario[scenario_type]  # Get the scenario details

    #         # Extract obstacles as parameters
    #         params = scenario_data.get("obstacles", [])
    #         numofobs = len(params)  # Count the number of obstacles

    #         # Structured format with "numofobs" before "params"
    #         structured_scenario = {
    #             "type": scenario_type,
    #             "numofobs": numofobs,  # Number of obstacles
    #             "params": params       # List of obstacles
    #         }

    #         for key, value in scenario.items():
    #             if key != scenario_type:
    #                 structured_scenario[key] = value  # Store additional fields

    #         self.scenarios.append(structured_scenario)

    #     print(f"Structured Scenarios: {self.scenarios}")

    def process_scenarios(self):
        self.scenarios = []  # Initialize the list

        for scenario in self.config["scenarios"]:
            print("Raw Scenario:", scenario, type(scenario))  # Debugging output
            
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

            # ✅ Store any additional fields in the scenario dictionary
            for key, value in scenario.items():
                if key not in {"name", "type", "obstacles"}:
                    structured_scenario[key] = value  

            self.scenarios.append(structured_scenario)

        print(f"Structured Scenarios: {self.scenarios}")




