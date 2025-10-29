import yaml
import random

def generate_random_scenarios(num_scenarios=5):
    scenario_types = ["static", "sudden_ped", "overtaking"]
    scenarios = []
    
    for i in range(num_scenarios):
        # scenario_type = random.choice(scenario_types)
        scenario_type = "sudden_ped"
        safety_dis = round(random.uniform(1.5, 4), 1)
        scenario = {"name": f"{i+1}", "type": scenario_type, "SAFETY_DIS": safety_dis,"obstacles": []}
        
        num_obstacles = random.randint(1, 3)
        for _ in range(num_obstacles):
            obstacle = {"OBSTACLE_R": round(random.uniform(1.5, 2.5), 1)}
            
            if scenario_type == "static":
                obstacle["PLACE_DIS"] = random.randint(20, 80)
            elif scenario_type == "sudden_ped":
                obstacle.update({
                    "PED_VEL": round(random.uniform(3, 5), 1),
                    "START_POS": [random.randint(-100, -20), random.randint(40, 100)],
                    "SUDDEN_DIS": random.randint(10, 20),
                    "END_POS_1": [random.randint(-80, -30), random.randint(50, 110)],
                    "END_POS_2": [random.randint(-60, -20), random.randint(40, 90)],
                })
            elif scenario_type == "overtaking":
                obstacle.update({
                    "OBS_VEL": random.randint(1, 4),
                    "PLACE_DIS": random.randint(20, 50),
                })
            
            scenario["obstacles"].append(obstacle)
        
        scenarios.append(scenario)
    
    return scenarios

def generate_yaml():
    data = {
        "vehicle": {
            "name": "Chevy Bolt",
            "L_F": 1.5213,
            "L_R": 1.4987,
            "V_MIN": 0.0,
            "V_MAX": 10.0,
            "A_MIN": -2.0,
            "A_MAX": 0.7,
            "DF_MIN": -0.3,
            "DF_MAX": 0.3,
            "A_DOT_MIN": -1.5,
            "A_DOT_MAX": 1.5,
            "DF_DOT_MIN": -0.3,
            "DF_DOT_MAX": 0.3,
        },
        "trajectory": {
            "path": "trajectory data/12_09_buckeyelot_striaght.csv",
            "sample": 0.05,
            "V_REF": 7,

            # "V_REF": [round(random.uniform(2, 10)) for _ in range(round(random.randint(2, 8),1))],
        },
        # "controller": ["MPC-DCBF", "MPC-DRCBF", "MPC-DECBF","MPC-CBF-QP", "MPC-RCBF-QP", "MPC-ECBF-QP"],
        "controller": ["MPC-DCBF", "MPC-CBF-QP"],
        "dynamics": ["RK2"],
        "mpc": {
            "Qposx": 500,
            "Qposy": 500,
            "Qpsi": 100,
            "Qvel": 500,
            "R": {
                "type": "diag",
                "values": [100, 50]
            },
            "N": 7,
            "DT": 0.1,
            "LHD": 10,
            "cbf": {
                "SAFETY_DIS": 3,
                "VEHICLE_R": 1.8,
                "GAMMA": 0.3,
                # ECBF, Quan 2016
                "KB": 2,
                # ECBF, one degree
                "ALPHA": 7,
                "LAMBDA": 1,
                "H": {
                    "type": "eye",
                    "values": [1, 1]
                },
                "G": {"values": None},
            },
        },
        # "scenarios": generate_random_scenarios(random.randint(2, 5)),
        "scenarios": generate_random_scenarios(5),

    }
    
    with open("utilities/config.yaml", "w") as file:
        yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    
    print("YAML file generated: generated_scenario.yaml")

if __name__ == "__main__":
    generate_yaml()
