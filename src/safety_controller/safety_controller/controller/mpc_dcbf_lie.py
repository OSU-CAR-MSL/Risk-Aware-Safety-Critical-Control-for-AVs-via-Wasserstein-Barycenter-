import numpy as np
import casadi
import time
from .constraints import Constraints as cs
from .cost import Cost as ct


class MPC_DCBF_LIE:

    def _parameter_assign(self, all_para):

        self.para  = all_para
        self.N = all_para.mpc_params.get("N")
        self.DT = all_para.mpc_params.get("DT")
        self.LHD = all_para.mpc_params.get("LHD")
        for scenario in all_para.scenarios:
            self.numobs = scenario.get("numofobs", 0)  # Extract the number of obstacles for this scenario
            self.mpc_max_cpu_time = scenario.get("mpc_max_cpu_time", 0.01)  # Extract the number of obstacles for this scenario
            self.mpc_max_iteration = scenario.get("mpc_max_iteration", 0.01)  # Extract the number of obstacles for this scenario

    def __init__(self, para, obs_init):

        print('initialize MPC-DCBF algorithm')

        # get parameters from config

        self._parameter_assign(para)

        self.opti = casadi.Opti()

 
        self.u_prev  = self.opti.parameter(2) # previous input: [u_{acc, -1}, u_{df, -1}]
        self.z_curr  = self.opti.parameter(4) # current state:  [x_0, y_0, psi_0, v_0]

        self.z_ref = self.opti.parameter(self.N, 4)

        self.x_ref   = self.z_ref[:,0]
        self.y_ref   = self.z_ref[:,1]
        self.psi_ref = self.z_ref[:,2]
        self.v_ref   = self.z_ref[:,3]

        self.z_dv = self.opti.variable(self.N+1, 4)
        self.u_dv = self.opti.variable(self.N, 2)
        self.sl_dv  = self.opti.variable(self.N , 2)

        self.obs = self.opti.parameter(self.N+1, self.numobs * 4)

        constraints = cs(self.opti, self.para, self.u_prev, self.z_curr, self.z_dv, self.u_dv, self.sl_dv, self.obs)
        constraints.all_nocbf_constraints()
        constraints.add_cbf_lie_constraints()

        self.cost = ct(self.opti, self.para, self.z_dv, self.z_ref, self.u_dv, self.sl_dv)
        self.cost.add_cost()


        self.update_initial_condition(0., 0., 0., 1.)
        
        self.update_reference([self.DT * (x+1) for x in range(self.N)],
                                self.N*[0.], 
                                self.N*[0.], 
                                self.N*[1.]
                                )

        self.update_previous_input(0., 0.)

       
        initial_obs_values = np.zeros((self.N+1, self.numobs * 4))          # Initialize to zeros, replace with your values
        # print('self.numobs ', self.numobs )
        # Loop over each obstacle
        for i in range(self.numobs):  
            initial_obs_values[:, i * 4]     = obs_init[:, i * 4]  # x positions
            initial_obs_values[:, i * 4 + 1] = obs_init[:, i * 4 + 1]  # y positions
            initial_obs_values[:, i * 4 + 2] = obs_init[:, i * 4 + 2]  # constant velocity
            initial_obs_values[:, i * 4 + 3] = obs_init[:, i * 4 + 3]  # constant radius
        
        self.obs_states = obs_init
        self.update_obs_state(initial_obs_values)

        p_opts = {'expand': True}
        s_opts = {'max_cpu_time': self.mpc_max_cpu_time, 'print_level': 0} 

        # s_opts = {'max_iter':self.mpc_max_iteration, "tol": 1e-6,'acceptable_tol': 1e-5, 'bound_relax_factor': 1e-6, 'constr_viol_tol': 1e-5,'print_level': 0} 
        # s_opts = {'max_iter':self.mpc_max_iteration,'print_level': 0} 

        self.opti.solver('ipopt', p_opts, s_opts)
    

    def update_initial_condition(self, x0, y0, psi0, vel0):
        self.opti.set_value(self.z_curr, [x0, y0, psi0, vel0])

    def update_n_solve(self, ref_states, cur_states, obs_states, pre_input, sensor_meas, veh_pos_samples):

        self.update_obs_state(obs_states)
        self.update_reference(ref_states[0], ref_states[1], ref_states[2], ref_states[3])
        self.update_initial_condition(cur_states[0],cur_states[1], cur_states[2], cur_states[3])
        self.update_previous_input(pre_input[0], pre_input[1])

        sol_dict = self.solve()

        return sol_dict


    def update_reference(self, x_ref, y_ref, psi_ref, v_ref):
        self.opti.set_value(self.x_ref,   x_ref)
        self.opti.set_value(self.y_ref,   y_ref)
        self.opti.set_value(self.psi_ref, psi_ref)
        self.opti.set_value(self.v_ref,   v_ref)

    def update_obs_state(self, obs_states):
        self.opti.set_value(self.obs, obs_states) # Update the obstacle states here in real time 

    def update_previous_input(self, acc_prev, df_prev):
        self.opti.set_value(self.u_prev, [acc_prev, df_prev])

    def solve(self):
        st = time.time()
        infeas = []
        try:
            self.sol = self.opti.solve()
            # Optimal solution.
            u_mpc  = self.sol.value(self.u_dv)
            z_mpc  = self.sol.value(self.z_dv)
            sl_mpc = self.sol.value(self.sl_dv)
            z_ref  = self.sol.value(self.z_ref)
            mpc_cost  = self.sol.value(self.cost.total_cost)
            is_opt = True
        except:
            # Suboptimal solution (e.g. timed out).
            u_mpc  = self.opti.debug.value(self.u_dv)
            z_mpc  = self.opti.debug.value(self.z_dv)
            sl_mpc = self.opti.debug.value(self.sl_dv)
            z_ref  = self.opti.debug.value(self.z_ref)
            mpc_cost  = self.opti.debug.value(self.cost.total_cost)
            g_values =self.opti.debug.value(self.opti.g)  # Get constraint values
            g_lbound = self.opti.debug.value(self.opti.debug.lbg)
            g_ubound = self.opti.debug.value(self.opti.debug.ubg)            # infeasible solution check!!!!

            infeas_results = [
                (i, g_values[i], g_lbound[i], g_ubound[i], self.opti.debug.g_describe(i))
                for i in range(len(g_values))
                # if g_values[i] < g_lbound[i] or g_values[i] > g_ubound[i]
            ]        
            infeas = {
                "infeasible_constraints": [
                    {
                        "index": i,
                        "description": desc,
                        "value": g_val,
                        "lower_bound": lb,
                        "upper_bound": ub
                    }
                    for i, g_val, lb, ub, desc in infeas_results
                ]
            }    
            # for i in range(self.opti.ng):
            #     constraint_value = self.opti.value(self.opti.g[i])  # Get current constraint value
            #     lower_bound = self.opti.lbg[i]
            #     upper_bound = self.opti.ubg[i]

            #     if constraint_value < lower_bound or constraint_value > upper_bound:
            #         infeas.append(self.opti.debug.g_describe(i))
            #         print('g_describe', self.opti.debug.g_describe(i))
            is_opt = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict['u_control']  = u_mpc[0,:]     # control input to apply based on solution
        sol_dict['optimal']    = is_opt         # whether the solution is optimal or not
        sol_dict['solve_time'] = solve_time     # how long the solver took in seconds
        sol_dict['u_mpc']      = u_mpc          # solution inputs (N by 2, see self.u_dv above) 
        sol_dict['z_mpc']      = z_mpc          # solution states (N+1 by 4, see self.z_dv above)
        sol_dict['sl_mpc']     = sl_mpc         # solution slack vars (N by 2, see self.sl_dv above)
        sol_dict['z_ref']      = z_ref          # state reference (N by 4, see self.z_ref above)
        sol_dict['mpc_cost']    = mpc_cost      # cost
        sol_dict['obs_states']  = self.obs_states       
        if infeas:
            sol_dict['infeas']      = infeas        # state reference (N by 4, see self.z_ref above) 

        return sol_dict