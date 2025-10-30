import numpy as np
import casadi
import time
from .constraints import Constraints as cs
from .cost import Cost as ct


class MPC_HOCBF_QP:

    def _parameter_assign(self, all_para):

        self.para  = all_para
        self.N = all_para.mpc_params.get("N")
        self.DT = all_para.mpc_params.get("DT")
        self.LHD = all_para.mpc_params.get("LHD")

    def __init__(self, para, obs_init):
        
        self._parameter_assign(para)

        self.mpc = MPC(para)
        self.qp  = HOCBFQP(para, obs_init)
        print('initialize MPC-QP algorithm')

    def update_n_solve(self, ref_states, cur_states, obs_states, pre_input, sensor_meas, veh_pos_samples):

        self.mpc.update_reference(ref_states[0], ref_states[1], ref_states[2], ref_states[3])
        self.mpc.update_initial_condition(cur_states[0],cur_states[1], cur_states[2], cur_states[3])
        self.mpc.update_previous_input(pre_input[0], pre_input[1])

        mpc_sol_dict = self.mpc.solve()

        acc   = mpc_sol_dict['u_control'][0]
        steer = mpc_sol_dict['u_control'][1]
        # prediction point
        z_predx = mpc_sol_dict['z_mpc'][self.N-1][0]
        z_predy = mpc_sol_dict['z_mpc'][self.N-1][1]
        u_nom = mpc_sol_dict['u_control']

        self.qp.update_nominal_obs(u_nom, obs_states, cur_states)
        qp_sol_dict = self.qp.solve()

        sol_dict = {}
        sol_dict['u_control']  = qp_sol_dict['u_control']     # control input to apply based on solution
        sol_dict['optimal']    = qp_sol_dict['optimal'] and mpc_sol_dict['optimal']        # whether the solution is optimal or not
        sol_dict['solve_time'] = mpc_sol_dict['solve_time'] + qp_sol_dict['solve_time']
        sol_dict['u_mpc']      = mpc_sol_dict['u_mpc']          # solution inputs (N by 2, see self.u_dv above) 
        sol_dict['z_mpc']      = mpc_sol_dict['z_mpc']          # solution states (N+1 by 4, see self.z_dv above)
        sol_dict['sl_mpc']     = mpc_sol_dict['sl_mpc']         # solution slack vars (N by 2, see self.sl_dv above)
        sol_dict['z_ref']      = mpc_sol_dict['z_ref']          # state reference (N by 4, see self.z_ref above)
        sol_dict['mpc_cost']   = mpc_sol_dict['mpc_cost'] + qp_sol_dict['cbf_qp_cost']
        sol_dict['obs_states'] = qp_sol_dict['obs_states']     # solution inputs (N by 2, see self.u_dv above) 
        sol_dict['mpc_qp_optimal']    = {'mpc': mpc_sol_dict['optimal'], 'qp': qp_sol_dict['optimal']}        # whether the solution is optimal or not
        sol_dict['mpc_iteration'] = mpc_sol_dict['iteration_count']
        sol_dict['qp_iteration'] = qp_sol_dict['iteration_count']
        sol_dict['total_iteration'] = mpc_sol_dict['iteration_count'] + qp_sol_dict['iteration_count']
        sol_dict['mpc_solve_time'] = mpc_sol_dict['solve_time']
        sol_dict['qp_solve_time'] = qp_sol_dict['solve_time']

        return sol_dict


class MPC:

    def _parameter_assign(self, all_para):

        self.para  = all_para
        self.L_F = all_para.vehicle_params.get("L_F")
        self.L_R = all_para.vehicle_params.get("L_R")
        self.N = all_para.mpc_params.get("N")
        self.DT = all_para.mpc_params.get("DT")
        self.LHD = all_para.mpc_params.get("LHD")
        scenarios = all_para.scenarios
        for scenario in scenarios:
            self.numofobs = scenario.get("numofobs", 0)
            self.obstacles = scenario.get("params", {})  # Get the parameters of the scenario
            self.mpc_max_cpu_time = scenario.get("mpc_max_cpu_time", 0.1)  # Extract the number of obstacles for this scenario
            self.mpc_max_iteration = scenario.get("mpc_max_iteration", 100)  # Extract the number of obstacles for this scenario

    def __init__(self, para):

        self.para = para
        self.mpc_opti = casadi.Opti()

        self._parameter_assign(para)
        self.u_prev  = self.mpc_opti.parameter(2) # previous input: [u_{acc, -1}, u_{df, -1}]
        self.z_curr  = self.mpc_opti.parameter(4) # current state:  [x_0, y_0, psi_0, v_0]

        self.z_ref = self.mpc_opti.parameter(self.N, 4)

        self.x_ref   = self.z_ref[:,0]
        self.y_ref   = self.z_ref[:,1]
        self.psi_ref = self.z_ref[:,2]
        self.v_ref   = self.z_ref[:,3]

        self.z_dv = self.mpc_opti.variable(self.N+1, 4)
        self.u_dv = self.mpc_opti.variable(self.N, 2)
        self.sl_dv  = self.mpc_opti.variable(self.N , 2)
        self.obs = self.mpc_opti.parameter(self.N+1, 4) # obs, x, y, v, r

        constraints = cs(self.mpc_opti, self.para, self.u_prev, self.z_curr, self.z_dv, self.u_dv, self.sl_dv, self.obs)
        constraints.all_nocbf_constraints()

        self.cost = ct(self.mpc_opti, self.para, self.z_dv, self.z_ref, self.u_dv, self.sl_dv)
        self.cost.add_cost()


        self.update_initial_condition(0., 0., 0., 1.)
        
        self.update_reference([self.DT * (x+1) for x in range(self.N)],
                                self.N*[0.], 
                                self.N*[0.], 
                                self.N*[1.]
                                )

        self.update_previous_input(0., 0.)
        p_opts = {'expand': True}
        s_opts = {'max_cpu_time': self.mpc_max_cpu_time, 'print_level': 0} 
        # s_opts = {'max_iter':self.mpc_max_iteration, 'print_level': 0} 

        self.mpc_opti.solver('ipopt', p_opts, s_opts)

        print('initialize MPC-QP - MPC optimizer ')

    def update_initial_condition(self, x0, y0, psi0, vel0):
        self.mpc_opti.set_value(self.z_curr, [x0, y0, psi0, vel0])

    def update_reference(self, x_ref, y_ref, psi_ref, v_ref):
        self.mpc_opti.set_value(self.x_ref,   x_ref)
        self.mpc_opti.set_value(self.y_ref,   y_ref)
        self.mpc_opti.set_value(self.psi_ref, psi_ref)
        self.mpc_opti.set_value(self.v_ref,   v_ref)

    def update_previous_input(self, acc_prev, df_prev):
        self.mpc_opti.set_value(self.u_prev, [acc_prev, df_prev])

    def solve(self):
        st = time.time()
        infeas = []
        iteration_count = 0
        try:
            self.sol = self.mpc_opti.solve()
            # Optimal solution.
            u_mpc  = self.sol.value(self.u_dv)
            z_mpc  = self.sol.value(self.z_dv)
            sl_mpc = self.sol.value(self.sl_dv)
            z_ref  = self.sol.value(self.z_ref)
            mpc_cost  = self.sol.value(self.cost.total_cost)
            iteration_count = self.mpc_opti.stats()["iter_count"]
            is_opt = True
        except:
            # Suboptimal solution (e.g. timed out).
            u_mpc  = self.mpc_opti.debug.value(self.u_dv)
            z_mpc  = self.mpc_opti.debug.value(self.z_dv)
            sl_mpc = self.mpc_opti.debug.value(self.sl_dv)
            z_ref  = self.mpc_opti.debug.value(self.z_ref)
            mpc_cost  = self.mpc_opti.debug.value(self.cost.total_cost)
            iteration_count = self.mpc_opti.stats()["iter_count"]

            # infeasible solution check!!!!
            infeas = infeas.append(self.mpc_opti.debug.show_infeasibilities())
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
        # sol_dict['infeas']      = infeas        # state reference (N by 4, see self.z_ref above) 
        sol_dict['iteration_count']      = iteration_count        # state reference (N by 4, see self.z_ref above) 

        return sol_dict


class HOCBFQP:

    def _parameter_assign(self, all_para):

        self.para  = all_para

        self.V_MIN = all_para.vehicle_params.get("V_MIN")
        self.V_MAX = all_para.vehicle_params.get("V_MAX")
        self.A_MIN = all_para.vehicle_params.get("A_MIN")
        self.A_MAX = all_para.vehicle_params.get("A_MAX")
        self.DF_MIN = all_para.vehicle_params.get("DF_MIN")
        self.DF_MAX = all_para.vehicle_params.get("DF_MAX")

        self.N = all_para.mpc_params.get("N")
        self.DT = all_para.mpc_params.get("DT")
        self.LHD = all_para.mpc_params.get("LHD")


        cbfpara = all_para.mpc_params.get("cbf", {})
        self.SAFETY_DIS = cbfpara.get("SAFETY_DIS", {})
        self.VEHICLE_R = cbfpara.get("VEHICLE_R", {})
        self.GAMMA = cbfpara.get("GAMMA", {})

        H_config = cbfpara.get("H", {})
        if H_config["type"] == "eye":
            self.H = np.eye(2)
        else:
            raise ValueError(f"Unsupported R type: {H_config['type']}")
        G_config = cbfpara.get("G", {})
        self.g = np.zeros(2)
        
        scenarios = all_para.scenarios
        for scenario in scenarios:
            self.numofobs = scenario.get("numofobs", 0)
            self.obstacles = scenario.get("params", {})  # Get the parameters of the scenario
            self.qp_max_cpu_time = scenario.get("qp_max_cpu_time", 0.1)  # Extract the number of obstacles for this scenario
            self.qp_max_iteration = scenario.get("qp_max_iteration", 100)  # Extract the number of obstacles for this scenario
            self.SAFETY_DIS = scenario.get("SAFETY_DIS", None)

    def __init__(self,para, obs_init):

        self.qp_opti = casadi.Opti()
        self._parameter_assign(para)
        self.u_var = self.qp_opti.variable(2)
        self.u_nom = self.qp_opti.parameter(2)
        self.optiobs = self.qp_opti.parameter(self.numofobs * 4)
        # self.optiobs_x = self.optiobs[::4]
        # self.optiobs_y = self.optiobs[1::4]
        self.x_var = self.qp_opti.parameter(4)

        self.define_cost()
        self.add_cbf_constraint()
        self.add_input_constraint()
        

        p_opts = {'expand': True}
        s_opts = {'max_cpu_time': 0.1, 'print_level': 0} 

        # s_opts = {'max_cpu_time': self.qp_max_cpu_time, 'max_iter':20, 'print_level': 0} 
        # s_opts = {'max_iter':self.qp_max_iteration, 'print_level': 0} 

        self.qp_opti.solver('ipopt', p_opts, s_opts)
        
    def define_cost(self):
        # Penalize deviations from u_nom
        self.cost = 0.5 * casadi.mtimes((self.u_var - self.u_nom).T, casadi.mtimes(self.H, (self.u_var - self.u_nom)))
        self.qp_opti.minimize(self.cost)

    def add_input_constraint(self):

        u1 = self.u_var[0]
        u2 = self.u_var[1]
        # Input Bound Constraints
        self.qp_opti.subject_to( self.qp_opti.bounded(self.A_MIN,  u1, self.A_MAX) )
        self.qp_opti.subject_to( self.qp_opti.bounded(self.DF_MIN, u2,  self.DF_MAX) )

    def define_dynamics(self, x, y, psi, v, a, delta):
        """
        Vehicle dynamics equations.
        """
        f = casadi.vertcat(
            v * casadi.cos(psi),  # dx/dt
            v * casadi.sin(psi),  # dy/dt
            0,                # dpsi/dt (no drift for simplicity)
            0                 # dv/dt (no control input)
        )
        # g = casadi.vertcat(
        #     casadi.horzcat(0, -v * casadi.sin(psi)),  # Control influence on x
        #     casadi.horzcat(0, v * casadi.cos(psi)),  # Control influence on y
        #     casadi.horzcat(0, v/self.L_R),                # Control influence on psi
        #     casadi.horzcat(1, 0)                 # Control influence on v
        # )
        
        g = casadi.vertcat(
            casadi.horzcat(0, 0),  # Control influence on x
            casadi.horzcat(0, 0),  # Control influence on y
            casadi.horzcat(0, 1),                # Control influence on psi
            casadi.horzcat(1, 0)                 # Control influence on v
        )
        return f, g
    

    def add_cbf_constraint(self):
        """
        Add CBF constraints to ensure safe distance from the obstacle.
        """

        x = self.x_var[0]
        y = self.x_var[1]
        psi = self.x_var[2]
        v = self.x_var[3]

        u1 = self.u_var[0]
        u2 = self.u_var[1] # delta (highly approximate delta and beta in our case)
        
        # Define system dynamics
        f, g = self.define_dynamics(x, y, psi, v, u1, u2)

        for obs_idx, obs in enumerate(self.obstacles):  # Loop through all obstacles
            OBSTACLE_R = obs.get("OBSTACLE_R", 0)  # Get radius for the specific obstacle
            
            obs_x = self.optiobs[obs_idx*4]
            obs_y = self.optiobs[obs_idx*4+1]

            # Define h(x)
            r = casadi.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            h = casadi.sqrt((x - obs_x)**2 + (y - obs_y)**2) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

            # Compute gradients
            grad_h_x = (x - obs_x) / r              # dh/dx
            grad_h_y = (y - obs_y) / r              # dh/dy
            # grad_h_psi = 0, grad_h_v = 0
            
            grad_h = casadi.vertcat(grad_h_x, grad_h_y, 0, 0)
            L_f_h = grad_h.T @ f  # Lie derivative of h along f
            L_g_h = grad_h.T @ g  # Lie derivative of h along g

            # Lf(Lf_h) = d(Lf_h)/dx
            L_f_h_dx = v*casadi.cos(psi)*((y-obs_y)**2-(x-obs_x)*(y-obs_y)*casadi.tan(psi))/(r**3)
            L_f_h_dy = v*casadi.sin(psi)*((x-obs_x)**2-(x-obs_x)*(y-obs_y)*casadi.cos(psi)/casadi.sin(psi))/(r**3)
            # Lfh_dpsi, Lfh_dv no need to calculate, because f_psi = 0, f_v = 0
            Lf2_h = L_f_h_dx*f[0] + L_f_h_dy*f[1]

            # LgLfhu = d(Lfh)/dpsi*delta+d(Lfh)/dv*a
            Lg_Lfhu = (v*(-(x-obs_x)*casadi.sin(psi)+(y-obs_y)*casadi.cos(psi))*u2+ ((x-obs_x)*casadi.cos(psi)+(y-obs_y)*casadi.sin(psi))*u1)/r

            term1 = Lf2_h
            term2 = Lg_Lfhu
            term3 = 2*h*L_f_h
            term4 = L_f_h**2
            term5  = 2*(h**2)*L_f_h
            term6 = h**4


            # Define constraint
            cbf_constraint = term1+term2+term3+term4+term5+term6
            self.qp_opti.subject_to(cbf_constraint >= 0)  

        print("CBF Constraint:", cbf_constraint)

    def update_nominal_obs(self, u_nom, obs_states, veh_states):
        print('enter update')
        self.qp_opti.set_value(self.u_nom, u_nom)
        self.qp_opti.set_value(self.x_var, veh_states)
        obs_cur_states = obs_states[0,:]
        self.qp_opti.set_value(self.optiobs, obs_cur_states)
        self.obs_states = obs_cur_states

    def solve(self):

        st = time.time()
        infeas = []
        iteration_count = 0
        try:
            self.sol = self.qp_opti.solve()
            # Optimal solution.
            u_cbfqp  = self.sol.value(self.u_var)
            obs_cbfqp  = self.sol.value(self.optiobs)
            cbfqp_cost  = self.sol.value(self.cost)
            iteration_count = self.qp_opti.stats()["iter_count"]

            is_opt = True
        except:
            # Suboptimal solution (e.g. timed out).
            u_cbfqp  = self.qp_opti.debug.value(self.u_var)
            obs_cbfqp  = self.qp_opti.debug.value(self.optiobs)
            cbfqp_cost  = self.qp_opti.debug.value(self.cost)
            iteration_count = self.qp_opti.stats()["iter_count"]

            # infeasible solution check!!!!
            # infeas = infeas.append(self.qp_opti.debug.show_infeasibilities())
            is_opt = False

        solve_time = time.time() - st

        sol_dict = {}
        sol_dict['u_control']       = u_cbfqp       # control input to apply based on solution
        sol_dict['optimal']         = is_opt        # whether the solution is optimal or not
        sol_dict['solve_time']      = solve_time    # how long the solver took in seconds
        sol_dict['obs_states']      = self.obs_states     # solution inputs (N by 2, see self.u_dv above) 
        sol_dict['cbf_qp_cost']     = cbfqp_cost    # state reference (N by 4, see self.z_ref above)
        sol_dict['iteration_count']          = iteration_count        # state reference (N by 4, see self.z_ref above)



        return sol_dict

