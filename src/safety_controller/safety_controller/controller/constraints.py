import casadi
import numpy as np


class Constraints:

    def _parameter_assign(self, all_para):

        self.para  = all_para

        self.L_F = all_para.vehicle_params.get("L_F")
        self.L_R = all_para.vehicle_params.get("L_R")
        self.V_MIN = all_para.vehicle_params.get("V_MIN")
        self.V_MAX = all_para.vehicle_params.get("V_MAX")
        self.A_MIN = all_para.vehicle_params.get("A_MIN")
        self.A_MAX = all_para.vehicle_params.get("A_MAX")
        self.DF_MIN = all_para.vehicle_params.get("DF_MIN")
        self.DF_MAX = all_para.vehicle_params.get("DF_MAX")
        self.A_DOT_MIN = all_para.vehicle_params.get("A_DOT_MIN")
        self.A_DOT_MAX = all_para.vehicle_params.get("A_DOT_MAX")
        self.DF_DOT_MIN = all_para.vehicle_params.get("DF_DOT_MIN")
        self.DF_DOT_MAX = all_para.vehicle_params.get("DF_DOT_MAX")

        self.Qposx = all_para.mpc_params.get("Qposx")
        self.Qposy = all_para.mpc_params.get("Qposy")
        self.Qpsi = all_para.mpc_params.get("Qpsi")
        self.Qvel = all_para.mpc_params.get("Qvel")
        self.N = all_para.mpc_params.get("N")
        self.DT = all_para.mpc_params.get("DT")
        self.LHD = all_para.mpc_params.get("LHD")
        R_config = all_para.mpc_params.get("R", {})
        if R_config["type"] == "diag":
            self.R = np.diag(R_config["values"])
        else:
            raise ValueError(f"Unsupported R type: {R_config['type']}")
        
        
        cbfpara = all_para.mpc_params.get("cbf", {})
        self.SAFETY_DIS = cbfpara.get("SAFETY_DIS", {})
        self.VEHICLE_R = cbfpara.get("VEHICLE_R", {})
        self.GAMMA = cbfpara.get("GAMMA", {})
        self.KB = cbfpara.get("KB", {})
        self.LAMBDA = cbfpara.get("LAMBDA", {})
        self.ALPHA = cbfpara.get("ALPHA", {})

        scenarios = all_para.scenarios
        for scenario in scenarios:
            self.numofobs = scenario.get("numofobs", 0)
            self.obstacles = scenario.get("params", {})  # Get the parameters of the scenario
            self.SAFETY_DIS = scenario.get("SAFETY_DIS", None)


    def __init__(self, opti, all_para, u_prev, z_curr, z_dv, u_dv, sl_dv, optiobs):

        self.opti = opti
        self._parameter_assign(all_para)

        self.u_prev = u_prev
        self.z_curr = z_curr

        self.x_dv   = z_dv[:, 0]
        self.y_dv   = z_dv[:, 1]
        self.psi_dv = z_dv[:, 2]
        self.v_dv   = z_dv[:, 3]

        self.acc_dv = u_dv[:,0]
        self.df_dv  = u_dv[:,1]

        self.sl_acc_dv = sl_dv[:,0]
        self.sl_df_dv  = sl_dv[:,1]

        self.optiobs_x = optiobs[:,::4]
        self.optiobs_y = optiobs[:,1::4]

    def all_nocbf_constraints(self):
        self.add_initial_constraints()
        self.add_state_constraints()
        self.add_input_constraints()
        self.add_input_rate_constraints()
        self.add_dynamics_constraints()
        self.add_slack_constraints()

    def add_all_constraints(self):

        self.add_initial_constraints()
        self.add_state_constraints()
        self.add_input_constraints()
        self.add_input_rate_constraints()
        self.add_dynamics_constraints()
        self.add_slack_constraints()
        self.add_cbf_constraints()


    def add_initial_constraints(self):
        self.opti.subject_to( self.x_dv[0]   == self.z_curr[0] )
        self.opti.subject_to( self.y_dv[0]   == self.z_curr[1] )
        self.opti.subject_to( self.psi_dv[0] == self.z_curr[2] )
        self.opti.subject_to( self.v_dv[0]   == self.z_curr[3] )
        print('Add Initial Constraints ---- Done')

    def add_state_constraints(self):
        self.opti.subject_to(self.opti.bounded(self.V_MIN, self.v_dv, self.V_MAX))
        print('Add States Constraints ---- Done')

    def add_input_constraints(self):
        self.opti.subject_to(self.opti.bounded(self.A_MIN, self.acc_dv, self.A_MAX))
        self.opti.subject_to(self.opti.bounded(self.DF_MIN, self.df_dv, self.DF_MAX))
        print('Add Input Constraints ---- Done')

    def add_input_rate_constraints(self):

        self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.DT -  self.sl_acc_dv[0], 
                                                    self.acc_dv[0] - self.u_prev[0],
                                                    self.A_DOT_MAX*self.DT   + self.sl_acc_dv[0]) )

        self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.DT  -  self.sl_df_dv[0], 
                                                    self.df_dv[0] - self.u_prev[1],
                                                    self.DF_DOT_MAX*self.DT  + self.sl_df_dv[0]) )

        for i in range(self.N - 1):
            self.opti.subject_to( self.opti.bounded( self.A_DOT_MIN*self.DT   -  self.sl_acc_dv[i+1], 
                                                        self.acc_dv[i+1] - self.acc_dv[i],
                                                        self.A_DOT_MAX*self.DT   + self.sl_acc_dv[i+1]) )
            self.opti.subject_to( self.opti.bounded( self.DF_DOT_MIN*self.DT  -  self.sl_df_dv[i+1], 
                                                        self.df_dv[i+1]  - self.df_dv[i],
                                                        self.DF_DOT_MAX*self.DT  + self.sl_df_dv[i+1]) )
        print('Add Input Rate Constraints ---- Done')


    def add_dynamics_constraints(self):
        for i in range(self.N):
            beta = casadi.atan( self.L_R / (self.L_F + self.L_R) * casadi.tan(self.df_dv[i]) )

            self.opti.subject_to( self.x_dv[i+1]   == self.x_dv[i]   + self.DT * (self.v_dv[i] * casadi.cos(self.psi_dv[i] + beta)) )
            self.opti.subject_to( self.y_dv[i+1]   == self.y_dv[i]   + self.DT * (self.v_dv[i] * casadi.sin(self.psi_dv[i] + beta)) )
            self.opti.subject_to( self.v_dv[i+1]   == self.v_dv[i]   + self.DT * (self.acc_dv[i]) )
            self.opti.subject_to( self.psi_dv[i+1] == self.psi_dv[i] + self.DT * (self.v_dv[i] / self.L_R * casadi.sin(0.01)) )
        print('Add Dynamics Constraints ---- Done')

    def add_slack_constraints(self):
        self.opti.subject_to( 0 <= self.sl_df_dv )
        self.opti.subject_to( 0 <= self.sl_acc_dv )
        print('Add Slack Constraints ---- Done')

    def add_dc_constraints(self):

        for obs_idx, obs in enumerate(self.obstacles):  # Loop through all obstacles
            OBSTACLE_R = obs.get("OBSTACLE_R", 0)  # Get radius for the specific obstacle
            
            for i in range(self.N):

                h_current = casadi.sqrt(
                    (self.x_dv[i] - self.optiobs_x[i, obs_idx])**2 + 
                    (self.y_dv[i] - self.optiobs_y[i, obs_idx])**2
                ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

                dc_constraint = h_current
                self.opti.subject_to(dc_constraint > 0)  # Apply constraint

        print('Add DC Constraints ---- Done')


    def add_cbf_constraints(self):

        for obs_idx, obs in enumerate(self.obstacles):  # Loop through all obstacles
            OBSTACLE_R = obs.get("OBSTACLE_R", 0)  # Get radius for the specific obstacle
            
            for i in range(self.N):
                h_next = casadi.sqrt(
                    (self.x_dv[i+1] - self.optiobs_x[i+1, obs_idx])**2 + 
                    (self.y_dv[i+1] - self.optiobs_y[i+1, obs_idx])**2
                ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

                h_current = casadi.sqrt(
                    (self.x_dv[i] - self.optiobs_x[i, obs_idx])**2 + 
                    (self.y_dv[i] - self.optiobs_y[i, obs_idx])**2
                ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

                cbf_constraint = h_next - (1 - self.GAMMA) * h_current
                self.opti.subject_to(cbf_constraint > 0)  # Apply constraint

        print('Add CBF Constraints ---- Done')

    def add_rcbf_constraints(self):
        eps = 1e-9
        for obs_idx, obs in enumerate(self.obstacles):  # Loop through all obstacles
            OBSTACLE_R = obs.get("OBSTACLE_R", 0)  # Get radius for the specific obstacle
            
            for i in range(self.N):
                h_next = casadi.sqrt(
                    (self.x_dv[i+1] - self.optiobs_x[i+1, obs_idx])**2 + 
                    (self.y_dv[i+1] - self.optiobs_y[i+1, obs_idx])**2
                ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

                h_current = casadi.sqrt(
                    (self.x_dv[i] - self.optiobs_x[i, obs_idx])**2 + 
                    (self.y_dv[i] - self.optiobs_y[i, obs_idx])**2
                ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

                B_current = -casadi.log((h_current+eps)/(h_current+1+eps))
                B_next = -casadi.log((h_next+eps)/(h_next+1+eps))

                # B_current = -casadi.log(1/(h_current+eps))
                # B_next = -casadi.log(1/(h_next+eps))
                rcbf_constraint = (B_next-B_current) - (self.GAMMA) * h_current
                self.opti.subject_to(rcbf_constraint <= 0)  # Apply constraint

        print('Add RCBF Constraints ---- Done')

    def add_ecbf_constraints(self):

        eps = 1e-9

        for obs_idx, obs in enumerate(self.obstacles):  # Loop through all obstacles
            OBSTACLE_R = obs.get("OBSTACLE_R", 0)  # Get radius for the specific obstacle
            
            for i in range(self.N):
                h_next = casadi.sqrt(
                    (self.x_dv[i+1] - self.optiobs_x[i+1, obs_idx])**2 + 
                    (self.y_dv[i+1] - self.optiobs_y[i+1, obs_idx])**2
                ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

                h_current = casadi.sqrt(
                    (self.x_dv[i] - self.optiobs_x[i, obs_idx])**2 + 
                    (self.y_dv[i] - self.optiobs_y[i, obs_idx])**2
                ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

                B_current = -casadi.log((h_current+eps)/(h_current+1+eps))
                B_next = -casadi.log((h_next+eps)/(h_next+1+eps))

                # Quan 2016
                # B_current = -casadi.log(1/(h_current+eps))
                # B_next = -casadi.log(1/(h_next+eps))
                # ecbf_constraint = (B_next-B_current) + (self.KB) * B_current

                #
                ecbf_constraint = (h_next-h_current) + (self.ALPHA) * casadi.exp(-self.LAMBDA*self.DT)*h_current

                self.opti.subject_to(ecbf_constraint >= 0)  # Apply constraint


        print('Add ECBF Constraints ---- Done')
        
    def add_dhocbf_constraints(self):

        alpha1 = 0.5
        alpha2 = 0.5
        alpha3 = 0.5
        for obs_idx, obs in enumerate(self.obstacles):  # Loop through all obstacles
            OBSTACLE_R = obs.get("OBSTACLE_R", 0)  # Get radius for the specific obstacle
            
            for i in range(self.N-1): # there is i+2 for 2 steps cbf

                h_2next = casadi.sqrt(
                    (self.x_dv[i+2] - self.optiobs_x[i+2, obs_idx])**2 + 
                    (self.y_dv[i+2] - self.optiobs_y[i+2, obs_idx])**2
                ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

                h_next = casadi.sqrt(
                    (self.x_dv[i+1] - self.optiobs_x[i+1, obs_idx])**2 + 
                    (self.y_dv[i+1] - self.optiobs_y[i+1, obs_idx])**2
                ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

                h_current = casadi.sqrt(
                    (self.x_dv[i] - self.optiobs_x[i, obs_idx])**2 + 
                    (self.y_dv[i] - self.optiobs_y[i, obs_idx])**2
                ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

                psi0 = h_current
                d_psi0 = h_next - h_current
                d_npsi0 = h_2next-h_next

                psi1 = d_psi0 + alpha1*h_current
                psi1_next = d_npsi0+alpha1*h_next

                d_psi1 = psi1_next - psi1
                psi2 = d_psi1+alpha2*psi1

                self.opti.subject_to(psi0 >= 0)  # Apply constraint
                self.opti.subject_to(psi1 >= 0)  # Apply constraint
                self.opti.subject_to(psi2 >= 0)  # Apply constraint


        print('Add ECBF Constraints ---- Done')
        
    def add_cbf_lie_constraints(self):

        LR = self.L_R

        def define_dynamics(psi, v):
            """
            Vehicle dynamics equations.
            """
            f = casadi.vertcat(
                v * casadi.cos(psi),  # dx/dt
                v * casadi.sin(psi),  # dy/dt
                0,                # dpsi/dt (no drift for simplicity)
                0                 # dv/dt (no control input)
            )
            g = casadi.vertcat(
                casadi.horzcat(0, -v * casadi.sin(psi)),  # Control influence on x
                casadi.horzcat(0, v * casadi.cos(psi)),  # Control influence on y
                casadi.horzcat(0, v/LR),                # Control influence on psi
                casadi.horzcat(1, 0)                 # Control influence on v
            )
            return f, g
        
        x = self.z_curr[0]
        y = self.z_curr[1]
        psi = self.z_curr[2]
        v = self.z_curr[3]
        u1 = self.u_prev[0]

        # u2 = np.atan((self.L_R/(self.L_F+self.L_R))*np.tan(self.u_var[1])) # beta
        u2 = self.u_prev[1] # delta (highly approximate delta and beta in our case)
        

        # Define system dynamics
        f, g = define_dynamics(psi, v)

        # for obs_idx, obs in enumerate(self.obstacles):  # Loop through all obstacles
        #     OBSTACLE_R = obs.get("OBSTACLE_R", 0)  # Get radius for the specific obstacle
            
        #     for i in range(self.N):
        #         h_next = casadi.sqrt(
        #             (self.x_dv[i+1] - self.optiobs_x[i+1, obs_idx])**2 + 
        #             (self.y_dv[i+1] - self.optiobs_y[i+1, obs_idx])**2
        #         ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

        #         h_current = casadi.sqrt(
        #             (self.x_dv[i] - self.optiobs_x[i, obs_idx])**2 + 
        #             (self.y_dv[i] - self.optiobs_y[i, obs_idx])**2
        #         ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

        #         cbf_constraint = h_next - (1 - self.GAMMA) * h_current
        #         self.opti.subject_to(cbf_constraint > 0)  # Apply constraint


        for obs_idx, obs in enumerate(self.obstacles):  # Loop through all obstacles
            OBSTACLE_R = obs.get("OBSTACLE_R", 0)  # Get radius for the specific obstacle
            
            obs_x = self.optiobs_x[obs_idx*4]
            obs_y = self.optiobs_y[obs_idx*4+1]

            # Define h(x)
            r = casadi.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            h = casadi.sqrt((x - obs_x)**2 + (y - obs_y)**2) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

            # Compute gradients
            grad_h_x = (x - obs_x) / casadi.sqrt((x - obs_x)**2 + (y - obs_y)**2)
            grad_h_y = (y - obs_y) / casadi.sqrt((x - obs_x)**2 + (y - obs_y)**2)

            grad_h = casadi.vertcat(grad_h_x, grad_h_y, 0, 0)
            L_f_h = grad_h.T @ f  # Lie derivative of h along f
            L_g_h = grad_h.T @ g  # Lie derivative of h along g

            # Define constraint
            cbf_constraint = L_f_h + L_g_h @ casadi.vertcat(self.acc_dv[0], self.df_dv[0]) + h
            self.opti.subject_to(cbf_constraint > 0)  

        print('Add CBF Constraints ---- Done')

    def add_cbf_nlie_constraints(self):

        LR = self.L_R

        def define_dynamics(psi, v):
            """
            Vehicle dynamics equations.
            """
            f = casadi.vertcat(
                v * casadi.cos(psi),  # dx/dt
                v * casadi.sin(psi),  # dy/dt
                0,                # dpsi/dt (no drift for simplicity)
                0                 # dv/dt (no control input)
            )
            g = casadi.vertcat(
                casadi.horzcat(0, -v * casadi.sin(psi)),  # Control influence on x
                casadi.horzcat(0, v * casadi.cos(psi)),  # Control influence on y
                casadi.horzcat(0, v/LR),                # Control influence on psi
                casadi.horzcat(1, 0)                 # Control influence on v
            )
            return f, g
                

        # Define system dynamics

        for obs_idx, obs in enumerate(self.obstacles):  # Loop through all obstacles
            OBSTACLE_R = obs.get("OBSTACLE_R", 0)  # Get radius for the specific obstacle
            cbf_constraint = []
            for i in range(self.N):

                f, g = define_dynamics(self.psi_dv[i], self.v_dv[i])

                h = casadi.sqrt(
                    (self.x_dv[i] - self.optiobs_x[i, obs_idx])**2 + 
                    (self.y_dv[i] - self.optiobs_y[i, obs_idx])**2
                ) - (self.VEHICLE_R + OBSTACLE_R + self.SAFETY_DIS)

                grad_h_x = (self.x_dv[i] - self.optiobs_x[i, obs_idx]) / casadi.sqrt((self.x_dv[i] - self.optiobs_x[i, obs_idx])**2 + (self.y_dv[i] - self.optiobs_y[i, obs_idx])**2)
                grad_h_y = (self.y_dv[i] - self.optiobs_y[i, obs_idx]) / casadi.sqrt((self.x_dv[i] - self.optiobs_x[i, obs_idx])**2 + (self.y_dv[i] - self.optiobs_y[i, obs_idx])**2)

                grad_h = casadi.vertcat(grad_h_x, grad_h_y, 0, 0)
                L_f_h = grad_h.T @ f  # Lie derivative of h along f
                L_g_h = grad_h.T @ g  # Lie derivative of h along g

                cbf_constraint.append(L_f_h + L_g_h @ casadi.vertcat(self.acc_dv[i], self.df_dv[i]) + h)
                self.opti.subject_to(cbf_constraint[i] > 0)  


        print('Add CBF Constraints ---- Done')