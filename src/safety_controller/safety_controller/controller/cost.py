import casadi
import numpy as np

class Cost:
    
    def _parameter_assign(self, all_para):

        self.para  = all_para

        self.Qposx = all_para.mpc_params.get("Qposx")
        self.Qposy = all_para.mpc_params.get("Qposy")
        self.Qpsi = all_para.mpc_params.get("Qpsi")
        self.Qvel = all_para.mpc_params.get("Qvel")
        self.N = all_para.mpc_params.get("N")
        R_config = all_para.mpc_params.get("R", {})
        if R_config["type"] == "diag":
            self.R = np.diag(R_config["values"])
        else:
            raise ValueError(f"Unsupported R type: {R_config['type']}")


    def __init__(self, opti, all_para, z_dv, z_ref, u_dv, sl_dv):

        self.opti = opti
        self._parameter_assign(all_para)

        self.z_dv = z_dv
        self.z_ref = z_ref
        self.u_dv = u_dv

        self.sl_acc_dv = sl_dv[:,0]
        self.sl_df_dv  = sl_dv[:,1]


    def add_cost(self):
        def _quad_form(z, Q):
            return casadi.mtimes(z, casadi.mtimes(Q, z.T))
        
        cost = 0

        for i in range(self.N):
            if i == 0:
                cost += _quad_form(self.z_ref[i,2], self.Qpsi) # psi is already error here, so we don't need to calculate again
            cost += _quad_form(self.z_dv[i+1, 0] - self.z_ref[i,0], self.Qposx) # tracking cost
            cost += _quad_form(self.z_dv[i+1, 1] - self.z_ref[i,1], self.Qposy) # tracking cost
            cost += _quad_form(self.z_dv[i+1, 3] - self.z_ref[i,3], self.Qvel) # tracking cost

        for i in range(self.N - 1):
            cost += _quad_form(self.u_dv[i+1, :] - self.u_dv[i,:], self.R)  # input derivative cost

        cost += (casadi.sum1(self.sl_df_dv) + casadi.sum1(self.sl_acc_dv))  # slack cost
        self.total_cost = cost

        self.opti.minimize( cost )

        print('Add Cost ---- Done')
