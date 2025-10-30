from .mpc import MPC
from .mpc_dc import MPC_DC
from .mpc_dcbf import MPC_DCBF
from .mpc_qp import MPC_CBF_QP  # Example for another controller
from .mpc_drcbf import MPC_DRCBF  # Example for another controller
from .mpc_rcbf_qp import MPC_RCBF_QP  # Example for another controller
from .mpc_decbf import MPC_DECBF  # Example for another controller
from .mpc_ecbf_qp import MPC_ECBF_QP  # Example for another controller
from .mpc_dhocbf import MPC_DHOCBF
from .mpc_dqp import MPC_CBF_DQP
from .mpc_hocbf_qp import MPC_HOCBF_QP
from .mpc_dcbf_lie import MPC_DCBF_LIE
from .mpc_dcbf_nlie import MPC_DCBF_NLIE
from .mpc_wb_qp import MPC_WB_QP

def get_controller(controller_type, config, obs_init):
    """Factory method to get the appropriate controller based on the type."""
    if controller_type == "MPC-DCBF":
        return MPC_DCBF(config, obs_init)
    if controller_type == "MPC-DCBF-LIE":
        return MPC_DCBF_LIE(config, obs_init)
    if controller_type == "MPC-DCBF-NLIE":
        return MPC_DCBF_NLIE(config, obs_init)
    elif controller_type == "MPC-CBF-QP":
        return MPC_CBF_QP(config, obs_init)
    elif controller_type == "MPC-WB-QP":
        return MPC_WB_QP(config, obs_init) 
    elif controller_type == "MPC-DRCBF":
        return MPC_DRCBF(config, obs_init)
    elif controller_type == "MPC-RCBF-QP":
        return MPC_RCBF_QP(config, obs_init)
    elif controller_type == "MPC-DECBF":
        return MPC_DECBF(config, obs_init)
    elif controller_type == "MPC-ECBF-QP":
        return MPC_ECBF_QP(config, obs_init)
    elif controller_type == "MPC-DHOCBF":
        return MPC_DHOCBF(config, obs_init)
    elif controller_type == "MPC-HOCBF-QP":
        return MPC_HOCBF_QP(config, obs_init)
    elif controller_type == "MPC":
        return MPC(config, obs_init)   
    elif controller_type == "MPC-DC":
        return MPC_DC(config, obs_init)  
    else:
        raise ValueError(f"Unknown controller type: {controller_type}")
