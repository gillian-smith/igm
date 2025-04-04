
from igm.modules.utils import *

def params_optimize(parser):
   
    parser.add_argument(
        "--opti_vars_to_save",
        type=list,
        default=[
            "usurf",
            "thk",
            "slidingco",
            "velsurf_mag",
            "velsurfobs_mag",
            "divflux",
            "icemask",
        ],
        help="List of variables to be recorded in the ncdef file",
    )
    parser.add_argument(
        "--opti_init_zero_thk",
        type=str2bool,
        default="False",
        help="Initialize the optimization with zero ice thickness",
    )
    parser.add_argument(
        "--opti_regu_param_thk",
        type=float,
        default=10.0,
        help="Regularization weight for the ice thickness in the optimization",
    )
    parser.add_argument(
        "--opti_regu_param_slidingco",
        type=float,
        default=1,
        help="Regularization weight for the slidingco field in the optimization",
    )
    parser.add_argument(
        "--opti_regu_param_arrhenius",
        type=float,
        default=10.0,
        help="Regularization weight for the arrhenius field in the optimization",
    )
    parser.add_argument(
        "--opti_regu_param_div",
        type=float,
        default=1,
        help="Regularization weight for the divrgence field in the optimization",
    )
    parser.add_argument(
        "--opti_smooth_anisotropy_factor",
        type=float,
        default=0.2,
        help="Smooth anisotropy factor for the ice thickness regularization in the optimization",
    )
    parser.add_argument(
        "--opti_smooth_anisotropy_factor_sl",
        type=float,
        default=1.0,
        help="Smooth anisotropy factor for the slidingco regularization in the optimization",
    )
    parser.add_argument(
        "--opti_convexity_weight",
        type=float,
        default=0.002,
        help="Convexity weight for the ice thickness regularization in the optimization",
    )
    parser.add_argument(
        "--opti_convexity_power",
        type=float,
        default=1.3,
        help="Power b in the area-volume scaling V ~ a * A^b taking fom 'An estimate of global glacier volume', A. Grinste, TC, 2013",
    )
    parser.add_argument(
        "--opti_usurfobs_std",
        type=float,
        default=2.0,
        help="Confidence/STD of the top ice surface as input data for the optimization",
    )
    parser.add_argument(
        "--opti_velsurfobs_std",
        type=float,
        default=1.0,
        help="Confidence/STD of the surface ice velocities as input data for the optimization (if 0, velsurfobs_std field must be given)",
    )
    parser.add_argument(
        "--opti_thkobs_std",
        type=float,
        default=3.0,
        help="Confidence/STD of the ice thickness profiles (unless given)",
    )
    parser.add_argument(
        "--opti_divfluxobs_std",
        type=float,
        default=1.0,
        help="Confidence/STD of the flux divergence as input data for the optimization (if 0, divfluxobs_std field must be given)",
    )
    parser.add_argument(
        "--opti_divflux_method",
        type=str,
        default="upwind",
        help="Compute the divergence of the flux using the upwind or centered method",
    )
    parser.add_argument(
        "--opti_force_zero_sum_divflux",
        type=str2bool,
        default="False",
        help="Add a penalty to the cost function to force the sum of the divergence of the flux to be zero",
    )
    parser.add_argument(
        "--opti_save_divflux_res",
        type=str2bool,
        default="False",
        help="Save linear regression (slope and intercept) of div flux?"
    )
    parser.add_argument(
        "--opti_scaling_thk",
        type=float,
        default=2.0,
        help="Scaling factor for the ice thickness in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_scaling_usurf",
        type=float,
        default=0.5,
        help="Scaling factor for the ice thickness in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_scaling_slidingco",
        type=float,
        default=0.0001,
        help="Scaling factor for the slidingco in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_scaling_arrhenius",
        type=float,
        default=0.1,
        help="Scaling factor for the Arrhenius in the optimization, serve to adjust step-size of each controls relative to each other",
    )
    parser.add_argument(
        "--opti_control",
        type=list,
        default=["thk"],  # "slidingco", "usurf"
        help="List of optimized variables for the optimization",
    )
    parser.add_argument(
        "--opti_cost",
        type=list,
        default=["velsurf", "thk", "icemask"],  # "divfluxfcz", ,"usurf"
        help="List of cost components for the optimization",
    )
    parser.add_argument(
        "--opti_nbitmin",
        type=int,
        default=50,
        help="Min iterations for the optimization",
    )
    parser.add_argument(
        "--opti_nbitmax",
        type=int,
        default=500,
        help="Max iterations for the optimization",
    )
    parser.add_argument(
        "--opti_step_size",
        type=float,
        default=1,
        help="Step size for the optimization",
    )
    parser.add_argument(
        "--opti_step_size_decay",
        type=float,
        default=0.9,
        help="Decay step size parameter for the optimization",
    )
    parser.add_argument(
        "--opti_output_freq",
        type=int,
        default=50,
        help="Frequency of the output for the optimization",
    )
    parser.add_argument(
        "--opti_save_result_in_ncdf",
        type=str,
        default="geology-optimized.nc",
        help="Geology input file",
    )
    parser.add_argument(
        "--opti_plot2d_live",
        type=str2bool,
        default=True,
        help="plot2d_live_inversion",
    )
    parser.add_argument(
        "--opti_plot2d",
        type=str2bool,
        default=True,
        help="plot 2d inversion",
    )
    parser.add_argument(
        "--opti_plot2d_masked_vel",
        type=str2bool,
        default=False,
        help="plot masked modelled velocity in 2d inversion",
    )
    parser.add_argument(
        "--opti_save_iterat_in_ncdf",
        type=str2bool,
        default=True,
        help="write_ncdf_optimize",
    )
    parser.add_argument(
        "--opti_editor_plot2d",
        type=str,
        default="vs",
        help="optimized for VS code (vs) or spyder (sp) for live plot",
    )
    parser.add_argument(
        "--opti_uniformize_thkobs",
        type=str2bool,
        default=True,
        help="uniformize the density of thkobs",
    )
    parser.add_argument(
        "--sole_mask",
        type=str2bool,
        default=False,
        help="sole_mask",
    )
    parser.add_argument(
        "--opti_thk_positive",
        type=str2bool,
        default=True,
        help="include term in cost function to enforce non-negative thickness"
    )
    parser.add_argument(
        "--opti_mask_thk",
        type=str2bool,
        default=True,
        help="mask thickness using icemask"
    )
    parser.add_argument(
        "--opti_retrain_iceflow_model",
        type=str2bool,
        default=True,
        help="Retrain the iceflow model simulatounously ?",
    )
    parser.add_argument(
        "--opti_retrain_stop_iter",
        type=int,
        default=100000,
        help="Iteration at which to stop retraining the iceflow model"
    )
    parser.add_argument(
        "--opti_reset_optimizer_after_retrain",
        type=str2bool,
        default=True,
        help="Reset the optimizer after retraining finished?"
    )
    # parser.add_argument(
    #     "--opti_reset_optimizer_lr",
    #     type=str2bool,
    #     default=False,
    #     help="Reset the learning rate when resetting the optimizer"
    # )
    parser.add_argument(
        "--opti_switch_to_SGD_iter",
        type=int,
        default=100000, # more than we would run in practice
        help="Iteration at which we stop optimizing with Adam and switch to SGD"
    )
    parser.add_argument(
       "--opti_to_regularize",
       type=str,
       default='topg',
       help="Field to regularize : topg or thk",
   )
    parser.add_argument(
       "--opti_include_low_speed_term",
       type=str2bool,
       default=False,
       help="opti_include_low_speed_term",
   ) 
    parser.add_argument(
        "--opti_infer_params",
        type=str2bool,
        default=False,
        help="infer slidingco and convexity weight from velocity observations",
    )
    parser.add_argument(
        "--opti_tidewater_glacier",
        type=str2bool,
        default=False,
        help="Is the glacier you're trying to infer parameters for a tidewater type?",
    )
    parser.add_argument(
        "--opti_vol_std",
        type=float,
        default=1000.0,
        help="Confidence/STD of the volume estimates from volume-area scaling",
    )
    parser.add_argument(
        "--fix_opti_normalization_issue",
        type=str2bool,
        default=False,
        help="formerly, the oce was mixing reduce_mean and l2_loss leadinf to dependence to the resolution of the grid",
    )
    parser.add_argument(
        "--opti_velsurfobs_thr",
        type=float,
        default=0.0,
        help="Threshold for the surface ice velocities as input data for the optimization, anything below this value will be ignored",
    )
    parser.add_argument(
        "--opti_use_thkobs_std",
        type=str2bool,
        default=False,
        help="Use thkobs_std as multiplier in thk cost",
    )
    parser.add_argument(
        "--opti_regu_thk_split",
        type=str2bool,
        default=False,
        help="Split thk_regu cost into along-flow and across flow; ignoring convexity term",
    )

    