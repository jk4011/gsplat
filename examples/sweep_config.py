import wandb
from easydict import EasyDict

SWEEP_WHOLE_ID = "jh11/GESI_sweep/kuzw3ao5"

sweep_config = {
    "method": "bayes",  # grid, random, bayes 중 선택
    "metric": {"name": "psnr_mean", "goal": "maximize"},
    "parameters": {
        "n_anchor"             : {"values": [300]},
        "coef_drag"            : {"values": [0.5, 1]},
        "coef_arap_drag"       : {"values": [1000, 2000, 5000, 10000]},
        "coef_group_arap"      : {"values": [100, 200, 500, 1000, 2000]},
        "coef_rgb"             : {"values": [2000, 5000, 10000, 20000, 50000]},
        "coef_drag_3d"         : {"values": [3000]},
        "lr_q"                 : {"values": [0.01, 0.02, 0.05]},
        "lr_t"                 : {"values": [0.001, 0.002, 0.005, 0.01]},
        "rigidity_k"           : {"values": [10, 25, 50, 75, 100]},
        "reprojection_error"   : {"values": [5, 6, 7, 8, 10]},
        "anchor_k"             : {"values": [10, 15]},
        "rbf_gamma"            : {"values": [50]},
        "cycle_threshold"      : {"values": [10]},
        "vis_threshold"        : {"values": [0.5]},
        "min_inlier_ratio"     : {"values": [0.5, 0.6, 0.7, 0.8]},
        "confidence"           : {"values": [0.95, 0.97, 0.99]},
        "refine_radius"        : {"values": [0.03, 0.05, 0.07, 0.10]},
        "refine_threhold"      : {"values": [0.002, 0.005, 0.01, 0.02, 0.03, 0.05]},
        "voxel_size"           : {"values": [0.02, 0.03, 0.04, 0.05, 0.06]},
    },
}

best_config_dict = EasyDict({
    "DFA": {
        "anchor_k": 9,
        "coef_arap_drag": 2000,
        "coef_drag": 0.5,
        "coef_drag_3d": 3000,
        "coef_group_arap": 500,
        "coef_rgb": 5000,
        "confidence": 0.99,
        "cycle_threshold": 10,
        "lr_motion": 3e-4,
        "lr_q": 0.03,
        "lr_t": 0.003,
        "min_inlier_ratio": 0.7,
        "n_anchor": 300,
        "rbf_gamma": 50,
        "reprojection_error": 6,
        "rigidity_k": 50,
        "vis_threshold": 0.5,
        "refine_radius": 0.05,
        "refine_threhold": 0.01,
        "voxel_size": 0.02,
        "filter_distance": 1,
        "min_group_size": 100,
    },
    "diva360": {
        "anchor_k": 10,
        "coef_arap_drag": 10000,
        "coef_drag": 0.5,
        "coef_drag_3d": 3000,
        "coef_group_arap": 1000,
        "coef_rgb": 50000,
        "confidence": 0.97,
        "cycle_threshold": 10,
        "lr_motion": 1e-3,
        "lr_q": 0.05,
        "lr_t": 0.01,
        "n_anchor": 300,
        "min_inlier_ratio": 0.7,
        "rbf_gamma": 50,
        "refine_radius": 0.1,
        "refine_threhold": 0.01,
        "reprojection_error": 8,
        "rigidity_k": 25,
        "vis_threshold": 0.3,
        "voxel_size": 0.06,
        "filter_distance": 1,
        "min_group_size": 25,
    }
})


def print_new_sweep_id(config, project_name):
    sweep_id = wandb.sweep(sweep=config, project=project_name)
    sweep_id_full = f"jh11/{project_name}/{sweep_id}"
    print(sweep_id_full)


def print_best_sweep_config(sweep_full_id):
    api = wandb.Api()

    sweep = api.sweep(sweep_full_id)
    best_run = sweep.best_run()
    # best_config = best_run.config
    best_config = best_config_dict["DFA"]
    print("{")
    for key in sorted(best_config):
        print(f"    \"{key}\": {best_config[key]},")
    print("}")


if __name__ == "__main__":
    print_new_sweep_id(sweep_config, project_name="GESI_sweep")
    # print_best_sweep_config(SWEEP_WHOLE_ID)

