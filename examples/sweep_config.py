import wandb
from easydict import EasyDict

SWEEP_WHOLE_ID = "jh11/GESI_sweep/uvrwj3xb"

sweep_config = {
    "method": "bayes",  # grid, random, bayes 중 선택
    "metric": {"name": "psnr_mean", "goal": "maximize"},
    "parameters": {
        "n_anchor"             : {"values": [200, 250, 300, 400, 500]},
        "decay_rate"           : {"values": [1]},
        "coef_drag"            : {"values": [0.5, 1, 2]},
        "coef_arap_drag"       : {"values": [3000]},
        "coef_group_arap"      : {"values": [100, 200, 500, 1000, 2000]},
        "coef_rgb"             : {"values": [50000]},
        "coef_drag_3d"         : {"values": [3000]},
        "lr_q"                 : {"values": [0.05]},
        "lr_t"                 : {"values": [0.01]},
        "rigidity_k"           : {"values": [100]},
        "reprojection_error"   : {"values": [10]},
        "anchor_k"             : {"values": [15]},
        "rbf_gamma"            : {"values": [50]},
        "cycle_threshold"      : {"values": [10]},
        "vis_threshold"        : {"values": [0.5]},
        "min_inlier_ratio"     : {"values": [0.7]},
        "confidence"           : {"values": [0.97, 0.98, 0.99]},
        "refine_radius"        : {"values": [0.03, 0.05, 0.07, 0.10]},
        "refine_threhold"      : {"values": [0.005, 0.01, 0.02, 0.03]},
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
        "decay_rate": 1,
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
    },
    "diva360": {
        "anchor_k": 15,
        "coef_arap_drag": 20000,
        "coef_drag": 0.5,
        "coef_drag_3d": 3000,
        "coef_group_arap": 500,
        "coef_rgb": 50000,
        "confidence": 0.97,
        "cycle_threshold": 10,
        "decay_rate": 1,
        "lr_motion": 1e-3,
        "lr_q": 0.05,
        "lr_t": 0.01,
        "n_anchor": 300,
        "min_inlier_ratio": 0.7,
        "rbf_gamma": 50,
        "reprojection_error": 10,
        "rigidity_k": 100,
        "vis_threshold": 0.5,
        "refine_radius": 0.03,
        "refine_threhold": 0.01,
        "voxel_size": 0.04,
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

