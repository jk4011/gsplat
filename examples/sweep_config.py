import wandb
from easydict import EasyDict

SWEEP_WHOLE_ID = "jh11/GESI_sweep/adxyxcte"

sweep_config = {
    "method": "bayes",  # grid, random, bayes 중 선택
    "metric": {"name": "psnr_mean", "goal": "maximize"},
    "parameters": {
        "n_anchor_list"        : {"values": [[300, 300, 300, 300, 300]]},
        "decay_rate"           : {"values": [1]},
        "coef_drag"            : {"values": [0.5, 1]},
        "coef_arap_drag"       : {"values": [1e3, 2e3, 5e3, 1e4, 2e4]},
        "coef_group_arap"      : {"values": [2e2, 5e2, 1e3, 2e3, 5e3, 1e4, 2e4, 5e4]},
        "coef_rgb"             : {"values": [5e3, 1e4, 2e4, 5e4, 1e5, 2e5]},
        "coef_drag_3d"         : {"values": [3000]},
        "lr_q"                 : {"values": [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]},
        "lr_t"                 : {"values": [1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]},
        "rigidity_k"           : {"values": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]},
        "reprojection_error"   : {"values": [10]},
        "anchor_k"             : {"values": [10, 12, 15]},
        "rbf_gamma"            : {"values": [50]},
        "cycle_threshold"      : {"values": [10]},
        "vis_threshold"        : {"values": [0.4, 0.5]},
        "min_inlier_ratio"     : {"values": [0.5, 0.6, 0.7, 0.8, 0.9]},
        "confidence"           : {"values": [0.97, 0.98, 0.99]},
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
        "lr_q": 0.03,
        "lr_t": 0.003,
        "min_inlier_ratio": 0.7,
        "n_anchor_list": [300, 300, 300, 300, 300],
        "rbf_gamma": 50,
        "reprojection_error": 6,
        "rigidity_k": 50,
        "vis_threshold": 0.5,
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
        "lr_q": 0.05,
        "lr_t": 0.01,
        "n_anchor_list": [300, 300, 300, 300, 300],
        "min_inlier_ratio": 0.7,
        "rbf_gamma": 50,
        "reprojection_error": 10,
        "rigidity_k": 100,
        "vis_threshold": 0.5,
    }
})


def print_new_sweep_id(config, project_name):
    sweep_id = wandb.sweep(sweep=config, project=project_name)
    sweep_id_full = f"jh11/{project_name}/{sweep_id}"
    from jhutil import color_log; color_log(1111, sweep_id_full)


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
    # print_new_sweep_id(sweep_config, project_name="GESI_sweep")
    print_best_sweep_config(SWEEP_WHOLE_ID)

