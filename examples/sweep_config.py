import wandb
from easydict import EasyDict

SWEEP_WHOLE_ID = "jh11/GESI_sweep/spq30kb0"

sweep_config = {
    "method": "bayes",  # grid, random, bayes 중 선택
    "metric": {"name": "psnr_mean", "goal": "maximize"},
    "parameters": {
        "n_anchor_list" : {"values": [
            [300, 300, 300, 300, 300],
            [200, 300, 400, 500, 1000],
            [200, 250, 300, 350, 400],
            [100, 200, 300, 400, 500],
            [100, 200, 300, 500, 1000],
            [300, 300, 300, 300, 1000],
            [300, 300, 300, 500, 500],
            [250, 250, 250, 500, 500],
            [250, 250, 250, 1000, 1000],
            [200, 200, 200, 400, 400],
            [200, 200, 200, 400, 400],
            [500, 500, 500, 1000, 1000],
            [300, 300, 300, 1000, 1000],
            [300, 300, 300, 1500, 1500],
            [300, 300, 300, 2000, 2000],
            [300, 300, 300, 3000, 3000],
        ]},
        "decay_rate"           : {"values": [0.998, 0.999, 1]},
        "coef_drag"            : {"values": [0.5, 1]},
        "coef_arap_drag"       : {"values": [1e3, 2e3, 5e3]},
        "coef_group_arap"      : {"values": [1e4, 2e4, 5e4]},
        "coef_rgb"             : {"values": [1e2, 2e2, 5e2, 1e3, 2e3, 5e3]},
        "coef_drag_3d"         : {"values": [3000]},
        "lr_q"                 : {"values": [3e-2]},
        "lr_t"                 : {"values": [1e-34, 3e-3, 1e-2, 2e-2, 3e-2]},
        "rigidity_k"           : {"values": [40, 50]},
        "reprojection_error"   : {"values": [5, 6]},
        "anchor_k"             : {"values": [9]},
        "rbf_gamma"            : {"values": [50, 60, 70]},
        "cycle_threshold"      : {"values": [10, 15, 20, 25]},
        "vis_threshold"        : {"values": [0.4, 0.5, 0.6, 0.7]},
    },
}

best_config_dict = EasyDict({
    "DFA": {
        "lr_q": 0.03,
        "lr_t": 0.003,
        "anchor_k": 9,
        "coef_rgb": 5000,
        "coef_drag": 0.5,
        "rbf_gamma": 50,
        "decay_rate": 1,
        "rigidity_k": 50,
        "coef_drag_3d": 3000,
        "n_anchor_list": [300, 300, 300, 300, 300],
        "coef_arap_drag": 2000,
        "coef_group_arap": 50000,
        "reprojection_error": 6,
        "cycle_threshold": 10,
        "vis_threshold": 0.5,
    },
    "diva360": {
        "lr_q": 0.03,
        "lr_t": 0.03,
        "anchor_k": 9,
        "n_anchor_list": [300, 300, 300, 300, 300],
        "coef_drag": 1,
        "coef_drag_3d": 3000,
        "rbf_gamma": 50,
        "decay_rate": 1,
        "rigidity_k": 40,
        "coef_rgb": 5000,
        "coef_arap_drag": 3000,
        "reprojection_error": 5,
        "coef_group_arap": 3000,
        "cycle_threshold": 10,
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
    best_config = best_run.config
    
    from jhutil import color_log; color_log(1111, best_config)


if __name__ == "__main__":
    print_new_sweep_id(sweep_config, project_name="GESI_sweep")
    # print_best_sweep_config(SWEEP_WHOLE_ID)

