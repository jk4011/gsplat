import wandb
from easydict import EasyDict

SWEEP_WHOLE_ID = "jh11/GESI_sweep/9w0ul3z0"

sweep_config = {
    "method": "bayes",  # grid, random, bayes 중 선택
    "metric": {"name": "psnr_mean", "goal": "maximize"},
    "parameters": {
        "n_anchor"             : {"values": [100, 200, 300, 400, 500]},
        "coef_drag"            : {"values": [0.3, 0.5, 1, 1.5, 2]},
        "coef_arap_drag"       : {"values": [5e2, 1e3, 2e3, 3e3, 5e3, 1e4]},
        "coef_group_arap"      : {"values": [1e3, 2e3, 3e3, 1e4, 2e4, 3e4]},
        "coef_arap_rgb"        : {"values": [1]},
        "lr_q"                 : {"values": [1e-2, 2e-2, 3e-2, 5e-2, 1e-1]},
        "lr_t"                 : {"values": [1e-2, 2e-2, 3e-2, 5e-2, 1e-1]},
        "rigidity_k"           : {"values": [10, 20, 30, 40, 50]},
        "reprojection_error"   : {"values": [3, 4, 5, 6, 7, 8]},
        "anchor_k"             : {"values": [7, 8, 9, 10, 12, 15]},
        "rbf_gamma"            : {"values": [30, 40, 50, 60, 70]},
        "cycle_threshold"      : {"values": [5, 7, 10, 15, 20, 25]},
    },
}

best_config = EasyDict({
    "lr_q": 0.03,
    "lr_t": 0.03,
    "anchor_k": 9,
    "n_anchor": 300,
    "coef_drag": 1,
    "coef_drag_3d": 3000,
    "rbf_gamma": 50,
    "rigidity_k": 40,
    "coef_arap_rgb": 1,
    "coef_arap_drag": 3000,
    "reprojection_error": 5,
    "coef_group_arap": 3000,
    "cycle_threshold": 20,
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

