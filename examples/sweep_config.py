import wandb

SWEEP_WHOLE_ID = "jh11/GESI_sweep/td80bymb"

sweep_config = {
    "method": "bayes",  # grid, random, bayes 중 선택
    "metric": {"name": "psnr_mean", "goal": "maximize"},
    "parameters": {
        "n_anchor"             : {"values": [300, 500, 1000]},
        "coef_drag"            : {"values": [0.1, 0.3, 1, 3, 10]},
        "coef_arap_drag"       : {"values": [3e3, 1e4, 3e4, 1e5, 3e5]},
        "coef_group_arap_drag" : {"values": [1e3, 3e3, 1e4, 3e4, 1e5]},
        "coef_arap_rgb"        : {"values": [1]},
        "lr_q"                 : {"values": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]},
        "lr_t"                 : {"values": [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]},
        "rigidity_k"           : {"values": [10, 20, 30, 40, 50]},
        "reprojection_error"   : {"values": [0.5, 1, 1.5, 2, 2.5, 3, 4, 5]},
        "anchor_k"             : {"values": [3, 4, 5, 6, 7, 8, 9]},
        "rbf_gamma"            : {"values": [5, 10, 20, 30, 40, 50]},
    },
}
best_sweep_config = {
    "lr_q": 0.03,
    "lr_t": 0.03,
    "anchor_k": 7,
    "n_anchor": 300,
    "coef_drag": 3,
    "rbf_gamma": 30,
    "rigidity_k": 50,
    "coef_arap_rgb": 1,
    "coef_arap_drag": 10000,
    "reprojection_error": 3,
    "coef_group_arap_drag": 30000
}


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

