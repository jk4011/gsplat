from simple_trainer import *
from simple_trainer import Runner


class ManipulationRunner(Runner):
    def human_manipulation(self, drag_source, drag_target, group_id_all, wo_group_arap=False):

        self.backgrounds = torch.ones(1, 3, device=self.device)   # white

        # get haraparameter
        coef_drag            = self.hpara.coef_drag
        coef_arap_drag       = self.hpara.coef_arap_drag
        coef_group_arap      = 0 if wo_group_arap else self.hpara.coef_group_arap * 10
        coef_drag            = self.hpara.coef_drag * 30
        lr_q                 = self.hpara.lr_q
        lr_t                 = self.hpara.lr_t
        anchor_k             = 10
        rbf_gamma            = self.hpara.rbf_gamma
        refine_radius        = self.hpara.refine_radius
        refine_threhold      = self.hpara.refine_threhold
        voxel_size           = self.hpara.voxel_size
        filter_distance      = 5
        n_anchor             = self.hpara.n_anchor
        
        self.splats = dict(self.splats)

        with torch.no_grad():
            image_source, _ = self.fetch_comparable_two_image(return_rgba=True, use_gt_pose=True)
            self.height = image_source.shape[1]
            self.width = image_source.shape[2]
        ##########################################################
        ############### 2. filter points and drag  ###############
        ##########################################################
        from jhutil import color_log; color_log(2222, "filter points and start drag")
        points_3d = self.splats["means"].clone().detach()
        points_3d.requires_grad = True

        # set drag
        with torch.no_grad():
            points_2d, _ = self.project_to_2d(points_3d, use_gt_pose=True)

        vis_mask = self.get_visibility_mask()

        points_mask, drag_indice = get_drag_mask(
            points_2d, vis_mask, drag_source, filter_distance, one_to_one=True
        )

        # points_3d_filtered = points_3d[points_mask]
        drag_target_filtered = drag_target[drag_indice]
        # drag_source_filtered = drag_source[drag_mask]

        ##########################################################
        ########### 3. initialize anchor and optimizer ###########
        ##########################################################
        from jhutil import color_log; color_log(3333, "initialize anchor and optimizer")

        anchor = voxelize_pointcloud_and_get_means(points_3d, voxel_size=voxel_size)
        assert anchor.shape[0] > n_anchor
        anchor = anchor.to(self.device)

        N = anchor.shape[0]
        q_init = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
        q_init = q_init.unsqueeze(0).repeat(N, 1, 1)
        t_init = torch.zeros((N, 3), dtype=torch.float32, device=self.device)

        quats = nn.Parameter(q_init)  # (N, 3, 3)
        t = nn.Parameter(t_init)  # (N, 3)

        # anchor_optimizer = torch.optim.SGD(
        anchor_optimizer = torch.optim.Adam(
            [
                {"params": quats, "lr": lr_q * 0.3},
                {"params": t, "lr": lr_t * 0.3},
            ]
        )

        ##########################################################
        #################### 5. drag optimize ####################
        ##########################################################
        from jhutil import color_log; color_log(5555, "drag optimize ")
        
        quats_origin = F.normalize(self.splats["quats"].clone().detach(), dim=-1)
        
        with torch.no_grad():
            distances, indices_knn = knn_jh(anchor, anchor, k=anchor_k)
            weight = rbf_weight(distances, gamma=rbf_gamma)
        video = []
        for i in tqdm(range(100)):
            R = quaternion_to_matrix(F.normalize(quats, dim=-1)).squeeze()  # (N, 3, 3)
            anchor_translated = anchor + t  # (N, 3)

            loss_arap = arap_loss(anchor, anchor_translated, R, weight, indices_knn)

            points_lbs, quats_lbs = linear_blend_skinning_knn(points_3d, anchor, R, t)
            updated_quaternions = quaternion_multiply(quats_lbs, quats_origin)

            R_points = quaternion_to_matrix(F.normalize(quats_lbs, dim=-1)).squeeze()

            if wo_group_arap:
                loss_group_arap = 0
            else:
                loss_group_arap = arap_loss_grouped(
                    points_3d, points_lbs, R_points, group_id_all
                )

            points_lbs_filtered = points_lbs[points_mask]
            points_lbs_filtered_2d, _ = self.project_to_2d(points_lbs_filtered, use_gt_pose=True)
            loss_drag = drag_loss(points_lbs_filtered_2d, drag_target_filtered)

            self.splats["means"] = points_lbs
            self.splats["quats"] = updated_quaternions

            loss = (
                coef_drag * loss_drag
                + coef_arap_drag * loss_arap
                + coef_group_arap * loss_group_arap
            )

            loss.backward(retain_graph=True)
            anchor_optimizer.step()
            anchor_optimizer.zero_grad()
        
            with torch.no_grad():
                img, _ = self.fetch_comparable_two_image(return_rgba=True, use_gt_pose=True)
            video.append(img[0].permute(2, 0, 1).cpu())
        
        from jhutil import save_video
        object_name = cfg.object_name.split("_[")[0]
        if wo_group_arap:
            output_path = f"/data2/wlsgur4011/GESI/output_manipulation/{object_name}_naive.mp4"
        else:
            output_path = f"/data2/wlsgur4011/GESI/output_manipulation/{object_name}_ours.mp4"

        save_video(video, output_path)
        return video[-1]


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = ManipulationRunner(local_rank, world_rank, world_size, cfg)


    images = []
    for wo_group_arap in [True, False]:
        if cfg.ckpt is not None:
            # run eval only
            ckpts = [
                torch.load(file, map_location=runner.device, weights_only=True)
                for file in cfg.ckpt
            ]
            for i, ckpt in enumerate(ckpts):
                if "clustered" not in ckpt:
                    ckpts[i]["splats"] = cluster_largest(ckpts[i]["splats"])
            for k in runner.splats.keys():
                runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
        
        [drag_source, drag_target] = torch.load(f"/tmp/.cache/human_manipulation_{cfg.object_name}.pt")
        drag_target = (drag_target * 2 + drag_source) / 3

        group_id_all = torch.load(f"{cfg.result_dir}/ckpt_finetune.pt")["group_id_all_init"]

        img = runner.human_manipulation(drag_source, drag_target, group_id_all, wo_group_arap)
        images.append(img)

    _, img1, img2 = crop_two_image_with_alpha(images[0], images[1])

    wandb.log({"train_diff": [wandb.Image(img1, caption="without_group"),wandb.Image(img2, caption="with_group"),]})


    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)


all_psnr_for_sweep = {}


def run_all_data(cfg: Config):
    if cfg.data_name == "DFA":
        image_indices = {
            ("beagle_dog", "s1_24fps"): {"32": [[250, 260]]},
            ("beagle_dog", "s1"): {"16": [[520, 525]]},
            ("bear", "run"): {"16": [[0, 2]]},
            ("bear", "walk"): {"16": [[110, 140]]},
            ("cat", "run"): {"32": [[25, 30]]},
            ("cat", "walk_final"): {"32": [[10, 20]]},
            ("cat", "walkprogressive_noz"): {"32": [[25, 30]]},
            ("cat", "walksniff"): {"32": [[30, 110]]},
            ("duck", "eat_grass"): {"24": [[0, 10]]},
            ("duck", "swim"): {"16": [[145, 160]]},
            ("duck", "walk"): {"16": [[200, 230]]},
            ("fox", "attitude"): {"24": [[95, 145]]},
            ("fox", "run"): {"32": [[25, 30],]},
            ("fox", "walk"): {"24": [[70, 75]]},
            ("panda", "acting"):{"32": [[95, 100]]},
            ("panda", "run"): {"32": [[5, 10]]},
            ("panda", "walk"):{"32": [[15, 25]]},
            ("lion", "Run"): {"24": [[70, 75]]},
            ("lion", "Walk"): {"32": [[30, 35]]},
            ("whiteTiger", "roaringwalk"): {"32": [[15, 25]]},
            ("whiteTiger", "run"): {"32": [[70, 80]]},
            ("wolf", "Damage"): {"32": [[0, 110]]},
            ("wolf", "Howling"): {"24": [[10, 60]]},
            ("wolf", "Run"): {"16": [[20, 25]]},
            ("wolf", "Walk"): {"32": [[10, 20]]},
        }
        for name, sub_name in image_indices.keys():
            index_dict = image_indices[(name, sub_name)]
            for cam_idx, pair_list in index_dict.items():
                
                for index_from, index_to in pair_list:
                    
                    object_name = f"{name}({sub_name})"
                    data_dir = f"/data2/wlsgur4011/GESI/gsplat/data/DFA_processed/{object_name}/{index_to}"
                    result_dir = f"./results/dfa/{object_name}_sweep"
                    ckpt = [f"./results/dfa/{object_name}_{index_from}/ckpts/ckpt_best_psnr.pt"]
                    
                    cfg.result_dir = result_dir
                    cfg.object_name = f"{object_name}_[{index_from},{index_to}]"
                    cfg.data_dir = data_dir
                    cfg.ckpt = ckpt
                    cfg.cam_idx = int(cam_idx)
                    
                    runner = Runner(0, 0, 1, cfg)

                    if cfg.ckpt is not None:
                        ckpts = [
                            torch.load(file, map_location=runner.device, weights_only=True)
                            for file in cfg.ckpt
                        ]
                        for k in runner.splats.keys():
                            runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
                        step = ckpts[0]["step"]
                        # runner.eval(step=step)
                        # runner.render_traj(step=step)
                        if cfg.compression is not None:
                            runner.run_compression(step=step)
                        if cfg.single_finetune:
                            runner.train_drag()
                    else:
                        runner.train()

        psnr_mean = np.mean(list(all_psnr_for_sweep.values()))
        wandb.log(all_psnr_for_sweep)
        wandb.log({"psnr/psnr_mean": psnr_mean})
    
    elif cfg.data_name == "diva360":
        image_indices = {
            "blue_car" : ("00", "0142", "0214"),
            "bunny" : ("00", "0000", "1000"),
            "dog" : ("00", "0177", "0279"),
            "horse" : ("00", "0120", "0375"),
            "k1_double_punch" :  ("01", "0270", "0282"),
            "k1_hand_stand" :  ("01", "0412", "0426"),
            "k1_push_up" :  ("01", "0541", "0557"),
            "music_box" : ("00", "0100", "0125"),
            "penguin" : ("00", "0217", "0239"),
            "hour_glass" :  ("00", "0100", "0200"),
            "wolf" :  ("00", "0357", "1953"),
            "trex" :  ("00", "0135", "0250"),
            "truck" : ("00", "0078", "0171"),
            "wall_e" : ("00", "0222", "0286"),
            "red_car" : ("00", "0042", "0250"),
            "clock" : ("00", "0000", "1500"),
            "world_globe" : ("00", "0020", "0074"),
            "stirling" : ("00", "0000", "0045"),
            "tornado" : ("00", "0000",  "0456"),
        }
        for object_name, (cam_idx, index_from, index_to) in image_indices.items():
            
            data_dir = f"/data2/wlsgur4011/GESI/gsplat/data/diva360_processed/{object_name}_{index_to}/"
            ckpt = [f"./results/diva360/{object_name}_{index_from}/ckpts/ckpt_best_psnr.pt"]
            result_dir = f"./results/diva360/{object_name}_sweep"
            cfg.cam_idx = int(cam_idx)
            cfg.result_dir = result_dir
            cfg.object_name = object_name
            cfg.data_dir = data_dir
            cfg.ckpt = ckpt
            
            runner = Runner(0, 0, 1, cfg)

            if cfg.ckpt is not None:
                # run eval only
                ckpts = [
                    torch.load(file, map_location=runner.device, weights_only=True)
                    for file in cfg.ckpt
                ]
                for k in runner.splats.keys():
                    runner.splats[k].data = torch.cat([ckpt["splats"][k] for ckpt in ckpts])
                step = ckpts[0]["step"]
                # runner.eval(step=step)
                # runner.render_traj(step=step)
                if cfg.compression is not None:
                    runner.run_compression(step=step)
                if cfg.single_finetune:
                    runner.train_drag()
            else:
                runner.train()

    psnr_mean = np.mean(list(all_psnr_for_sweep.values()))
    wandb.log(all_psnr_for_sweep)
    wandb.log({"psnr_mean": psnr_mean})


if __name__ == "__main__":
    """
    Usage:

    ```bash
    # Single GPU training
    CUDA_VISIBLE_DEVICES=0 python simple_trainer.py default

    # Distributed training on 4 GPUs: Effectively 4x batch size so run 4x less steps.
    CUDA_VISIBLE_DEVICES=0,1,2,3 python simple_trainer.py default --steps_scaler 0.25

    """

    # Config objects we can choose between.
    # Each is a tuple of (CLI description, config object).
    configs = {
        "default": (
            "Gaussian splatting training using densification heuristics from the original paper.",
            Config(
                strategy=DefaultStrategy(verbose=True),
            ),
        ),
        "mcmc": (
            "Gaussian splatting training using densification from the paper '3D Gaussian Splatting as Markov Chain Monte Carlo'.",
            Config(
                init_opa=0.5,
                init_scale=0.1,
                opacity_reg=0.01,
                scale_reg=0.01,
                strategy=MCMCStrategy(verbose=True),
            ),
        ),
    }
    cfg = tyro.extras.overridable_config_cli(configs)
    cfg.adjust_steps(cfg.steps_scaler)

    # try import extra dependencies
    if cfg.compression == "png":
        try:
            import plas
            import torchpq
        except:
            raise ImportError(
                "To use PNG compression, you need to install "
                "torchpq (instruction at https://github.com/DeMoriarty/TorchPQ?tab=readme-ov-file#install) "
                "and plas (via 'pip install git+https://github.com/fraunhoferhhi/PLAS.git') "
            )

    # cli(main, cfg, verbose=True)

    # Logger
    if cfg.wandb_sweep:
        wandb.agent(SWEEP_WHOLE_ID, function=lambda: run_all_data(cfg), count=100)
    else:
        main(0, 0, 1, cfg)

