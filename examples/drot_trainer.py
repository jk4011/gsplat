from simple_trainer import *
from simple_trainer import Runner

import torch.nn as nn
from gesi.loss import arap_loss, drag_loss, arap_loss_grouped
from gesi.mini_pytorch3d import quaternion_to_matrix
from jhutil.algorithm import knn as knn_jh
from gesi.helper import (
    load_points_and_anchor,
    save_points_and_anchor,
    make_simple_goal,
    rbf_weight,
    deform_point_cloud_arap,
    voxelize_pointcloud_and_get_means,
    linear_blend_skinning_knn,
    cluster_largest,
    get_visible_mask_by_depth,
    get_target_indices_drag,
    project_pointcloud_to_2d,
    deform_point_cloud_arap_2d,
    knn_djastra,
    get_drag_mask,
    get_front_mask_by_depth,
)
from gesi.loss import drot_loss_with_means2d, drot_loss
from gesi.visibility import compute_visibility
from gesi.mini_pytorch3d import quaternion_multiply, quaternion_invert
from jhutil import show_matching, show_groups
from jhutil import (
    get_img_diff,
    crop_two_image_with_background,
    crop_two_image_with_alpha,
)
from gesi.roma import get_drag_roma
import warnings
import torch_fpsample
from sweep_config import sweep_config, best_config_dict, SWEEP_WHOLE_ID
from gesi.rigid_grouping import local_rigid_grouping, naive_rigid_grouping, refine_rigid_group
from jhutil import save_video
from torch.nn import SmoothL1Loss
from torch.optim.lr_scheduler import LambdaLR
from gesi.visibility import _fully_fused_projection2
from gesi.torch_splat import render_uv_coordinate, render

warnings.simplefilter("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning) # certain warning



class DrotRunner(Runner):
    
    def train_drag(
        self,
        rgb_iteration=500,
        filter_distance=1,
        rgb_optimize=False,
    ) -> None:

        # get haraparameter
        drot_iterations      = 3
        coef_arap_drag       = self.hpara.coef_arap_drag
        coef_rgb             = self.hpara.coef_rgb * 0.1
        lr_q                 = self.hpara.lr_q
        lr_t                 = self.hpara.lr_t
        coef_drag            = 1
        coef_arap_drag       = 1e5
        coef_arap_rgb        = 5e4
        coef_mask_arap       = 10
        coef_mask_rot        = 10
        coef_mask_dist       = 10
        
        self.splats = dict(self.splats)

        if not self.cfg.skip_eval:
            step = 0
            self.eval(step=step)

        points = self.splats["means"].clone().detach()
        points.requires_grad = False

        ##########################################################
        ########### 1. initialize anchor and optimizer ###########
        ##########################################################
        anchor = voxelize_pointcloud_and_get_means(points, 0.04)

        N = anchor.shape[0]
        q_init = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
        q_init = q_init.unsqueeze(0).repeat(N, 1, 1)
        t_init = torch.zeros((N, 3), dtype=torch.float32, device=self.device)

        mask_arap_init = torch.ones((N, 1), dtype=torch.float32, device=self.device)
        mask_rot_init = torch.ones((N, 1), dtype=torch.float32, device=self.device)
        mask_dist_init = torch.ones((N, 1), dtype=torch.float32, device=self.device)

        q = nn.Parameter(q_init)  # (N, 3, 3)
        t = nn.Parameter(t_init)  # (N, 3)
        mask_arap = nn.Parameter(mask_arap_init)  # (N, 3)
        mask_rot = nn.Parameter(mask_rot_init)  # (N, 3)
        mask_dist = nn.Parameter(mask_dist_init)  # (N, 3)

        anchor_optimizer = torch.optim.Adam(
            [
                {"params": q, "lr": lr_q},
                {"params": t, "lr": lr_t},
            ]
        )

        with torch.no_grad():
            # Eq.11 in "3D Gaussian Editing with A Single Image"
            distances, indices_knn = knn_jh(anchor, anchor, k=5)
            weight = rbf_weight(distances, gamma=30)

            # Eq.12 in "3D Gaussian Editing with A Single Image"
            weight = weight * torch.sigmoid(mask_arap)

        quats_origin = F.normalize(self.splats['quats'].clone().detach(), dim=-1)

        from jhutil import color_log; color_log("aaaa", "drot optimize start")

        ##########################################################
        ##################### 2. coarse stage ####################
        ##########################################################
        for i in tqdm(range(drot_iterations + 1)):
            R = quaternion_to_matrix(F.normalize(q, dim=-1)).squeeze()  # (N, 3, 3)
            anchor_translated = anchor + t  # (N, 3)

            # Eq.10 in "3D Gaussian Editing with A Single Image"
            loss_arap = arap_loss(anchor, anchor_translated, R, weight, indices_knn)

            points_lbs, quats_lbs = linear_blend_skinning_knn(points, anchor, R, t)
            quats_multiplied = quaternion_multiply(quats_lbs, quats_origin)

            self.splats["means"] = points_lbs
            self.splats["quats"] = quats_multiplied


            with torch.no_grad():
                image_from, image_to = self.fetch_comparable_two_image()

            loss_drag = self.gesi_loss(image_from, image_to)

            # means2d, depths = self.project_to_2d(points_lbs)
            # target_indices = get_front_mask_by_depth(means2d, depths)
            # means2d_target = means2d[target_indices]
            # loss_drag = drot_loss_with_means2d(image_from, image_to, means2d_target)

            # Eq.17 in "3D Gaussian Editing with A Single Image"
            loss_mask_arap = (torch.sigmoid(mask_arap) - 1).abs().mean()
            loss_mask_rot = (torch.sigmoid(mask_rot) - 1).abs().mean()
            loss_mask_dist = (torch.sigmoid(mask_dist) - 1).abs().mean()

            loss = (
                # Eq.18 in "3D Gaussian Editing with A Single Image"
                loss_arap * coef_arap_drag +
                loss_drag * coef_drag +
                # Eq.19 in "3D Gaussian Editing with A Single Image"
                loss_mask_rot * coef_mask_rot +
                loss_mask_dist * coef_mask_dist +
                loss_mask_arap * coef_mask_arap
            )

            loss.backward(retain_graph=True)
            anchor_optimizer.step()
            anchor_optimizer.zero_grad()

            wandb.log({"loss_arap": loss_arap, "loss_drag": loss_drag}, step=i, commit=True)

            if i % 100 == 0:
                image_source, image_target = self.fetch_comparable_two_image(
                    return_rgba=True, return_shape="chw"
                )
                _, img1, img2 = crop_two_image_with_alpha(
                    image_source, image_target
                )
                diff_img = get_img_diff(img1[:3], img2[:3])
                wandb.log(
                    {"train_img_diff": wandb.Image(diff_img, caption="train_img_diff")},
                    step=i+1,
                    commit=True,
                )

        if not self.cfg.skip_eval:
            step += drot_iterations
            self.eval(step=step)

        ##########################################################
        ###################### 3. fine stage #####################
        ##########################################################
        from jhutil import color_log; color_log("bbbb", "rgb optimize start")

        means_origin = self.splats['means'].clone().detach()
        quats_origin = F.normalize(self.splats['quats'].clone().detach(), dim=-1)
        quats_origin_invert = quaternion_invert(quats_origin)
        with torch.no_grad():
            distances, indices_knn = knn_jh(means_origin, means_origin, k=5)
            weight = rbf_weight(distances, gamma=30)

        for i in tqdm(range(rgb_iteration)):
            quats_current = F.normalize(self.splats['quats'], dim=-1)
            q = quaternion_multiply(quats_current, quats_origin_invert)
            R = quaternion_to_matrix(q).squeeze()  # (N, 3, 3)
            # arap_loss(anchor, anchor_translated, R, weight, indices_knn)
            loss_rgb = self.render_and_calc_rgb_loss()
            loss_arap = arap_loss(means_origin, self.splats['means'], R, weight, indices_knn)

            loss = coef_arap_rgb * loss_arap # + loss_rgb * coef_rgb
            loss.backward(retain_graph=True)

            for var_name in ['means', 'quats']:
                optimizer = self.optimizers[var_name]
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            
            if i % 100 == 0:
                image_source, image_target = self.fetch_comparable_two_image(
                    return_rgba=True, return_shape="chw"
                )
                _, img1, img2 = crop_two_image_with_alpha(
                    image_source, image_target
                )
                diff_img = get_img_diff(img1[:3], img2[:3])
                wandb.log(
                    {"train_img_diff": wandb.Image(diff_img, caption="train_img_diff")},
                    step=i+drot_iterations+1,
                    commit=True,
                )

        if not self.cfg.skip_eval:
            step += rgb_iteration
            self.eval(step=step)


    def gesi_loss(self, image_from, image_to):

        camtoworlds = self.data["camtoworld"].to(self.device)
        Ks = self.data["K"].to(self.device)
        width = image_from.shape[2]
        height = image_from.shape[1]
        means3d = self.splats["means"]
        opacities = self.splats["opacities"]
        scales = self.splats["scales"]
        quats = self.splats["quats"]

        # cam_origin = camtoworlds[:, :3, 3]
        viewmats = torch.linalg.inv(camtoworlds)
        opacities = torch.sigmoid(opacities)
        scales = torch.exp(scales)

        from gsplat.cuda._torch_impl import _quat_to_rotmat
        R = _quat_to_rotmat(quats)  # (..., 3, 3)
        M = R * scales[..., None, :]  # (..., 3, 3)
        covars = torch.bmm(M, M.transpose(-1, -2))  # (..., 3, 3)

        means2d, depths, cov2d = _fully_fused_projection2(means3d, covars, viewmats, Ks, width, height)

        means2d, cov2d, opacities, depths = [x.squeeze() for x in [means2d, cov2d, opacities, depths]]
        opacities = opacities[:, None]

        uv = render_uv_coordinate(width, height, means2d, cov2d, opacities, depths)

        # Eq.6 in "3D Gaussian Editing with A Single Image"
        return drot_loss(image_from, image_to, uv, downsample=2)


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = DrotRunner(local_rank, world_rank, world_size, cfg)

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
        step = ckpts[0]["step"]
        # runner.eval(step=step)
        # runner.render_traj(step=step)
        if cfg.compression is not None:
            runner.run_compression(step=step)
        if cfg.single_finetune:
            if cfg.motion_video:
                runner.make_motion_video(idx=0, threhold_early_stop=5e-6, scheduler_step=500, min_rigid_coef=0)
                runner.make_motion_video(idx=1, threhold_early_stop=1e-5, scheduler_step=800, min_rigid_coef=0)
                runner.make_motion_video(idx=2, threhold_early_stop=1e-5, scheduler_step=500, min_rigid_coef=0)
            else:
                runner.train_drag()

        if cfg.render_traj_all:
            if cfg.data_name == "diva360":
                traj_path_list = ["diva360_spiral"]  # "interp", "ellipse", "spiral"
            elif cfg.data_name == "dfa":
                traj_path_list = ["interp", "ellipse", "spiral"]
            for render_traj_path in traj_path_list:
                runner.cfg.render_traj_path = render_traj_path

                video_dir = "/data2/wlsgur4011/GESI/output_video_traj"
                video_paths = [
                    f"{video_dir}/traj_{cfg.object_name}_{cfg.render_traj_path}.mp4",
                    f"{video_dir}/traj_{cfg.object_name}_{cfg.render_traj_path}_init.mp4",
                    f"{video_dir}/traj_{cfg.object_name}_{cfg.render_traj_path}_final.mp4"
                ]
                group_ids = [
                    None,
                    ckpts[0]["group_id_all_init"],
                    ckpts[0]["group_id_all"],
                ]
                for i in range(3):
                    group_id_all = group_ids[i]
                    video_path = video_paths[i]
                    runner.render_traj(step=step, group_id_all=group_id_all, video_path=video_path)
                    runner.cfg.sh_degree = 0

    else:
        runner.train()

    if not cfg.disable_viewer:
        print("Viewer running... Ctrl+C to exit.")
        time.sleep(1000000)



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

