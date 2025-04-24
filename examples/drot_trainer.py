from simple_trainer import *
from simple_trainer import Runner

import torch.nn as nn
from gesi.loss import arap_loss, arap_loss_rot, arap_loss_dist
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
torch.autograd.set_detect_anomaly(True)

warnings.simplefilter("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning) # certain warning


def inverse_sigmoid(y, eps=1e-6):
    y = torch.clamp(torch.tensor(y), eps, 1 - eps)  # 0이나 1에 너무 가까운 값 방지
    return torch.log(y / (1 - y)).item()


class DrotRunner(Runner):    
    def train_drag(self) -> None:

        # get haraparameter
        shotten_by           = 1
        drot_iterations      = 500 // shotten_by
        coef_arap_drag       = self.hpara.coef_arap_drag
        coef_rgb             = self.hpara.coef_rgb
        lr_q                 = self.hpara.lr_q * shotten_by
        lr_t                 = self.hpara.lr_t * shotten_by
        lr_rest              = 2.5e-3 * shotten_by
        coef_drag            = 1
        coef_arap_drag       = 1e5
        coef_mask            = 10
        coef_arap_rot        = 1e5
        coef_arap_dist       = 1e5
        coef_fine_reg        = 1e5
        min_thresh           = 0.1
        min_mask_value = inverse_sigmoid(min_thresh)
        
        self.splats = dict(self.splats)

        if not self.cfg.skip_eval:
            step = 0
            self.eval(step=step)

        points = self.splats["means"].clone().detach()
        points.requires_grad = False

        cur_step = 0

        scale_origin = self.splats["scales"].clone().detach()
        sh0_origin = self.splats["sh0"].clone().detach()

        for stage in ["coarse", "fine"]:
            ##########################################################
            ########### 1. initialize anchor and optimizer ###########
            ##########################################################
            if stage == "coarse":
                anchor = voxelize_pointcloud_and_get_means(points, 0.04)
            elif stage == "fine":
                anchor = self.splats["means"].clone().detach()

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
                    {"params": self.splats["opacities"], "lr": lr_rest},
                    {"params": self.splats["scales"], "lr": lr_rest},
                    {"params": self.splats["sh0"], "lr": lr_rest},
                ]
            )
            quats_origin = F.normalize(self.splats['quats'].clone().detach(), dim=-1)

            ##########################################################
            ##################### 2. optimization ####################
            ##########################################################

            with torch.no_grad():
                # Eq.11 in "3D Gaussian Editing with A Single Image"
                distances, indices_knn = knn_jh(anchor, anchor, k=5)
                weight = rbf_weight(distances, gamma=30)

            for i in tqdm(range(drot_iterations + 1)):
                R = quaternion_to_matrix(F.normalize(q, dim=-1)).squeeze()  # (N, 3, 3)
                anchor_translated = anchor + t  # (N, 3)

                # Eq.15 in "3D Gaussian Editing with A Single Image
                mask_arap.data.clamp_(min=min_mask_value)
                mask_rot.data.clamp_(min=min_mask_value)
                mask_dist.data.clamp_(min=min_mask_value)

                # Eq.12 in "3D Gaussian Editing with A Single Image"
                weight_arap = weight * torch.sigmoid(mask_arap)
                weight_rot = weight * torch.sigmoid(mask_rot)
                weight_dist = weight * torch.sigmoid(mask_dist)

                # Eq.10 in "3D Gaussian Editing with A Single Image"
                loss_arap = arap_loss(anchor, anchor_translated, R, weight_arap, indices_knn)

                if stage == "coarse":
                    points_updated, quats_lbs = linear_blend_skinning_knn(points, anchor, R, t)
                    quats_updated = quaternion_multiply(quats_lbs, quats_origin)
                elif stage == "fine":
                    points_updated = anchor
                    q_normalized = F.normalize(q, dim=-1).squeeze()
                    quats_updated = quaternion_multiply(quats_origin, q_normalized)

                self.splats["means"].data.copy_(points_updated)
                self.splats["quats"].data.copy_(quats_updated)

                with torch.no_grad():
                    image_from, image_to = self.fetch_comparable_two_image()

                loss_drot = self.drot_loss_reparameter(image_from, image_to, points_updated, quats_updated)

                # Eq.13 in "3D Gaussian Editing with A Single Image"
                loss_rot = arap_loss_rot(q, weight_rot, indices_knn)
                # Eq.14 in "3D Gaussian Editing with A Single Image"
                loss_dist = arap_loss_dist(anchor, anchor_translated, weight_dist, indices_knn)

                # Eq.16 in "3D Gaussian Editing with A Single Image"
                loss_rgb = self.render_and_calc_rgb_loss()
                loss_match = loss_rgb * coef_rgb + loss_drot * coef_drag

                # Eq.17 in "3D Gaussian Editing with A Single Image"
                loss_mask_arap = (torch.sigmoid(mask_arap) - 1).abs().mean()
                loss_mask_rot = (torch.sigmoid(mask_rot) - 1).abs().mean()
                loss_mask_dist = (torch.sigmoid(mask_dist) - 1).abs().mean()

                loss = (
                    # Eq.18 in "3D Gaussian Editing with A Single Image"
                    loss_match +
                    loss_arap * coef_arap_drag +
                    # Eq.19 in "3D Gaussian Editing with A Single Image"
                    loss_rot * coef_arap_rot +
                    loss_dist * coef_arap_dist +
                    (loss_mask_rot + loss_mask_dist + loss_mask_arap) * coef_mask
                )

                if stage == "fine":
                    loss_fine_reg = (
                        (torch.sigmoid(self.splats["scale"]) / torch.sigmoid(scale_origin) - 1).abs().mean()
                        + (torch.sigmoid(self.splats["sh0"]) / torch.sigmoid(sh0_origin) - 1).abs().mean()
                    )
                    loss = loss + loss_fine_reg * coef_fine_reg

                loss.backward()
                anchor_optimizer.step()
                anchor_optimizer.zero_grad()

                wandb.log({"loss_arap": loss_arap, "loss_drag": loss_drot}, step=cur_step)
            
                if cur_step % 100 == 0:
                    image_source, image_target = self.fetch_comparable_two_image(
                        return_rgba=True, return_shape="chw"
                    )
                    _, img1, img2 = crop_two_image_with_alpha(
                        image_source, image_target
                    )
                    diff_img = get_img_diff(img1[:3], img2[:3])
                    wandb.log({"train_img_diff": wandb.Image(diff_img, caption="train_img_diff"),}, step=cur_step)
                
                cur_step += 1

            if not self.cfg.skip_eval:
                self.eval(step=cur_step)


    def render_and_calc_rgb_loss(self):
        renders, gt_images = self.fetch_comparable_two_image()
        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None

        l1loss = F.l1_loss(colors, gt_images)
        ssimloss = 1.0 - fused_ssim(
            colors.permute(0, 3, 1, 2), gt_images.permute(0, 3, 1, 2), padding="valid"
        )
        loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda

        return loss

    def drot_loss_reparameter(self, image_from, image_to, means3d, quats):

        camtoworlds = self.data["camtoworld"].to(self.device)
        Ks = self.data["K"].to(self.device)
        width = image_from.shape[2]
        height = image_from.shape[1]
        # means3d = self.splats["means"]
        opacities = self.splats["opacities"]
        scales = self.splats["scales"]
        # quats = self.splats["quats"]

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
        loss = drot_loss(image_from, image_to, uv, downsample=4)
        return loss


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

