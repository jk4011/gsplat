import json
import math
import os
import time
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import imageio
import nerfview
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import tyro
import viser
import yaml
import wandb
from easydict import EasyDict
from datasets.colmap import Dataset, Parser
from datasets.traj import (
    generate_interpolated_path,
    generate_ellipse_path_z,
    generate_spiral_path,
)
from torch import Tensor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from fused_ssim import fused_ssim
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from typing_extensions import Literal, assert_never
from utils import AppearanceOptModule, CameraOptModule, knn, rgb_to_sh, set_random_seed
from lib_bilagrid import (
    BilateralGrid,
    slice,
    color_correct,
    total_variation_loss,
)

from gsplat.compression import PngCompression
from gsplat.distributed import cli
from gsplat.rendering import rasterization
from gsplat.strategy import DefaultStrategy, MCMCStrategy

# from gsplat.optimizers import SelectiveAdam
from einops import rearrange
import torch.optim as optim
import sys

sys.path.append("/data2/wlsgur4011/GESI/")
sys.path.append("/data2/wlsgur4011/GESI/RoMa")

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
)
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

warnings.simplefilter("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning) # certain warning

from .simple_trainer import Runner

class DrotRunner(Runner):
    
    def train_drag(
        self,
        drag_iterations=500,
        rgb_iteration=500,
        filter_distance=1,
        rgb_optimize=False,
    ) -> None:
        if not self.cfg.skip_eval:
            step = 0
            self.eval(step=step)

        # get haraparameter
        n_anchor             = self.hpara.n_anchor
        coef_drag            = self.hpara.coef_drag
        coef_arap_drag       = self.hpara.coef_arap_drag
        coef_group_arap      = 0 if self.cfg.without_group else self.hpara.coef_group_arap
        coef_rgb             = self.hpara.coef_rgb
        coef_drag            = self.hpara.coef_drag
        lr_q                 = self.hpara.lr_q
        lr_t                 = self.hpara.lr_t
        rigidity_k           = self.hpara.rigidity_k
        reprojection_error   = self.hpara.reprojection_error
        anchor_k             = self.hpara.anchor_k
        rbf_gamma            = self.hpara.rbf_gamma
        cycle_threshold      = self.hpara.cycle_threshold
        decay_rate           = self.hpara.decay_rate
        min_inlier_ratio     = self.hpara.min_inlier_ratio
        confidence           = self.hpara.confidence
        refine_radius        = self.hpara.refine_radius
        refine_threhold      = self.hpara.refine_threhold
        
        self.splats = dict(self.splats)
        points_init = self.splats["means"].clone().detach()
        quats_init = self.splats["quats"].clone().detach()
        
        ##########################################################
        ################## 1. get drag via RoMa ##################
        ##########################################################
        
        # with torch.no_grad():
        #     image_source, image_target = self.fetch_with_orientation()

        from jhutil import color_log; color_log(1111, "get drag via RoMa")
        with torch.no_grad():
            image_source, image_target = self.fetch_comparable_two_image(
                return_rgba=True
            )
            drag_source, drag_target, bbox = get_drag_roma(
                image_source, image_target, device=self.device, cycle_threshold=cycle_threshold
            )
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
            points_2d, points_depth = self.project_to_2d(points_3d)

        # 18.front_mask.ipynb
        # torch.save([points_3d, points_2d, points_depth, drag_source, H, W], "/tmp/.cache/drag_source.pt")
        # exit()

        vis_mask = self.get_visibility_mask()
        points_mask, drag_mask = get_drag_mask(
            points_2d, vis_mask, drag_source, filter_distance
        )

        points_3d_filtered = points_3d[points_mask]
        drag_target_filtered = drag_target[drag_mask]
        drag_source_filtered = drag_source[drag_mask]

        ##########################################################
        ########### 3. initialize anchor and optimizer ###########
        ##########################################################
        from jhutil import color_log; color_log(3333, "initialize anchor and optimizer")

        anchor = voxelize_pointcloud_and_get_means(points_3d, voxel_size=0.04)
        # anchor, anchor_indice = torch_fpsample.sample(points_3d.cpu(), n_anchor)
        anchor = anchor.to(self.device)

        camtoworlds = self.data["camtoworld"]
        Ks = self.data["K"]
        data_all = [anchor, points_mask, drag_mask, points_2d, drag_target, drag_source, image_target, image_source, points_3d, bbox, camtoworlds, Ks]
        torch.save(data_all, f"/tmp/.cache/data_all_{self.cfg.object_name}.pt")

        N = anchor.shape[0]
        q_init = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
        q_init = q_init.unsqueeze(0).repeat(N, 1, 1)
        t_init = torch.zeros((N, 3), dtype=torch.float32, device=self.device)

        quats = nn.Parameter(q_init)  # (N, 3, 3)
        t = nn.Parameter(t_init)  # (N, 3)

        anchor_optimizer = torch.optim.Adam(
            [
                {"params": quats, "lr": lr_q},
                {"params": t, "lr": lr_t},
            ]
        )

        ##########################################################
        #################### 4. rigid grouping ###################
        ##########################################################
        from jhutil import color_log; color_log(4444, "rigid grouping")

        if self.cfg.naive_group:
            groups = naive_rigid_grouping(
                points_3d_filtered,
                drag_target_filtered,
                reprojection_error,
                camera_matrix=self.data["K"][0]
            )
        else:
            groups, outliers, group_trans = local_rigid_grouping(
                points_3d_filtered,
                drag_target_filtered,
                k=rigidity_k,
                min_inlier_ratio=min_inlier_ratio,
                confidence=confidence,
                min_inlier_size=100,
                max_expansion_iterations=100,
                reprojection_error=reprojection_error,
                iterations_count=100,
                camera_matrix=self.data["K"][0],
            )

        groud_id = -torch.ones(
            points_3d_filtered.shape[0], dtype=torch.long, device=self.device
        )
        for i, group in enumerate(groups):
            groud_id[group] = i
        group_id_all = -torch.ones(
            points_3d.shape[0], dtype=torch.long, device=self.device
        )
        group_id_all[points_mask] = groud_id

        if wandb.run and not cfg.wandb_sweep:
            n_drag = len(drag_source)
            n_pts = 5000

            img1 = rearrange(image_source[0], "h w c -> c h w")
            img2 = rearrange(image_target[0], "h w c -> c h w")
            
            origin_image = show_matching(img1[:3], img2[:3], bbox=bbox, skip_line=True)
            matching_image = show_matching(
                img1[:3],
                img2[:3],
                drag_source[:: n_drag // n_pts],
                drag_target[:: n_drag // n_pts],
                bbox=bbox,
                skip_line=True,
            )
            group_image = show_groups(
                img1[:3],
                img2[:3],
                drag_source_filtered,
                drag_target_filtered,
                groups=groups,
                bbox=bbox,
            )
            
            images = [
                wandb.Image(origin_image, caption="origin_image"),
                wandb.Image(matching_image, caption="matching_image"),
                wandb.Image(group_image, caption="group_image"),
            ]
            wandb.log({"matching": images}, commit=True)

        ##########################################################
        #################### 5. drag optimize ####################
        ##########################################################
        from jhutil import color_log; color_log(5555, "drag optimize ")
        
        quats_origin = F.normalize(self.splats["quats"].clone().detach(), dim=-1)
        
        with torch.no_grad():
            distances, indices_knn = knn_jh(anchor, anchor, k=anchor_k)
            weight = rbf_weight(distances, gamma=rbf_gamma)

        for i in tqdm(range(drag_iterations + 1)):
            
            R = quaternion_to_matrix(F.normalize(quats, dim=-1)).squeeze()  # (N, 3, 3)
            anchor_translated = anchor + t  # (N, 3)

            loss_arap = arap_loss(anchor, anchor_translated, R, weight, indices_knn)

            points_lbs, quats_lbs = linear_blend_skinning_knn(points_3d, anchor, R, t)
            updated_quaternions = quaternion_multiply(quats_lbs, quats_origin)

            R_points = quaternion_to_matrix(F.normalize(quats_lbs, dim=-1)).squeeze()
            loss_group_arap = arap_loss_grouped(
                points_3d, points_lbs, R_points, group_id_all
            )

            points_lbs_filtered = points_lbs[points_mask]
            points_lbs_filtered_2d, _ = self.project_to_2d(points_lbs_filtered)
            # loss_drag = drag_loss(points_lbs_filtered_2d, drag_target_filtered)
            # TODO: drot loss

            self.splats["means"] = points_lbs
            self.splats["quats"] = updated_quaternions
            loss_rgb = self.render_and_calc_rgb_loss() if i > 300 else 0

            loss = (
                (decay_rate ** i * coef_drag) * loss_drag
                + coef_arap_drag * loss_arap
                + coef_group_arap * loss_group_arap
                + coef_rgb * loss_rgb
            )

            loss.backward(retain_graph=True)
            anchor_optimizer.step()
            anchor_optimizer.zero_grad()

            if not self.cfg.without_group_refine and i > 300 and i % 10 == 0:
                group_id_all = refine_rigid_group(
                    points_3d,
                    points_lbs,
                    group_id_all,
                    R_points,
                    radius=refine_radius,
                    outlier_threhold=refine_threhold,
                )
                
            if wandb.run and not cfg.wandb_sweep:
                wandb.log({
                    "loss_arap"      : loss_arap,
                    "loss_drag"      : loss_drag,
                    "loss_group_arap": loss_group_arap,
                    "loss_rgb"       : loss_rgb
                }, step=i)

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
                        step=i,
                        commit=True,
                    )
                

        if not self.cfg.skip_eval:
            self.eval(step=drag_iterations+1)

        # data for motion
        motion_data = [points_init.detach(), quats_init.detach(), anchor.detach(), R.detach(), t.detach(), group_id_all, bbox]
        torch.save(motion_data, f"{self.cfg.result_dir}/motion_data.pt")

        # save checkpoint
        data = {"step": step, "splats": (torch.nn.ParameterDict(self.splats).state_dict())}
        torch.save(data, f"{self.cfg.result_dir}/ckpt_finetune.pt")




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

