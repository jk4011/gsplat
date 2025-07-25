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
from datasets.colmap import Dataset, Parser, json_to_cam2world
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
    count_covered_patches,
)
from gesi.visibility import compute_visibility
from gesi.mini_pytorch3d import quaternion_multiply, quaternion_invert
from jhutil import show_matching, show_groups
from jhutil import (
    get_img_diff,
    crop_two_image_with_background,
    crop_two_image_with_alpha,
    # save_motion_img,
)
from gesi.roma import get_drag_roma
import warnings
import torch_fpsample
from sweep_config import sweep_config, best_config_dict, SWEEP_WHOLE_ID
from gesi.rigid_grouping import local_rigid_grouping, naive_rigid_grouping, refine_rigid_group
from jhutil import save_video, bkgd2white
from torch.nn import SmoothL1Loss
from torch.optim.lr_scheduler import LambdaLR

warnings.simplefilter("ignore")
# warnings.filterwarnings("ignore", category=DeprecationWarning) # certain warning


@dataclass
class Config:
    # Disable viewer
    disable_viewer: bool = False
    # Path to the .pt files. If provide, it will skip training and run evaluation only.
    ckpt: Optional[List[str]] = None
    # Name of compression strategy to use
    compression: Optional[Literal["png"]] = None

    render_traj_all: bool = False

    render_traj_simple: bool = False

    video_path: str = None
    # Render trajectory path
    render_traj_path: str = "diva360_spiral"  # "interp", "ellipse", "spiral"

    # Path to the Mip-NeRF 360 dataset
    data_dir: str = "data/360_v2/garden"
    # Downsample factor for the dataset
    data_factor: int = 4
    # Directory to save results
    result_dir: str = "results/garden"
    # Every N images there is a test image
    test_every: int = 8
    # Random crop size for training  (experimental)
    patch_size: Optional[int] = None
    # A global scaler that applies to the scene size related parameters
    global_scale: float = 1.0
    # Normalize the world space
    normalize_world_space: bool = True
    # Camera model
    camera_model: Literal["pinhole", "ortho", "fisheye"] = "pinhole"

    # Port for the viewer server
    port: int = 8080

    # Batch size for training. Learning rates are scaled automatically
    batch_size: int = 1
    # A global factor to scale the number of training steps
    steps_scaler: float = 1.0

    # Number of training steps
    max_steps: int = 10000
    # Steps to evaluate the model
    eval_steps: List[int] = field(
        default_factory=lambda: [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    )
    # Steps to save the model
    save_steps: List[int] = field(
        default_factory=lambda: [3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
    )

    # Initialization strategy
    init_type: str = "sfm"
    # Initial number of GSs. Ignored if using sfm
    init_num_pts: int = 100_000
    # Initial extent of GSs as a multiple of the camera extent. Ignored if using sfm
    init_extent: float = 3.0
    # Degree of spherical harmonics
    sh_degree: int = 3
    # Turn on another SH degree every this steps
    sh_degree_interval: int = 1000
    # Initial opacity of GS
    init_opa: float = 0.1
    # Initial scale of GS
    init_scale: float = 1.0
    # Weight for SSIM loss
    ssim_lambda: float = 0.2

    # Near plane clipping distance
    near_plane: float = 0.01
    # Far plane clipping distance
    far_plane: float = 1e10

    # Strategy for GS densification
    strategy: Union[DefaultStrategy, MCMCStrategy] = field(
        default_factory=DefaultStrategy
    )
    # Use packed mode for rasterization, this leads to less memory usage but slightly slower.
    packed: bool = False
    # Use sparse gradients for optimization. (experimental)
    sparse_grad: bool = False
    # Use visible adam from Taming 3DGS. (experimental)
    visible_adam: bool = False
    # Anti-aliasing in rasterization. Might slightly hurt quantitative metrics.
    antialiased: bool = False

    # Use random background for training to discourage transparency
    random_bkgd: bool = False

    # Opacity regularization
    opacity_reg: float = 0.0
    # Scale regularization
    scale_reg: float = 0.0

    # Enable camera optimization.
    pose_opt: bool = False
    # Learning rate for camera optimization
    pose_opt_lr: float = 0
    # Regularization for camera optimization as weight decay
    pose_opt_reg: float = 1e-6
    # Add noise to camera extrinsics. This is only to test the camera pose optimization.
    pose_noise: float = 0.0

    # Enable appearance optimization. (experimental)
    app_opt: bool = False
    # Appearance embedding dimension
    app_embed_dim: int = 16
    # Learning rate for appearance optimization
    app_opt_lr: float = 1e-3
    # Regularization for appearance optimization as weight decay
    app_opt_reg: float = 1e-6

    # Enable bilateral grid. (experimental)
    use_bilateral_grid: bool = False
    # Shape of the bilateral grid (X, Y, W)
    bilateral_grid_shape: Tuple[int, int, int] = (16, 16, 8)

    # Enable depth loss. (experimental)
    depth_loss: bool = False
    # Weight for depth loss
    depth_lambda: float = 1e-2

    # Dump information to tensorboard every this steps
    tb_every: int = 100
    # Save training images to tensorboard
    tb_save_image: bool = False

    lpips_net: Literal["vgg", "alex"] = "vgg"

    data_name: str = "diva360"

    single_finetune: bool = False

    cam_idx: int = 0

    wandb: bool = False

    object_name: str = None

    wandb_group: str = None

    wandb_sweep: bool = False
    
    without_group: bool = False

    without_group_refine: bool = False

    naive_group: bool = False
    
    motion_video: bool = False

    video_name: str = "video"
    
    skip_eval: bool = False

    simple_video: bool = False

    finetune_with_only_rgb: bool = False


    def adjust_steps(self, factor: float):
        self.eval_steps = [int(i * factor) for i in self.eval_steps]
        self.save_steps = [int(i * factor) for i in self.save_steps]
        self.max_steps = int(self.max_steps * factor)
        self.sh_degree_interval = int(self.sh_degree_interval * factor)

        strategy = self.strategy
        if isinstance(strategy, DefaultStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.reset_every = int(strategy.reset_every * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        elif isinstance(strategy, MCMCStrategy):
            strategy.refine_start_iter = int(strategy.refine_start_iter * factor)
            strategy.refine_stop_iter = int(strategy.refine_stop_iter * factor)
            strategy.refine_every = int(strategy.refine_every * factor)
        else:
            assert_never(strategy)


def create_splats_with_optimizers(
    parser: Parser,
    init_type: str = "sfm",
    init_num_pts: int = 100_000,
    init_extent: float = 3.0,
    init_opacity: float = 0.1,
    init_scale: float = 1.0,
    scene_scale: float = 1.0,
    sh_degree: int = 3,
    sparse_grad: bool = False,
    visible_adam: bool = False,
    batch_size: int = 1,
    feature_dim: Optional[int] = None,
    device: str = "cuda",
    world_rank: int = 0,
    world_size: int = 1,
) -> Tuple[torch.nn.ParameterDict, Dict[str, torch.optim.Optimizer]]:
    if init_type == "sfm":
        points = torch.from_numpy(parser.points).float()
        rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()
    elif init_type == "random":
        points = init_extent * scene_scale * (torch.rand((init_num_pts, 3)) * 2 - 1)
        rgbs = torch.rand((init_num_pts, 3))
    else:
        raise ValueError("Please specify a correct init_type: sfm or random")

    # Initialize the GS size to be the average dist of the 3 nearest neighbors
    dist2_avg = (knn(points, 4)[:, 1:] ** 2).mean(dim=-1)  # [N,]
    dist_avg = torch.sqrt(dist2_avg)
    scales = torch.log(dist_avg * init_scale).unsqueeze(-1).repeat(1, 3)  # [N, 3]

    # Distribute the GSs to different ranks (also works for single rank)
    points = points[world_rank::world_size]
    rgbs = rgbs[world_rank::world_size]
    scales = scales[world_rank::world_size]

    N = points.shape[0]
    quats = torch.rand((N, 4))  # [N, 4]
    opacities = torch.logit(torch.full((N,), init_opacity))  # [N,]

    params = [
        # name, value, lr
        ("means", torch.nn.Parameter(points), 1.6e-4 * scene_scale),
        ("scales", torch.nn.Parameter(scales), 5e-3),
        ("quats", torch.nn.Parameter(quats), 1e-3),
        ("opacities", torch.nn.Parameter(opacities), 5e-2),
    ]

    if feature_dim is None:
        # color is SH coefficients.
        colors = torch.zeros((N, (sh_degree + 1) ** 2, 3))  # [N, K, 3]
        colors[:, 0, :] = rgb_to_sh(rgbs)
        params.append(("sh0", torch.nn.Parameter(colors[:, :1, :]), 2.5e-3))
        params.append(("shN", torch.nn.Parameter(colors[:, 1:, :]), 2.5e-3 / 20))
    else:
        # features will be used for appearance and view-dependent shading
        features = torch.rand(N, feature_dim)  # [N, feature_dim]
        params.append(("features", torch.nn.Parameter(features), 2.5e-3))
        colors = torch.logit(rgbs)  # [N, 3]
        params.append(("colors", torch.nn.Parameter(colors), 2.5e-3))

    splats = torch.nn.ParameterDict({n: v for n, v, _ in params}).to(device)
    # Scale learning rate based on batch size, reference:
    # https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Note that this would not make the training exactly equivalent, see
    # https://arxiv.org/pdf/2402.18824v1
    BS = batch_size * world_size
    optimizer_class = None
    if sparse_grad:
        optimizer_class = torch.optim.SparseAdam
    elif visible_adam:
        optimizer_class = SelectiveAdam
    else:
        optimizer_class = torch.optim.Adam
    optimizers = {
        name: optimizer_class(
            [{"params": splats[name], "lr": lr * math.sqrt(BS), "name": name}],
            eps=1e-15 / math.sqrt(BS),
            # TODO: check betas logic when BS is larger than 10 betas[0] will be zero.
            betas=(1 - BS * (1 - 0.9), 1 - BS * (1 - 0.999)),
        )
        for name, _, lr in params
    }
    return splats, optimizers


class Runner:
    """Engine for training and testing."""

    def __init__(
        self, local_rank: int, world_rank, world_size: int, cfg: Config
    ) -> None:
        set_random_seed(42 + local_rank)

        self.cfg = cfg
        self.world_rank = world_rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = f"cuda:{local_rank}"

        # Where to dump results.
        os.makedirs(cfg.result_dir, exist_ok=True)

        # Setup output directories.
        self.ckpt_dir = f"{cfg.result_dir}/ckpts"
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.stats_dir = f"{cfg.result_dir}/stats"
        os.makedirs(self.stats_dir, exist_ok=True)
        self.render_dir = f"{cfg.result_dir}/renders"
        os.makedirs(self.render_dir, exist_ok=True)

        # Tensorboard
        self.writer = SummaryWriter(log_dir=f"{cfg.result_dir}/tb")

        # Load data: Training data should contain initial points and colors.
        self.parser = Parser(
            data_dir=cfg.data_dir,
            factor=cfg.data_factor,
            normalize=cfg.normalize_world_space,
            test_every=cfg.test_every,
            data_name=cfg.data_name,
        )
        self.trainset = Dataset(
            self.parser,
            split="train",
            patch_size=cfg.patch_size,
            load_depths=cfg.depth_loss,
            single_finetune=cfg.single_finetune,
            cam_idx=cfg.cam_idx,
        )
        self.valset = Dataset(
            self.parser, split="val", single_finetune=cfg.single_finetune
        )
        self.scene_scale = self.parser.scene_scale * 1.1 * cfg.global_scale
        print("Scene scale:", self.scene_scale)

        # Model
        feature_dim = 32 if cfg.app_opt else None
        self.splats, self.optimizers = create_splats_with_optimizers(
            self.parser,
            init_type=cfg.init_type,
            init_num_pts=cfg.init_num_pts,
            init_extent=cfg.init_extent,
            init_opacity=cfg.init_opa,
            init_scale=cfg.init_scale,
            scene_scale=self.scene_scale,
            sh_degree=cfg.sh_degree,
            sparse_grad=cfg.sparse_grad,
            visible_adam=cfg.visible_adam,
            batch_size=cfg.batch_size,
            feature_dim=feature_dim,
            device=self.device,
            world_rank=world_rank,
            world_size=world_size,
        )
        print("Model initialized. Number of GS:", len(self.splats["means"]))

        # Densification Strategy
        self.cfg.strategy.check_sanity(self.splats, self.optimizers)

        if isinstance(self.cfg.strategy, DefaultStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state(
                scene_scale=self.scene_scale
            )
        elif isinstance(self.cfg.strategy, MCMCStrategy):
            self.strategy_state = self.cfg.strategy.initialize_state()
        else:
            assert_never(self.cfg.strategy)

        # Compression Strategy
        self.compression_method = None
        if cfg.compression is not None:
            if cfg.compression == "png":
                self.compression_method = PngCompression()
            else:
                raise ValueError(f"Unknown compression strategy: {cfg.compression}")

        self.pose_optimizers = []
        if cfg.pose_opt:
            self.pose_adjust = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_adjust.zero_init()
            self.pose_optimizers = [
                torch.optim.Adam(
                    self.pose_adjust.parameters(),
                    lr=cfg.pose_opt_lr * math.sqrt(cfg.batch_size),
                    weight_decay=cfg.pose_opt_reg,
                )
            ]
            if world_size > 1:
                self.pose_adjust = DDP(self.pose_adjust)

        if cfg.pose_noise > 0.0:
            self.pose_perturb = CameraOptModule(len(self.trainset)).to(self.device)
            self.pose_perturb.random_init(cfg.pose_noise)
            if world_size > 1:
                self.pose_perturb = DDP(self.pose_perturb)

        self.app_optimizers = []
        if cfg.app_opt:
            assert feature_dim is not None
            self.app_module = AppearanceOptModule(
                len(self.trainset), feature_dim, cfg.app_embed_dim, cfg.sh_degree
            ).to(self.device)
            # initialize the last layer to be zero so that the initial output is zero.
            torch.nn.init.zeros_(self.app_module.color_head[-1].weight)
            torch.nn.init.zeros_(self.app_module.color_head[-1].bias)
            self.app_optimizers = [
                torch.optim.Adam(
                    self.app_module.embeds.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size) * 10.0,
                    weight_decay=cfg.app_opt_reg,
                ),
                torch.optim.Adam(
                    self.app_module.color_head.parameters(),
                    lr=cfg.app_opt_lr * math.sqrt(cfg.batch_size),
                ),
            ]
            if world_size > 1:
                self.app_module = DDP(self.app_module)

        self.bil_grid_optimizers = []
        if cfg.use_bilateral_grid:
            self.bil_grids = BilateralGrid(
                len(self.trainset),
                grid_X=cfg.bilateral_grid_shape[0],
                grid_Y=cfg.bilateral_grid_shape[1],
                grid_W=cfg.bilateral_grid_shape[2],
            ).to(self.device)
            self.bil_grid_optimizers = [
                torch.optim.Adam(
                    self.bil_grids.parameters(),
                    lr=2e-3 * math.sqrt(cfg.batch_size),
                    eps=1e-15,
                ),
            ]

        # Losses & Metrics.
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(self.device)

        if cfg.lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="alex", normalize=True
            ).to(self.device)
        elif cfg.lpips_net == "vgg":
            # The 3DGS official repo uses lpips vgg, which is equivalent with the following:
            self.lpips = LearnedPerceptualImagePatchSimilarity(
                net_type="vgg", normalize=False
            ).to(self.device)
        else:
            raise ValueError(f"Unknown LPIPS network: {cfg.lpips_net}")

        # Viewer
        if not self.cfg.disable_viewer:
            self.server = viser.ViserServer(port=cfg.port, verbose=False)
            self.viewer = nerfview.Viewer(
                server=self.server,
                render_fn=self._viewer_render_fn,
                mode="training",
            )

        if cfg.wandb_sweep:
            wandb.init(
                project=f"GESI_{cfg.data_name}",
                dir="./wandb",
                group=cfg.wandb_group,
                name=cfg.object_name,
                settings=wandb.Settings(start_method="fork"),
            )
            self.hpara = wandb.config
        elif cfg.wandb:
            wandb.init(
                project=f"GESI_{cfg.data_name}",
                dir="./wandb",
                group=cfg.wandb_group,
                name=cfg.object_name,
                settings=wandb.Settings(start_method="fork"),
                config=best_config_dict[cfg.data_name],
            )
            print(EasyDict(asdict(cfg)))
            self.hpara = best_config_dict[cfg.data_name]
        else:
            self.hpara = best_config_dict[cfg.data_name]
        
        if cfg.data_name == "DFA":
            self.backgrounds = torch.ones(1, 3, device=self.device)   # white
        else:
            self.backgrounds = torch.zeros(1, 3, device=self.device)  # black

    def rasterize_splats(
        self,
        camtoworlds: Tensor,
        Ks: Tensor,
        width: int,
        height: int,
        masks: Optional[Tensor] = None,
        interactive: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Tensor, Dict]:
        means = self.splats["means"]  # [N, 3]
        # quats = F.normalize(self.splats["quats"], dim=-1)  # [N, 4]
        # rasterization does normalization internally
        quats = self.splats["quats"]  # [N, 4]
        scales = torch.exp(self.splats["scales"])  # [N, 3]
        opacities = torch.sigmoid(self.splats["opacities"])  # [N,]

        image_ids = kwargs.pop("image_ids", None)
        if self.cfg.app_opt:
            colors = self.app_module(
                features=self.splats["features"],
                embed_ids=image_ids,
                dirs=means[None, :, :] - camtoworlds[:, None, :3, 3],
                sh_degree=kwargs.pop("sh_degree", self.cfg.sh_degree),
            )
            colors = colors + self.splats["colors"]
            colors = torch.sigmoid(colors)
        else:
            colors = torch.cat([self.splats["sh0"], self.splats["shN"]], 1)  # [N, K, 3]

            
        rasterize_mode = "antialiased" if self.cfg.antialiased else "classic"
        render_colors, render_alphas, info = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=torch.linalg.inv(camtoworlds),  # [C, 4, 4]
            Ks=Ks,  # [C, 3, 3]
            width=width,
            height=height,
            packed=self.cfg.packed,
            absgrad=(
                self.cfg.strategy.absgrad
                if isinstance(self.cfg.strategy, DefaultStrategy)
                else False
            ),
            sparse_grad=self.cfg.sparse_grad,
            rasterize_mode=rasterize_mode,
            distributed=self.world_size > 1,
            camera_model=self.cfg.camera_model,
            backgrounds=self.backgrounds if not interactive else None,
            **kwargs,
        )
        if masks is not None:
            render_colors[~masks] = 0
        return render_colors, render_alphas, info

    def train(self):
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        # Dump cfg.
        if world_rank == 0:
            with open(f"{cfg.result_dir}/cfg.yml", "w") as f:
                yaml.dump(vars(cfg), f)

        max_steps = cfg.max_steps
        init_step = 0

        schedulers = [
            # means has a learning rate schedule, that end at 0.01 of the initial value
            torch.optim.lr_scheduler.ExponentialLR(
                self.optimizers["means"], gamma=0.01 ** (1.0 / max_steps)
            ),
        ]
        if cfg.pose_opt:
            # pose optimization has a learning rate schedule
            schedulers.append(
                torch.optim.lr_scheduler.ExponentialLR(
                    self.pose_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                )
            )
        if cfg.use_bilateral_grid:
            # bilateral grid has a learning rate schedule. Linear warmup for 1000 steps.
            schedulers.append(
                torch.optim.lr_scheduler.ChainedScheduler(
                    [
                        torch.optim.lr_scheduler.LinearLR(
                            self.bil_grid_optimizers[0],
                            start_factor=0.01,
                            total_iters=1000,
                        ),
                        torch.optim.lr_scheduler.ExponentialLR(
                            self.bil_grid_optimizers[0], gamma=0.01 ** (1.0 / max_steps)
                        ),
                    ]
                )
            )

        trainloader = DataLoader(
            self.trainset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
            pin_memory=True,
        )
        trainloader_iter = iter(trainloader)

        # Training loop.
        global_tic = time.time()
        pbar = tqdm(range(init_step, max_steps))
        for step in pbar:
            if not cfg.disable_viewer:
                while self.viewer.state.status == "paused":
                    time.sleep(0.01)
                self.viewer.lock.acquire()
                tic = time.time()

            try:
                data = next(trainloader_iter)
            except StopIteration:
                trainloader_iter = iter(trainloader)
                data = next(trainloader_iter)

            camtoworlds = camtoworlds_gt = data["camtoworld"].to(device)  # [1, 4, 4]
            Ks = data["K"].to(device)  # [1, 3, 3]
            pixels = data["image"].to(device) / 255.0  # [1, H, W, 3]
            alphas_gt = data["alpha"].to(device) / 255.0  # [1, H, W, 3]
            num_train_rays_per_step = (
                pixels.shape[0] * pixels.shape[1] * pixels.shape[2]
            )
            image_ids = data["image_id"].to(device)
            masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
            if cfg.depth_loss:
                points = data["points"].to(device)  # [1, M, 2]
                depths_gt = data["depths"].to(device)  # [1, M]

            height, width = pixels.shape[1:3]

            if cfg.pose_noise:
                camtoworlds = self.pose_perturb(camtoworlds, image_ids)

            if cfg.pose_opt:
                camtoworlds = self.pose_adjust(camtoworlds, image_ids)

            # sh schedule
            sh_degree_to_use = min(step // cfg.sh_degree_interval, cfg.sh_degree)

            # forward
            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=sh_degree_to_use,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB+ED" if cfg.depth_loss else "RGB",
                masks=masks,
            )
            if renders.shape[-1] == 4:
                colors, depths = renders[..., 0:3], renders[..., 3:4]
            else:
                colors, depths = renders, None

            if cfg.use_bilateral_grid:
                grid_y, grid_x = torch.meshgrid(
                    (torch.arange(height, device=self.device) + 0.5) / height,
                    (torch.arange(width, device=self.device) + 0.5) / width,
                    indexing="ij",
                )
                grid_xy = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)
                colors = slice(self.bil_grids, grid_xy, colors, image_ids)["rgb"]

            if cfg.random_bkgd:
                bkgd = torch.rand(1, 3, device=device)
                colors = colors + bkgd * (1.0 - alphas)
                pixels = pixels + bkgd * (1.0 - alphas_gt)
            
            # bkgd = torch.ones(1, 3, device=device)
            # colors = colors + bkgd * (1.0 - alphas)
            # pixels = pixels + bkgd * (1.0 - alphas_gt)

            self.cfg.strategy.step_pre_backward(
                params=self.splats,
                optimizers=self.optimizers,
                state=self.strategy_state,
                step=step,
                info=info,
            )

            # loss
            l1loss = F.l1_loss(colors, pixels)
            ssimloss = 1.0 - fused_ssim(
                colors.permute(0, 3, 1, 2), pixels.permute(0, 3, 1, 2), padding="valid"
            )
            loss = l1loss * (1.0 - cfg.ssim_lambda) + ssimloss * cfg.ssim_lambda
            if cfg.depth_loss:
                # query depths from depth map
                points = torch.stack(
                    [
                        points[:, :, 0] / (width - 1) * 2 - 1,
                        points[:, :, 1] / (height - 1) * 2 - 1,
                    ],
                    dim=-1,
                )  # normalize to [-1, 1]
                grid = points.unsqueeze(2)  # [1, M, 1, 2]
                depths = F.grid_sample(
                    depths.permute(0, 3, 1, 2), grid, align_corners=True
                )  # [1, 1, M, 1]
                depths = depths.squeeze(3).squeeze(1)  # [1, M]
                # calculate loss in disparity space
                disp = torch.where(depths > 0.0, 1.0 / depths, torch.zeros_like(depths))
                disp_gt = 1.0 / depths_gt  # [1, M]
                depthloss = F.l1_loss(disp, disp_gt) * self.scene_scale
                loss += depthloss * cfg.depth_lambda
            if cfg.use_bilateral_grid:
                tvloss = 10 * total_variation_loss(self.bil_grids.grids)
                loss += tvloss

            # regularizations
            if cfg.opacity_reg > 0.0:
                loss = (
                    loss
                    + cfg.opacity_reg
                    * torch.abs(torch.sigmoid(self.splats["opacities"])).mean()
                )
            if cfg.scale_reg > 0.0:
                # loss = (
                #     loss
                #     + cfg.scale_reg * torch.abs(torch.exp(self.splats["scales"])).mean()
                # )

                # Gaussian marble
                loss = (
                    loss
                    + cfg.scale_reg
                    * torch.abs(
                        torch.exp(self.splats["scales"].max(dim=-1)[0])
                        - torch.exp(self.splats["scales"].min(dim=-1)[0])
                    ).mean()
                )

            loss.backward()

            desc = f"loss={loss.item():.3f}| " f"sh degree={sh_degree_to_use}| "
            if cfg.depth_loss:
                desc += f"depth loss={depthloss.item():.6f}| "
            if cfg.pose_opt and cfg.pose_noise:
                # monitor the pose error if we inject noise
                pose_err = F.l1_loss(camtoworlds_gt, camtoworlds)
                desc += f"pose err={pose_err.item():.6f}| "
            pbar.set_description(desc)

            # write images (gt and render)
            # if world_rank == 0 and step % 800 == 0:
            #     canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
            #     canvas = canvas.reshape(-1, *canvas.shape[2:])
            #     imageio.imwrite(
            #         f"{self.render_dir}/train_rank{self.world_rank}.png",
            #         (canvas * 255).astype(np.uint8),
            #     )

            if world_rank == 0 and cfg.tb_every > 0 and step % cfg.tb_every == 0:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                self.writer.add_scalar("train/loss", loss.item(), step)
                self.writer.add_scalar("train/l1loss", l1loss.item(), step)
                self.writer.add_scalar("train/ssimloss", ssimloss.item(), step)
                self.writer.add_scalar("train/num_GS", len(self.splats["means"]), step)
                self.writer.add_scalar("train/mem", mem, step)
                if cfg.depth_loss:
                    self.writer.add_scalar("train/depthloss", depthloss.item(), step)
                if cfg.use_bilateral_grid:
                    self.writer.add_scalar("train/tvloss", tvloss.item(), step)
                if cfg.tb_save_image:
                    canvas = torch.cat([pixels, colors], dim=2).detach().cpu().numpy()
                    canvas = canvas.reshape(-1, *canvas.shape[2:])
                    self.writer.add_image("train/render", canvas, step)
                self.writer.flush()

            # save checkpoint before updating the model
            if step in [i - 1 for i in cfg.save_steps] or step == max_steps - 1:
                mem = torch.cuda.max_memory_allocated() / 1024**3
                stats = {
                    "mem": mem,
                    "ellipse_time": time.time() - global_tic,
                    "num_GS": len(self.splats["means"]),
                }
                print("Step: ", step, stats)
                with open(
                    f"{self.stats_dir}/train_step{step:04d}_rank{self.world_rank}.json",
                    "w",
                ) as f:
                    json.dump(stats, f)
                data = {"step": step, "splats": self.splats.state_dict()}
                if cfg.pose_opt:
                    if world_size > 1:
                        data["pose_adjust"] = self.pose_adjust.module.state_dict()
                    else:
                        data["pose_adjust"] = self.pose_adjust.state_dict()
                if cfg.app_opt:
                    if world_size > 1:
                        data["app_module"] = self.app_module.module.state_dict()
                    else:
                        data["app_module"] = self.app_module.state_dict()
                torch.save(
                    data, f"{self.ckpt_dir}/ckpt_{step}_rank{self.world_rank}.pt"
                )

            # Turn Gradients into Sparse Tensor before running optimizer
            if cfg.sparse_grad:
                assert cfg.packed, "Sparse gradients only work with packed mode."
                gaussian_ids = info["gaussian_ids"]
                for k in self.splats.keys():
                    grad = self.splats[k].grad
                    if grad is None or grad.is_sparse:
                        continue
                    self.splats[k].grad = torch.sparse_coo_tensor(
                        indices=gaussian_ids[None],  # [1, nnz]
                        values=grad[gaussian_ids],  # [nnz, ...]
                        size=self.splats[k].size(),  # [N, ...]
                        is_coalesced=len(Ks) == 1,
                    )

            if cfg.visible_adam:
                gaussian_cnt = self.splats.means.shape[0]
                if cfg.packed:
                    visibility_mask = torch.zeros_like(
                        self.splats["opacities"], dtype=bool
                    )
                    visibility_mask.scatter_(0, info["gaussian_ids"], 1)
                else:
                    visibility_mask = (info["radii"] > 0).any(0)

            # optimize
            for optimizer in self.optimizers.values():
                if cfg.visible_adam:
                    optimizer.step(visibility_mask)
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.pose_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.app_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for optimizer in self.bil_grid_optimizers:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            for scheduler in schedulers:
                scheduler.step()

            # Run post-backward steps after backward and optimizer
            if isinstance(self.cfg.strategy, DefaultStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    packed=cfg.packed,
                )
            elif isinstance(self.cfg.strategy, MCMCStrategy):
                self.cfg.strategy.step_post_backward(
                    params=self.splats,
                    optimizers=self.optimizers,
                    state=self.strategy_state,
                    step=step,
                    info=info,
                    lr=schedulers[0].get_last_lr()[0],
                )
            else:
                assert_never(self.cfg.strategy)

            # eval the full set
            if step in [i - 1 for i in cfg.eval_steps]:
                self.eval(step)
                # self.render_traj(step)

            # run compression
            if cfg.compression is not None and step in [i - 1 for i in cfg.eval_steps]:
                self.run_compression(step=step)

            if not cfg.disable_viewer:
                self.viewer.lock.release()
                num_train_steps_per_sec = 1.0 / (time.time() - tic)
                num_train_rays_per_sec = (
                    num_train_rays_per_step * num_train_steps_per_sec
                )
                # Update the viewer state.
                self.viewer.state.num_train_rays_per_sec = num_train_rays_per_sec
                # Update the scene.
                self.viewer.update(step, num_train_rays_per_step)
            
            if wandb.run and not cfg.wandb_sweep:
                if step % 100 == 0:
                    image_source, image_target = self.fetch_comparable_two_image(
                        return_rgba=True, return_shape="chw", use_gt_pose=True
                    )
                    _, img1, img2 = crop_two_image_with_alpha(
                        image_source, image_target
                    )
                    diff_img = get_img_diff(img1, img2)
                    
                    wandb.log(
                        {"train_diff": [
                            wandb.Image(diff_img, caption="train_diff"),
                        ]}, step=step,
                        commit=True
                    )
        # self.render_traj(step=step, group_id_all=None, video_path=cfg.video_path)

    def train_drag(
        self,
        drag_iterations=500,
    ) -> None:
        if not self.cfg.skip_eval:
            step = 0
            self.eval(step=step)

        # get haraparameter
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
        min_inlier_ratio     = self.hpara.min_inlier_ratio
        confidence           = self.hpara.confidence
        refine_radius        = self.hpara.refine_radius
        refine_threhold      = self.hpara.refine_threhold
        voxel_size           = self.hpara.voxel_size
        filter_distance      = self.hpara.filter_distance
        min_group_size       = self.hpara.min_group_size
        
        self.splats = dict(self.splats)
        points_init = self.splats["means"].clone().detach()
        quats_init = self.splats["quats"].clone().detach()


        
        ##########################################################
        ############ 1. get campose and drag via RoMa ############
        ##########################################################

        from jhutil import color_log; color_log(1111, "get campose and drag via RoMa")
        with torch.no_grad():

            self.image_target = self.fetch_target_image(return_rgba=True)
            self.camtoworlds_pred = self.estimate_camtoworlds(self.image_target)

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
        points_mask, drag_indice = get_drag_mask(
            points_2d, vis_mask, drag_source, filter_distance
        )

        points_3d_filtered = points_3d[points_mask]
        drag_target_filtered = drag_target[drag_indice]
        drag_source_filtered = drag_source[drag_indice]
        ##########################################################
        ########### 3. initialize anchor and optimizer ###########
        ##########################################################
        from jhutil import color_log; color_log(3333, "initialize anchor and optimizer")

        anchor = voxelize_pointcloud_and_get_means(points_3d, voxel_size=voxel_size)
        
        anchor = anchor.to(self.device)


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
                min_group_size=min_group_size,
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
        group_id_all_init = group_id_all.clone().detach()
        if wandb.run and not cfg.wandb_sweep:
            n_drag = len(drag_source)
            n_pts = 5000

            img1 = rearrange(image_source[0], "h w c -> c h w")
            img2 = rearrange(image_target[0], "h w c -> c h w")
            
            origin_image = show_matching(img1, img2, bbox=bbox, skip_line=True)
            matching_image = show_matching(
                img1,
                img2,
                drag_source[:: n_drag // n_pts],
                drag_target[:: n_drag // n_pts],
                bbox=bbox,
                skip_line=True,
            )
            for i in range(0, 5):
                group_id_all_tmp = torch.where(
                    group_id_all == i, group_id_all, -1
                )
                sh0_origin = self.update_sh_with_group_id(group_id_all_tmp)
                with torch.no_grad():
                    group_image, _ = self.fetch_comparable_two_image(return_rgba=True)

                group_image = rearrange(group_image[0], "h w c -> c h w")
                group_image = show_matching(img1, group_image, bbox=bbox, skip_line=True)
                self.splats["sh0"] = sh0_origin
                
                images = [
                    wandb.Image(origin_image, caption="origin_image"),
                    wandb.Image(matching_image, caption="matching_image"),
                ]
                wandb.log({"matching": images})
                wandb.log({"group_image": [wandb.Image(group_image, caption="initial_group")]})
        

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
            loss_drag = drag_loss(points_lbs_filtered_2d, drag_target_filtered)

            self.splats["means"] = points_lbs
            self.splats["quats"] = updated_quaternions
            loss_rgb = self.render_and_calc_rgb_loss() if i > 300 else 0

            loss = (
                coef_drag * loss_drag
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
                    diff_img = get_img_diff(img1, img2)
                    
                    wandb.log(
                        {"train_diff": [
                            wandb.Image(diff_img, caption="train_diff"),
                        ]}, step=i,
                    )

        if not self.cfg.skip_eval:
            self.eval(step=drag_iterations+1, is_pred_camtoworld=True)


        sh0_origin = self.update_sh_with_group_id(group_id_all)
        self.log_final_group(points_init, quats_init, bbox)
        
        self.splats["sh0"] = sh0_origin

        # data for motion
        motion_data = [points_init.detach(), quats_init.detach(), anchor.detach(), R.detach(), t.detach(), group_id_all, bbox, self.camtoworlds_pred]
        torch.save(motion_data, f"{self.cfg.result_dir}/motion_data.pt")

        # save checkpoint
        data = {"step": step, "splats": (torch.nn.ParameterDict(self.splats).state_dict()), "group_id_all": group_id_all, "group_id_all_init": group_id_all_init}
        torch.save(data, f"{self.cfg.result_dir}/ckpt_finetune.pt")
        print("save checkpoint to ", f"{self.cfg.result_dir}/ckpt_finetune.pt")


        # data for anything
        camtoworlds = self.data["camtoworld"]
        Ks = self.data["K"]
        data_all = {
            "width" : self.width,
            "height" : self.height,
            "points_3d" : points_3d,
            "points_2d" : points_2d,
            "quats" : self.splats["quats"],
            "opacities" : self.splats["opacities"],
            "scales" : self.splats["scales"],
            "sh0" : self.splats["sh0"],
            "Ks" : Ks,
            "camtoworlds" : camtoworlds,
            "anchor" : anchor,
            "points_mask" : points_mask,
            "drag_mask" : drag_indice,
            "drag_target" : drag_target,
            "drag_source" : drag_source,
            "image_target" : image_target,
            "image_source" : image_source,
            "bbox" : bbox,
            "group_id_all_init" : group_id_all_init,
            "group_id_all" : group_id_all
        }
        torch.save(data_all, f"/tmp/.cache/data_all_{self.cfg.object_name}.pt")


    def log_final_group(self, points_init, quats_init, bbox):
        with torch.no_grad():
            group_image, image_target = self.fetch_comparable_two_image(return_rgba=True)

        im1 = rearrange(image_target[0], "h w c -> c h w")
        im2 = rearrange(group_image[0], "h w c -> c h w")

        image_final_group = show_matching(im1, im2, bbox=bbox, skip_line=True)
        wandb.log({"group_image": [wandb.Image(image_final_group, caption="final_group")]})
        
        points_final = self.splats["means"].clone().detach()
        quats_final = self.splats["quats"].clone().detach()

        with torch.no_grad():
            self.splats["means"] = points_init
            self.splats["quats"] = quats_init
            group_image, image_target = self.fetch_comparable_two_image(return_rgba=True)
            self.splats["means"] = points_final
            self.splats["quats"] = quats_final
            im2 = rearrange(group_image[0], "h w c -> c h w")
        wandb.log({"group_image": [wandb.Image(im2, caption="final_group_2")]})


    def get_visibility_mask(self):

        means3d = self.splats["means"]
        quats = self.splats["quats"]
        opacities = self.splats["opacities"]
        scales = self.splats["scales"]
        Ks = self.data["K"].to(self.device)
        camtoworlds = self.data["camtoworld"].to(self.device)
        visibility, means2d = compute_visibility(self.width, self.height, means3d, quats, opacities, scales, Ks, camtoworlds)
        visible_mask = visibility > self.hpara.vis_threshold

        return visible_mask


    def make_motion_video(self, idx=0, n_iter=1000, threhold_early_stop=1e-5 , scheduler_step=300, min_rigid_coef=0):

        motion_data = torch.load(f"{self.cfg.result_dir}/motion_data.pt")

        # TODO: delete
        try:
            points_init, quats_init, anchor, R_goal, t_goal, group_id_all, bbox, self.camtoworlds_pred = motion_data
        except:
            points_init, quats_init, anchor, R_goal, t_goal, group_id_all, bbox = motion_data
            breakpoint()
            trainloader = DataLoader(self.trainset, batch_size=1)
            trainloader_iter = iter(trainloader)
            self.data = next(trainloader_iter)
            self.camtoworlds_pred = self.data["camtoworld"].to(self.device)
    

        self.splats["means"].data.copy_(points_init.detach())
        self.splats["quats"].data.copy_(quats_init.detach())
        
        # get haraparameter
        coef_drag            = self.hpara.coef_drag_3d
        coef_group_arap      = self.hpara.coef_group_arap * 0.01
        coef_arap_drag       = self.hpara.coef_arap_drag * 0.2
        if self.cfg.data_name == "dfa":
            coef_arap_drag *= 0.1
        elif "dog" in self.cfg.result_dir:
            coef_arap_drag *= 5

        lr                   = self.hpara.lr_motion
        anchor_k             = self.hpara.anchor_k
        rbf_gamma            = self.hpara.rbf_gamma
        
        
        ##########################################################
        ########### a. initialize anchor and optimizer ###########
        ##########################################################
        
        quats_origin = F.normalize(self.splats["quats"].clone().detach(), dim=-1)
        points_3d = self.splats["means"].clone().detach()
        
        self.splats["quats"].data.copy_(quats_origin)
        

        anchor = anchor.to(self.device)
        N = anchor.shape[0]
        # re-init parameter
        q_init = torch.tensor([1, 0, 0, 0], dtype=torch.float32, device=self.device)
        q_init = q_init.unsqueeze(0).repeat(N, 1, 1)
        t_init = torch.zeros((N, 3), dtype=torch.float32, device=self.device)

        q = nn.Parameter(q_init)  # (N, 3, 3)
        t = nn.Parameter(t_init)  # (N, 3)

        anchor_optimizer = torch.optim.Adam([q, t], lr=lr, weight_decay=0.01)
        scheduler = LambdaLR(anchor_optimizer, lr_lambda=lambda step: 0.3 + min(0.7, (step + 1) / scheduler_step))
        ##########################################################
        #################### b. drag optimize ####################
        ##########################################################
        if not self.cfg.disable_viewer:
            breakpoint()
        
        with torch.no_grad():
            distances, indices_knn = knn_jh(anchor, anchor, k=anchor_k)
            weight = rbf_weight(distances, gamma=rbf_gamma)

        loss_fn = SmoothL1Loss(beta=0.1)
        
        # change into until convergence
        image_list = []
        for i in tqdm(range(n_iter)):
            R = quaternion_to_matrix(F.normalize(q, dim=-1)).squeeze()  # (N, 3, 3)
            anchor_translated = anchor + t  # (N, 3)

            loss_arap = arap_loss(anchor, anchor_translated, R, weight, indices_knn)
            loss_drag = loss_fn(R, R_goal) + loss_fn(t, t_goal)
            
            if loss_drag < threhold_early_stop:
                break

            points_lbs, quats_lbs = linear_blend_skinning_knn(points_3d, anchor, R, t)
            updated_quaternions = quaternion_multiply(quats_lbs, quats_origin)

            R_points = quaternion_to_matrix(F.normalize(quats_lbs, dim=-1)).squeeze()
            loss_group_arap = arap_loss_grouped(
                points_3d, points_lbs, R_points, group_id_all
            )

            self.splats["means"].data.copy_(points_lbs)
            self.splats["quats"].data.copy_(updated_quaternions)
            
            loss = (
                coef_drag * loss_drag
                # + max(min_rigid_coef, 1 - i / scheduler_step) * coef_group_arap * loss_group_arap
                + max(min_rigid_coef, 1 - i / scheduler_step) * coef_arap_drag * loss_arap
            )

            loss.backward(retain_graph=True)
            anchor_optimizer.step()
            scheduler.step()
            anchor_optimizer.zero_grad()

            with torch.no_grad():
                image_source, image_target = self.fetch_comparable_two_image(return_rgba=True)
            
            if self.cfg.simple_video:
                w_from, h_from, w_to, h_to = bbox
                image = image_source[:, h_from-5:h_to+5, w_from-10:w_to].squeeze()
            else:
                # crop
                # w_from, h_from, w_to, h_to = bbox
                # image_source = image_source[:, h_from-30:h_to+30, w_from-45:w_to+15]
                # image_target = image_target[:, h_from-30:h_to+30, w_from-45:w_to+15]
                image_target[..., :3] = image_target[..., :3] + (1 - image_target[..., 3:])
                # src and tgt
                image = torch.cat(
                    [image_source.squeeze(), image_target.squeeze()], dim=1
                )
            image_list.append(image.cpu())


        stride = max(len(image_list), 75) // 75
        if self.cfg.simple_video:
            output_path = f"/data2/wlsgur4011/GESI/output_video/{self.cfg.data_name}_simple/{self.cfg.video_name}_{idx}.mp4"
            
            image_list = torch.stack(image_list, dim=0)
            
            image_list_path = f"/data2/wlsgur4011/GESI/output_overlay_img/{self.cfg.data_name}/{self.cfg.video_name}.pt"
            torch.save(image_list, image_list_path)

            img_path = f"/data2/wlsgur4011/GESI/output_overlay_img/{self.cfg.data_name}/{self.cfg.video_name}.png"
            os.makedirs(os.path.dirname(img_path), exist_ok=True)

            indice = [0, int(len(image_list) * 0.2), int(len(image_list) * 0.35), -1]
            save_motion_img(image_list[indice], img_path)

            img_path = f"/data2/wlsgur4011/GESI/output_overlay_img/{self.cfg.data_name}/{self.cfg.video_name}_low_alpha.png"
            indice = [0, int(len(image_list) * 0.2), -1]
            save_motion_img(image_list[indice], img_path, alpha_range=(0.2, 1))


        else:
            output_path = f"/data2/wlsgur4011/GESI/output_video/{self.cfg.data_name}/{self.cfg.video_name}_{idx}.mp4"
            # rewind
            image_list = image_list + [image_list[-1]] * 30 + image_list[::-1]
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        save_video(image_list[::stride], output_path, fps=30)

    def project_to_2d(self, points, use_gt_pose=False):
        if not hasattr(self, "data"):
            trainloader = DataLoader(self.trainset, batch_size=1)
            trainloader_iter = iter(trainloader)
            self.data = next(trainloader_iter)
        data = self.data
        device = self.device
        # camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]

        if use_gt_pose:
            camtoworlds = data["camtoworld"].to(device)
        else:
            camtoworlds = self.camtoworlds_pred
        
        Ks = data["K"].to(device)  # [1, 3, 3]
        means2d, depth = project_pointcloud_to_2d(points, camtoworlds, Ks)
        return means2d, depth

    def fectch_query_image(self):
        device = self.device
        
        if not hasattr(self, "data"):
            trainloader = DataLoader(self.trainset, batch_size=1)
            trainloader_iter = iter(trainloader)
            data = next(trainloader_iter)
        gt_images = data["image"].to(device) / 255.0  # [1, H, W, 3]
        gt_alphas = data["alpha"].to(device) / 255.0  # [1, H, W, 1]
        
        gt_images = torch.concat([gt_images, gt_alphas], dim=-1)
        
        return gt_images
    
    def estimate_camtoworlds(self, image_target):
        trainset = Dataset(
            self.parser,
            split="train",
            patch_size=self.cfg.patch_size,
            load_depths=self.cfg.depth_loss,
            single_finetune=False,
            cam_idx=self.cfg.cam_idx,
        )
        trainloader = DataLoader(trainset, batch_size=1)
        device = self.device

        max_covered_path = -1
        for data in trainloader:
            Ks = data["K"].to(device)  # [1, 3, 3]
            image_ids = data["image_id"].to(device)
            gt_images = data["image"].to(device) / 255.0  # [1, H, W, 3]
            height, width = gt_images.shape[1:3]
            camtoworlds = data["camtoworld"].to(device)  # [1, 4, 4]

            renders, alphas, info = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=3,
                near_plane=self.cfg.near_plane,
                far_plane=self.cfg.far_plane,
                image_ids=image_ids,
                render_mode="RGB",
            )
            image_source = torch.concat([renders, alphas], dim=-1)
            
            img1 = rearrange(image_source[0], "h w c -> c h w")
            img2 = rearrange(image_target[0], "h w c -> c h w")

            assert img1.shape == img2.shape, f"img1: {img1.shape}, img2: {img2.shape}"
            try:
                drag_from, drag_to, bbox = get_drag_roma(img1, img2, cycle_threshold=self.hpara.cycle_threshold, device=device)
            except:
                try:
                    drag_from, drag_to, bbox = get_drag_roma(img1, img2, cycle_threshold=self.hpara.cycle_threshold, device=device)
                except:
                    continue
                    
            n_covered_patch = count_covered_patches(drag_to, patch_size=20)
            if n_covered_patch > max_covered_path:
                max_covered_path = n_covered_patch
                camtoworld_pred = camtoworlds

        return camtoworld_pred


    def fetch_target_image(self, return_rgba=False, return_shape="bhwc"):
        if not hasattr(self, "data"):
            trainloader = DataLoader(self.trainset, batch_size=1)
            trainloader_iter = iter(trainloader)
            self.data = next(trainloader_iter)
        data = self.data
        device = self.device

        target_images = data["image"].to(device) / 255.0  # [1, H, W, 3]
        target_alphas = data["alpha"].to(device) / 255.0  # [1, H, W, 1]

        if return_rgba:
            target_images = torch.concat([target_images, target_alphas], dim=-1)
        if return_shape == "bhwc":
            pass
        elif return_shape == "chw":
            target_images = target_images[0].permute(2, 0, 1)
        else:
            raise ValueError(f"Invalid shape: {return_shape}")

        return target_images

    
    def fetch_comparable_two_image(self, return_rgba=False, return_shape="bhwc", use_gt_pose=False):
        if not hasattr(self, "data"):
            trainloader = DataLoader(self.trainset, batch_size=1)
            trainloader_iter = iter(trainloader)
            self.data = next(trainloader_iter)

        data = self.data
        device = self.device

        Ks = data["K"].to(device)  # [1, 3, 3]
        gt_images = data["image"].to(device) / 255.0  # [1, H, W, 3]
        gt_alphas = data["alpha"].to(device) / 255.0  # [1, H, W, 1]
        image_ids = data["image_id"].to(device)
        masks = data["mask"].to(device) if "mask" in data else None  # [1, H, W]
        height, width = gt_images.shape[1:3]

        if use_gt_pose:
            camtoworlds = data["camtoworld"].to(device)
        else:
            camtoworlds = self.camtoworlds_pred
        
        renders, alphas, info = self.rasterize_splats(
            camtoworlds=camtoworlds,
            Ks=Ks,
            width=width,
            height=height,
            sh_degree=3,
            near_plane=self.cfg.near_plane,
            far_plane=self.cfg.far_plane,
            image_ids=image_ids,
            render_mode="RGB+ED" if self.cfg.depth_loss else "RGB",
            masks=masks,
        )

        if return_rgba:
            renders = torch.concat([renders, alphas], dim=-1)
            gt_images = torch.concat([gt_images, gt_alphas], dim=-1)

        if return_shape == "bhwc":
            pass
        elif return_shape == "chw":
            renders = renders[0].permute(2, 0, 1)
            gt_images = gt_images[0].permute(2, 0, 1)
        else:
            raise ValueError(f"Invalid shape: {return_shape}")

        return renders, gt_images

    def render_and_calc_rgb_loss(self):
        renders, gt_images = self.fetch_comparable_two_image()
        if renders.shape[-1] == 4:
            colors, depths = renders[..., 0:3], renders[..., 3:4]
        else:
            colors, depths = renders, None
        rgb_loss = F.l1_loss(colors, gt_images)

        return rgb_loss

    @torch.no_grad()
    def eval(self, step: int, stage: str = "val", is_pred_camtoworld=False):
        """Entry for evaluation."""
        print("Running evaluation...")
        cfg = self.cfg
        device = self.device
        world_rank = self.world_rank
        world_size = self.world_size

        valloader = DataLoader(
            self.valset, batch_size=1, shuffle=False, num_workers=1
        )
        ellipse_time = 0
        metrics = defaultdict(list)
        img_diff_list = []
        for i, data in enumerate(valloader):
            camtoworlds = data["camtoworld"].to(device)
            Ks = data["K"].to(device)
            img = data["image"].to(device) / 255.0
            alpha = data["alpha"].to(device) / 255.0
            pixels = torch.concat([img, alpha], dim=-1)
            
            masks = data["mask"].to(device) if "mask" in data else None
            height, width = pixels.shape[1:3]

            torch.cuda.synchronize()
            tic = time.time()
            if is_pred_camtoworld:
                camtoworlds_origin = self.data["camtoworld"].to(device)
                camtoworlds_residual = self.camtoworlds_pred @ camtoworlds_origin.inverse()
                camtoworlds = camtoworlds_residual @ camtoworlds

            colors, alpha, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                masks=masks,
            )  # [1, H, W, 3]
            colors = torch.concat([colors, alpha], dim=-1)
            torch.cuda.synchronize()
            ellipse_time += time.time() - tic

            colors = torch.clamp(colors, 0.0, 1.0)
            canvas_list = [pixels, colors]

            if world_rank == 0:
                # write images
                canvas = torch.cat(canvas_list, dim=2).squeeze(0).cpu().numpy()
                canvas = (canvas * 255).astype(np.uint8)
                imageio.imwrite(
                    f"{self.render_dir}/{stage}_step{step}_{i:04d}.png",
                    canvas,
                )

                pixels_p = pixels.permute(0, 3, 1, 2)  # [1, 4, H, W]
                colors_p = colors.permute(0, 3, 1, 2)  # [1, 4, H, W]
                try:
                    _, img1, img2 = crop_two_image_with_alpha(colors_p[0], pixels_p[0])
                    img1 = img1[:, ::2, ::2]
                    img2 = img2[:, ::2, ::2]
                except:
                    continue
                img_diff_list.append(get_img_diff(img1, img2))
                colors_p = colors_p[:, :3]
                pixels_p = pixels_p[:, :3]
                metrics["psnr"].append(self.psnr(colors_p, pixels_p))
                metrics["ssim"].append(self.ssim(colors_p, pixels_p))
                metrics["lpips"].append(self.lpips(colors_p, pixels_p))
                if cfg.use_bilateral_grid:
                    cc_colors = color_correct(colors, pixels)
                    cc_colors_p = cc_colors.permute(0, 3, 1, 2)  # [1, 3, H, W]
                    metrics["cc_psnr"].append(self.psnr(cc_colors_p, pixels_p))

        if world_rank == 0:
            ellipse_time /= len(valloader)

            stats = {k: torch.stack(v).mean().item() for k, v in metrics.items()}
            stats.update(
                {
                    "ellipse_time": ellipse_time,
                    "num_GS": len(self.splats["means"]),
                }
            )
            from jhutil import color_log; color_log(0000, f"PSNR: {stats['psnr']:.3f}, SSIM: {stats['ssim']:.4f}, LPIPS: {stats['lpips']:.3f} ")
            print(
                f"Time: {stats['ellipse_time']:.3f}s/image "
                f"Number of GS: {stats['num_GS']}"
            )
            if wandb.run and not cfg.wandb_sweep:
                img_diffs = [
                    wandb.Image(img_diff_list[i], caption="00") for i in range(0, len(img_diff_list), 3)
                ]
                logging_data = {
                    "psnr": stats["psnr"],
                    "ssim": stats["ssim"],
                    "lpips": stats["lpips"],
                    "img_diff": img_diffs,
                }
                wandb.log(logging_data, step=step)
                
            if cfg.wandb_sweep:
                all_psnr_for_sweep[cfg.object_name] = stats["psnr"]
                
            # save stats as json
            with open(f"{self.stats_dir}/{stage}_step{step:04d}.json", "w") as f:
                json.dump(stats, f)
            # save stats to tensorboard
            for k, v in stats.items():
                self.writer.add_scalar(f"{stage}/{k}", v, step)
            self.writer.flush()

        if not hasattr(self, "max_psnr"):
            self.max_psnr = 0

        if not self.cfg.single_finetune and stats["psnr"] > self.max_psnr:
            self.max_psnr = stats["psnr"]
            splats = self.splats.state_dict()
            eps = 0.05 if self.cfg.data_name == "diva360" else 0.015
            splats = cluster_largest(splats, eps=eps)
            ckpt = {"step": step, "splats": splats, "clustered": True}
            torch.save(ckpt, f"{self.ckpt_dir}/ckpt_best_psnr.pt")
            print("Save best checkpoint to ", f"{self.ckpt_dir}/ckpt_best_psnr.pt")
            
        torch.cuda.empty_cache()
        
    def update_sh_with_group_id(self, group_id_all, only_idx=-1):
        from gesi.torch_splat import rgb_to_sh
        from jhutil import PALATTE

        sh0 = torch.zeros_like(self.splats["sh0"])
        # sh0 = rgb_to_sh()
        for i in range(0, group_id_all.max() + 1):
            group_id = group_id_all == i
            if group_id.sum() == 0:
                continue

            if only_idx != -1:
                color_idx = 0  # red
            else:
                color_idx = i

            rgb = torch.tensor(PALATTE[color_idx], device=self.device).float() / 255.0
            rgb = rgb.unsqueeze(0).repeat(group_id.sum(), 1)
            sh0[group_id] = rgb_to_sh(rgb)

        if only_idx != -1:
            with torch.no_grad():
                self.splats["opacities"][group_id_all != only_idx] = - 100

        sh0_origin = self.splats["sh0"].clone().detach()
        self.splats["sh0"] = sh0

        return sh0_origin
    

    @torch.no_grad()
    def render_traj(self, step, group_id_all=None, video_path="/data2/wlsgur4011/GESI/output_video_traj/output.mp4"):

        self.backgrounds = torch.ones(1, 3, device=self.device)   # white
        
        """Entry for trajectory rendering."""
        print("Running trajectory rendering...")
        cfg = self.cfg
        device = self.device

        if group_id_all is not None:
            sh0_origin = self.update_sh_with_group_id(group_id_all)

        camtoworlds_all = self.parser.camtoworlds
        if cfg.render_traj_path == "interp":
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 4
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "ellipse":
            height = camtoworlds_all[:, 2, 3].mean()
            camtoworlds_all = generate_ellipse_path_z(
                camtoworlds_all, height=height
            )  # [N, 3, 4]
        elif cfg.render_traj_path == "spiral":
            camtoworlds_all = generate_spiral_path(
                camtoworlds_all,
                bounds=self.parser.bounds * self.scene_scale,
                spiral_scale_r=self.parser.extconf["spiral_radius_scale"],
            )
        elif cfg.render_traj_path == "diva360_circle":
            object_name = cfg.object_name.split("_[")[0]
            json_path = f"/data2/wlsgur4011/GESI/gsplat/data/Diva360_data/processed_data/{object_name}/transforms_circle.json"
            camtoworlds_all = json_to_cam2world(json_path, self.parser.transform)
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]

        elif cfg.render_traj_path == "diva360_spiral":
            object_name = cfg.object_name.split("_[")[0]
            json_path = f"/data2/wlsgur4011/GESI/gsplat/data/Diva360_data/processed_data/{object_name}/transforms_spiral_hr.json"
            camtoworlds_all = json_to_cam2world(json_path, self.parser.transform)
            camtoworlds_all = generate_interpolated_path(
                camtoworlds_all, 1
            )  # [N, 3, 4]
            camtoworlds_all = camtoworlds_all[0:220]

        else:
            raise ValueError(
                f"Render trajectory type not supported: {cfg.render_traj_path}"
            )

        camtoworlds_all = np.concatenate(
            [
                camtoworlds_all,
                np.repeat(
                    np.array([[[0.0, 0.0, 0.0, 1.0]]]), len(camtoworlds_all), axis=0
                ),
            ],
            axis=1,
        )  # [N, 4, 4]

        camtoworlds_all = torch.from_numpy(camtoworlds_all).float().to(device)
        K = torch.from_numpy(list(self.parser.Ks_dict.values())[0]).float().to(device)
        width, height = list(self.parser.imsize_dict.values())[0]

        # video_dir from video_path
        video_dir = os.path.dirname(video_path)
        os.makedirs(video_dir, exist_ok=True)
        writer = imageio.get_writer(video_path, fps=30)
        for i in tqdm(range(len(camtoworlds_all)), desc="Rendering trajectory"):
            camtoworlds = camtoworlds_all[i : i + 1]
            Ks = K[None]

            renders, _, _ = self.rasterize_splats(
                camtoworlds=camtoworlds,
                Ks=Ks,
                width=width,
                height=height,
                sh_degree=cfg.sh_degree,
                near_plane=cfg.near_plane,
                far_plane=cfg.far_plane,
                render_mode="RGB+ED",
            )  # [1, H, W, 4]
            colors = torch.clamp(renders[..., 0:3], 0.0, 1.0)  # [1, H, W, 3]
            H, W = colors.shape[1:3]
            # colors = colors[:, :, W//4:3*W//4]  # crop by width

            canvas = colors.squeeze(0).cpu().numpy()
            canvas = (canvas * 255).astype(np.uint8)

            writer.append_data(canvas)
        writer.close()

        video_dir = os.path.abspath(video_dir)
        print(f"Video saved to {video_path}")

        if group_id_all is not None:
            self.splats["sh0"] = sh0_origin

    @torch.no_grad()
    def run_compression(self, step: int):
        """Entry for running compression."""
        print("Running compression...")
        world_rank = self.world_rank

        compress_dir = f"{cfg.result_dir}/compression/rank{world_rank}"
        os.makedirs(compress_dir, exist_ok=True)

        self.compression_method.compress(compress_dir, self.splats)

        # evaluate compression
        splats_c = self.compression_method.decompress(compress_dir)
        for k in splats_c.keys():
            self.splats[k].data = splats_c[k].to(self.device)
        self.eval(step=step, stage="compress")

    @torch.no_grad()
    def _viewer_render_fn(
        self, camera_state: nerfview.CameraState, img_wh: Tuple[int, int]
    ):
        """Callable function for the viewer."""
        W, H = img_wh
        c2w = camera_state.c2w
        K = camera_state.get_K(img_wh)
        c2w = torch.from_numpy(c2w).float().to(self.device)
        K = torch.from_numpy(K).float().to(self.device)

        render_colors, _, _ = self.rasterize_splats(
            camtoworlds=c2w[None],
            Ks=K[None],
            width=W,
            height=H,
            sh_degree=self.cfg.sh_degree,  # active all SH degrees
            radius_clip=3.0,  # skip GSs that have small image radius (in pixels)
            interactive=True,
        )  # [1, H, W, 3]
        return render_colors[0].cpu().numpy()


def main(local_rank: int, world_rank, world_size: int, cfg: Config):
    if world_size > 1 and not cfg.disable_viewer:
        cfg.disable_viewer = True
        if world_rank == 0:
            print("Viewer is disabled in distributed training.")

    runner = Runner(local_rank, world_rank, world_size, cfg)

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

            if cfg.finetune_with_only_rgb:
                runner.cfg.max_steps = 1001
                runner.train()
                runner.eval(step=1001)
            elif cfg.motion_video:
                runner.make_motion_video(idx=0, threhold_early_stop=5e-6, scheduler_step=500, min_rigid_coef=0)
            else:
                start_time = time.time()  # 시작 시간 기록
                runner.train_drag()
                end_time = time.time()    # 종료 시간 기록

                elapsed_time = end_time - start_time
                wandb.log({"drag_time": elapsed_time})
                
                if cfg.render_traj_simple:
                    if cfg.data_name == "diva360":
                        render_traj_path = "diva360_spiral"
                    elif cfg.data_name == "DFA":
                        render_traj_path = "interp"
                    runner.cfg.render_traj_path = render_traj_path
                    runner.render_traj(step=step, group_id_all=None, video_path=cfg.video_path)
        

        if cfg.render_traj_all:
            if cfg.data_name == "diva360":
                traj_path_list = ["diva360_spiral"]  # "interp", "ellipse", "spiral"
            elif cfg.data_name == "DFA":
                traj_path_list = ["interp"]  # "interp", "ellipse", "spiral"
            for render_traj_path in traj_path_list:
                runner.cfg.render_traj_path = render_traj_path

                # video_dir = f"/data2/wlsgur4011/GESI/output_video_traj/{cfg.data_name}"
                video_dir = cfg.result_dir
                os.makedirs(video_dir, exist_ok=True)
                video_paths = [
                    f"{video_dir}/traj_{cfg.object_name}_{cfg.render_traj_path}.mp4",
                    f"{video_dir}/traj_{cfg.object_name}_{cfg.render_traj_path}_init.mp4",
                    f"{video_dir}/traj_{cfg.object_name}_{cfg.render_traj_path}_final.mp4"
                ]
                try:
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
                
                    # concat three video by width 
                    video_path = f"{video_dir}/traj_{cfg.object_name}_{cfg.render_traj_path}_concat.mp4"
                    os.system(f'ffmpeg -y -i "{video_paths[0]}" -i "{video_paths[1]}" -i "{video_paths[2]}" -filter_complex hstack=inputs=3 "{video_path}"')
                    
                    for video_path in video_paths:
                        os.remove(video_path)
                except:
                    # group_id_all = group_ids[i]
                    # video_path = video_paths[i]
                    video_path = f"{video_dir}/traj_{cfg.object_name}.mp4"
                    runner.render_traj(step=step, group_id_all=None, video_path=video_path)


    else:
        runner.train()

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

