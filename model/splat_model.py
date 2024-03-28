from typing import Any, Mapping
from pytorch_lightning import LightningModule, Trainer
import torch
from torch import optim, nn, utils, Tensor
from .utils.rasterization import _RasterizeGaussians
from .utils.gaussians import GaussianModel
from .utils.gssplat.loss_utils import l1_loss, ssim
from .splat_render import GaussianRasterizer
from .utils.dataloader import DataLoader


class SplatHyperParams:
    random_background : bool
    background : Tensor # (1, 3)
    img_size :  Tensor # (1, 2)
    lambda_dssim : float
    near_far : Tensor # (1, 2)

class SplatModel(LightningModule):
    def __init__(self, gaussians : GaussianModel, hyper_params : SplatHyperParams):
        super().__init__()
        self.gaussians = gaussians
        self.h = hyper_params
        self.render = GaussianRasterizer()
    

    def training_step(self, gt_with_pose):
        iteration, gt_image, camera = gt_with_pose

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Render
        bg = torch.rand((3), device="cuda") if self.h.random_background else self.h.background

        image_out = self.render(
            self.gaussians, 
            camera,
            self.h.img_size,
            bg, #(1, 3)
            self.h.near_far, #(1, 2)
            self.gaussians.active_sh_degree, 
            )

        # Loss
        Ll1 = l1_loss(image_out, gt_image)
        loss = (1.0 - self.h.lambda_dssim) * Ll1 + self.h.lambda_dssim * (1.0 - ssim(image_out, gt_image))
        loss.backward()

        return loss
    
    def validation_step(self, gt_with_pose):
        iteration, gt_image, camera = gt_with_pose

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            self.gaussians.oneupSHdegree()

        # Render

        bg = torch.rand((3), device="cuda") if self.h.random_background else self.h.background

        image_out = self.render(
            self.gaussians, 
            camera,
            self.h.img_size,
            bg, #(1, 3)
            self.h, #(1, 2)
            self.gaussians.active_sh_degree, 
            )

        # Loss
        Ll1 = l1_loss(image_out, gt_image)
        loss = (1.0 - self.lambda_dssim) * Ll1 + self.lambda_dssim * (1.0 - ssim(image_out, gt_image))

        return loss

    def configure_optimizers(self):
        return self.gaussians.optimizer

