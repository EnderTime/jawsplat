import torch
import torch.nn as nn
from .utils.rasterization import rasterize_gaussians

class GaussianRasterizer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
            # model
            gaussians,
            # render
            camera,
            # no-grad
            image_size, # (1, 2)
            bg_color, #(1, 3)
            near_far, #(1, 2)
            sh_degree, #(1, 1)
        ):
        
        means3D = gaussians.means3D
        sh = gaussians.sh
        opacities = gaussians.opacities
        scales = gaussians.scales
        rotations = gaussians.rotations

        cam_pos = camera.pos
        cam_rot = camera.rot
        tan_half_fov = camera.tan_half_fov        

        # Invoke C++/CUDA rasterization routine
        return rasterize_gaussians(
            # model
            means3D, # (P, 3)
            sh, # (P, k)
            opacities, # (P, 1)
            scales, # (P, 3)
            rotations, # (P, 4)
            # render
            cam_pos, # (1, 3)
            cam_rot, # (1, 4)
            tan_half_fov, # (1, 2)
            # no-grad
            image_size, # (1, 2)
            bg_color, #(1, 3)
            near_far, #(1, 2)
            sh_degree, #(1, 1)
        )