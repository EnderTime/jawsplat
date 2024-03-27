import torch
from .camera import get_viewport_matrix, get_projection_matrix
from diff_gaussian_rasterization import _C

def rasterize_gaussians(
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
):
    return _RasterizeGaussians.apply(
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

class _RasterizeGaussians(torch.autograd.Function):
    
    @staticmethod
    def forward(
        ctx,
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
    ):
        # P = number of gaussain points
        # k = rank of sh (d0 = k1, d1 = k4, d2 = k9, d3 = k16)

        viewport_matrix = get_viewport_matrix(cam_pos, cam_rot )
        projection_matrix = get_projection_matrix(near_far[0], near_far[1], tan_half_fov[0], tan_half_fov[1])
        args = (
            bg_color, 
            means3D,
            torch.Tensor([]),
            opacities,
            scales,
            rotations,
            1.0,
            torch.Tensor([]),
            viewport_matrix,
            projection_matrix,
            tan_half_fov[0],
            tan_half_fov[1],
            image_size[0],
            image_size[1],
            sh,
            sh_degree,
            cam_pos,
            False,
            False
        ) 

        # Invoke C++/CUDA rasterizer
        num_rendered, image_out, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.save_for_backward(
            # model
            means3D, 
            scales, 
            rotations, 
            sh, 
            geomBuffer, 
            binningBuffer, 
            imgBuffer,
            # render
            cam_pos, # (1, 3)
            tan_half_fov, # (1, 2)
            viewport_matrix,
            projection_matrix,
            # no-grad
            bg_color, #(1, 3)
            sh_degree, #(1, 1)
            # other
            num_rendered,
            radii, 
        )
        return image_out

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        # num_rendered = ctx.num_rendered
        # raster_settings = ctx.raster_settings
        # colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        (
            # model
            means3D, 
            scales, 
            rotations, 
            sh, 
            geomBuffer, 
            binningBuffer, 
            imgBuffer,
            # render
            cam_pos, # (1, 3)
            tan_half_fov, # (1, 2)
            viewport_matrix,
            projection_matrix,
            # no-grad
            bg_color, #(1, 3)
            sh_degree, #(1, 1)
            # other
            num_rendered,
            radii, 
        ) = ctx.saved_tensors
        # Restructure args as C++ method expects them
        args = (
            bg_color,
            means3D, 
            radii, 
            torch.Tensor([]), 
            scales, 
            rotations, 
            1.0, 
            torch.Tensor([]), 
            viewport_matrix, 
            projection_matrix, 
            tan_half_fov[0], 
            tan_half_fov[1], 
            grad_out_color, 
            sh, 
            sh_degree, 
            cam_pos,
            geomBuffer,
            num_rendered,
            binningBuffer,
            imgBuffer,
            False
        )

        # Compute gradients for relevant tensors by invoking backward method
        (
            grad_means2D, 
            grad_colors_precomp, 
            grad_opacities, 
            grad_means3D, 
            grad_cov3Ds_precomp, 
            grad_sh, 
            grad_scales, 
            grad_rotations
        ) = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads