import torch
from gsplat.project_gaussians import project_gaussians

to_tensor = lambda x : torch.tensor(x, dtype=torch.float).to(device='cuda')

scale_activate = lambda x : torch.exp(x)
rot_normalize = lambda x : x / x.norm(dim=-1, keepdim=True)

_xyz = [ -1.6114, -52.7745,  25.7178]
_scale = [-1.5645, -1.5645, -1.5645]
_rot = [0.3277, 0.1174, 0.6795, 0.6458]

_rot[0], _rot[1], _rot[2], _rot[3] = _rot[3], _rot[0], _rot[1], _rot[2]

# ===============

BLOCK_WIDTH = 16

W, H   = (1080, 1920)
fx, fy = (1406.5775627775, 1406.5775627775)
cx, cy = (W / 2, H / 2)
viewmat = to_tensor(
   [[ 9.9999e-01,  2.0965e-03, -3.9813e-03, -2.6973e+00],
    [-2.1137e-03,  9.9999e-01, -4.3284e-03,  7.8322e-01],
    [ 3.9722e-03,  4.3367e-03,  9.9998e-01, -3.6006e+00],
    [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]]
)
xyz   = to_tensor([_xyz])
scale = to_tensor([_scale])
rot   = to_tensor([_rot])

xys, depths, radii, conics, _, num_tiles_hit, _ = project_gaussians(
    means3d = xyz,
    scales = scale_activate(scale),
    glob_scale = 1,
    quats = rot_normalize(rot),
    viewmat = viewmat,
    fx = fx,
    fy = fy,
    cx = cx,
    cy = cy,
    img_width = W,
    img_height = H,
    block_width = BLOCK_WIDTH,
)

print(f"=> xys    = {xys}")
print(f"=> depths = {depths}")
print(f"=> radii  = {radii}")
print(f"=> conics = {conics}")
print(f"=> tiles  = {num_tiles_hit}")

# Returns:
# => xys    = tensor([[0., 0.]], device='cuda:0')
# => depths = tensor([0.], device='cuda:0')
# => radii  = tensor([0], device='cuda:0', dtype=torch.int32)
# => conics = tensor([[0.0352, 0.0016, 0.0055]], device='cuda:0')
# => tiles  = tensor([0], device='cuda:0', dtype=torch.int32)

# ==================

# R
rot = rot_normalize(torch.tensor(_rot))

r = rot[0]
x = rot[1]
y = rot[2]
z = rot[3]
# x = rot[0]
# y = rot[1]
# z = rot[2]
# r = rot[3]

R = torch.zeros((3, 3))
R[0, 0] = 1 - 2 * (y*y + z*z)
R[0, 1] = 2 * (x*y - r*z)
R[0, 2] = 2 * (x*z + r*y)
R[1, 0] = 2 * (x*y + r*z)
R[1, 1] = 1 - 2 * (x*x + z*z)
R[1, 2] = 2 * (y*z - r*x)
R[2, 0] = 2 * (x*z - r*y)
R[2, 1] = 2 * (y*z + r*x)
R[2, 2] = 1 - 2 * (x*x + y*y)

# S
scale = scale_activate(torch.tensor(_scale))

S = torch.zeros((3,3))
S[0,0] = scale[0]
S[1,1] = scale[1]
S[2,2] = scale[2]

# W
viewmat = viewmat.detach().cpu()
W = viewmat[:3,:3]

# project mean3D to 2D
xyz_w = torch.tensor(_xyz + [1])
intrinsics = torch.tensor(
    [[ fx,  0, cx],
     [  0, fy, cy],
     [  0,  0,  1]]
)
xyz_c = viewmat @ xyz_w
xyz_c = xyz_c[:3] / xyz_c[3]
xy = intrinsics @ xyz_c

# J
tx = xyz_c[0]
ty = xyz_c[1]
tz = xyz_c[2]

J = torch.zeros((3,3))
J[0,0] = fx / tz
J[1,1] = fy / tz
J[0,2] = -fx * tx / tz ** 2
J[1,2] = -fy * ty / tz ** 2

# project cov
Sigma = R @ S @ S.T @ R.T
Sigma_ = J @ W @ Sigma @ W.T @ J.T


print(f"============")
print(f"=> xyz_camera   = {xyz_c}")
print(f"=> xy_projected = {xy}")
print(f"=> conics = {Sigma_[:2,:2]}")

# Returns:
# => xyz_camera   = tensor([ -4.5217, -52.0987,  21.8814])
# => xy_projected = tensor([ 5.4558e+03, -5.2275e+04,  2.1881e+01])
# => conics = tensor([[ 188.5513,   88.9704],
#                     [  88.9705, 1205.9365]])