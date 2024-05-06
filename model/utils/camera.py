import torch
import numpy as np

def get_viewport_matrix(R, t, translate=None, scale=1.0):
    # translate=np.array([.0, .0, .0]), scale=1.0
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    if translate != None:
        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def get_projection_matrix(znear, zfar, tanHalfFovX, tanHalfFovY):

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

class Camera:
    def __init__(self,
        cam_pos,
        cam_rot,
        tan_half_fov
    ) -> None:
        self._pos = torch.tensor(cam_pos, dtype = torch.float, device="cuda")
        self._rot = torch.tensor(cam_rot, dtype = torch.float, device="cuda")
        self._tan_half_fov = torch.tensor(tan_half_fov, dtype = torch.float, device="cuda")   
    
    @property
    def pos(self):
        return self._pos

    @property
    def rot(self):
        return self._rot
    
    @property
    def tan_half_fov(self):
        return self._tan_half_fov
