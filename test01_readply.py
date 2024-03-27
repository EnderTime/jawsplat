from model.utils.gaussians import GaussianModel

model = GaussianModel(3)
model.load_ply(r"resrc\fox\raw.ply")
print(type(model.get_xyz))
print(model.get_xyz.shape)
print(type(model.get_scaling))