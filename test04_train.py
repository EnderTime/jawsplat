from model.splat_model import SplatHyperParams, SplatModel
from model.utils.gaussians import GaussianModel
import torch
from pytorch_lightning import Trainer
from model.utils.dataloader import DataLoader


resolution = 8

hypers = SplatHyperParams()

hypers.random_background = False
hypers.background = torch.FloatTensor([ 0, 0, 0 ], device="cuda")
hypers.img_size = torch.IntTensor([ 1920/resolution, 1080/resolution ], device="cuda")
hypers.lambda_dssim = 0.01
hypers.near_far = torch.FloatTensor([ 0.01, 100 ], device="cuda")

gaussians = GaussianModel(3)
gaussians.load_ply(r"resrc\fox\raw.ply")
model = SplatModel(gaussians, hypers)

trainer = Trainer()
train_data = DataLoader(r"resrc\fox\points\train.json", 1000, 1/resolution)
val_data = DataLoader(r"resrc\fox\points\val.json", None, 1/resolution)
trainer.fit(model, train_data, val_data)