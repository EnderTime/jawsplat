from .camera import Camera
import pandas as pd
import torch
from PIL import Image
from .gssplat.general_utils import PILtoTorch

class DataLoader:
    def __init__(self, json_path, max_iter = None, resolution = 1.0) -> None:
        self.iteration = 0
        self.load_from_path(json_path)
        self.max_iter = max_iter
        self.resolution = resolution

    def load_from_path(self, json_path):
        data_str = open(json_path).read()
        self.df = pd.read_json(data_str, orient = 'records')
        self.img_n = self.df.shape[0]
        if self.max_iter is None:
            self.max_iter = self.img_n
        self.img_size = self.df.loc[0,['camera_height','camera_width']].to_list()
        self.df.drop(columns=["camera_width"], inplace=True)
        self.df.drop(columns=["camera_height"], inplace=True)




    def __iter__(self):
        return self
    
    def _get_idx(self,iteration):
        return iteration % self.img_n
    
    def __next__(self):
        self.iteration += 1
        if self.iteration > self.max_iter:
            raise StopIteration
        
        idx = self._get_idx(self.iteration)
        # print(type(self.df.loc[idx]))
        # print(self.df.loc[idx])
        # print(type(self.df.loc[idx,'rot']))
        # print(self.df.loc[idx,'rot'])
        img_info = self.df.loc[idx]
        img = Image.open(img_info['image_path'])
        img = PILtoTorch(img, self.resolution)
        camera = Camera(img_info['pos'], img_info['rot'], img_info['tan_half_fov'])

        return self.iteration, img, camera
