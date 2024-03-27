import os
import pandas as pd
import json
import numpy as np
import argparse
import struct
import collections
from model.utils.COLMAP_reader.read_model import read_model

def quaternion_to_rotation_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
    ])

workplace = r"resrc\fox"
base_path = workplace + r"\sparse\0"
image_path = workplace + r"\images"
output_dir = workplace + r"\points"

images,cameras,points = read_model(base_path)

point_cloud = points[['x', 'y', 'z']].values
point_cloud = point_cloud.T
point_cloud_color = points[['r', 'g', 'b']].values
point_cloud_color = point_cloud_color.T
print(point_cloud.shape)
print(point_cloud_color.shape)

data = []
idx = 0
length = len(images)
for name, image in images.items():
    idx += 1
    print(f"done {idx}/{length}")
    camera = cameras.loc[int(image['camera_id'])]
    # Extract quaternion and translation vector
    # qvec = np.array(image['qvec'])
    # tvec = np.array(image['tvec'])
    # Convert quaternion to rotation matrix
    # R = np.zeros((4, 4))
    # R[:3, :3] = quaternion_to_rotation_matrix(qvec)
    # R[:3, 3] = tvec
    # R[3, 3] = 1.0
    # T_pointcloud_camera = np.linalg.inv(R)
    K = camera['K']
    # Construct the JSON data
    image_full_path = os.path.join(image_path, name)
    data.append({
        'image_path': image_full_path,
        'pos': list(image['tvec']),
        'rot': list(image['qvec']),
        'tan_half_fov': [ camera['width'] / 2 / K[0,0], camera['height'] / 2 / K[1,1] ],
        # 'focal': [K[0,0], K[1,1]],
        # 'camera_intrinsics': K.tolist(),
        'camera_height': camera['height'],
        'camera_width': camera['width'],
        'camera_id': camera.name,
    })


df = pd.DataFrame(data)
# taking every 8th photo for test
df["is_train"] = df.index % 8 != 0
    
# test_images = [f"00{idx}.png" for idx in range(175, 250)]
# select training data and validation data, have a val every 3 frames
train_df = df[df["is_train"]].copy()
val_df = df[~df["is_train"]].copy()
print(train_df.shape)
print(val_df.shape)

train_df.drop(columns=["is_train"], inplace=True)
val_df.drop(columns=["is_train"], inplace=True)
train_df.to_json(os.path.join(output_dir, "train.json"), orient="records", indent=2)
val_df.to_json(os.path.join(output_dir, "val.json"), orient="records", indent=2)
points.to_parquet(os.path.join(output_dir, "point_cloud.parquet"))