import os

import torch
from tqdm import tqdm

feature_dir_0 = 'video_features/official_slowfast'

feature_dir_1 = 'video_features/ego_vlp_reshape'

feature_dir_2 = 'video_features/official_omnivore'

output_dir = 'video_features/fusion'
os.makedirs(output_dir, exist_ok=True)

file_list = [item for item in os.listdir(feature_dir_1) if item.endswith('.pt')]

for clip in tqdm(file_list):
    vlp_clip = os.path.join(feature_dir_1, clip)
    sf_clip = os.path.join(feature_dir_0, clip)
    om_clip = os.path.join(feature_dir_2, clip)

    vlp_clip = torch.load(vlp_clip)
    sf_clip = torch.load(sf_clip)
    om_clip = torch.load(om_clip)

    output_clip = torch.concat([vlp_clip, om_clip, sf_clip], dim=1)
    out_path = os.path.join(output_dir, clip)

    torch.save(output_clip, out_path)

print('Down!')
