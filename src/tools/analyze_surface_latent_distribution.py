import numpy as np
from pathlib import Path

from tqdm import tqdm

files = list(Path('../data/logan_jsons_latent_1126/abc').rglob('*.npz'))

template_data = np.load(files[0])

data_dict = {key:[] for key in template_data.keys()}

for file in tqdm(files):
    data = np.load(file)
    for key in data_dict.keys():
        data_dict[key].append(data[key])

for key in data_dict.keys():
    data_dict[key] = np.concatenate(data_dict[key], axis=0)


print(len(list(data_dict.values())[0]))