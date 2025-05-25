import torch
from torch.utils.data import Dataset
import numpy as np
from einops import rearrange


from src.utils.helpers import (
    get_file_list, get_filename_wo_ext, get_file_list_with_extension, 
)
from src.dataset.dataset_fn import (
    scale_and_jitter_pc, scale_and_jitter_wireframe_set, curve_yz_scale,
    random_viewpoint, hidden_point_removal,
    aug_pc_by_idx,
    surface_scale_and_jitter,
    surface_rotate,
)

class LatentDataset(Dataset):
    def __init__(
        self, 
        dataset_file_path = None,
        is_train: bool = True,
        replication: int = 1,
        sample: bool = False,
        condition_on_points: bool = False,
        transform=scale_and_jitter_pc,
        use_partial_pc: bool = False,
        use_sampled_zs: bool = False,
        use_pc_noise: bool = False,
        condition_on_img: bool = False,
        k=48,
        pc_dir = '/root/pc_for_condition',
        img_dir = '/root/img_latents',
        point_num = 1024,
    ):
        super().__init__()
        self.is_train = is_train
        self.replica = replication
        self.data_path = dataset_file_path
        self.sample = sample
        self.condition_on_points = condition_on_points
        self.transform = transform
        self.use_partial_pc = use_partial_pc
        self.use_sampled_zs = use_sampled_zs
        self.use_pc_noise = use_pc_noise
        self.condition_on_img = condition_on_img
        self.pc_dir = pc_dir
        self.img_dir = img_dir
        self.point_num = point_num

        eval_mode = not is_train
        if sample or eval_mode:
            self.transform = None

        self.data = self._load_data(dataset_file_path)
        
        print(f'load {len(self.data)} data')
        
        self.samples_per_file = k
        
        if self.condition_on_img:
            self.total_samples = len(self.data) * self.samples_per_file
        else:
            self.total_samples = len(self.data) 
        
        

    def _load_data(self, data_path):
        file_list = get_file_list(data_path)
        
        return file_list

    def __len__(self):
        if self.is_train != True:
            return self.total_samples
        else:
            return self.total_samples * self.replica

    def __getitem__(self, idx):
        idx = idx % len(self.data)

        if self.condition_on_points:            
            
            wireframe_zs_file_path = self.data[idx]
            uid = get_filename_wo_ext(wireframe_zs_file_path)
            
            if self.is_train:
                uuid = uid.split("_")[0]
                aug_idx = uid.split("_")[1]
            else:
                uuid = uid.split("_")[0]
                aug_idx = 0
                
            pc_file_path = self.pc_dir + '/' + uuid + '.npy'
            pc = np.load(pc_file_path)

            if self.use_partial_pc:
                pc = pc[np.random.choice(len(pc), 2 * self.point_num, replace=False)]

                campos = random_viewpoint()
                points = pc[:, :3]  
                indices = hidden_point_removal(points, campos, only_return_indices=True)
                pc = pc[indices]
                            
            if len(pc) < self.point_num:
                pc = pc[np.random.choice(len(pc), self.point_num, replace=True)]
            else:
                pc = pc[np.random.choice(len(pc), self.point_num, replace=False)]

            pc = aug_pc_by_idx(pc, int(aug_idx))

            if self.transform is not None:
                if self.use_pc_noise:
                    noise_level = 0.02
                else:   
                    noise_level = 0.01
                
                pc = self.transform(pc, is_rotation=True, noise_level=noise_level)
            
            pc = torch.from_numpy(pc).to(torch.float32)
            context = pc
        elif self.condition_on_img:
            file_idx = idx // self.samples_per_file
            sample_idx = idx % self.samples_per_file

            wireframe_zs_file_path = self.data[file_idx]
            uid = get_filename_wo_ext(wireframe_zs_file_path)
            uuid = uid.split("_")[0]

            img_latent_file_path = self.img_dir + '/' + uuid + '/img_feature_dinov2_' + str(sample_idx) + '.npy'
            img_latent = np.load(img_latent_file_path)

            img_latent = torch.from_numpy(img_latent).to(torch.float32)
            context = img_latent
        else:
            wireframe_zs_file_path = self.data[idx]
            context = None

        mu_and_std = np.load(wireframe_zs_file_path)['zs']
        mu = mu_and_std[:,:16]
        std = mu_and_std[:,16:]
        zs = mu + std * np.random.randn(*std.shape)
            
        zs = torch.from_numpy(zs).to(torch.float32)

        if self.sample:
            uid = get_filename_wo_ext(wireframe_zs_file_path)
            return dict(zs=zs, context=context, uid=uid)

        return dict(zs=zs, context=context)

class WireframeDataset(Dataset):
    def __init__(
        self, 
        dataset_file_path = None,
        transform = scale_and_jitter_wireframe_set,
        is_train: bool = True,
        replication: float = 1.,
        sample: bool = False,
        max_num_lines: int = 128,
    ):
        super().__init__()
        self.transform = transform
        self.is_train = is_train
        self.replica = replication
        self.sample = sample
        self.max_num_lines = max_num_lines
        
        if self.sample:
            self.transform = None
        
        self.data = self._load_data(dataset_file_path)
        print(f'load {len(self.data)} valid data')
        
    def _load_data(self, data_path):
        file_list = get_file_list_with_extension(data_path, '.npz')
       
        return file_list

    def __len__(self):
        if self.is_train != True:
            return len(self.data)
        else:
            return int(len(self.data) * self.replica) 

    def __getitem__(self, idx):
        
        idx = idx % len(self.data)
        sample_path = self.data[idx]

        with np.load(sample_path) as sample:
            diffs = sample['diffs']
            segments = sample['segments']
            curve_latent = sample['curve_latent']
        
        num_lines = segments.shape[0]
        
        if self.transform is not None:
            segments = self.transform(segments)        
        
        segments = rearrange(segments, 'n v c -> n (v c)')
        xs = np.concatenate([segments, curve_latent], axis=1)

        padding_cols = self.max_num_lines - num_lines
        xs = np.pad(xs, ((0, padding_cols), (0, 0)), mode='constant', constant_values=0)
        diffs = np.pad(diffs, ((0, padding_cols), (0, 0)), mode='constant', constant_values=0)
        
        xs = torch.from_numpy(xs).to(torch.float32)
        diffs = torch.from_numpy(diffs).to(torch.long)        

        curveset = {
            'xs': xs,
            'flag_diffs': diffs, 
        }
        if self.sample:
            folder_name = sample_path.split('/')[-2]
            uid = get_filename_wo_ext(sample_path)
            curveset['uid'] = f'{folder_name}_{uid}'

        return curveset



class CurveDataset(Dataset):
    def __init__(
        self, 
        dataset_file_path = '',
        transform = curve_yz_scale,
        is_train: bool = True,
        replication: int = 1,
    ):
        super().__init__()
        self.transform = transform
        self.is_train = is_train
        self.replica = replication
        self.data_path = dataset_file_path
        
        self.data = self._load_data()

    def _load_data(self):
        data = np.load(self.data_path, allow_pickle=True)
        
        print(f'load {data.shape[0]} data')
        
        return data

    def __len__(self):
        if self.is_train:
            return len(self.data) * self.replica
        else:
            return len(self.data)

    def __getitem__(self, idx):
        idx = idx % len(self.data)
        vertices = self.data[idx]
        
        if self.transform is not None:
            vertices = self.transform(vertices)

        vertices = torch.from_numpy(vertices).to(torch.float32)

        return vertices

class SurfaceDataset(CurveDataset):
    def __init__(self, dataset_file_path, transform=surface_scale_and_jitter, is_train=True, replication=1, num_samples=32):
        super().__init__(dataset_file_path, transform, is_train, replication)
        self.num_samples = num_samples

    def __getitem__(self, idx):
        vertices = super().__getitem__(idx)
        if vertices.shape[0] > self.num_samples:
            assert vertices.shape[1] % self.num_samples == 0
            step = vertices.shape[1] // self.num_samples
            vertices = vertices[::step, ::step]
        elif vertices.shape[0] < self.num_samples:
            raise ValueError(f'vertices.shape[0] ({vertices.shape[0]}) < num_samples ({self.num_samples})')
        else:
            pass
        return vertices


