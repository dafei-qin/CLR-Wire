import torch
import os
import json
from typing import Dict, List, Iterator
from pathlib import Path
import numpy as np
import dataclasses
import trimesh

from sft.datasets.data_utils import  process_mesh
from sft.datasets.serializaitonDEEMOS import serialize

from utils.common import init_logger, import_module_or_data
from hy3dshape.surface_loaders import SharpEdgeSurfaceLoader
# from hy3dshape.models.autoencoders import ShapeVAE

import pymeshlab
import igl
import h5py
import random

logger = init_logger()
from torch.utils.data import Sampler

SYNSET_DICT_DIR = Path(__file__).resolve().parent  

class DynamicAttributes:
    def __init__(self):
        self._storage = {}

    def __setattr__(self, name, value):
        if name == '_storage':
            super().__setattr__(name, value)
        else:
            self._storage[name] = value

    def __getattr__(self, name):
        if name not in self._storage:
            new_obj = DynamicAttributes()
            self._storage[name] = new_obj
        return self._storage[name]
    
def sample_pc(verts, faces, pc_num, with_normal=False):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points
    points, face_idx = mesh.sample(50000, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points[:,[2,0,1]], normals[:,[2,0,1]]], axis=-1, dtype=np.float16)
    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    return pc_normal




def sample_pc_noTrans(verts, faces, pc_num, with_normal=False):
    mesh = trimesh.Trimesh(vertices=verts, faces=faces)
    if not with_normal:
        points, _ = mesh.sample(pc_num, return_index=True)
        return points
    points, face_idx = mesh.sample(pc_num, return_index=True)
    normals = mesh.face_normals[face_idx]
    pc_normal = np.concatenate([points, normals], axis=-1, dtype=np.float16)
    # random sample point cloud
    ind = np.random.choice(pc_normal.shape[0], pc_num, replace=False)
    pc_normal = pc_normal[ind]
    return pc_normal

def merge_uv_points(mesh): 
    uvs = mesh.visual.uv
    vertices = mesh.vertices
    faces = mesh.faces

    uvs_and_vertices = np.hstack((uvs, vertices))

    # Find and merge duplicate UV points and same vertices position
    unique_uvs, inverse_indices = np.unique(uvs_and_vertices, axis=0, return_inverse=True)
    merged_vertices = vertices[np.unique(inverse_indices, return_index=True)[1]]
    merged_faces = inverse_indices[faces]

    # Update the mesh with merged vertices and faces
    mesh.vertices = merged_vertices
    mesh.faces = merged_faces
    mesh.visual.uv = unique_uvs[:, :2]  # Keep only UV coordinates

    return mesh

def rescale_verts(verts):
    # Transpose so that z-axis is vertical.
    verts = verts[:, [2, 0, 1]]

    # Translate the vertices so that bounding box is centered at zero.
    center = (verts.max(0) + verts.min(0)) / 2
    verts = verts - center

    # Scale the vertices so that the long diagonal of the bounding box is equal
    # to one.
    scale = np.sqrt(np.sum((verts.max(0) - verts.min(0)) ** 2))
    verts = verts / scale

    return verts, center, scale

def normalize_mesh(mesh, scale=0.9999):
    """
    Normalize the mesh to fit inside a centered cube with a specified scale.

    The mesh is translated so that its bounding box center is at the origin,
    then uniformly scaled so that the longest side of the bounding box fits within [-scale, scale].

    Args:
        mesh (trimesh.Trimesh): Input mesh to normalize.
        scale (float, optional): Scaling factor to slightly shrink the mesh inside the unit cube. Default is 0.9999.

    Returns:
        trimesh.Trimesh: The normalized mesh with applied translation and scaling.
    """
    bbox = mesh.bounds
    center = (bbox[1] + bbox[0]) / 2
    scale_ = (bbox[1] - bbox[0]).max()

    mesh.apply_translation(-center)
    mesh.apply_scale(1 / scale_ * 2 * scale)

    return mesh

@dataclasses.dataclass
class MeshMeta:
    model_id: str
    raw_obj: str = None
    face_info: dict = None
    category: str = None
    category_path_en: str = None
    obj_path: str = None
    model_type: str = None
    face_cnt: int = None

class Sample_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        quant_bit: int     = 9,
        point_num:int      = 81920,
        path:str           = "/deemos-research-area-d/meshgen/code/Samba/data/overfit1",
        uid_list:list      = [],
        csv_path:str       = "/deemos-research-area-d/meshgen/code/DEEMOS/sft/datasets/glb_analysis_report.csv",
        use_uid:bool       = True,
        use_H5:bool        = False,
    ) -> None:
        super().__init__()
        self.quant_bit    = quant_bit
        self.point_num    = point_num
        self.path         = path
        self.csv_path     = csv_path
        self.useUID       = use_uid
        self.useH5        = use_H5
        self.enforce_same_folder_batches = True
        self.scaleup      = True
        # 读取CSV文件，过滤掉没有贴图数据的文件
        import csv
        self.csv_data = []
        self.valid_files = []  # 存储有效的文件信息
        self.fixtoken = None
        self.PCloader = SharpEdgeSurfaceLoader(
            num_sharp_points=0,
            num_uniform_points=81920,
        )

        from collections import Counter
        json_path1 = "/deemos-research-area-d/meshgen/code/DEEMOS_debug/available_files_final_label.json"
        json_path2 = "/deemos-research-area-d/meshgen/code/DEEMOS_debug/available_files_final_1027.json"
        
        if (os.path.exists(json_path1) or os.path.exists(json_path2)) and self.useH5 == False:
            # 读取两个JSON文件并合并
            all_json_data = []
            seen_paths = set()  # 用于去重
            
            # 读取第一个JSON文件
            if os.path.exists(json_path1):
                with open(json_path1, "r", encoding="utf-8") as f:
                    json_data1 = json.load(f)
                    for item in json_data1:
                        file_path = item.get('file_path', '')
                        file_path = file_path.replace("/deemos-research/meshgen", "/deemos-research-area-d/meshgen")
                        if file_path and file_path not in seen_paths:
                            seen_paths.add(file_path)
                            all_json_data.append(item)
            
            # 读取第二个JSON文件并去重
            if os.path.exists(json_path2):
                with open(json_path2, "r", encoding="utf-8") as f:
                    json_data2 = json.load(f)
                    for item in json_data2:
                        file_path = item.get('file_path', '')
                        file_path = file_path.replace("/deemos-research/meshgen", "/deemos-research-area-d/meshgen")
                        if file_path and file_path not in seen_paths:
                            seen_paths.add(file_path)
                            all_json_data.append(item)
            
            print(f"合并后去重前的数据量: {len(all_json_data)}")
            print(f"去重后的数据量: {len(seen_paths)}")
            
            label_counter = Counter()
            for item in all_json_data:
                    file_path = item['file_path']
                    file_path = file_path.replace("/deemos-research/meshgen", "/deemos-research-area-d/meshgen")
                    file_dir = item['file_dir']
                    file_dir = file_dir.replace("/deemos-research/meshgen", "/deemos-research-area-d/meshgen")
                    file_name = item['file_name']
                    base_name = item['base_name']
                    comp_num = item['num_connected_components']
                    face_num = item['num_faces']
                    vert_num = item['num_vertices']
                    label = item['label']
                    max_face = 16000  if self.scaleup else 6000

                    if face_num > max_face or face_num < 500 or vert_num > 16000:
                        continue
                    # if comp_num > 200:
                    #     continue
                    if float(vert_num) / float(face_num) > 1.0:
                        continue
                    
                    # keywords = ["ckpackageupload" ,"saic", "zeng", "amos", "toys", "cgtrader"]
                    # if not any(k in file_dir.lower() for k in keywords):
                    #     continue

                    
                    
                    def norm_label(x):
                        if x is None:
                            return "none"
                        return str(x).strip().lower()
                    
                    repeattimes = 1
                    times0 = [ "abstract design"]
                    times2 = [ "instrument", "electronics", "logo","none","book","container","accessories", "sports"]
                    times4 = ["shape","character", "furniture"]
                    times5 = [  ]
                    times7 = ["weapon"]
                    times10 = [ "food","vehicle" ]
                    times12 = ["human head","clothing"] 
                    L = norm_label(label)
                    filters = ["electronics","none","book","logo","accessories"]
                    if any(k in L for k in filters):
                        continue
                    if L in times10:
                        repeattimes = 3
                    elif L in times7:
                        repeattimes = 2
                    elif L in times12:
                        repeattimes = 4
                    elif L in times5:
                        repeattimes = 1
                    elif L in times4:
                        repeattimes = 1
                    elif L in times2:
                        repeattimes = 1
                    else:
                        repeattimes = 1

                    if L in times0:
                        repeattimes = 0
                        continue
                    
                    if face_num > 6000:
                        repeattimes = repeattimes * 4
                    for _ in range(repeattimes):
                        label_counter[L] += 1
                        # 直接添加到有效文件列表，因为JSON中已经过滤了有贴图数据的文件
                        self.valid_files.append({
                            'file_path': file_path,
                            'file_dir': file_dir,
                            'file_name': file_name,
                            'base_name': base_name,
                            'face_num': face_num,
                            'vert_num': vert_num,
                            'label': label
                        })
            print("Label 分布：")
            for label, count in label_counter.most_common():
                print(f"{label}: {count}")


            # output_path = "/deemos-research-area-d/meshgen/code/Samba/6000data.json"
            # with open(output_path, "w", encoding="utf-8") as f:
            #     json.dump(self.valid_files, f, indent=4, ensure_ascii=False)
            # print(f"✅ Saved {len(self.valid_files)} valid files to: {output_path}")

            # self.valid_files.sort(key=lambda x: int(x['face_num']))
            # chunk_size = len(self.valid_files) // 50
            # if chunk_size == 0:
            #     chunk_size = 1
            
            # # 创建片段列表
            # chunks = []
            # for i in range(0, len(self.valid_files), chunk_size):
            #     chunk = self.valid_files[i:i + chunk_size]
            #     chunks.append(chunk)
            
            # # 打乱片段顺序
            # random.seed(0)
            # random.shuffle(chunks)
            
            # # 重新组合
            # self.valid_files = []
            # for chunk in chunks:
            #     self.valid_files.extend(chunk)
            # print(f"Reading form json: {json_path}")
            # print(f"总数据条数: {len(self.csv_data)}")
            print(f"Selected data len: {len(self.valid_files)}")
        elif self.useH5 == True:
            self.h5_path = "/deemos-research-area-d/meshgen/H5Data_FullToken_small_DM"
            if self.useH5 and os.path.exists(self.h5_path):
                # Build mapping from base name -> folder name (start pos)
                self.uid_to_folder = {}
                self.folder_names = []
                self.uid_list = []
                try:
                    for entry in os.scandir(self.h5_path):
                        if entry.is_dir():
                            folder_name = os.path.basename(entry.path)
                            self.folder_names.append(folder_name)
                            # accept any file type inside; we only need the base name for grouping
                            for f in os.scandir(entry.path):
                                if f.is_file():
                                    base_name = os.path.splitext(os.path.basename(f.path))[0]
                                    if base_name[0] == '0':
                                    # if True:
                                        self.uid_to_folder[base_name] = folder_name
                                        self.uid_list.append(base_name)
                except Exception as e:
                    print(f"Failed scanning H5Data: {e}")
                print(f"dataset init,H5 dataset length: {len(self.uid_list)},folders: {len(self.folder_names)}")
                # Defer building folder_to_indices until uid_list is built below
        else:
            print("no valid files, use uid.")
        # self.valid_files = self.valid_files[:1000]
        # 使用CSV中的有效文件列表
       
        if self.useUID:
            name              = os.listdir(path)
            if uid_list == [] or uid_list == [""]:
                self.uid_list = [i for i in name if len(i.split("."))>1 and i.split(".")[-1] == "obj"]
            else:
                self.uid_list = uid_list
            print(f"dataset init,UID dataset length: {len(self.uid_list)}")
        else:
            print(f"dataset init,valid files dataset length: {len(self.valid_files)}")
        # print("dataset init, dataset length: ", len(self.uid_list))
    
        # Optional lazy cache for start -> indices grouping
        self._start_cache_ready = False
        self._start_value_by_index = {}
        self._start_to_indices = {}

    def __len__(self) -> int:
        if self.useUID or self.useH5:
            return len(self.uid_list) 
        else:
            return len(self.valid_files)

    def _preprocess_mesh(self, mesh):
        mesh = self._mesh_filter(mesh)
        
        # 归一化处理
        vertices = mesh.vertices
        bbmin, bbmax = vertices.min(0), vertices.max(0)
        center, scale = (bbmin + bbmax)*0.5, 2.0 * 0.9 / (bbmax - bbmin).max()
        mesh.vertices = (vertices - center) * scale
        
        # 确保三角形网格
        if mesh.faces.shape[1] == 4:
            mesh.faces = np.vstack([mesh.faces[:, :3], mesh.faces[:, [0,2,3]]])
        
        return mesh

    def _mesh_filter(self, mesh: trimesh.Trimesh) -> trimesh.Trimesh:
        """使用PyMeshLab进行网格预处理"""
        ml_mesh = pymeshlab.Mesh(vertex_matrix=mesh.vertices, face_matrix=mesh.faces)
        ms = pymeshlab.MeshSet()
        ms.add_mesh(ml_mesh, "raw_mesh")
            
        ms.apply_filter('meshing_remove_duplicate_faces')
        # ms.apply_filter('meshing_remove_duplicate_vertices')
        # ms.apply_filter('meshing_merge_close_vertices', 
                    # threshold=pymeshlab.PercentageValue(0.5))
        # ms.meshing_repair_non_manifold_edges(method=1)
        # ms.meshing_repair_non_manifold_edges(method=0)
        # ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        # ms.meshing_repair_non_manifold_edges(method=1)
        # ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        # ms.meshing_repair_non_manifold_edges(method=1)
        # ms.meshing_repair_non_manifold_vertices(vertdispratio=0)
        # ms.meshing_repair_non_manifold_edges(method=1)
        ms.apply_filter('meshing_remove_unreferenced_vertices')
            
        processed = ms.current_mesh()
        mesh.vertices = processed.vertex_matrix()
        mesh.faces = processed.face_matrix()
            
        return mesh


    def split_mesh_into_connected_components(self, mesh_vertices, mesh_faces):
        """
        将网格拆分为连通体。
        参数:
        mesh_path (str): 网格文件路径。
        返回:
        tuple: 连通体列表（vertices, faces）。
        """
        if (mesh_faces.shape[0] == 0 or mesh_vertices.shape[0] == 0):
            print(mesh_faces.shape)
            return None
        
        if mesh_faces.shape[0] > self.max_triangles:
            return [(mesh_vertices, mesh_faces)], mesh_vertices, mesh_faces
        
        adjacency_matrix = igl.adjacency_matrix(mesh_faces)
        num_components, components, _ = igl.connected_components(adjacency_matrix)
        component_faces = [[] for _ in range(num_components)]

        # 确保components是数组格式
        if hasattr(components, 'shape'):
            if components.shape[0] == 0:
                return None
        else:
            # 如果components是set或其他格式，转换为数组
            components = np.array(list(components)) if hasattr(components, '__iter__') else np.array([components])
            
        # 将每个面分配到对应的连通体
        for i in range(mesh_faces.shape[0]):
            face = mesh_faces[i]
            # 检查面上的所有顶点是否属于同一个连通体
            if components[face[0]] == components[face[1]] == components[face[2]]:
                component_faces[components[face[0]]].append(face)
                
        processed_components = []
        for faces in component_faces:
            if len(faces) > 0:
                faces = np.array(faces)
                unique_vertices = np.unique(faces)
                vertex_map = {old_idx: new_idx for new_idx, old_idx in enumerate(unique_vertices)}
                new_faces = np.vectorize(vertex_map.get)(faces)
                new_vertices = mesh_vertices[unique_vertices]

                data = DynamicAttributes()
                data.vertices = new_vertices
                data.faces = new_faces
                if len(new_faces) > 50:
                    processed_components.append(data)
                
        return processed_components
    
    def __getitem__(self, idx: int) -> Dict:
        # idx = 0
        # idx += 600000
        # print(idx)
        # Defensive: if idx is a list/tuple/ndarray (shouldn't be for __getitem__), pick first
        if isinstance(idx, (list, tuple, np.ndarray)):
            if not hasattr(self, '_warned_list_idx'):
                self._warned_list_idx = True
                print("Warning: __getitem__ received a list index; taking the first element. Ensure to use StartGroupBatchSampler as batch_sampler, not sampler.")
            idx = int(idx[0])

        if self.useH5:
            data = {}
            file_path = os.path.join(self.h5_path, f'{self.uid_to_folder[self.uid_list[idx]]}', f"{self.uid_list[idx]}.h5")
            # print(f"file_path: {file_path}")
            filename = self.uid_list[idx].split(".")[0]
            # read h5 file
            with h5py.File(file_path, 'r') as f:
                # data['pc'] = torch.tensor(f['pc'][()])
                # data['token_list_0'] = torch.tensor(f['token_list'][0], dtype=torch.long)
                # data['token_list_target'] = torch.tensor(f['token_list_target'][0], dtype=torch.long)
                # data['len_tk'] = len(f['token_list'][0])
                # derive start directly from folder name to avoid H5 reads
                try:
                    folder_name = self.uid_to_folder[self.uid_list[idx]]
                    start_val = int(folder_name)
                except Exception:
                    fn = self.uid_to_folder[self.uid_list[idx]]
                    digits = ''.join([c for c in fn if c.isdigit()])
                    start_val = int(digits) if digits else -1
                data['start'] = torch.tensor(start_val, dtype=torch.long)
                # Decode H5 bytes dataset to Python str
                try:
                    path_str = f['path'].asstr()[0]
                except AttributeError:
                    p0 = f['path'][0]
                    path_str = p0.decode('utf-8') if isinstance(p0, (bytes, np.bytes_)) else str(p0)
                data['path'] = path_str
                # print(path_str)
                try:
                    raw_obj = trimesh.load(path_str, force='mesh', process=False)
                    try:
                        uv_obj = merge_uv_points(raw_obj)
                        processed_mesh = uv_obj
                    except:
                        processed_mesh = raw_obj
                    mesh = process_mesh(processed_mesh.vertices, processed_mesh.faces, quantization_bits=8, augment=True)
                    indices      = np.random.choice(50000, 16384, replace=False)
                    pc_normal    = sample_pc_noTrans(mesh["vertices"], mesh["faces"], pc_num=50000, with_normal=True)[indices]
                    
                    token_list = serialize(trimesh.Trimesh(vertices=mesh["vertices"], faces=mesh["faces"], process=True))
                    token_list = np.concatenate([[4736], token_list, [4737]])
                    
                except:
                    print("Error...!!!!")
                    data['token_list_0'] = torch.ones_like(data['token_list_0']) * 4737
                    #  data['token_list_target'] = torch.ones_like(data['token_list_target']) * 4737
                    data['pc'] = torch.zeros((16384, 6))
                    #  data['start'] = torch.tensor(-1, dtype=torch.long)
                    data['name'] = filename
                    return data
                data['token_list_0'] = torch.tensor(token_list, dtype=torch.long)
                data['len_tk'] = len(token_list)
                data['pc'] = torch.tensor(pc_normal)
                data['name'] = filename
                data['filename'] = filename
                data['filepath'] = path_str
            return data

        # 使用CSV中的文件信息
        useUID = self.useUID
        if useUID:
            # import h5py
            # h5_path = "/deemos-research-area-d/meshgen/code/Samba/h5/cowboycopy18.h5"
            # try:
            #     with h5py.File(h5_path, "r") as f:
            #         pc_from_h5 = f["pc"][:]
            #         token_list_0_from_h5 = f["token_list_0"][:]
            #     # print("成功从h5文件读取'pc'和'token_list_0'")
            # except Exception as e:
            #     print(f"读取h5文件失败: {h5_path}, 错误: {e}")
            #     pc_from_h5 = None
            #     token_list_0_from_h5 = None

            # # 检查数据是否成功加载
            # if pc_from_h5 is None or token_list_0_from_h5 is None:
            #     print(f"警告: h5文件读取失败，尝试使用备用方法获取数据 (idx={idx})")
            #     # 如果h5读取失败，递归调用获取下一个样本
            #     return self.__getitem__((idx + 1) % self.__len__())

            # if len(token_list_0_from_h5) > 9000:
            #     # 随机选取起始位置：从0到len(token_list) - 15000
            #     max_start = max(0, len(token_list_0_from_h5) - 5000)
            #     start_idx = np.random.randint(0, max_start + 1)
            #     # 从start_idx开始截取剩余部分
            #     remaining_length = len(token_list_0_from_h5) - start_idx
            #     if remaining_length >= 9000:
            #         # 如果剩余长度足够，截取20000个
            #         token_list_0_from_h5 = token_list_0_from_h5[start_idx:start_idx + 9000]
            #     else:
            #         # 如果剩余长度不足20000，直接截取剩余部分
            #         token_list_0_from_h5 = token_list_0_from_h5[start_idx:]
            
            # data = {}
            # # print(pc_from_h5.shape, token_list_0_from_h5.shape)
            # data['pc'] = torch.tensor(pc_from_h5)
            # data['token_list_0'] = torch.tensor(token_list_0_from_h5, dtype=torch.long)
            # data['len_tk'] = len(token_list_0_from_h5)

            # return data




            file_path = os.path.join(self.path, f"{self.uid_list[idx]}")
            filename = self.uid_list[idx].split(".")[0]
            # print(file_path)
            try:
                raw_obj = trimesh.load(file_path, force='mesh', file_type='obj',process=False)
            except Exception as e:
                print(f"无法加载GLB文件 {file_path}: {e}")

            
        
        if useUID == False and hasattr(self, 'valid_files') and len(self.valid_files) > 0:
            file_info = self.valid_files[idx]
            file_path = file_info['file_path']
            filename = file_info['base_name']
            # print(file_path)
            # 尝试加载GLB文件
            if os.path.isfile(file_path):
                try:
                    raw_obj = trimesh.load(file_path, force='mesh', file_type='glb', process=False)
                    raw_obj = merge_uv_points(raw_obj)

                    # if raw_obj.vertices.shape[0] >= raw_obj.faces.shape[0]:
                    #     print(f"vertices.shape[0] >= faces.shape[0], {raw_obj.vertices.shape[0]} >= {raw_obj.faces.shape[0]}, {file_path}")
                    #     return self.__getitem__(idx+1 % self.__len__())
                except: 
                    print("load error or mergeuv error")
                    return self.__getitem__(idx+1 % self.__len__())
            else:
                # print(f"无法加载GLB文件 {file_path}")
                # print("no valid files, get item again")
                return self.__getitem__(idx+1 % self.__len__())
                    # raise FileNotFoundError(f"找不到文件: {file_path} 或 {obj_path}")
        else:
            if useUID == False:
                print("no valid files, get item again")
                return self.__getitem__(idx+1 % self.__len__())

        # 将raw_obj转为obj格式并重新读取，但不写入本地文件
        
        data = {}
        try:
            # processed_mesh = self._preprocess_mesh(raw_obj)
            processed_mesh = raw_obj
            
                
            raw_obj = trimesh.Trimesh(processed_mesh.vertices, processed_mesh.faces, process=False)
            if useUID:
                mesh = process_mesh(raw_obj.vertices, raw_obj.faces, quantization_bits=8, quan=False, augment=False)
            else:
                mesh = process_mesh(raw_obj.vertices, raw_obj.faces, quantization_bits=8, quan=True, augment=True)
            # verts, faces = mesh['vertices'], mesh['faces']
            # mesh = normalize_mesh(processed_mesh)

            # indices      = np.random.choice(50000, 16384, replace=False)
            # pc_normal    = sample_pc_noTrans(mesh["vertices"], mesh["faces"], pc_num=50000, with_normal=True)[indices]

            pointcloud = self.PCloader(trimesh.Trimesh(mesh["vertices"], mesh["faces"],process=False)).squeeze(0)
            # print(pointcloud.shape)
            if pointcloud.shape[0] != 81920 or pointcloud.shape[1] != 7:
                print("Sample error.")
                return self.__getitem__(idx+1 % self.__len__())
        # mesh = trimesh.Trimesh(vertices=verts, faces=faces)
        # faces = mesh.faces
        # face_mask = np.ones(faces.shape[0], dtype=bool)
        # for f in range(faces.shape[0]):
        #     if faces[f, 0] == faces[f, 1] or faces[f, 1] == faces[f, 2] or faces[f, 0] == faces[f, 2]:
        #         face_mask[f] = False
        # faces = faces[face_mask]
        
        # mesh = trimesh.Trimesh(vertices=verts, faces=faces)

        # mesh = self._mesh_filter(mesh)
        # print(verts.shape, faces.shape)
        
        # points = sample_pc(verts, faces, pc_num=self.point_num, with_normal=True)
        
            
            # raw_obj = mesh.process(True)
            # raw_obj = mesh
            # obj_bytes = raw_obj.export(file_type='obj')
            # obj_buffer = BytesIO(obj_bytes.encode('utf-8') if isinstance(obj_bytes, str) else obj_bytes)
            # raw_obj = trimesh.load(obj_buffer, file_type='obj')
            
            # INSERT_YOUR_CODE
            # 从指定的h5文件读取'pc'和'token_list_0'
            
            
            data['pc'] = torch.tensor(pointcloud)

            # raw_obj.export(f"/deemos-research-area-d/meshgen/code/DEEMOS/DataDeToken/{filename}.obj")
        except:
            print("Fail to Quan and sample.")
            return self.__getitem__((idx+1) % self.__len__())
        # print(pointcloud.shape)

        
        try:
            if not useUID:
            # if True:
                token_list = serialize(trimesh.Trimesh(mesh["vertices"], mesh["faces"],process=False))
                token_list = np.concatenate([[4736], token_list, [4737]])
                
                # 如果长度超过20000，从完整序列中随机截取一段长度为20000的连续序列
                if len(token_list) > 9001:
                    # 随机选取起始位置：从0到len(token_list) - 15000
                    max_start = max(0, len(token_list) - 5000)
                    start_idx = np.random.randint(0, max_start + 1)
                    # 从start_idx开始截取剩余部分
                    remaining_length = len(token_list) - start_idx
                    if remaining_length >= 9001:
                        # 如果剩余长度足够，截取20000个
                        token_list = token_list[start_idx:start_idx + 9001]
                    else:
                        # 如果剩余长度不足20000，直接截取剩余部分
                        token_list = token_list[start_idx:]
                
                data['token_list_0'] = torch.tensor(token_list, dtype=torch.long)
                data['len_tk'] = len(token_list)
            # INSERT_YOUR_CODE
            # 将 'pc' 和 'token_list_0' 保存到本地 h5 文件

            # import os

            # save_dir = "/deemos-research-area-d/meshgen/code/Samba/h5"
            # os.makedirs(save_dir, exist_ok=True)
            # h5_path = os.path.join(save_dir, f"{filename}.h5")
            # try:
            #     with h5py.File(h5_path, "w") as f:
            #         f.create_dataset("pc", data=data['pc'])
            #         f.create_dataset("token_list_0", data=token_list)

            #     # break
            # except Exception as e:
            #     print(f"保存到h5文件失败: {h5_path}, 错误: {e}")



            data['path'] = file_path
            # if not useUID and (len(token_list) < 500 or len(token_list) > 20000):
            #     print(f"token too long: {len(token_list)}. ")
            #     return self.__getitem__(idx+1 % self.__len__())
        except:
            print("tokenizer fail.")
            return self.__getitem__(idx+1 % self.__len__())

        
        data['name'] = filename
        data['filename'] = filename
        data['filepath'] = file_path
        
        return data

    # --- Lazy start cache utilities ---
    def _ensure_start_cache(self):
        if self._start_cache_ready:
            return
        self._start_value_by_index = {}
        self._start_to_indices = {}
        for i, uid in enumerate(self.uid_list):
            # derive start from folder name; no H5 reads
            try:
                folder_name = self.uid_to_folder[uid]
                start_val = int(folder_name)
            except Exception:
                digits = ''.join([c for c in self.uid_to_folder.get(uid, '') if c.isdigit()])
                start_val = int(digits) if digits else -1
            self._start_value_by_index[i] = start_val
            if start_val not in self._start_to_indices:
                self._start_to_indices[start_val] = []
            self._start_to_indices[start_val].append(i)
        self._start_cache_ready = True

    def get_start_for_index(self, idx: int) -> int:
        self._ensure_start_cache()
        return self._start_value_by_index.get(idx, -1)


class StartGroupBatchSampler(Sampler[List[int]]):
    def __init__(self, dataset: Sample_Dataset, batch_size: int, drop_last: bool = False, shuffle: bool = True, num_replicas: int = 1, rank: int = 0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        
        # 准备数据
        self.dataset._ensure_start_cache()
        self._prepare_iterations()

    def set_epoch(self, epoch: int):
        self.epoch = int(epoch)
        self._prepare_iterations()

    def _prepare_iterations(self):
        """准备迭代数据：扫描全部数据，计算每次迭代需要的数据量，划分文件夹数据"""
        # 计算每次迭代需要的数据量
        self.data_per_iteration = self.batch_size * self.num_replicas
        print(f"Data per iteration: {self.data_per_iteration} (batch_size={self.batch_size} * num_replicas={self.num_replicas})")
        # 获取所有start文件夹的数据
        original_groups = self.dataset._start_to_indices
        
        # 存储所有迭代信息：(start_val, iteration_idx, data_indices)
        self.all_iterations = []
        
        # 确保跨进程/卡一致：按start键排序遍历，避免dict遍历顺序不一致
        for start_val in sorted(original_groups.keys()):
            all_indices = original_groups[start_val]
            if len(all_indices) == 0:
                continue
                
            # 对当前start文件夹的数据进行shuffle
            if self.shuffle:
                rng = np.random.default_rng(self.epoch + start_val)
                shuffled_indices = all_indices[:]
                rng.shuffle(shuffled_indices)
            else:
                shuffled_indices = all_indices[:]
            
           
            
            # 如果数据不能整除一次迭代大小，则在同文件夹内重复填充，使其可整除
            if (len(shuffled_indices) % self.data_per_iteration) != 0:
                if len(shuffled_indices) == 0:
                    continue
                # 重复填充到足够一次迭代
                needed = self.data_per_iteration - (len(shuffled_indices) % self.data_per_iteration)
                if self.shuffle:
                    rng = np.random.default_rng(self.epoch + start_val + 1000)
                    extra_indices = rng.choice(shuffled_indices, size=needed, replace=True).tolist()
                else:
                    extra_indices = (shuffled_indices * ((needed // len(shuffled_indices)) + 1))[:needed]
                shuffled_indices = shuffled_indices + extra_indices
            # 计算这个文件夹可以分成多少份（每次迭代的数据量）
            num_iterations = len(shuffled_indices) // self.data_per_iteration
            
            # 为这个文件夹创建所有迭代
            for iteration_idx in range(num_iterations):
                start_idx = iteration_idx * self.data_per_iteration
                end_idx = start_idx + self.data_per_iteration
                iteration_data = shuffled_indices[start_idx:end_idx]
                
                # 为当前rank提取数据（统一shuffle后再按rank步长切分，步长=num_replicas）
                # 每个迭代块大小为 batch_size*num_replicas，切分后每rank恰好 batch_size 个样本
                rank_data = iteration_data[self.rank::self.num_replicas]
                
                if len(rank_data) > 0:
                    self.all_iterations.append((start_val, iteration_idx, rank_data))
        
        # # 打乱所有迭代的顺序
        if self.shuffle:
            rng = np.random.default_rng(self.epoch)
            rng.shuffle(self.all_iterations)
        
        # Debug info
        print(f"Rank {self.rank}: Prepared {len(self.all_iterations)} iterations")
        if self.rank == 0:
            total_samples = sum(len(indices) for indices in original_groups.values())
            print(f"Total dataset size: {total_samples} samples across {len(original_groups)} start groups")
            print(f"Data per iteration: {self.data_per_iteration} (batch_size={self.batch_size} * num_replicas={self.num_replicas})")

    def __iter__(self) -> Iterator[List[int]]:
        """按准备好的迭代顺序生成batch"""
        for start_val, iteration_idx, rank_data in self.all_iterations:
            # 为当前迭代的数据创建batch
            for i in range(0, len(rank_data), self.batch_size):
                batch = rank_data[i:i + self.batch_size]
                if len(batch) < self.batch_size:
                    print("Error: should not happen")
                    if self.drop_last:
                        continue
                    if len(rank_data) == 0:
                        continue
                    need = self.batch_size - len(batch)
                    if self.shuffle:
                        rng = np.random.default_rng(self.epoch + start_val + 3000 + iteration_idx)
                        extra = rng.choice(rank_data, size=need, replace=True).tolist()
                    else:
                        extra = (rank_data * ((need // len(rank_data)) + 1))[:need]
                    batch = list(batch) + list(extra)
                
                if len(batch) == 0:
                    continue
                yield batch

    def __len__(self) -> int:
        """返回当前rank的batch数量"""
        total_batches = 0
        for start_val, iteration_idx, rank_data in self.all_iterations:
            n = len(rank_data) // self.batch_size
            if not self.drop_last and (len(rank_data) % self.batch_size) != 0:
                print("Error: should not happen")
                n += 1
            total_batches += n
        return total_batches
