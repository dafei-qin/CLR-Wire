import importlib
import os
from typing import Any, Dict, Tuple, Union
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch


def _get_obj_from_str(string: str, reload: bool = False) -> Any:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def _to_dict(cfg: Any) -> Dict[str, Any]:
    # Support OmegaConf / DictConfig or plain dict
    if hasattr(cfg, "to_container"):
        return cfg.to_container(resolve=True)
    return dict(cfg)


def _clean_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Strip common prefixes like 'module.' or 'model.'
    def strip_prefix(k: str) -> str:
        for p in ("module.", "model."):
            if k.startswith(p):
                return k[len(p) :]
        return k

    return {strip_prefix(k): v for k, v in state_dict.items()}


def load_model_from_config(
    cfg: Union[Dict[str, Any], Any],
    device: Union[str, torch.device, None] = None,
    strict: bool = True,
    section: str = 'model',
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Instantiate model from config and load checkpoint.

    Expected config structure (YAML):
    model:
        name: src.vae.vae_v1.SurfaceVAE   # full module path to class
        params:
            param_raw_dim: [17, 18, 19, 18, 19]
            latent_dim: 128
            ...
        checkpoint_folder: path/to/folder
        checkpoint_file_name: model-xx.pt
    """
    cfg_dict = _to_dict(cfg)
    model_cfg = cfg_dict.get(section, {})

    target = model_cfg.get("name", None)
    if target is None or "." not in target:
        raise ValueError(f"model.name must be a full module path like 'src.vae.vae_v1.SurfaceVAE', got: {target}")

    cls = _get_obj_from_str(target)

    params = dict(model_cfg.get("params", {}))
    # Also allow top-level fields in model_cfg (e.g., param_raw_dim) without nesting under params
    # for k, v in model_cfg.items():
    #     if k not in {"name", "checkpoint_folder", "checkpoint_file_name", "target", "params"}:
    #         params.setdefault(k, v)

    model = cls(**params)
    print(f'Init model {target } with class: ', cls.__name__)

    if device is not None:
        model.to(device)

    ckpt_folder = model_cfg.get("checkpoint_folder", None)
    ckpt_name = model_cfg.get("checkpoint_file_name", None)
    if ckpt_folder is None or ckpt_name is None:
        return model
    if model_cfg.get('load_ckpt', True) == False:
        return model
    ckpt_path = os.path.join(ckpt_folder, ckpt_name)
    ckpt = torch.load(ckpt_path, map_location=device or "cpu", weights_only=True)
    print('Loading checkpoint from: ', ckpt_path)
    if isinstance(ckpt, dict) and any(k in ckpt for k in ("state_dict", "model")):
        state_dict = ckpt.get("state_dict", ckpt.get("model"))
    else:
        state_dict = ckpt

    state_dict = _clean_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=strict)

    # return model, ckpt if isinstance(ckpt, dict) else {}
    return model


def load_dataset_from_config(
    cfg: Union[Dict[str, Any], Any],
    section: str = "data",
    **extra_kwargs,
):
    """
    Instantiate dataset from config.

    Expected config structure (YAML):
    data:
        name: src.dataset.dataset_v1.V1   # full module path to Dataset class
        params:
            train_json_dir: ...
            val_json_dir: ...
            ...
    Or you can put args directly under `data` instead of nesting in `params`.
    """
    cfg_dict = _to_dict(cfg)
    data_cfg = cfg_dict.get(section, {})

    target = data_cfg.get("name", None)
    if target is None or "." not in target:
        raise ValueError(f"{section}.name must be a full module path like 'src.dataset.dataset_v1.V1', got: {target}")

    cls = _get_obj_from_str(target)


    params = dict(data_cfg.get("params", {}))
    # Also allow top-level fields in data_cfg (e.g., train_json_dir) without nesting under params
    # for k, v in data_cfg.items():
    #     if k not in {"name", "params"}:
    #         params.setdefault(k, v)

    params.update(extra_kwargs)
    print(f"Loading dataset {target} with class: {cls.__name__}...", end='')
    dataset = cls(**params)
    print('Done!')
    return dataset


if __name__ == '__main__':
    from omegaconf import OmegaConf

    cfg = OmegaConf.load('src/configs/train_vae_v1_canonical_logvar_l2norm_pred_closed.yaml')
    model, _ = load_model_from_config(cfg, device='cuda')

    dataset = load_dataset_from_config(cfg)
    
    #print(model)