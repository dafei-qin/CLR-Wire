import torch
from chamferdist import ChamferDistance


def compute_cd_rt(source, target):
    CD_loss = ChamferDistance()
    