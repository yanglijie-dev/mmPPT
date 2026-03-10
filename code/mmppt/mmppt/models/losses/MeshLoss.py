"""
Mesh Loss

Author: Lijie Yang
Please cite our work if the code is helpful to you.
"""


import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from .builder import LOSSES
from mmppt.utils import SMPLXModel
from mmppt.utils.SMPLXModel import *


@LOSSES.register_module()
class MeshLoss(_Loss):
    smplx_model = [None, None, None]
    def __init__(self, size_average=None, reduce=None, reduction: str = 'mean', scale: float = 1):
        """Mesh loss for segmentation task.
        """
        super().__init__(size_average=size_average, reduce=reduce, reduction=reduction)

        if self.smplx_model[0] is None:
            self.smplx_model[0] = SMPLXModel(bm_fname=SMPLX_MODEL_FEMALE_PATH, num_betas=16, num_expressions=0)
        if self.smplx_model[1] is None:
            self.smplx_model[1] = SMPLXModel(bm_fname=SMPLX_MODEL_MALE_PATH, num_betas=16, num_expressions=0)
        if self.smplx_model[2] is None:
            self.smplx_model[2] = SMPLXModel(bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0)
        self.scale = scale

    def forward(self, input: torch.Tensor, target: torch.Tensor, use_gender: int = 0, train: bool = True, use_rodrigues=False):
        _input = input * self.scale  # [32,217]
        _target = target * self.scale  # [32,217]

        if not use_gender:
            input_model = target_model = self.smplx_model[2]
        else:
            input_model = target_model = self.smplx_model[0 if target[0][-1] < 0.5 else 1]

        input_result = input_model(pose_body=_input[:, :-16], betas=_input[:, -16:], use_rodrigues=use_rodrigues,data_type="model_output")
        input_verts = input_result['vertices']
        input_joints = input_result['joints']

        target_result = target_model(pose_body=_target[:, :-16], betas=_target[:, -16:], use_rodrigues=use_rodrigues, data_type="model_target")
        target_verts = target_result['vertices']
        target_joints = target_result['joints']

        per_joint_err = torch.norm((input_joints - target_joints), dim=-1)
        per_vertex_err = torch.norm((input_verts - target_verts), dim=-1)

        if train:
            return (F.l1_loss(input_verts, target_verts, reduction=self.reduction),
                    F.l1_loss(input_joints, target_joints, reduction=self.reduction),
                    {"output_verts":input_verts,"target_verts":target_verts},
                    {"output_joints":input_joints, "target_joints":target_joints})
        else:
            return (torch.sqrt(F.mse_loss(input_verts, target_verts, reduction=self.reduction)),
                    torch.sqrt(F.mse_loss(input_joints, target_joints, reduction=self.reduction)),
                    (per_joint_err, per_vertex_err),{"output_verts":input_verts,"target_verts":target_verts},
                    {"output_joints":input_joints, "target_joints":target_joints})
