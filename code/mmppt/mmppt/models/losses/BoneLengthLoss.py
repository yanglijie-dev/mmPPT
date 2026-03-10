import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds

from typing import Union
from .builder import LOSSES




@LOSSES.register_module()
class BoneLengthLoss(torch.nn.Module):
    def __init__(self, start_epoch):
        super(BoneLengthLoss, self).__init__()
        self.start_epoch = int(start_epoch)
    def forward( self, output_and_target_joints):
        '''
        Args:
            target_joints: [Batch,55,3] represent 55 body joints' coordination. Refer to mapping_joint_to_body_part.py for the mapping between 55 joints and real body parts.
        '''
        a = self.start_epoch
        l_shoulder, l_elbow, l_wrist = 16, 18, 20
        r_shoulder, r_elbow, r_wrist = 17, 19, 21
        l_hip, l_knee, l_ankle = 1, 4, 7
        r_hip, r_knee, r_ankle = 2, 5, 8
        #Recording  the indices of
                        # left_shoulder,left_elbow,left_wrist,
                       # right_shoulder,right_elbow,right_wrist,
                       # left_hip,left_knee,left_ankle,
                       # righ_hip,righ_knee,righ_ankle
        # in joints

        target_joints = output_and_target_joints["target_joints"]
        output_joints = output_and_target_joints["output_joints"]

        tgt_B_up_1 = torch.sum(torch.abs( target_joints[:, l_shoulder,:]-target_joints[:, l_elbow,:] ), dim=1 ) +\
                torch.sum(torch.abs(target_joints[:, r_shoulder, :] - target_joints[:, r_elbow, :]), dim=1)
        tgt_B_up_1 = torch.unsqueeze(tgt_B_up_1, dim=-1)

        tgt_B_up_2 = torch.sum(torch.abs( target_joints[:, l_elbow,:]-target_joints[:, l_wrist,:] ), dim=1 ) +\
                torch.sum(torch.abs(target_joints[:, r_elbow, :] - target_joints[:, r_wrist, :]), dim=1)
        tgt_B_up_2 = torch.unsqueeze(tgt_B_up_2, dim=-1)

        out_B_up_1 = torch.sum(torch.abs( output_joints[:, l_shoulder,:]-output_joints[:, l_elbow,:] ), dim=1 ) +\
                torch.sum(torch.abs(output_joints[:, r_shoulder, :] - output_joints[:, r_elbow, :]), dim=1)
        out_B_up_1 = torch.unsqueeze(out_B_up_1, dim=-1)

        out_B_up_2 = torch.sum(torch.abs( output_joints[:, l_elbow,:]-output_joints[:, l_wrist,:] ), dim=1 ) +\
                torch.sum(torch.abs(output_joints[:, r_elbow, :] - output_joints[:, r_wrist, :]), dim=1)
        out_B_up_2 = torch.unsqueeze(out_B_up_2, dim=-1)


        tgt_B_low_1 = torch.sum(torch.abs( target_joints[:, l_hip,:]-target_joints[:, l_knee,:] ), dim=1 ) +\
                torch.sum(torch.abs(target_joints[:, r_hip, :] - target_joints[:, r_knee, :]), dim=1)
        tgt_B_low_1 = torch.unsqueeze(tgt_B_low_1, dim=-1)

        tgt_B_low_2 = torch.sum(torch.abs( target_joints[:, l_knee,:]-target_joints[:, l_ankle,:] ), dim=1 ) +\
                torch.sum(torch.abs(target_joints[:, r_knee, :] - target_joints[:, r_ankle, :]), dim=1)
        tgt_B_low_2 = torch.unsqueeze(tgt_B_low_2, dim=-1)

        out_B_low_1 = torch.sum(torch.abs( output_joints[:, l_hip,:]-output_joints[:, l_knee,:] ), dim=1 ) +\
                torch.sum(torch.abs(output_joints[:, r_hip, :] - output_joints[:, r_knee, :]), dim=1)
        out_B_low_1 = torch.unsqueeze(out_B_low_1, dim=-1)

        out_B_low_2 = torch.sum(torch.abs( output_joints[:, l_knee,:]-output_joints[:, l_ankle,:] ), dim=1 ) +\
                torch.sum(torch.abs(output_joints[:, r_knee, :] - output_joints[:, r_ankle, :]), dim=1)
        out_B_low_2 = torch.unsqueeze(out_B_low_2, dim=-1)

        BoneLengthLoss_up = torch.sum(torch.abs(tgt_B_up_1 - out_B_up_1), dim=1)+\
                            torch.sum(torch.abs(tgt_B_up_2 - out_B_up_2),dim=1)
        BoneLengthLoss_low = torch.sum(torch.abs(tgt_B_low_1-out_B_low_1), dim=1)+\
                             torch.sum(torch.abs(tgt_B_low_2-out_B_low_2) ,dim=1)
        BoneLengthLoss = torch.mean(BoneLengthLoss_up + BoneLengthLoss_low)
        return BoneLengthLoss
