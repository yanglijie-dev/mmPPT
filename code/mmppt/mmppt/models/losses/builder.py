"""
Criteria Builder

"""
import torch
from mmppt.utils.registry import Registry
from mmppt.utils.misc import rotation6d_2_rot_mat, rodrigues_2_rot_mat
LOSSES = Registry("losses")


class Criteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.criteria = []
        for loss_cfg in self.cfg:
            self.criteria.append(LOSSES.build(cfg=loss_cfg))

    def __call__(self, pred, target):
        if len(self.criteria) == 0:
            # loss computation occur in model
            return pred
        loss = 0
        for c in self.criteria:
            loss += c(pred, target)
        return loss


def build_criteria(cfg):
    return Criteria(cfg)


class LossManager():
    def __init__(self, ding_bot=None) -> None:
        super(LossManager).__init__()
        self.loss_dict = {}
        self.batch_loss = []
        self.ding_bot = ding_bot

    def update_loss(self, name, loss):
        if name not in self.loss_dict:
            self.loss_dict.update({name: [loss]})
        else:
            self.loss_dict[name].append(loss)

    def calculate_total_loss(self):
        batch_loss = []
        for loss in self.loss_dict.values():
            batch_loss.append(loss[-1])
        total_loss = torch.sum(torch.stack(batch_loss))
        self.batch_loss.append(total_loss.detach())
        for key in self.loss_dict:
            self.loss_dict[key][-1] = self.loss_dict[key][-1].detach()
        return total_loss




class mmBodyCriteria(object):
    def __init__(self, cfg=None):
        self.cfg = cfg if cfg is not None else []
        self.losses = LossManager()
        self.criteria = {}
        for loss_cfg in self.cfg:
            loss_type = loss_cfg['type']
            criterion = LOSSES.build(cfg=loss_cfg)
            self.criteria[loss_type] = criterion
    def __call__(self, output, target,args, curr_epoch=None):
        if target.shape[0] != output.shape[0]:
            target = target.reshape(output.shape[0],-1)
        if len(self.criteria) == 0:
            # loss computation occur in model
            return output
        self.losses.update_loss("trans_loss", args.loss_weight[0]*self.criteria["MSELoss"](output[:,0:3], target[:,0:3]))
        if args.use_6d_pose:
            output_mat = rotation6d_2_rot_mat(output[:,3:-16])#[32, 132]→[32, 198] #Convert the 6D rotation vectors of the 22 body key points into rotation matrices, i.e., transforming from 226=132 dimensions to 223*3=198 dimensions.
            target_mat = rodrigues_2_rot_mat(target[:,3:-16])#[32, 66]→[32, 198] Convert Rodrigues vectors (rotation vectors) to rotation matrices. The calculation results include 32 batches, where each batch contains 22 human body key points, and each key point corresponds to one [3, 3] rotation matrix.
            self.losses.update_loss("pose_loss", args.loss_weight[1]*self.criteria["GeodesicLoss"](output_mat, target_mat))#GeodesicLoss
            v_loss, j_loss, smpl_vertices, output_and_target_joints = self.criteria["MeshLoss"](torch.cat((output[:,:3], output_mat, output[:,-16:]), -1),
                                        torch.cat((target[:,:3], target_mat, target[:,-16:]), -1), args.use_gender, )
        else:
            self.losses.update_loss("pose_loss", args.loss_weight[1]*self.criteria["MSELoss"](output[:,3:-16],target[:,3:-16]))
            v_loss, j_loss, smpl_vertices, output_and_target_joints = self.criteria["MeshLoss"](output, target, args.use_gender, use_rodrigues=True)
        # shape loss
        self.losses.update_loss("shape_loss", args.loss_weight[2]*self.criteria["MSELoss"](output[:,-16:], target[:,-16:]))
        # joints loss
        self.losses.update_loss("joints_loss", args.loss_weight[3]*j_loss)
        # vertices loss
        self.losses.update_loss("vertices_loss", args.loss_weight[4]*v_loss)
        # gender loss
        if args.use_gender:
            self.losses.update_loss("gender_loss", args.loss_weight[5]*self.criteria["BCEWithLogitsLoss"](output[:,-1], target[:,-1]))

        #calculate chamfer loss
        # if "ChamferDistance" in args.criteria.criteria:
        #     chamferloss, _ ,_,_,_ = self.criteria["ChamferDistance"](smpl_vertices['output_verts'], smpl_vertices['target_verts'])
        #     chamferloss *= args.loss_weight[6]
        #     self.losses.update_loss("chamfer_loss", chamferloss)

        #计算BoneLengthLoss
        if "BoneLengthLoss" in args.criteria.criteria and curr_epoch>=self.criteria["BoneLengthLoss"].start_epoch:
            BoneLengthLoss= self.criteria["BoneLengthLoss"](output_and_target_joints)

            BoneLengthLoss *= args.loss_weight[6]
            self.losses.update_loss("BoneLengthLoss", BoneLengthLoss)


        loss = self.losses.calculate_total_loss()
        return loss, smpl_vertices


def build_mmBody_criteria(cfg):
    return mmBodyCriteria(cfg)
