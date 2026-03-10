import smplx
import torch
from .human_body_prior.body_model.body_model import BodyModel
from .human_body_prior.body_model.lbs import lbs
from .misc import rotation6d_2_rot_mat, rodrigues_2_rot_mat

SMPLX_MODEL_PATH = "./mmppt/utils/smplx_locked_head"
SMPLX_MODEL_FEMALE_PATH = SMPLX_MODEL_PATH + "/female/model.npz"
SMPLX_MODEL_MALE_PATH = SMPLX_MODEL_PATH + "/male/model.npz"
SMPLX_MODEL_NEUTRAL_PATH = SMPLX_MODEL_PATH + "/neutral/model.npz"

class SMPLXModel(BodyModel):
    def __init__(self, bm_fname=SMPLX_MODEL_NEUTRAL_PATH, num_betas=16, num_expressions=0, **kwargs):
        super().__init__(bm_fname=bm_fname, num_betas=num_betas, num_expressions=num_expressions, **kwargs)

    def forward(self, pose_body, betas, use_rodrigues=True, data_type = "model_output"):
        assert data_type in ["model_output","model_target"]
        device = pose_body.device
        for name in ['init_pose_hand', 'init_pose_jaw','init_pose_eye', 'init_v_template', 'init_expression',
                    'shapedirs', 'exprdirs', 'posedirs', 'J_regressor', 'kintree_table', 'weights', ]:
            _tensor = getattr(self, name)
            setattr(self, name, _tensor.to(device))

        batch_size = pose_body.shape[0]
        trans = pose_body[:, :3]
        pose_hand = self.init_pose_hand.expand(batch_size, -1)
        pose_jaw = self.init_pose_jaw.expand(batch_size, -1)
        pose_eye = self.init_pose_eye.expand(batch_size, -1)
        v_template = self.init_v_template.expand(batch_size, -1, -1)
        expression = self.init_expression.expand(batch_size, -1)

        init_pose = torch.cat([pose_jaw, pose_eye, pose_hand], dim=-1)
        if not use_rodrigues:
            init_pose = rodrigues_2_rot_mat(init_pose)
        full_pose = torch.cat([pose_body[:, 3:], init_pose], dim=-1)
        shape_components = torch.cat([betas, expression], dim=-1)
        shapedirs = torch.cat([self.shapedirs, self.exprdirs], dim=-1)

        verts, joints = lbs(betas=shape_components, pose=full_pose, v_template=v_template,
                        shapedirs=shapedirs, posedirs=self.posedirs, J_regressor=self.J_regressor,
                        parents=self.kintree_table[0].long(), lbs_weights=self.weights, pose2rot=use_rodrigues)

        joints = joints + trans.unsqueeze(dim=1)
        verts = verts + trans.unsqueeze(dim=1)
        '''
        if False:
            vert_t = verts.cpu()
            vert_t = vert_t.detach()
            vert_t = vert_t.numpy()

            model = smplx.create('/home/caffe/Documents/SMPL-X/original_code/models/smplx/SMPLX_MALE.npz',
                             model_type='smplx')
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vert_t[0,:,:])
            mesh.triangles = o3d.utility.Vector3iVector(model.faces)
            o3d.visualization.draw_geometries([mesh],window_name=open3d_window_name)
        '''
        return dict(vertices=verts, joints=joints)
