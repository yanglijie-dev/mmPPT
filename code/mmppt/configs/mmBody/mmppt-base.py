_base_ = ["../_base_/default_runtime.py"]

batch_size =32
num_worker = 16

empty_cache = True
enable_amp = True

empty_cache_per_epoch = True

#you can also perform test here
test_only = False
test_only_pth_path="../output/hybrid_clip=1_BoneLoss_20250306_113331"#only effective when test_only=True. Read model_best.pth from designated folder

smpl_vertices_store = True


# model settings
model = dict(
    type="DefaultPointCloud",
    num_classes=151,
    backbone=dict(
        type="PT-v3m1",
        in_channels=6,
        order=["hybrid-type0", "hybrid-type0-trans","hybrid-type1", "hybrid-type1-trans"],#type0 adopts z-order curving， type1 adopts hilbert curving
        serialization_mode ="xyz",
        shuffle_orders=True,
        anchor_span = 2,
        stride=(2, 2, 2, 2),
        enc_depths=(2, 2, 2, 6, 2),
        enc_channels=(32, 64, 128, 256, 512),
        enc_num_head=(2, 4, 8, 16, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(2, 2, 2, 2),
        dec_channels=(152, 256, 256, 512),
        dec_num_head=(4, 8, 8, 16),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        pre_norm=True,
        enable_temporal_encoding = True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        cls_mode=False,
        pdnorm_bn=False,
        pdnorm_ln=False,
        pdnorm_decouple=True,
        pdnorm_adaptive=False,
        pdnorm_affine=True,
        pdnorm_conditions=("ScanNet", "S3DIS", "Structured3D", "mmBody"),
    ),
    criteria=[
        dict(type="MSELoss"),
        dict(type="BCEWithLogitsLoss"),
        dict(type="MeshLoss"),
        dict(type="GeodesicLoss"),
        dict(type="BoneLengthLoss",start_epoch=0),
    ],
    loss_weight = [1,1,1,1,1,1,1],#weight for trans_loss, pose_loss, shape_loss, joints_loss, vertices_loss, gender_loss, BoneLengthLoss
    use_6d_pose = True,
    use_gender = False,
)
model["backbone_embed_dim"] = model["backbone"]["dec_channels"][0] if not model["backbone"]["cls_mode"] else model["backbone"]["enc_channels"][-1]


# scheduler settings
eval_epoch = epoch = 100
optimizer = dict(type="AdamW", lr=0.001, weight_decay=0.01)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.001, 0.0001],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0001)]

# dataset settings
dataset_type = "mmBodyDataset"

data_path = "../../data/mmBody"

clip_len = skip_head =5,
num_points = 1024,
normal_scale = 1,
new_gmm = "store_true",
output_dim = 151,
use_6d_pose = 1,
input_data = "radar",
test_scene = "lab1",  # type of test data, choose from: furnished  lab1  lab2  occlusion  poor_lighting  rain  smoke
seq_idxes = range(20),
feat_dim = model["backbone"]["in_channels"]-3,

test = dict(type="PointCloudTester")

basic_data_cfg = dict(
    type = dataset_type,
    data_path = data_path,
    train = True,
    clip_frames=clip_len,
    skip_head=skip_head,
    input_data= input_data,
    test_scene = test_scene,
    seq_idxes = seq_idxes,
    num_points = num_points,
    output_dim = output_dim,
    feat_dim = feat_dim,
    point_shuffle = False,
    transform=[
        dict(
            type="GridSample",
            grid_size=0.1,
            hash_type="fnv",
            mode="train",
            use_all_grid_points =True,
            keys=("coord", "features"),
            return_grid_coord=True,
        ),
        dict(type="ToTensor"),
        dict(
            type="Collect",
            keys=("coord", "grid_coord", "label"),
            feat_keys=("coord", "features"),
        ),
    ],
)


data = dict(
    train=dict(
        basic_data_cfg,
    ),
    test=dict(
        basic_data_cfg,
        data_path=data_path+"/test",
        train=False,
    ),
)

hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="mmBodyInformationWriter"),
    dict(type="mmBodyEvaluator"),
    dict(type="mmBodyCheckpointSaver", save_freq=5),
    dict(type="PreciseEvaluator", test_last=False),
]