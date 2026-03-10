import torch

import time

from .z_order import xyz2key as z_order_encode_
from .z_order import xyz2key_with_t as z_order_encode_with_t
from .z_order import key2xyz as z_order_decode_
# from .z_order import key2xyz_with_t as z_order_decode_with_t

from .hilbert import encode as hilbert_encode_
from .hilbert import decode as hilbert_decode_


@torch.inference_mode()
def encode(grid_coord, t=None, batch=None, depth=16, order="z", anchor_span=None):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans","hybrid-type0","hybrid-type0-trans","hybrid-type1","hybrid-type1-trans"}

    if order == "z":
        code = z_order_encode(grid_coord, t=t, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], t=t, depth=depth)

    elif order == "hilbert":
        code = hilbert_encode(grid_coord, t=t, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], t=t, depth=depth)
    elif order == "hybrid-type0":
        code, depth_after_hybrid = hybrid_encode(points=grid_coord, depth=depth, anchor_curve_mode="z", anchor_span=anchor_span)
    elif order == "hybrid-type0-trans":
        code, depth_after_hybrid = hybrid_encode(points=grid_coord[:,[1,0,2]], depth=depth, anchor_curve_mode='z', anchor_span=anchor_span)
    elif order == "hybrid-type1":
        code, depth_after_hybrid = hybrid_encode(points=grid_coord, depth=depth, anchor_curve_mode="hilbert", anchor_span=anchor_span)
    elif order == "hybrid-type1-trans":
        code, depth_after_hybrid = hybrid_encode(points=grid_coord[:,[1,0,2]], depth=depth, anchor_curve_mode='hilbert', anchor_span=anchor_span)
    else:
        raise NotImplementedError


    if order in {"z", "z-trans", "hilbert", "hilbert-trans"}:
        if t is None:
            return code, depth * 3
        else:
            return code, depth * 4
    elif order in {"hybrid-type0","hybrid-type0-trans","hybrid-type1","hybrid-type1-trans"}:
        return code, depth_after_hybrid
    else:
        raise NotImplementedError


@torch.inference_mode()
def decode(code, depth=16, order="z"):
    raise NotImplementedError("todo: this function has not yet been verified-----Lijie Yang")

    assert order in {"z", "hilbert"}
    batch = code >> depth * 3
    code = code & ((1 << (depth * 3)) - 1)

    if order == "z":
        grid_coord = z_order_decode(code, depth=depth)
    elif order == "hilbert":
        grid_coord = hilbert_decode(code, depth=depth)
    else:
        raise NotImplementedError
    return grid_coord, batch

def z_order_encode(grid_coord: torch.Tensor, t=None, depth: int = 16):
    x, y, z = grid_coord[:, 0].long(), grid_coord[:, 1].long(), grid_coord[:, 2].long()
    # we block the support to batch, maintain batched code in Point class
    if t is None:
        code = z_order_encode_(x=x, y=y, z=z, b=None, depth=depth)
    else:
        t = t.long()
        code = z_order_encode_with_t(x=x, y=y, z=z, t=t, b=None, depth=depth)
    return code


def z_order_decode(code: torch.Tensor, depth):
    x, y, t, z,_ = z_order_decode_(code, depth=depth)
    grid_coord = torch.stack([x, y, z], dim=-1)  # (N,  3)
    return grid_coord


def hilbert_encode(grid_coord: torch.Tensor, t=None, depth: int = 16):#todo
    if t is None:
        return hilbert_encode_(grid_coord, num_dims=3, num_bits=depth)
    else:
        grid_coord_with_t = torch.cat((grid_coord, t.unsqueeze(-1)), dim=1)
        grid_coord_with_t[:,[-1,-2]] = grid_coord_with_t[:,[-2,-1]]#[x,y,z,t] -> [x,y,t,z]
        return hilbert_encode_(grid_coord_with_t, num_dims=3+1, num_bits=depth)

def hilbert_decode(code: torch.Tensor, depth: int = 16):#todo
    return hilbert_decode_(code, num_dims=3, num_bits=depth)


def hybrid_encode(points:torch.Tensor, depth:int, anchor_curve_mode, anchor_span:int ):

    coord_max = 2**depth-1
    anchor_coord = torch.arange(0, coord_max, step=anchor_span)

    # start_time = time.time()
    x_anchor, y_anchor, z_anchor = torch.meshgrid(anchor_coord, anchor_coord, anchor_coord)
    # end_time = time.time()
    # print(f"torch.meshgrid : {end_time-start_time} 秒")



    anchor_points = torch.stack((x_anchor, y_anchor, z_anchor),dim=3).view(-1,3).to(points.device)

    if anchor_curve_mode=="z":
        code_anchor = z_order_encode(grid_coord=anchor_points / anchor_span, depth=depth)
    elif anchor_curve_mode=="hilbert":
        code_anchor = hilbert_encode(grid_coord= anchor_points / anchor_span, depth=depth)
    else:
        raise NotImplementedError


    depth_code_anchor = int(code_anchor.max() + 1).bit_length()


    cube_idx_for_points = (points / anchor_span).to(int)#
    points_coord_relative_to_anchor = torch.remainder(points, anchor_span)#
    code_z_points_coord_relative_to_anchor,_ = encode(grid_coord=points_coord_relative_to_anchor, order='z')#
    code_hilbert_points_coord_relative_to_anchor,_ = encode(grid_coord=points_coord_relative_to_anchor, order='hilbert')#


    code_group_points_relative = []
    originPoint_to_groupPoint_map = []
    unique_idx = torch.unique(cube_idx_for_points, dim=0)
    depth_code_relative = 0
    for cube_idx in unique_idx:
        mask = torch.all((cube_idx_for_points==cube_idx), dim=1)
        originPoint_to_groupPoint_map.append(torch.where(mask)[0])
        use_z_order = sum(cube_idx) % 2 == 1
        if use_z_order:
            code_points_coord_relative_to_anchor = code_z_points_coord_relative_to_anchor[mask]
        else:
            code_points_coord_relative_to_anchor = code_hilbert_points_coord_relative_to_anchor[mask]

        code_group_points_relative.append(code_points_coord_relative_to_anchor)
        depth_code_relative = max(int(code_points_coord_relative_to_anchor.max() + 1).bit_length(), depth_code_relative)




    remainder = depth_code_relative % 3#
    if remainder != 0:
        depth_code_relative += 3 - remainder
    used_depth = depth_code_anchor + depth_code_relative
    assert used_depth < 16*3, "depth_code_anchor + depth_code_relative is too large"




    for idx,used_anchor in enumerate(unique_idx):
        combined_code = (code_anchor.view(x_anchor.size())[used_anchor[0],used_anchor[1],used_anchor[2]] << depth_code_relative) | code_group_points_relative[idx]
        if combined_code.dim()==0:
            combined_code = combined_code.unsqueeze(-1).unsqueeze(-1)
        else:
            combined_code = combined_code.unsqueeze(-1)
        if idx == 0:
            code_final = combined_code.clone()
        else:
            code_final = torch.cat((code_final, combined_code),dim=0)
    code_final = code_final.squeeze(-1)#





    for idx, origin_points in enumerate(originPoint_to_groupPoint_map):
        if idx == 0:
            originPoint_to_groupPoint_map_tensor = origin_points.clone()
        else:
            originPoint_to_groupPoint_map_tensor = torch.cat((originPoint_to_groupPoint_map_tensor, origin_points), dim=0)

    groupPoint_to_originPoint_map_tensor = torch.argsort(originPoint_to_groupPoint_map_tensor)



    code_final_reordered =  code_final[groupPoint_to_originPoint_map_tensor]


    return code_final_reordered, used_depth

