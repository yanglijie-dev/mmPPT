import torch

from z_order import xyz2key as z_order_encode_
from z_order import xyz2key_with_t as z_order_encode_with_t

from z_order import key2xyz as z_order_decode_
# from .z_order import key2xyz_with_t as z_order_decode_with_t

from hilbert import encode as hilbert_encode_

from hilbert import decode as hilbert_decode_


@torch.inference_mode()#
def encode(grid_coord, t=None, batch=None, depth=16, order="z"):
    assert order in {"z", "z-trans", "hilbert", "hilbert-trans"}

    if order == "z":#
        code = z_order_encode(grid_coord, t=t, depth=depth)
    elif order == "z-trans":
        code = z_order_encode(grid_coord[:, [1, 0, 2]], t=t, depth=depth)
    elif order == "hilbert":
        code = hilbert_encode(grid_coord, t=t, depth=depth)
    elif order == "hilbert-trans":
        code = hilbert_encode(grid_coord[:, [1, 0, 2]], t=t, depth=depth)
    else:
        raise NotImplementedError

    if batch is not None:
        batch = batch.long()
        if t is None:
            code = batch << (depth * 3) | code
        else:
            code = batch << (depth * 4) | code
    return code


@torch.inference_mode()
def decode(code, depth=16, order="z"):#todo: this function has not yet been verified-----Lijie Yang
    #raise NotImplementedError("todo: this function has not yet been verified-----Lijie Yang")
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
        grid_coord_with_t = torch.cat((grid_coord, t.unsqueeze(-1)), dim=1)#
        grid_coord_with_t[:,[-1,-2]] = grid_coord_with_t[:,[-2,-1]]#[x,y,z,t] -> [x,y,t,z]
        return hilbert_encode_(grid_coord_with_t, num_dims=3+1, num_bits=depth)

def hilbert_decode(code: torch.Tensor, depth: int = 16):#todo
    return hilbert_decode_(code, num_dims=3, num_bits=depth)




def hybrid_encode(points:torch.Tensor, depth:int, anchor_mode, anchor_span:int, patch_mode):

    coord_max = 2**depth-1
    anchor_coord = torch.arange(0, coord_max, step=anchor_span)
    x_anchor, y_anchor, z_anchor = torch.meshgrid(anchor_coord, anchor_coord, anchor_coord)
    anchor_points = torch.stack((x_anchor, y_anchor, z_anchor),dim=3).view(-1,3).to(device)#
    code_anchor= encode(grid_coord=anchor_points/anchor_span, order=anchor_mode) #
    depth_code_anchor = int(code_anchor.max() + 1).bit_length()

    cube_idx_for_points = (points / anchor_span).to(int)#
    points_coord_relative_to_anchor = torch.remainder(points, anchor_span)#

    if patch_mode=='normal':
        code_z_points_coord_relative_to_anchor = encode(grid_coord=points_coord_relative_to_anchor, order='z')#
        code_hilbert_points_coord_relative_to_anchor = encode(grid_coord=points_coord_relative_to_anchor, order='hilbert')#
    elif patch_mode=='trans':
        code_z_points_coord_relative_to_anchor = encode(grid_coord=points_coord_relative_to_anchor, order='z-trans')#
        code_hilbert_points_coord_relative_to_anchor = encode(grid_coord=points_coord_relative_to_anchor, order='hilbert-trans')#
    else:
        raise NotImplementedError

    group_points_relative_to_anchor = []#
    code_group_points_relative = []#
    originPoint_to_groupPoint_map = []#
    unique_idx = torch.unique(cube_idx_for_points, dim=0)
    depth_code_relative = 0
    for cube_idx in unique_idx:
        mask = torch.all((cube_idx_for_points==cube_idx), dim=1)
        originPoint_to_groupPoint_map.append(torch.where(mask)[0])
        # group_points.append(points[mask])
        group_points_relative_to_anchor.append(points_coord_relative_to_anchor[mask])
        use_z_order = sum(cube_idx) % 2 == 1  #
        if use_z_order:
            code_points_coord_relative_to_anchor = code_z_points_coord_relative_to_anchor[mask]
        else:
            code_points_coord_relative_to_anchor = code_hilbert_points_coord_relative_to_anchor[mask]

        code_group_points_relative.append(code_points_coord_relative_to_anchor)
        depth_code_relative = max(int(code_points_coord_relative_to_anchor.max() + 1).bit_length(), depth_code_relative)#


    remainder = depth_code_relative % 3
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
    code_final = code_final.squeeze(-1)


    for idx, origin_points in enumerate(originPoint_to_groupPoint_map):
        if idx == 0:
            originPoint_to_groupPoint_map_tensor = origin_points.clone()
        else:
            originPoint_to_groupPoint_map_tensor = torch.cat((originPoint_to_groupPoint_map_tensor, origin_points), dim=0)
    groupPoint_to_originPoint_map_tensor = torch.argsort(originPoint_to_groupPoint_map_tensor)

    # _order_final = torch.argsort(code_final)
    # order_final = originPoint_to_groupPoint_map_tensor[_order_final]

    code_final_reordered =  code_final[groupPoint_to_originPoint_map_tensor]

    return code_final_reordered, used_depth


if __name__=="__main__":
    import torch
    if torch.cuda.is_available():
        device = torch.device("cuda")
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"

    point_mode = "2D"
    coord_max = 20
    depth = int(coord_max).bit_length()
    anchor_span =  5
    n_points = 100 #only available when point_distribution_mode=="random"

    anchor_mode = ["z", "z-trans", "hilbert", "hilbert-trans"]#
    patch_mode = ["normal", "trans"]#


    point_distribution_mode = "random"# "random" or "uniform"

    if point_distribution_mode=="uniform":
        #生成均匀分布的点云
        tmp = torch.arange(0, coord_max, 1, dtype=torch.int64).to(device)
        if point_mode=="2D":
            tmp_z = torch.tensor([0]).to(device)
            points = torch.stack(torch.meshgrid((tmp, tmp, tmp_z)), dim=3).view(-1,3)
        elif point_mode == "3D":
            points = torch.stack(torch.meshgrid((tmp, tmp, tmp)), dim=3).view(-1,3)
    elif point_distribution_mode=="random":
        points = torch.randint(0, coord_max, (n_points,3), dtype=torch.int64).to(device)
        if point_mode=="2D":
            points[:,2] = 0

    x = points[:, 0].to('cpu')
    y = points[:, 1].to('cpu')
    z = points[:, 2].to('cpu')

    for curr_anchor_mode in anchor_mode:
        for curr_patch_mode in patch_mode:
            code_final, _ = hybrid_encode(points=points, depth=depth, anchor_mode=curr_anchor_mode, anchor_span=anchor_span, patch_mode=curr_patch_mode)
            order_final = torch.argsort(code_final).to('cpu')

            # order_final = order_final.to('cpu')
            if point_mode=="3D":
                fig = go.Figure(
                    data=[go.Scatter3d(x=x[order_final], y=y[order_final], z=z[order_final], mode='markers+lines', marker=dict(size=10), line=dict(color='blue',width=3))])
                fig.update_layout(title='Hybrid mode, 3D'+"; anchor_mode="+curr_anchor_mode+"; patch_mode="+curr_patch_mode, scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z',xaxis=dict(range=[0,coord_max],dtick=1),yaxis=dict(range=[0,coord_max],dtick=1),zaxis=dict(range=[0,coord_max],dtick=1)))
                try:fig.show()
                except:
                    fig.write_html("tmp/Hybrid_mode_3D.html")
            elif point_mode=="2D":

                fig = go.Figure(
                    data=[go.Scatter(x=x[order_final], y=y[order_final], mode='markers', marker=dict(size=10), line=dict(color='blue', width=3))])
                fig.update_layout(title='Point', xaxis_title='X',yaxis_title='Y',xaxis=dict(range=[-1, coord_max], dtick=1),yaxis=dict(range=[-1, coord_max], dtick=1))
                try:fig.show()
                except:
                    fig.write_html("tmp/Point.html")

                fileName = 'Hybrid_mode_2D'+"--anchor_mode="+curr_anchor_mode+"--patch_mode="+curr_patch_mode
                fig = go.Figure(
                    data=[go.Scatter(x=x[order_final], y=y[order_final], mode='markers+lines', marker=dict(size=10), line=dict(color='blue', width=3))])
                fig.update_layout(title=fileName, xaxis_title='X',yaxis_title='Y',xaxis=dict(range=[0, coord_max], dtick=1),yaxis=dict(range=[0, coord_max], dtick=1))
                try:fig.show()
                except:
                    fig.write_html("tmp/"+fileName+".html")


            if point_distribution_mode=="uniform":
                random_values = (torch.rand(points.size(0), 2, device=device) - 0.5).to('cpu')
                x_random = x.float() + random_values[:,0]
                y_random = y.float() + random_values[:,1]
                x_random = torch.clamp(x_random, min=0, max=coord_max)
                y_random = torch.clamp(y_random, min=0, max=coord_max)
                fig = go.Figure(
                    data=[go.Scatter(x=x_random[order_final], y=y_random[order_final], mode='markers', marker=dict(size=10), line=dict(color='blue', width=3))])
                fig.update_layout(title='Point with jitter',xaxis_title='X',yaxis_title='Y',xaxis=dict(range=[-1, coord_max+1], dtick=1),yaxis=dict(range=[-1, coord_max+1], dtick=1))
                try:fig.show()
                except:
                    fig.write_html("tmp/Point_with_jitter.html")

                fileName = 'Hybrid_mode_pointJitter_2D'+"--anchor_mode="+curr_anchor_mode+"--patch_mode="+curr_patch_mode
                fig = go.Figure(
                    data=[go.Scatter(x=x_random[order_final], y=y_random[order_final], mode='markers+lines', marker=dict(size=10), line=dict(color='blue', width=3))])
                fig.update_layout(title=fileName,xaxis_title='X',yaxis_title='Y',xaxis=dict(range=[-1, coord_max+1], dtick=1),yaxis=dict(range=[-1, coord_max+1], dtick=1))
                try:fig.show()
                except:
                    fig.write_html("tmp/"+fileName+".html")

    code_0 = encode(grid_coord=points, order='z')  # 'hilbert'  'z'
    order_0 = torch.argsort(code_0).to('cpu')
    '''
    fig = go.Figure(
        data=[go.Scatter3d(x=x[order_0], y=y[order_0], z=z[order_0], mode='markers+lines', marker=dict(size=10), line=dict(color='blue'))])
    fig.update_layout(scene=dict(title='Z mode',xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    fig.show()
    '''
    fig = go.Figure(
        data=[go.Scatter(x=x[order_0], y=y[order_0], mode='markers+lines', marker=dict(size=10),
                         line=dict(color='blue', width=3))])
    fig.update_layout(title='Z mode',xaxis_title='X', yaxis_title='Y', xaxis=dict(range=[-1, coord_max+1], dtick=1),
                      yaxis=dict(range=[-1, coord_max+1], dtick=1))
    try:fig.show()
    except:
        fig.write_html("tmp/Z_mode.html")

    if point_distribution_mode == "uniform":
        fig = go.Figure(
            data=[go.Scatter(x=x_random[order_0], y=y_random[order_0], mode='markers+lines', marker=dict(size=10), line=dict(color='blue', width=3))])
        fig.update_layout(title='Z mode with position jitter',xaxis_title='X',yaxis_title='Y',xaxis=dict(range=[-1, coord_max+1], dtick=1),yaxis=dict(range=[-1, coord_max+1], dtick=1))
        try:fig.show()
        except:
            fig.write_html("tmp/Z_mode_with_position_jitter.html")



    code_1= encode(grid_coord=points, order='hilbert')  # 'hilbert'  'z'
    order_1 = torch.argsort(code_1).to('cpu')
    '''
    fig = go.Figure(
        data=[go.Scatter3d(x=x[order_1], y=y[order_1], z=z[order_1], mode='markers+lines', marker=dict(size=10), line=dict(color='blue'))])
    fig.update_layout(scene=dict(title='Hilbert mode',xaxis_title='X', yaxis_title='Y', zaxis_title='Z',xaxis=dict(range=[0,2],dtick=1),yaxis=dict(range=[0,2],dtick=1),zaxis=dict(range=[0,2],dtick=1)))
    fig.show()
    '''
    fig = go.Figure(
        data=[go.Scatter(x=x[order_1], y=y[order_1], mode='markers+lines', marker=dict(size=10), line=dict(color='blue', width=3))])
    fig.update_layout(title='hilbert mode',xaxis_title='X',yaxis_title='Y',xaxis=dict(range=[-1, coord_max+1], dtick=1),yaxis=dict(range=[-1, coord_max+1], dtick=1))
    try:fig.show()
    except:
        fig.write_html("tmp/hilbert_mode.html")

    if point_distribution_mode == "uniform":
        fig = go.Figure(
            data=[go.Scatter(x=x_random[order_1], y=y_random[order_1], mode='markers+lines', marker=dict(size=10), line=dict(color='blue', width=3))])
        fig.update_layout(title='hilbert mode with position jitter',xaxis_title='X',yaxis_title='Y',xaxis=dict(range=[-1, coord_max+1], dtick=1),yaxis=dict(range=[-1, coord_max+1], dtick=1))
        try:fig.show()
        except:
            fig.write_html("tmp/hilbert_mode_with_position_jitter.html")
