# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
from typing import Optional, Union


class KeyLUT:
    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                self.xyz2key(r256, zero, zero, 8),
                self.xyz2key(zero, r256, zero, 8),
                self.xyz2key(zero, zero, r256, 8),
            )
        }
        self._decode = {device: self.key2xyz(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key(self, x, y, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (2 * i + 2))
                | ((y & mask) << (2 * i + 1))
                | ((z & mask) << (2 * i + 0))
            )
        return key

    def key2xyz(self, key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (3 * i + 2))) >> (2 * i + 2))
            y = y | ((key & (1 << (3 * i + 1))) >> (2 * i + 1))
            z = z | ((key & (1 << (3 * i + 0))) >> (2 * i + 0))
        return x, y, z

_key_lut = KeyLUT()

def xyz2key(x: torch.Tensor, y: torch.Tensor, z: torch.Tensor, b: Optional[Union[torch.Tensor, int]] = None, depth: int = 16,):
    r"""Encodes :attr:`x`, :attr:`y`, :attr:`z` coordinates to the shuffled keys
    based on pre-computed look up tables. The speed of this function is much
    faster than the method based on for-loop.

    Args:
      x (torch.Tensor): The x coordinate.
      y (torch.Tensor): The y coordinate.
      z (torch.Tensor): The z coordinate.
      b (torch.Tensor or int): The batch index of the coordinates, and should be
          smaller than 32768. If :attr:`b` is :obj:`torch.Tensor`, the size of
          :attr:`b` must be the same as :attr:`x`, :attr:`y`, and :attr:`z`.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    EX, EY, EZ = _key_lut.encode_lut(x.device)
    x, y, z = x.long(), y.long(), z.long()

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | EZ[z & mask]
    if depth > 8:
        mask = (1 << (depth - 8)) - 1
        key16 = EX[(x >> 8) & mask] | EY[(y >> 8) & mask] | EZ[(z >> 8) & mask]
        key = key16 << 24 | key

    if b is not None:
        b = b.long()
        key = b << 48 | key

    return key

def key2xyz(key: torch.Tensor, depth: int = 16):
    r"""Decodes the shuffled key to :attr:`x`, :attr:`y`, :attr:`z` coordinates
    and the batch index based on pre-computed look up tables.

    Args:
      key (torch.Tensor): The shuffled key.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    DX, DY, DZ = _key_lut.decode_lut(key.device)
    x, y, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)

    b = key >> 48
    key = key & ((1 << 48) - 1)

    n = (depth + 2) // 3
    for i in range(n):
        k = key >> (i * 9) & 511
        x = x | (DX[k] << (i * 3))
        y = y | (DY[k] << (i * 3))
        z = z | (DZ[k] << (i * 3))

    return x, y, z, b

class KeyLUT_with_t:

    def __init__(self):
        r256 = torch.arange(256, dtype=torch.int64)
        r512 = torch.arange(512, dtype=torch.int64)
        zero = torch.zeros(256, dtype=torch.int64)
        device = torch.device("cpu")

        self._encode = {
            device: (
                self.xyz2key_with_t(r256, zero, zero, zero, 8),
                self.xyz2key_with_t(zero, r256, zero, zero, 8),
                self.xyz2key_with_t(zero, zero, r256, zero, 8),
                self.xyz2key_with_t(zero, zero, zero, r256, 8),
            )
        }
        self._decode = {device: self.key2xyz_with_t(r512, 9)}

    def encode_lut(self, device=torch.device("cpu")):
        if device not in self._encode:
            cpu = torch.device("cpu")
            self._encode[device] = tuple(e.to(device) for e in self._encode[cpu])
        return self._encode[device]

    def decode_lut(self, device=torch.device("cpu")):
        if device not in self._decode:
            cpu = torch.device("cpu")
            self._decode[device] = tuple(e.to(device) for e in self._decode[cpu])
        return self._decode[device]

    def xyz2key_with_t(self, x, y, t, z, depth):
        key = torch.zeros_like(x)
        for i in range(depth):
            mask = 1 << i
            key = (
                key
                | ((x & mask) << (3 * i + 3))
                | ((y & mask) << (3 * i + 2))
                | ((t & mask) << (3 * i + 1))
                | ((z & mask) << (3 * i + 0))
            )
        return key

    def key2xyz_with_t(self, key, depth):
        x = torch.zeros_like(key)
        y = torch.zeros_like(key)
        t = torch.zeros_like(key)
        z = torch.zeros_like(key)
        for i in range(depth):
            x = x | ((key & (1 << (4 * i + 3))) >> (3 * i + 3))
            y = y | ((key & (1 << (4 * i + 2))) >> (3 * i + 2))
            t = t | ((key & (1 << (4 * i + 1))) >> (3 * i + 1))
            z = z | ((key & (1 << (4 * i + 0))) >> (3 * i + 0))
        return x, y, t, z

_key_lut_with_t = KeyLUT_with_t()

def xyz2key_with_t(x: torch.Tensor, y: torch.Tensor, t: torch.Tensor, z: torch.Tensor, b: Optional[Union[torch.Tensor, int]] = None, depth: int = 16,):
    r"""Encodes :attr:`x`, :attr:`y`, :attr:`z` coordinates to the shuffled keys; attr:'t' temporal information
    based on pre-computed look up tables. The speed of this function is much
    faster than the method based on for-loop.

    Args:
      x (torch.Tensor): The x coordinate.
      y (torch.Tensor): The y coordinate.
      t (torch.Tensor): The temporal info.
      z (torch.Tensor): The z coordinate.
      b (torch.Tensor or int): The batch index of the coordinates, and should be
          smaller than 32768. If :attr:`b` is :obj:`torch.Tensor`, the size of
          :attr:`b` must be the same as :attr:`x`, :attr:`y`, and :attr:`z`.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    EX, EY, ET, EZ = _key_lut_with_t.encode_lut(x.device)
    x, y, z, t = x.long(), y.long(), z.long(), t.long()

    assert depth<=8, "Currently only depth<=8 is supported for xyz_with_t"

    mask = 255 if depth > 8 else (1 << depth) - 1
    key = EX[x & mask] | EY[y & mask] | ET[t & mask] | EZ[z & mask]

    if b is not None:
        b = b.long()
        key = b << 48 | key

    return key

def key2xyz_with_t(key: torch.Tensor, depth: int = 16):
    r"""Decodes the shuffled key to :attr:`x`, :attr:`y`, :attr:`z` coordinates
    and the batch index based on pre-computed look up tables.

    Args:
      key (torch.Tensor): The shuffled key.
      depth (int): The depth of the shuffled key, and must be smaller than 17 (< 17).
    """

    DX, DY, DT, DZ = _key_lut_with_t.decode_lut(key.device)
    x, y, t, z = torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key), torch.zeros_like(key)

    b = key >> 48
    key = key & ((1 << 48) - 1)

    n = (depth + 3) // 4
    for i in range(n):
        k = key >> (i * 12) & 511
        x = x | (DX[k] << (i * 4))
        y = y | (DY[k] << (i * 4))
        t = t | (DT[k] << (i * 4))
        z = z | (DZ[k] << (i * 4))

    return x, y, t, z, b


def main():
    import plotly.graph_objects as go
    import plotly.io as pio
    pio.renderers.default = "browser"

    n_points_per_dim = 5
    x_values = torch.linspace(0,n_points_per_dim-1,n_points_per_dim)
    y_values = torch.linspace(0,n_points_per_dim-1,n_points_per_dim)
    z_values = torch.linspace(0,n_points_per_dim-1,n_points_per_dim)
    X,Y,Z = torch.meshgrid(x_values,y_values,z_values)

    point = torch.stack((X.flatten(),Y.flatten(),Z.flatten()), dim=1)
    x ,y, z= point[:,0],point[:,1],point[:,2]
    z_order_key = xyz2key(x=x, y=y, z=z)

    x_order, y_order, z_order = x[z_order_key],y[z_order_key],z[z_order_key]
    fig = go.Figure(
        data=[go.Scatter3d(x=x, y=y, z=z, mode='markers+lines', marker=dict(size=5), line=dict(color='blue'))])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))
    fig.show()
    return 0
if __name__=="__main__":
    main()
    print("sdsdasdas")