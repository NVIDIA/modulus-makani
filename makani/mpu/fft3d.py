import math
import torch
from torch import nn
import torch.nn.functional as F

from makani.utils import comm
from modulus.distributed.utils import compute_split_shapes
from torch_harmonics.distributed import distributed_transpose_azimuth as distributed_transpose_w
from torch_harmonics.distributed import distributed_transpose_polar as distributed_transpose_h

# 3D routines
# forward
class RealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nd,
                 nh,
                 nw,
                 ldmax = None,
                 lhmax = None,
                 lwmax = None):
        super(RealFFT3, self).__init__()
        
        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)
        
        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

    def forward(self, x):
        x = torch.fft.rfftn(x, s=(self.nd, self.nh, self.nw), dim=(-3, -2, -1), norm="ortho")
        
        # truncate in w
        x = x[..., :self.lwmax]
        
        # truncate in h
        x = torch.cat([x[..., :self.lhmax_high, :], x[..., -self.lhmax_low:, :]], dim=-2)
        
        # truncate in d
        x = torch.cat([x[..., :self.ldmax_high, :, :], x[..., -self.ldmax_low:, :, :]], dim=-3)
        
        return x

    
class DistributedRealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nd,
                 nh,
                 nw,
                 ldmax = None,
                 lhmax = None,
                 lwmax = None):
        super(DistributedRealFFT3, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)
        
        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

        # shapes, we assume the d-dim is always local
        self.lat_shapes = compute_split_shapes(self.nh, self.comm_size_h)
        self.lon_shapes = compute_split_shapes(self.nw, self.comm_size_w)
        self.l_shapes = compute_split_shapes(self.lhmax, self.comm_size_h)
        self.m_shapes = compute_split_shapes(self.lwmax, self.comm_size_w)


    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # make sure input is 5D
        assert(x.dim() == 5)
        
        # store number of chans
        num_chans = x.shape[1]
        
        # h and w is split. First we make w local by transposing into channel dim
        if self.comm_size_w > 1:
            x = distributed_transpose_w.apply(x, (1, -1), self.lon_shapes)
        
        # do first 2D FFT
        x = torch.fft.rfft2(x, s=(self.nd, self.nw), dim=(-3, -1), norm="ortho")
        
        # truncate width-modes
        x = x[..., :self.lwmax]
        
        # truncate depth-modes
        x = torch.cat([x[..., :self.ldmax_high, :, :],
                       x[..., -self.ldmax_low:, :, :]], dim=-3)
        
        # transpose: after this, m is split and c is local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose_w.apply(x, (-1, 1), chan_shapes)
            
        # transpose: after this, c is split and h is local
        if self.comm_size_h > 1:
            x = distributed_transpose_h.apply(x, (1, -2), self.lat_shapes)

        # do second FFT:
        x = torch.fft.fft(x, n=self.nh, dim=-2, norm="ortho")
        
        # truncate the modes
        x = torch.cat([x[..., :self.lhmax_high, :], 
                       x[..., -self.lhmax_low:, :]], dim=-2)
        
        # transpose: after this, l is split and c is local
        if self.comm_size_h > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_h)
            x = distributed_transpose_h.apply(x, (-2, 1), chan_shapes)

        return x
        
    
class InverseRealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nd,
                 nh,
                 nw,
                 ldmax = None,
                 lhmax = None,
                 lwmax = None):
        super(InverseRealFFT3, self).__init__()

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)
        
        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

    def forward(self, x):
           
        # pad in d 
        if (self.ldmax < self.nd):
            # pad
            xh = x[..., :self.ldmax_high, :, :]
            xl = x[..., -self.ldmax_low:, :, :]
            xhp = F.pad(xh, (0,0,0,0,0,self.nd-self.ldmax))
            x = torch.cat([xhp, xl], dim=-3)
            
        # pad in h
        if (self.lhmax < self.nh):
            # pad
            xh = x[..., :self.lhmax_high, :]
            xl = x[..., -self.lhmax_low:, :]
            xhp = F.pad(xh, (0,0,0,self.nh-self.lhmax))
            x = torch.cat([xhp, xl], dim=-2)
        
        x = torch.fft.irfftn(x, s=(self.nd, self.nh, self.nw), dim=(-3, -2, -1), norm="ortho")
        
        return x
        

class DistributedInverseRealFFT3(nn.Module):
    """
    Helper routine to wrap FFT similarly to the SHT
    """
    def __init__(self,
                 nd,
                 nh,
                 nw,
                 ldmax = None,
                 lhmax = None,
                 lwmax = None):
        super(DistributedInverseRealFFT3, self).__init__()

        # get the comms grid:
        self.comm_size_h = comm.get_size("h")
        self.comm_size_w = comm.get_size("w")
        self.comm_rank_w = comm.get_rank("w")

        # dimensions
        self.nd = nd
        self.nh = nh
        self.nw = nw
        self.ldmax = min(ldmax or self.nd, self.nd)
        self.lhmax = min(lhmax or self.nh, self.nh)
        self.lwmax = min(lwmax or self.nw // 2 + 1, self.nw // 2 + 1)
        
        # half-modes
        self.ldmax_high = math.ceil(self.ldmax / 2)
        self.ldmax_low = math.floor(self.ldmax / 2)
        self.lhmax_high = math.ceil(self.lhmax / 2)
        self.lhmax_low = math.floor(self.lhmax / 2)

        # shapes, we assume the d-dim is always local
        self.lat_shapes = compute_split_shapes(self.nh, self.comm_size_h)
        self.lon_shapes = compute_split_shapes(self.nw, self.comm_size_w)
        self.l_shapes = compute_split_shapes(self.lhmax, self.comm_size_h)
        self.m_shapes = compute_split_shapes(self.lwmax, self.comm_size_w)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # make sure input is 5D
        assert(x.dim() == 5)

        # store number of chans
        num_chans = x.shape[1]

        # transpose: after that, channels are split, lh is local:
        if self.comm_size_h > 1:
            x = distributed_transpose_h.apply(x, (1, -2), self.l_shapes)
            
        # we should pad the middle here manually, so that the inverse FFT is correct
        if self.lhmax < self.nh:
            xh = x[..., :self.lhmax_high, :]
            xl = x[..., -self.lhmax_low:, :]
            xhp = F.pad(xh, (0, 0, 0, self.nh-self.lhmax), mode="constant")
            x = torch.cat([xhp, xl], dim=-2)
            
        if self.ldmax < self.nd:
            xh = x[..., :self.ldmax_high, :, :]
            xl = x[..., -self.ldmax_low:, :, :]
            xhp = F.pad(xh, (0, 0, 0, 0, 0, self.nd-self.ldmax), mode="constant")
            x = torch.cat([xhp, xl], dim=-3)
        
        # do first fft
        x = torch.fft.ifft2(x, s=(self.nd, self.nh), dim=(-3, -2), norm="ortho")

        if self.comm_size_h > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_h)
            x = distributed_transpose_h.apply(x, (-2, 1), chan_shapes)

        # transpose: after this, channels are split and m is local
        if self.comm_size_w > 1:
            x = distributed_transpose_w.apply(x, (1, -1), self.m_shapes)

        # apply the inverse (real) FFT
        x = torch.fft.irfft(x, n=self.nw, dim=-1, norm="ortho")

        # transpose: after this, m is split and channels are local
        if self.comm_size_w > 1:
            chan_shapes = compute_split_shapes(num_chans, self.comm_size_w)
            x = distributed_transpose_w.apply(x, (-1, 1), chan_shapes)

        return x
