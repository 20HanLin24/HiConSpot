import sys
import os
  
  

import torch
import torch.nn as nn

from .....model.v2.HiConSpot.tools.pointops.functions import pointops


class PointTransformerLayer(nn.Module):
    """点云注意力机制层"""

    def __init__(self, in_planes, out_planes, share_planes=8, nsample=16):
        super().__init__()
          
        self.mid_planes = mid_planes = out_planes // 1
          
        self.out_planes = out_planes
          
        self.share_planes = share_planes
          
        self.nsample = nsample
          
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
          
        self.linear_p = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3), nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
          
        self.linear_w = nn.Sequential(nn.BatchNorm1d(mid_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, mid_planes // share_planes),
                                      nn.BatchNorm1d(mid_planes // share_planes), nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes // share_planes, out_planes // share_planes))
          
        self.softmax = nn.Softmax(dim=1)

    def forward(self, pxo):
        """
        pxo:[coords(n, 3), feats(n, c), offsets(b,)，None]
        n、c是不确定的，会随着过渡层而变化。c==in_planes
        【假设】batch_size==1，stride==4，m1=n/stride，nsample=16，n==man_num
        enc1输入：[coords(n, 3), feats(n, c), offsets(b,)，None]，n=max_num，c=64，b=1
        enc2输入：[coords(m1, 3), feats(m1, c), offsets(b,)，None]，m1=n/stride，c=128，b=1
        enc3输入：[coords(m2, 3), feats(m2, c), offsets(b,)，None]，m2=m1/stride=n/16，c=256，b=1
        enc4输入：[coords(m3, 3), feats(m3, c), offsets(b,)，None]，m3=m2/stride=n/64，c=512，b=1
        enc5输入：[coords(m4, 3), feats(m4, c), offsets(b,)，None]，m4=m3/stride=n/256，c=1024，b=1
        """
          
        p, x, o, knn_idx = pxo    
          
        x_q, x_k, x_v = self.linear_q(x), self.linear_k(x), self.linear_v(x)    
          
          
          
        x_k, knn_idx = pointops.queryandgroup(self.nsample, p, p, x_k, knn_idx, o, o, use_xyz=True)    
        x_v = pointops.queryandgroup(self.nsample, p, p, x_v, knn_idx, o, o, use_xyz=False)    
          
        p_r, x_k = x_k[:, :, 0:3], x_k[:, :, 3:]    
          
          
        for i, layer in enumerate(self.linear_p):
              
              
            p_r = layer(p_r.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i == 1 else layer(p_r)

          
          
          
          
          
          
        w = x_k - x_q.unsqueeze(1) + p_r.view(p_r.shape[0], p_r.shape[1], self.out_planes // self.mid_planes,
                                              self.mid_planes).sum(2)    
          
        for i, layer in enumerate(self.linear_w):
              
              
            w = layer(w.transpose(1, 2).contiguous()).transpose(1, 2).contiguous() if i % 3 == 0 else layer(w)
        w = self.softmax(w)    
          
        n, nsample, c = x_v.shape
        s = self.share_planes    
          
          
        x = ((x_v + p_r).view(n, nsample, s, c // s) * w.unsqueeze(2)).sum(1).view(n, c)
          
        return x, knn_idx    


class TransitionDown(nn.Module):
    """点云过渡下采样，来自pointnet++
    点云下采样：通过减少点的数量来降低计算复杂度，同时保留关键特征信息。
    特征聚合：使用局部邻域信息来丰富每个点的特征表示。
    多尺度处理：支持通过扇区分割（sectorized）进行多尺度特征提取。"""

    def __init__(self, in_planes, out_planes, stride=1, nsample=16, num_sector=1):
        """
        in_planes：输入特征维度。
        out_planes：输出特征维度。
        stride：下采样率，控制输出点数量（默认 1 表示不进行下采样）。
        nsample：每个点的邻居采样数。
        num_sector：扇区数量，用于多尺度处理（默认为 1，即不使用扇区）。
        """
        super().__init__()
          
          

        self.stride, self.nsample, self.num_sector = stride, nsample, num_sector
        if stride != 1:
            self.linear = nn.Linear(3 + in_planes, out_planes, bias=False)
            self.pool = nn.MaxPool1d(nsample)
        else:
            self.linear = nn.Linear(in_planes, out_planes, bias=False)
        self.bn = nn.BatchNorm1d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        """
        pxo:coords(n, 3), cat(coords,feats)及其new_feats(n, c), offsets(b)
            c==in_planes==输入的节点特征张量的特征维度
        第一次输入，即输入enc1[0]
        pxo：coords(max_num, 3)，(coords,feats)(max_num, 9)，offsets(1,)
        输入enc2[0]，如果stride=4，且c，则 m=n/4，n=max_num，此时c=64

        max_num/stride**(i-1)，stride==1时，不执行最远点采样。
        后续模块串联时，stride==4，enci(i>=2)，n==max_num/stride**(i-2)
        最远点采样：采样个数m==new_offsets[-1]==max_num/stride**(i-1)，在pi点集(max_num/stride**(i-1), 3)中采样max_num/stride**i个点，获取它们的在pi的位置id
        节点特征张量矩阵的节点数n是会变动的。上面的new_offsets指的是该方法中用到的变量
        """
          
        p, x, o = pxo    
        if self.stride != 1:
              
            assert isinstance(o, torch.Tensor), f"Expected tensor, got {type(o)}"
            assert o.device.type == 'cuda', f"Tensor is not on GPU: {o.device}"
            assert o.numel() > 0, "Tensor is empty!"
              
              
              
            n_o, count = [o[0].item() // self.stride], o[0].item() // self.stride

              
            for i in range(1, o.shape[0]):
                count += (o[i].item() - o[i - 1].item()) // self.stride
                n_o.append(count)

            n_o = torch.cuda.IntTensor(n_o)    

              
            if self.num_sector > 1 and self.training:
                  
                idx = pointops.sectorized_fps(p, o, n_o, self.num_sector)    
            else:
                  
                idx = pointops.furthestsampling(p, o, n_o)    
            n_p = p[idx.long(), :]    
              
              
            x, _ = pointops.queryandgroup(self.nsample, p, n_p, x, None, o, n_o, use_xyz=True)    
              
            x = self.relu(self.bn(self.linear(x).transpose(1, 2).contiguous()))    
              
            x = self.pool(x).squeeze(-1)    
            p, o = n_p, n_o    
              
            return [p, x, o], idx.long()    
        else:
              
              
            x = self.relu(self.bn(self.linear(x)))    
        return [p, x, o]    


class TransitionUp(nn.Module):
    """
    点云过渡上采样
    """

    def __init__(self, in_planes, out_planes=None):
        super().__init__()
        if out_planes is None:
            self.linear1 = nn.Sequential(nn.Linear(2 * in_planes, in_planes), nn.BatchNorm1d(in_planes),
                                         nn.ReLU(inplace=True))    
            self.linear2 = nn.Sequential(nn.Linear(in_planes, in_planes), nn.ReLU(inplace=True))    
        else:
            self.linear1 = nn.Sequential(nn.Linear(out_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))    
            self.linear2 = nn.Sequential(nn.Linear(in_planes, out_planes), nn.BatchNorm1d(out_planes),
                                         nn.ReLU(inplace=True))    

    def forward(self, pxo1, pxo2=None):
        """
        dec5[0]:
        pxo1=new_coords(m4, 3), new_feats(m4, c), new_offsets4(b,), m4==new_offsets4[-1]==max_num/stride**4，c==1024==planes[4]
        pxo2=None
        后续deci[0]，i=[1.2.3.4]:
        pxo1=new_coords(mi-1, 3), new_feats(mi-1, c1), new_offsetsi-1(b,), mi-1==new_offsetsi-1[-1]==max_num/stride**(i-1)，c1==planes[i-1]=[64,128,256,512]
        px02=new_coords(mi, 3), new_feats(mi, c2), new_offsetsi(b,), mi==new_offsetsi[-1]==max_num/stride**(i)，c2==planes[i]=[128,256,512,1024]
        m0==max_num
        """
          
        if pxo2 is None:
            _, x, o = pxo1    
            x_tmp = []
              
            for i in range(o.shape[0]):
                if i == 0:
                      
                      
                    s_i, e_i, cnt = 0, o[0], o[0]
                else:
                      
                      
                    s_i, e_i, cnt = o[i - 1], o[i], o[i] - o[i - 1]

                x_b = x[s_i:e_i, :]    
                  
                  
                x_b = torch.cat((x_b, self.linear2(x_b.sum(0, True) / cnt).repeat(cnt, 1)), 1)    
                x_tmp.append(x_b)
            x = torch.cat(x_tmp, 0)    
            x = self.linear1(x)    
        else:
            p1, x1, o1 = pxo1    
            p2, x2, o2 = pxo2    
              
              
            x = self.linear1(x1) + pointops.interpolation(p2, p1, self.linear2(x2), o2, o1)    
        return x    


class PointTransformerBlock(nn.Module):
    expansion = 1
    """PointTransformerBlock层仅对feats特征张量的特征维度进行处理，不改变节点数"""

    def __init__(self, in_planes, planes, share_planes=8, nsample=16):
        super(PointTransformerBlock, self).__init__()
        self.linear1 = nn.Linear(in_planes, planes, bias=False)    
        self.bn1 = nn.BatchNorm1d(planes)
        self.transformer2 = PointTransformerLayer(planes, planes, share_planes, nsample)    
        self.bn2 = nn.BatchNorm1d(planes)
        self.linear3 = nn.Linear(planes, planes * self.expansion, bias=False)    
        self.bn3 = nn.BatchNorm1d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, pxo):
        """
        pxo:[coords(n, 3), feats(n, c), offsets(b,)，None]
        n、c是不确定的，会随着过渡层而变化。c==in_planes
        【假设】batch_size==1，stride==4，m1=n/stride，nsample=16，n==man_num
        enc1输入：[coords(n, 3), feats(n, c), offsets(b,)，None]，n=max_num，c=64，b=1
        enc2输入：[coords(m1, 3), feats(m1, c), offsets(b,)，None]，m1=n/stride，c=128，b=1
        enc3输入：[coords(m2, 3), feats(m2, c), offsets(b,)，None]，m2=m1/stride=n/16，c=256，b=1
        enc4输入：[coords(m3, 3), feats(m3, c), offsets(b,)，None]，m3=m2/stride=n/64，c=512，b=1
        enc5输入：[coords(m4, 3), feats(m4, c), offsets(b,)，None]，m4=m3/stride=n/256，c=1024，b=1"""
          
        p, x, o, knn_idx = pxo    
        identity = x
        x = self.relu(self.bn1(self.linear1(x)))    
          
        x, knn_idx = self.transformer2([p, x, o, knn_idx])    
        x = self.relu(self.bn2(x))
        x = self.bn3(self.linear3(x))    
        x += identity
        x = self.relu(x)
        return [p, x, o, knn_idx]    


