# ------------------------------------------------------------------------------
# Portions of this code are from
# CornerNet (https://github.com/princeton-vl/CornerNet)
# Copyright (c) 2018, University of Michigan
# Licensed under the BSD 3-Clause License
# ------------------------------------------------------------------------------
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import torch
import torch.nn as nn
from .utils import _tranpose_and_gather_feat as _transpose_and_gather_feat
import torch.nn.functional as F
from utils.image import transform_preds_train
ep = 1e-10
def _slow_neg_loss(pred, gt):
  '''focal loss from CornerNet'''
  pos_inds = gt.eq(1)
  neg_inds = gt.lt(1)

  neg_weights = torch.pow(1 - gt[neg_inds], 4)

  loss = 0
  pos_pred = pred[pos_inds]
  neg_pred = pred[neg_inds]

  pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
  neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if pos_pred.nelement() == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss


def _neg_loss(pred, gt):
  ''' Modified focal loss. Exactly the same as CornerNet.
      Runs faster and costs a little bit more memory
    Arguments:
      pred (batch x c x h x w)
      gt_regr (batch x c x h x w)
  '''
  pos_inds = gt.eq(1).float()
  neg_inds = gt.lt(1).float()

  neg_weights = torch.pow(1 - gt, 4)

  loss = 0

  pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
  neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

  num_pos  = pos_inds.float().sum()
  pos_loss = pos_loss.sum()
  neg_loss = neg_loss.sum()

  if num_pos == 0:
    loss = loss - neg_loss
  else:
    loss = loss - (pos_loss + neg_loss) / num_pos
  return loss

def _not_faster_neg_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()    
    num_pos  = pos_inds.float().sum()
    neg_weights = torch.pow(1 - gt, 4)

    loss = 0
    trans_pred = pred * neg_inds + (1 - pred) * pos_inds
    weight = neg_weights * neg_inds + pos_inds
    all_loss = torch.log(1 - trans_pred) * torch.pow(trans_pred, 2) * weight
    all_loss = all_loss.sum()

    if num_pos > 0:
        all_loss /= num_pos
    loss -=  all_loss
    return loss

def _slow_reg_loss(regr, gt_regr, mask):
    num  = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr)

    regr    = regr[mask]
    gt_regr = gt_regr[mask]
    
    regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss

def _reg_loss(regr, gt_regr, mask):
  ''' L1 regression loss
    Arguments:
      regr (batch x max_objects x dim)
      gt_regr (batch x max_objects x dim)
      mask (batch x max_objects)
  '''
  num = mask.float().sum()
  mask = mask.unsqueeze(2).expand_as(gt_regr).float()

  regr = regr * mask
  gt_regr = gt_regr * mask
    
  regr_loss = nn.functional.smooth_l1_loss(regr, gt_regr, size_average=False)
  regr_loss = regr_loss / (num + 1e-4)
  return regr_loss

class FocalLoss(nn.Module):
  '''nn.Module warpper for focal loss'''
  def __init__(self):
    super(FocalLoss, self).__init__()
    self.neg_loss = _neg_loss

  def forward(self, out, target):
    return self.neg_loss(out, target)

class RegLoss(nn.Module):
  '''Regression loss for an output tensor
    Arguments:
      output (batch x dim x h x w)
      mask (batch x max_objects)
      ind (batch x max_objects)
      target (batch x max_objects x dim)
  '''
  def __init__(self):
    super(RegLoss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    loss = _reg_loss(pred, target, mask)
    return loss

class RegL1Loss(nn.Module):
  def __init__(self):
    super(RegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class NormRegL1Loss(nn.Module):
  def __init__(self):
    super(NormRegL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    pred = pred / (target + 1e-4)
    target = target * 0 + 1
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class RegWeightedL1Loss(nn.Module):
  def __init__(self):
    super(RegWeightedL1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.float()
    # loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    loss = F.l1_loss(pred * mask, target * mask, size_average=False)
    loss = loss / (mask.sum() + 1e-4)
    return loss

class L1Loss(nn.Module):
  def __init__(self):
    super(L1Loss, self).__init__()
  
  def forward(self, output, mask, ind, target):
    pred = _transpose_and_gather_feat(output, ind)
    mask = mask.unsqueeze(2).expand_as(pred).float()
    loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
    return loss

class BinRotLoss(nn.Module):
  def __init__(self):
    super(BinRotLoss, self).__init__()
  
  def forward(self, output, mask, ind, rotbin, rotres):
    pred = _transpose_and_gather_feat(output, ind)
    loss = compute_rot_loss(pred, rotbin, rotres, mask)
    return loss

def compute_res_loss(output, target):
    return F.smooth_l1_loss(output, target, reduction='elementwise_mean')

# TODO: weight
def compute_bin_loss(output, target, mask):
    mask = mask.expand_as(output)
    output = output * mask.float()
    return F.cross_entropy(output, target, reduction='elementwise_mean')

def compute_rot_loss(output, target_bin, target_res, mask):
    # output: (B, 128, 8) [bin1_cls[0], bin1_cls[1], bin1_sin, bin1_cos, 
    #                 bin2_cls[0], bin2_cls[1], bin2_sin, bin2_cos]
    # target_bin: (B, 128, 2) [bin1_cls, bin2_cls]
    # target_res: (B, 128, 2) [bin1_res, bin2_res]
    # mask: (B, 128, 1)
    # import pdb; pdb.set_trace()
    output = output.view(-1, 8)
    target_bin = target_bin.view(-1, 2)
    target_res = target_res.view(-1, 2)
    mask = mask.view(-1, 1)
    loss_bin1 = compute_bin_loss(output[:, 0:2], target_bin[:, 0], mask)
    loss_bin2 = compute_bin_loss(output[:, 4:6], target_bin[:, 1], mask)
    loss_res = torch.zeros_like(loss_bin1)
    if target_bin[:, 0].nonzero().shape[0] > 0:
        idx1 = target_bin[:, 0].nonzero()[:, 0]
        valid_output1 = torch.index_select(output, 0, idx1.long())
        valid_target_res1 = torch.index_select(target_res, 0, idx1.long())
        loss_sin1 = compute_res_loss(
          valid_output1[:, 2], torch.sin(valid_target_res1[:, 0]))
        loss_cos1 = compute_res_loss(
          valid_output1[:, 3], torch.cos(valid_target_res1[:, 0]))
        loss_res += loss_sin1 + loss_cos1
    if target_bin[:, 1].nonzero().shape[0] > 0:
        idx2 = target_bin[:, 1].nonzero()[:, 0]
        valid_output2 = torch.index_select(output, 0, idx2.long())
        valid_target_res2 = torch.index_select(target_res, 0, idx2.long())
        loss_sin2 = compute_res_loss(
          valid_output2[:, 6], torch.sin(valid_target_res2[:, 1]))
        loss_cos2 = compute_res_loss(
          valid_output2[:, 7], torch.cos(valid_target_res2[:, 1]))
        loss_res += loss_sin2 + loss_cos2
    return loss_bin1 + loss_bin2 + loss_res

# class GasussionLoss(nn.Module):
#     def __init__(self):
#         super(GasussionLoss, self).__init__()
      
#     def forward(self, output, mask, ind, target):
#         pred = _transpose_and_gather_feat(output, ind)
#         mask = mask.unsqueeze(2).expand_as(pred).float()
        
#         loss = torch.exp((pred[0] - target [0])**2 / pred[1])
#         loss = torch.sum(loss * mask)

#         #loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
#         return loss


class ConfidenceLoss(nn.Module):
    def __init__(self):
        super(ConfidenceLoss, self).__init__()
      
    def forward(self, output, mask, ind, target):
      pred = _transpose_and_gather_feat(output, ind)
      mask = mask.unsqueeze(2).expand_as(target).float()
      residual  =  - torch.abs(pred[:,:,0:1] - target)/target
      confidence = 1 - torch.exp(residual)
      loss_depth =       F.l1_loss(pred[:,:,0:1]*mask, target * mask, reduction='elementwise_mean')
      loss_confidence  =  F.l1_loss(pred[:,:,1:2]*mask,confidence *mask)
      loss = loss_depth + loss_confidence
      #loss = F.l1_loss(pred * mask, target * mask, reduction='elementwise_mean')
      return loss

class DisentangelLOSS(nn.Module):
    def __init__(self):
      super(DisentangelLOSS,self).__init__()
      self.bboxloss =  F.l1_loss
      
    def forward(self,output_1,output_2,batch,group):
        #import pdb;pdb.set_trace()
        target = batch['gt_bbox']
        inds   = batch['ind']
        mask   = batch['reg_mask']
        num  = torch.sum(batch['sample']).float()
        calib = batch['meta']['calib'].cpu().numpy()
        c = batch['meta']['c'].cpu().numpy()
        s = batch['meta']['s'].cpu().numpy()
        down_ration =4
        if(group == 'loc'):
           depth = _transpose_and_gather_feat(output_1, inds)

           depth =  depth *16.32 + 28.01
           h = batch['gt_dimension'][:,:,0:1]
           offset = _transpose_and_gather_feat(output_2,inds)
           ground_uv = batch['gt_uv']+offset
           ground_uv_transform = transform_preds_train(ground_uv, c, s, (320, 96))
           
           location = trans_to_location(depth,ground_uv_transform,calib,h)
           
           group_dim = batch['gt_dimension']
           group_rotation = batch['gt_rot_y']
           



           bbox_predict = trans_to_bbox(location,group_dim,group_rotation)
       
           gt = batch['gt_bbox']

           mask = mask.unsqueeze(2).expand_as(bbox_predict).float()
           loss = self.bboxloss(bbox_predict*mask,gt*mask,reduction='sum')
           return  loss/num
        if(group == 'rot'):
           #import pdb;pdb.set_trace()
           alpha_x = _transpose_and_gather_feat(output_1, inds)
           alpha_x_sin = alpha_x[:,:,0:1]
           alpha_x_cos = alpha_x[:,:,1:2]
           alpha_x_tan = alpha_x_sin / alpha_x_cos
           alpha_x_rad   = torch.atan(alpha_x_tan)
           
           rot = trans_to_rotation(alpha_x_rad,batch['gt_location'])
        
           group_dim = batch['gt_dimension']
           group_location =batch['gt_location']
           
           bbox_predict = trans_to_bbox(group_location,group_dim,rot)
           mask = mask.unsqueeze(2).expand_as(bbox_predict).float()

           loss = self.bboxloss(bbox_predict*mask,batch['gt_bbox']*mask,reduction = 'sum') 
           return  loss/num
        
        if(group == 'dim'):
#           import pdb;pdb.set_trace()
           dim = _transpose_and_gather_feat(output_1, inds)
          
           dim  = trans_to_dim(dim)
          
           group_location =batch['gt_location']
           group_rotation = batch['gt_rot_y']
           
           
           bbox_predict = trans_to_bbox(group_location,dim,group_rotation)
           mask = mask.unsqueeze(2).expand_as(bbox_predict).float()

           loss = self.bboxloss(bbox_predict*mask,batch['gt_bbox']*mask,reduction='sum')
           return  loss/num

def trans_to_bbox(loc,dim,rotation_y):
    '''
    loc:batch,number,3
    dim batch,number,3
    rot batch,numer,2
    '''
 
    c, s = torch.cos(rotation_y), torch.sin(rotation_y)
    ones  = torch.ones(c.shape[0],c.shape[1],1).cuda().float()
    zeros  = torch.zeros(c.shape[0],c.shape[1],1).cuda().float()
    rotMat =torch.cat((c,zeros,s,zeros,ones,zeros,-s,zeros,-c),2)
    rotMat = rotMat.reshape(c.shape[0],c.shape[1],3,3)

    #R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
    l, w, h = dim[:,:,2:], dim[:,:,1:2], dim[:,:,0:1]
    
    corners = torch.cat((l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2,zeros,zeros,zeros,zeros,-h,-h,-h,-h,w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2),2).reshape(c.shape[0],c.shape[1],3,8)

    #conners = torch.cat((x_corners,y_corners,zeros),0).reshape(1,-1)
    

    #corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
    corners_3d = torch.matmul(rotMat,corners).permute(0,1,3,2)
    loc = loc.unsqueeze(2).expand_as(corners_3d)
    corners_3d = corners_3d + loc
    return corners_3d.reshape(c.shape[0],c.shape[1],24)

def trans_to_rotation(alpha_z,loc):
  x, y, z = loc[:,:,0:1], loc[:,:,1:2], loc[:,:,2:]
  offset = torch.atan(x/(z+ep))
  yaw = alpha_z+offset + np.pi/2
  return yaw
  



def trans_to_dim(dim):
  eh , ew , el = dim[:,:,0] ,dim[:,:,1] ,dim[:,:,2]

  h_mean ,w_mean ,l_mean = 1.63,1.53,3.88

  dim[:,:,0] = h_mean*torch.exp(eh)
  dim[:,:,1] = w_mean*torch.exp(ew)
  dim[:,:,2] = l_mean*torch.exp(el)

  return dim




def trans_to_location(depth,pt_2d, P,h):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  #import pdb;pdb.set_trace()
  P = torch.tensor(P).float().cuda()
  z = depth
  x = (pt_2d[:,:,0:1] * depth[:,:,:]  - P[:,0:1, 2:3] * depth[:,:,:] ) / P[:,0:1, 0:1]
  y = (pt_2d[:,:,0:1] * depth[:,:,:]  - P[:,0:1, 2:3] * depth[:,:,:]) / P[:,1:2, 1:2] +0.5*h
  pt_3d = torch.cat((x,y,z),2)
  return pt_3d