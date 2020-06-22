import torch
from torch import nn
from torch.nn import functional as F
from .utils import _tranpose_and_gather_feat
#from .quaternion import points_quaternion_decode, orientation_quaternion_decode
#from .giou_3d_loss import giou_3d_loss
from .IoU import *

PI = 3.1415926

def depth_decode(depth_offset):
    device = depth_offset.device
    depth_pre = torch.tensor([28.01, 16.32]).to(device=device)
    depth = depth_offset * depth_pre[1] + depth_pre[0]
    return depth


def locations_decode(ct_3d, ct_3d_offsets, K, depth, trans_output):
    '''
    :param ct_3d: in (x,y) form on feature map, shape = (N, 3), the first column
    stores the image index
    :param ct_3d_offsets: in (x,y) form, shape = (N, 2)
    :param K: camera intrinsic matrix, shape = (N, 3, 4)
    :param depth: shape = (N, 1)
    :param trans_output: transformation matrix ---- from image to feature map
    :return: object locations, shape (N, 3)
    '''
    #import pdb;pdb.set_trace()
    device = ct_3d.device

    N_objs = ct_3d.shape[1]
    ct_3d = ct_3d.view(-1, 2)

    N = ct_3d.shape[0]
    N_img = K.shape[0]
    # img_idx = ct_3d[:, 0].long()
    ranges = torch.arange(0, N_img).view(N_img, -1)
    img_idx = ranges.repeat(1, N_objs).flatten()

    # ct_3d_ = torch.flip(ct_3d[:, 1:], (1,)).to(dtype=torch.float32)
    ct_3d_refine = ct_3d.float() + ct_3d_offsets
    extend_trans = torch.tensor([0., 0., 1.]).to(device=device)
    extend_trans = extend_trans.repeat(N_img, 1).view(N_img, 1, -1)
    trans_square = torch.cat((trans_output, extend_trans), dim=1)
    trans_inv = torch.zeros(trans_square.shape).to(device =device)
    for i in range(trans_square.shape[0]):
        trans_inv[i] = torch.inverse(trans_square[i,:,:])
    trans_inv_mapping = trans_inv[img_idx]
    ct_3d_homo = torch.cat((ct_3d_refine, torch.ones(N, 1).to(device=device)), dim=1)
    ct_3d_homo = ct_3d_homo.view(N, -1, 1)
    ct_3d_img = torch.matmul(trans_inv_mapping, ct_3d_homo)
    ct_3d_img = ct_3d_img * depth.view(N, 1, -1)
    K_inv = torch.zeros(K.shape[0],3,3).to(device =device)
    for i  in range(K.shape[0]):
        K_inv[i,:,:] = torch.inverse(K[i, :, :3])
    K_inv_mapping = K_inv[img_idx]

    locations = torch.matmul(K_inv_mapping, ct_3d_img)

    return locations.squeeze(2)


def orientation_decode(ori, locations, flip_mask=None):
    '''
    :param pred_ori: location orientations (N, 2)
    :param locations:  object locations (N, 3)
    :return: global orientations
    '''
    offsets = torch.atan(locations[:, 0] / locations[:, 2] + 1e-7)
    alpha = torch.atan(ori[:, 0] / (ori[:, 1] + 1e-7))

    cos_pos_idx = (ori[:, 1] >= 0).nonzero()
    cos_neg_idx = (ori[:, 1] < 0).nonzero()

    alpha[cos_pos_idx] = alpha[cos_pos_idx] - PI / 2
    alpha[cos_neg_idx] = alpha[cos_neg_idx] + PI / 2

    rotys = alpha + offsets

    roty_large_idx = (rotys > PI).nonzero()
    roty_small_idx = (rotys < -PI).nonzero()

    if len(roty_large_idx) != 0:
        rotys[roty_large_idx] = rotys[roty_large_idx] - 2*PI
    if len(roty_small_idx) != 0:
        rotys[roty_small_idx] = rotys[roty_small_idx] + 2*PI

    fm = flip_mask.view(-1, 1).squeeze(1)
    rotys_flip = fm.float() * rotys

    rotys_flip_pos_idx = rotys_flip > 0
    rotys_flip_neg_idx = rotys_flip < 0
    rotys_flip[rotys_flip_pos_idx] -=  PI
    rotys_flip[rotys_flip_neg_idx] +=  PI

    rotys_all = fm.float() * rotys_flip + (1 - fm.float()) * rotys

    return rotys_all.unsqueeze(1), alpha.unsqueeze(1)


def dimension_decode(cls_ids, dim_offsets):
    device = dim_offsets.device
    N = dim_offsets.shape[0]

    # Todo: remove hard code here
    dims_avg = torch.tensor([[1.63, 1.53, 3.88],
                             [1.73, 0.67, 0.88],
                             [1.70, 0.58, 1.78]]).to(device=device).unsqueeze(0)
    cls_ids = cls_ids.squeeze(1).long()
    dims_select = dims_avg[:, cls_ids, :]
    dims = torch.exp(dim_offsets) * dims_select.squeeze(0)

    return dims


def points_decode(rotys, dimensions, locations):
    '''
    :param rotys: global orientation rot_y, shape(N, 1)
    :param dimensions: object dimensions, shape (N, 3)
    :param locations: object locations, shape (N, 3)
    :return: 3d points in camera coordinate
    '''
    '''
    rotation matrix is (3, 3), in batch version is R_theta.shape = (N, 3, 3)
    8 points in object coordinate is corners.shape = (N, 3, 8)
    totally results in 24 values in N batch size (N, 3, 8)
    '''
    # import pdb;pdb.set_trace()asdasd



    device = rotys.device
    N = rotys.shape[0]
    I = torch.eye(3, 3).to(device=device)

    cos_rotys = torch.cos(rotys)
    sin_rotys = torch.sin(rotys)
    identity_flat = I.repeat(N, 1).view(N, -1)
    identity_flip_flat = torch.flip(I, dims=(0,)).repeat(N, 1).view(N, -1)
    rot_matrix_flat = cos_rotys * identity_flat + sin_rotys * identity_flip_flat
    rot_matrix_flat[:, 4] = 1.
    rot_matrix_flat[:, 6] *= -1.
    rot_matrix_flat = rot_matrix_flat.view(N, 3, 3)
    #import pdb;pdb.set_trace()
    # we need to make sure the order (x,y,z) ---- (l,h,w)
    #dims = torch.roll(dimensions, 1, 1)
    dims = torch.zeros(dimensions.shape[0],dimensions.shape[1]).to(device =device)
    dims[:,0],dims[:,1],dims[:,2] = dimensions[:,2],dimensions[:,1],dimensions[:,0] 
    dims = dims.view(-1, 1).repeat(1, 8)
    
    dims[::3, :4], dims[2::3, :4] = 0.5 * dims[::3, :4], 0.5 * dims[2::3, :4]
    dims[::3, 4:], dims[2::3, 4:] = -0.5 * dims[::3, 4:], -0.5 * dims[2::3, 4:]
    dims[1::3, :4], dims[1::3, 4:] = 0., -dims[1::3, 4:]
    # index = torch.tensor([[4, 0, 1, 2, 3, 5, 6, 7],
    #                       [4, 5, 0, 1, 6, 7, 2, 3],
    #                       [4, 5, 6, 0, 1, 2, 3, 7]]).repeat(N, 1).to(device=device)
    index = torch.tensor([[0, 1, 4, 5, 2, 3, 6, 7],
                          [0, 1, 2, 3, 4, 5, 6, 7],
                          [0, 4, 5, 1, 2, 6, 7, 3]]).repeat(N, 1).to(device=device)
    bbox_3d_object = torch.gather(dims, 1, index)
    bbox_3d = torch.matmul(rot_matrix_flat, bbox_3d_object.view(N, 3, -1))
    bbox_3d += locations.unsqueeze(-1).repeat(1, 1, 8)

    return bbox_3d


class Reg3dLoss(nn.Module):
    def __init__(self):
        super(Reg3dLoss, self).__init__()

    def forward(self, output, target):
        batch_size = len(target['img_id'])

        ind = target['ind']
        cls_ids = target['cls_id']
        mask = target['reg_mask']
        flip_mask = target['flip_mask']
        target_3dct = target['3dct']
        target_reg = target['reg_3d']
        target_roty = target['roty']
        # print(target['img_id'])
        pred = _tranpose_and_gather_feat(output, ind)
        mask_all = mask.unsqueeze(2).expand_as(pred).float()
        pred = pred * mask_all
        target_reg = target_reg * mask_all

        # squeeze all-zero elements
        # Todo: this is not correct
        # pred, target_reg = pred[pred != 0], target_reg[target_reg != 0]
        pred, target_reg = pred.view(-1, 8), target_reg.view(-1, 8)
        cls_ids = cls_ids.view(-1, 1)
        # target_hm_select = hm_select(target_hm, batch_size, ind.shape[1])
        # indices = torch.tensor([0, 2, 3]).to(device='cuda')
        # target_hm = (target_hm == 1).nonzero()
        # target_hm = torch.index_select(target_hm, 1, indices)

        pred_ct_3doffsets = pred[:, 1:3]
        pred_depthoffsets = pred[:, 0]
        pred_ori = pred[:, 6:]
        pred_dim_offsets = pred[:, 3:6]

        gt_ct_3doffsets = target_reg[:, 1:3]
        gt_depthoffsets = target_reg[:, 0]
        gt_ori = target_reg[:, 6:]
        gt_dim_offsets = target_reg[:, 3:6]

        pred_depth = depth_decode(pred_depthoffsets)
        gt_depth = depth_decode(gt_depthoffsets)

        # target_3dct = target_3dct.view(-1, 2)
        pred_locations = locations_decode(target_3dct, pred_ct_3doffsets, target['K'],
                                          pred_depth, target['trans'])
        gt_locations = locations_decode(target_3dct, gt_ct_3doffsets, target['K'],
                                        gt_depth, target['trans'])
        pred_dimensions = dimension_decode(cls_ids, pred_dim_offsets)
        gt_dimensions = dimension_decode(cls_ids, gt_dim_offsets)

        pred_locations[:, 1] += pred_dimensions[:, 0] / 2
        gt_locations[:, 1] += gt_dimensions[:, 0] / 2

        # Todo: gt locations or pred locations here
        # Todo: need to make sure pred_rotys lies in [-pi, pi]
        pred_rotys, _ = orientation_decode(pred_ori, gt_locations, flip_mask)
        # pred_rotys, _ = orientation_quaternion_decode(pred_ori, gt_locations, flip_mask)
        pred_rotys = mask.view(-1, 1).float() * pred_rotys
        # gt_rotys, _ = orientation_quaternion_decode(gt_ori, gt_locations, flip_mask)
        # gt_rotys = target_roty[target_roty != 0].view(batch_size, -1)
        gt_rotys = (mask.float() * target_roty).view(-1, 1)

        # rotation matrix decode
        pred_bbox3d_rotys = points_decode(pred_rotys, gt_dimensions, gt_locations)
        pred_bbox3d_dims = points_decode(gt_rotys, pred_dimensions, gt_locations)
        pred_bbox3d_locs = points_decode(gt_rotys, gt_dimensions, pred_locations)
        gt_bbox3d = points_decode(gt_rotys, gt_dimensions, gt_locations)
        #import pdb;pdb.set_trace()
        # quaternion decode
        # pred_bbox3d_rotys = points_quaternion_decode(pred_rotys, gt_dimensions, gt_locations)
        # pred_bbox3d_dims = points_quaternion_decode(gt_rotys, pred_dimensions, gt_locations)
        # pred_bbox3d_locs = points_quaternion_decode(gt_rotys, gt_dimensions, pred_locations)
        # gt_bbox3d = points_quaternion_decode(gt_rotys, gt_dimensions, gt_locations)

        # loss = giou_3d_loss(pred_bbox3d, gt_bbox3d, mask) * 20
        # loss = F.smooth_l1_loss(pred_bbox3d, gt_bbox3d, reduction='sum') / (ind.shape[1] * 10)
        loss_rotys = F.l1_loss(pred_bbox3d_rotys, gt_bbox3d, reduction='sum') / (ind.shape[1] * 10)
        loss_dims = F.l1_loss(pred_bbox3d_dims, gt_bbox3d, reduction='sum') / (ind.shape[1] * 10)
        loss_locs = F.l1_loss(pred_bbox3d_locs, gt_bbox3d, reduction='sum') / (ind.shape[1] * 10)
        return loss_rotys + loss_dims + loss_locs
        #return loss_locs
    

class Reg3dLoss_IOU(nn.Module):
    def __init__(self):
        super(Reg3dLoss_IOU, self).__init__()

    def forward(self, output, target):
        batch_size = len(target['img_id'])

        ind = target['ind']
        cls_ids = target['cls_id']
        mask = target['reg_mask']
        flip_mask = target['flip_mask']
        target_3dct = target['3dct']
        target_reg = target['reg_3d']
        target_roty = target['roty']
        # print(target['img_id'])
        pred = _tranpose_and_gather_feat(output, ind)
        mask_all = mask.unsqueeze(2).expand_as(pred).float()
        pred = pred * mask_all
        target_reg = target_reg * mask_all

        # squeeze all-zero elements
        # Todo: this is not correct
        # pred, target_reg = pred[pred != 0], target_reg[target_reg != 0]
        pred, target_reg = pred.view(-1, 8), target_reg.view(-1, 8)
        cls_ids = cls_ids.view(-1, 1)
        # target_hm_select = hm_select(target_hm, batch_size, ind.shape[1])
        # indices = torch.tensor([0, 2, 3]).to(device='cuda')
        # target_hm = (target_hm == 1).nonzero()
        # target_hm = torch.index_select(target_hm, 1, indices)

        pred_ct_3doffsets = pred[:, 1:3]
        pred_depthoffsets = pred[:, 0]
        pred_ori = pred[:, 6:]
        pred_dim_offsets = pred[:, 3:6]

        gt_ct_3doffsets = target_reg[:, 1:3]
        gt_depthoffsets = target_reg[:, 0]
        gt_ori = target_reg[:, 6:]
        gt_dim_offsets = target_reg[:, 3:6]

        pred_depth = depth_decode(pred_depthoffsets)
        gt_depth = depth_decode(gt_depthoffsets)

        # target_3dct = target_3dct.view(-1, 2)
        pred_locations = locations_decode(target_3dct, pred_ct_3doffsets, target['K'],
                                          pred_depth, target['trans'])
        gt_locations = locations_decode(target_3dct, gt_ct_3doffsets, target['K'],
                                        gt_depth, target['trans'])
        pred_dimensions = dimension_decode(cls_ids, pred_dim_offsets)
        gt_dimensions = dimension_decode(cls_ids, gt_dim_offsets)

        pred_locations[:, 1] += pred_dimensions[:, 0] / 2
        gt_locations[:, 1] += gt_dimensions[:, 0] / 2

        # Todo: gt locations or pred locations here
        # Todo: need to make sure pred_rotys lies in [-pi, pi]
        pred_rotys, _ = orientation_decode(pred_ori, gt_locations, flip_mask)
        # pred_rotys, _ = orientation_quaternion_decode(pred_ori, gt_locations, flip_mask)
        pred_rotys = mask.view(-1, 1).float() * pred_rotys
        # gt_rotys, _ = orientation_quaternion_decode(gt_ori, gt_locations, flip_mask)
        # gt_rotys = target_roty[target_roty != 0].view(batch_size, -1)
        gt_rotys = (mask.float() * target_roty).view(-1, 1)

        # rotation matrix decode
        pred_bbox3d_rotys = points_decode(pred_rotys, gt_dimensions, gt_locations)
        pred_bbox3d_dims = points_decode(gt_rotys, pred_dimensions, gt_locations)
        pred_bbox3d_locs = points_decode(gt_rotys, gt_dimensions, pred_locations)
        
        iou_bbox = points_decode(pred_rotys,pred_dimensions,pred_locations)
        gt_bbox3d = points_decode(gt_rotys, gt_dimensions, gt_locations)
        
        #import pdb;pdb.set_trace()
        iouloss = torch.sum(torch.tensor(giou_3D_loss(iou_bbox,gt_bbox3d)).cuda().float()) / ind.shape[1]
        location_loss = F.l1_loss(gt_locations,pred_locations,reduction='sum') / ind.shape[1]*10
        dim_loss = F.l1_loss(pred_dimensions,gt_dimensions,reduction = 'sum') /  ind.shape[1]*10
        
        #import pdb;pdb.set_trace()
        # iouloss_dim = giou_3D_loss(pred_bbox3d_dims,gt_bbox3d)
        # iouloss_dim = torch.tensor(iouloss_dim).cuda().float()
        # iouloss_dim = iouloss_dim*(mask.float().view(-1,1))
        # iouloss_dim = torch.sum(iouloss_dim)
        
        
        # iouloss_rotys = giou_3D_loss(pred_bbox3d_rotys,gt_bbox3d)
        # iouloss_rotys = torch.tensor(iouloss_rotys).cuda().float()     
        #iouloss_rotys = iouloss_rotys*(mask.float().view(-1,1))
        # iouloss_rotys = torch.sum(iouloss_rotys)
        
        
        # iouloss_locs = giou_3D_loss(pred_bbox3d_locs,gt_bbox3d)
        # iouloss_locs = torch.tensor(iouloss_locs).cuda().float()
        #iouloss_locs = iouloss_locs*(mask.float().view(-1,1))
        # iouloss_locs = torch.sum(iouloss_locs)

        # quaternion decode
        # pred_bbox3d_rotys = points_quaternion_decode(pred_rotys, gt_dimensions, gt_locations)
        # pred_bbox3d_dims = points_quaternion_decode(gt_rotys, pred_dimensions, gt_locations)
        # pred_bbox3d_locs = points_quaternion_decode(gt_rotys, gt_dimensions, pred_locations)
        # gt_bbox3d = points_quaternion_decode(gt_rotys, gt_dimensions, gt_locations)

        # loss = giou_3d_loss(pred_bbox3d, gt_bbox3d, mask) * 20
        # loss = F.smooth_l1_loss(pred_bbox3d, gt_bbox3d, reduction='sum') / (ind.shape[1] * 10)
        
        loss_rotys = F.l1_loss(pred_bbox3d_rotys, gt_bbox3d, reduction='sum') / (ind.shape[1] * 10)
        # loss_dims = F.l1_loss(pred_bbox3d_dims, gt_bbox3d, reduction='sum') / (ind.shape[1] * 10)
        # loss_locs = F.l1_loss(pred_bbox3d_locs, gt_bbox3d, reduction='sum') / (ind.shape[1] * 10)
        return  iouloss + location_loss + loss_rotys

def giou_3D_loss(predbbox,gtbbox):
    #import pdb;pdb.set_trace()
    num = predbbox.shape[0]
    pred_bbox = predbbox.clone().detach().cpu().numpy()
    gt_bbox  = gtbbox.clone().detach().cpu().numpy()
    # pred_coner = np.vstack((predbbox[:,0,:],pred_bbox[:,1,:]))
    # gt_bbox = np.vstack(gt_bbox[:,0,:],gt_bbox[:,1,:])
    Iou = box3d_iou_loss(pred_bbox,gt_bbox)
    
    return  Iou