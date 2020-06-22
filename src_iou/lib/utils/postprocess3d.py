import torch
from models.losses3d import depth_decode, dimension_decode, points_decode
#from models.quaternion import orientation_quaternion_decode_postprocess
from utils.image import get_affine_transform

PI = 3.1415926

def locations_decode_postprocess(ct_3d, ct_3d_offsets, K, depth, trans_output):
    N = ct_3d.shape[0]
    ct_3d_refine = ct_3d + ct_3d_offsets
    extend_trans = torch.tensor([[0., 0., 1.]])
    trans_square = torch.cat((trans_output, extend_trans), dim=0)
    trans_inv = torch.inverse(trans_square)
    ct_3d_homo = torch.cat((ct_3d_refine, torch.ones(N, 1)), dim=1)
    ct_3d_img = torch.matmul(trans_inv, ct_3d_homo.transpose(0, 1))
    ct_3d_img = ct_3d_img * depth
    K_inv = torch.inverse(K[:, :3])

    locations = torch.matmul(K_inv, ct_3d_img)

    return locations.transpose(0, 1)


def box2d_decode(K, rotys, dimensions, locations, s):
    #import pdb;pdb.set_trace()
    bbox3d = points_decode(rotys, dimensions, locations)
    ones = torch.ones(bbox3d.shape[0], 1, bbox3d.shape[2])
    corners_3D_1 = torch.cat((bbox3d, ones), dim=1)
    corners_2D = torch.matmul(K, corners_3D_1)
    corners_2D = corners_2D[:, :2, :] / corners_2D[:, 2, :].view(bbox3d.shape[0], 1, bbox3d.shape[2])

    xmins, _ = corners_2D[:, 0, :].min(dim=1)
    xmaxs, _ = corners_2D[:, 0, :].max(dim=1)
    ymins, _ = corners_2D[:, 1, :].min(dim=1)
    ymaxs, _ = corners_2D[:, 1, :].max(dim=1)

    # Todo: we need to add image shape here
    xmins = xmins.clamp(0, int(s[0]))
    xmaxs = xmaxs.clamp(0, int(s[0]))
    ymins = ymins.clamp(0, int(s[1]))
    ymaxs = ymaxs.clamp(0, int(s[1]))

    bboxfrom3d = torch.cat((xmins.unsqueeze(1), ymins.unsqueeze(1),
                            xmaxs.unsqueeze(1), ymaxs.unsqueeze(1)), dim=1)

    return bboxfrom3d


def orientation_decode(ori, locations):
    offsets = torch.atan(locations[:, 0] / (locations[:, 2] + 1e-7))
    alpha = torch.atan(ori[:, 0] / (ori[:, 1] + 1e-7))

    cos_pos_idx = (ori[:, 1] >= 0).nonzero()
    cos_neg_idx = (ori[:, 1] < 0).nonzero()

    alpha[cos_pos_idx] = alpha[cos_pos_idx] - PI / 2
    alpha[cos_neg_idx] = alpha[cos_neg_idx] + PI / 2

    rotys = alpha + offsets

    roty_large_idx = (rotys > PI).nonzero()
    roty_small_idx = (rotys < -PI).nonzero()

    if len(roty_large_idx) != 0:
        rotys[roty_large_idx] = rotys[roty_large_idx] - 2 * PI
    if len(roty_small_idx) != 0:
        rotys[roty_small_idx] = rotys[roty_small_idx] + 2 * PI

    return rotys.unsqueeze(1), alpha.unsqueeze(1)


def ddd_post_process_sincos_3d(dets, c, s, calibs,opt):
   # import pdb;pdb.set_trace()
    trans_output = get_affine_transform(
        c[0], s[0], 0, [opt.output_w, opt.output_h]
    )
    cls = dets[:, :, -1].transpose(0, 1)
    depths = depth_decode(dets[0, :, 5])
    calibs = torch.tensor(calibs[0])
    trans_output = torch.tensor(trans_output).to(dtype=torch.float32)
    # ct_3d_refine = dets[0, :, :2] + dets[0, :, 2:4]
    locations = locations_decode_postprocess(dets[0, :, :2], dets[0, :, 2:4], calibs, depths, trans_output)
    dimensions = dimension_decode(cls, dets[0, :, 6:9])
    locations[:, 1] += dimensions[:, 0] / 2
    rotys, alphas = orientation_decode(dets[0, :, 9:11], locations)
    # rotys, alphas = orientation_quaternion_decode_postprocess(dets[0, :, 9:11], locations)

    bboxfrom3d = box2d_decode(calibs, rotys, dimensions, locations, s[0])
    scores = dets[:, :, 4].transpose(0, 1)
    dets = torch.cat([cls, alphas, bboxfrom3d, dimensions, locations, rotys, scores], dim=1)

    return dets
