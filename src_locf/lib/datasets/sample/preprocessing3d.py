import numpy as np

def get_new_alpha(alpha):
    """
    change the range of orientation from [-pi, pi] to [0, 2pi]
    :param alpha: original orientation in KITTI
    :return: new alpha
    """
    new_alpha = float(alpha) + np.pi / 2.
    if new_alpha < 0:
        new_alpha = new_alpha + 2. * np.pi
        # make sure angle lies in [0, 2pi]
    # new_alpha = new_alpha - int(new_alpha / (2. * np.pi)) * (2. * np.pi)
    assert (0.<= new_alpha <= 2.* np.pi)

    return new_alpha

def compute_3d_center(ann, K, flipped):
    h, w, l = ann['dim'][0], ann['dim'][1], ann['dim'][2]
    x, y, z = ann['location'][0], ann['location'][1], ann['location'][2]
    ry = ann['rotation_y']

    if flipped:
        x, ry = -x, -ry
        ann['rotation_y'] = - ann['rotation_y']
        ann['new_alpha'] = 2 * np.pi - ann['new_alpha']
        ann['location'][0] = - ann['location'][0]

    x_corners = [0, l, l, l, l, 0, 0, 0]  # -l/2
    y_corners = [0, 0, h, h, 0, 0, h, h]  # -h
    z_corners = [0, 0, 0, w, w, w, w, 0]  # -w/2

    x_corners += - np.float32(l) / 2
    y_corners += - np.float32(h)
    z_corners += - np.float32(w) / 2

    # bounding box in object co-ordinate
    corners_3D = np.array([x_corners, y_corners, z_corners])

    R = np.array([[np.cos(ry), 0, np.sin(ry)],
                  [    0,      1,   0       ],
                  [-np.sin(ry),0, np.cos(ry)]])
    corners_3D = R.dot(corners_3D)
    # add translation
    corners_3D += np.array([x, y, z]).reshape((3, 1))

    # 3d center on image plane
    ct_3D = np.matmul(K[:3, :3], np.array([x, y-h/2, z]))
    ct_3D = ct_3D[:2] / ct_3D[2]

    corners_3D_1 = np.vstack((corners_3D, np.ones((corners_3D.shape[-1]))))
    corners_2D = K.dot(corners_3D_1)
    corners_2D = corners_2D / corners_2D[2]
    corners_2D = corners_2D[:2]

    # TODO: investage if negative value matters here
    bboxfrom3d = [min(corners_2D[0]), min(corners_2D[1]),
                 max(corners_2D[0]), max(corners_2D[1])]

    return ann, ct_3D, np.asarray(bboxfrom3d)

def depth_encode(depth, depth_pre):
    delta_depth = (depth - depth_pre[0]) / depth_pre[1]
    return delta_depth

def dimension_encode(cls_id, dims, dims_avg):
    #Todo: if we need add weights here
    delta_dim = np.log(dims/dims_avg[cls_id])

    return delta_dim

def orientation_encode(alpha):
    return np.array([np.sin(alpha), np.cos(alpha)])

def orientation_quaternion_encode(alpha):
    return np.array([np.sin(alpha/2), np.cos(alpha/2)])