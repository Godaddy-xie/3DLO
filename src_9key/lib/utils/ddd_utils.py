from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
from scipy.optimize import least_squares
import torch
def compute_box_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 8 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  l, w, h = dim[2], dim[1], dim[0]
  x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
  y_corners = [0,0,0,0,-h,-h,-h,-h]
  z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

  corners = np.array([x_corners, y_corners, z_corners], dtype=np.float32)
  corners_3d = np.dot(R, corners) 
  corners_3d = corners_3d + np.array(location, dtype=np.float32).reshape(3, 1)
  return corners_3d.transpose(1, 0)

def project_to_image(pts_3d, P):
  # pts_3d: n x 3
  # P: 3 x 4
  # return: n x 2
  pts_3d_homo = np.concatenate(
    [pts_3d, np.ones((pts_3d.shape[0], 1), dtype=np.float32)], axis=1)
  pts_2d = np.dot(P, pts_3d_homo.transpose(1, 0)).transpose(1, 0)
  pts_2d = pts_2d[:, :2] / pts_2d[:, 2:]
  # import pdb; pdb.set_trace()
  return pts_2d

def compute_orientation_3d(dim, location, rotation_y):
  # dim: 3
  # location: 3
  # rotation_y: 1
  # return: 2 x 3
  c, s = np.cos(rotation_y), np.sin(rotation_y)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)
  orientation_3d = np.array([[0, dim[2]], [0, 0], [0, 0]], dtype=np.float32)
  orientation_3d = np.dot(R, orientation_3d)
  orientation_3d = orientation_3d + \
                   np.array(location, dtype=np.float32).reshape(3, 1)
  return orientation_3d.transpose(1, 0)

def draw_box_3d(image, corners, c=(0, 0, 255)):
  face_idx = [[0,1,5,4],
              [1,2,6, 5],
              [2,3,7,6],
              [3,0,4,7]]
  for ind_f in range(3, -1, -1):
    f = face_idx[ind_f]
    for j in range(4):
      cv2.line(image, (corners[f[j], 0], corners[f[j], 1]),
               (corners[f[(j+1)%4], 0], corners[f[(j+1)%4], 1]), c, 2, lineType=cv2.LINE_AA)
    if ind_f == 0:
      cv2.line(image, (corners[f[0], 0], corners[f[0], 1]),
               (corners[f[2], 0], corners[f[2], 1]), c, 1, lineType=cv2.LINE_AA)
      cv2.line(image, (corners[f[1], 0], corners[f[1], 1]),
               (corners[f[3], 0], corners[f[3], 1]), c, 1, lineType=cv2.LINE_AA)
  return image

def unproject_2d_to_3d_bev(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth 
  x = (pt_2d[0] * depth  - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth  - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32)
  return pt_3d


def unproject_2d_to_3d(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth 
  x = (pt_2d[0] * depth  - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth  - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32)
  return pt_3d


def unproject_2d_to_3d_orginal(pt_2d, depth, P):
  # pts_2d: 2
  # depth: 1
  # P: 3 x 4
  # return: 3
  z = depth - P[2, 3]
  x = (pt_2d[0] * depth - P[0, 3] - P[0, 2] * z) / P[0, 0]
  y = (pt_2d[1] * depth - P[1, 3] - P[1, 2] * z) / P[1, 1]
  pt_3d = np.array([x, y, z], dtype=np.float32)
  return pt_3d





def alpha2rot_y(alpha, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    rot_y = alpha + np.arctan2(x - cx, fx)
    if rot_y > np.pi:
      rot_y -= 2 * np.pi
    if rot_y < -np.pi:
      rot_y += 2 * np.pi
    return rot_y

def rot_y2alpha(rot_y, x, cx, fx):
    """
    Get rotation_y by alpha + theta - 180
    alpha : Observation angle of object, ranging [-pi..pi]
    x : Object center x to the camera center (x-W/2), in pixels
    rotation_y : Rotation ry around Y-axis in camera coordinates [-pi..pi]
    """
    alpha = rot_y - np.arctan2(x - cx, fx)
    if alpha > np.pi:
      alpha -= 2 * np.pi
    if alpha < -np.pi:
      alpha += 2 * np.pi
    return alpha


def ddd2locrot_bev(center, alpha, dim, depth, calib):
  # single image
  
  locations = unproject_2d_to_3d(center, depth, calib)
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  return locations, rotation_y

def ddd2locrot(center, alpha, dim, depth, calib):

  # single image
  
  locations = unproject_2d_to_3d(center, depth, calib)
  locations[1] +=  dim[0]/2
  #locations[1] += dim[0] / 2
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  return locations, rotation_y


def post_error_project(x,*args):
  h,w,l = x[0:3]
  dim = np.array([h,w,l])
  rotation= x[3]
  loc_x = x[4]
  loc_y = x[5]
  loc_z = x[6]
  loc = x[4:]
  uv = args[2]
  calib = args[3]
  dimension_piror = args[0]
  rotataion_piror = args[1]
  sco = args[4]
  calib = calib.reshape(3,4)
  c, s = np.cos(rotation), np.sin(rotation)
  R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)

  offset = np.array([[l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2,0],
                     [0,  0, 0 , 0 ,-h ,-h,-h,-h,-0.5*h],
                     [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2,0]]
                     )
  
  point = np.dot(R,offset)  + loc.reshape(3,1)
  fx, cx, fy, cy = calib[0,0], calib[0,2], calib[1,1], calib[1,2]
  K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1])
  K = K.reshape(3,3)
  
  reproject_kp = np.dot(K, point)
  reproject_kp = (reproject_kp / reproject_kp[2])[0:2]
  reproject_kp = reproject_kp.T

  ## normal methods###
  err_weight = np.diag(torch.nn.functional.softmax(torch.tensor(sco)).numpy().tolist()[::-1])
  error_vec_kp = (reproject_kp - uv.reshape(9,2))
  
  error_vec_kp_trans = error_vec_kp.T
  erro_vec = np.dot(error_vec_kp,error_vec_kp_trans)
  err_v = np.sum(err_weight*erro_vec)
  error_rotation =  np.linalg.norm(rotation - rotataion_piror)
  erro_dim = np.linalg.norm(dim - dimension_piror)
  err_weight = torch.nn.functional.softmax(torch.tensor(sco)).numpy().tolist()[::-1]



  return err_v,error_rotation,erro_dim

def ddd2locrot_rtm(center, alpha, dim, depth, kp,kp_scores,calib):

  # single image
 
  calib = calib[0]
  locations_init = unproject_2d_to_3d(center, depth, calib)
  locations_init[1] +=  dim[0]/2
  #bev point#
  rotation_init = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  dim_init = dim
  x = np.concatenate((dim_init,np.array([rotation_init]),locations_init))
  # bbox = compute_box_3d(dim,locations_init,rotation_init)
  # kp_9 = np.vstack((bbox,locations_init))
  #post_error_project(x,dim_init,rotation_init,kp,calib,kp_scores)
  result =  least_squares(post_error_project,x,args = [dim_init,rotation_init,kp,calib,kp_scores])
  locations_p =   result['x'][4:7]
  rotation_y_=  result['x'][3]
  dim_p = result['x'][0:3]
  return locations_p, rotation_y_,dim_p



def ddd2locrot_xyz(center, alpha, dim, locations, calib):
  # single image
  #locations = unproject_2d_to_3d(center, depth, calib)
  #locations[1] += dim[0] / 2
  rotation_y = alpha2rot_y(alpha, center[0], calib[0, 2], calib[0, 0])
  return locations, rotation_y

def project_3d_bbox(location, dim, rotation_y, calib):
  box_3d = compute_box_3d(dim, location, rotation_y)
  box_2d = project_to_image(box_3d, calib)
  return box_2d


if __name__ == '__main__':
  calib = np.array(
    [[7.070493000000e+02, 0.000000000000e+00, 6.040814000000e+02, 4.575831000000e+01],
     [0.000000000000e+00, 7.070493000000e+02, 1.805066000000e+02, -3.454157000000e-01],
     [0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00, 4.981016000000e-03]],
    dtype=np.float32)
  alpha = -0.20
  tl = np.array([712.40, 143.00], dtype=np.float32)
  br = np.array([810.73, 307.92], dtype=np.float32)
  ct = (tl + br) / 2
  rotation_y = 0.01
  print('alpha2rot_y', alpha2rot_y(alpha, ct[0], calib[0, 2], calib[0, 0]))
  print('rotation_y', rotation_y)