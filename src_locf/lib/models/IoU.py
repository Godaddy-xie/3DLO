import numpy as np
from scipy.spatial import ConvexHull
from numpy import *
import shapely
from shapely.geometry import Polygon,MultiPoint
def polygon_clip(subjectPolygon, clipPolygon):
   
   def inside(p):
      return(cp2[0]-cp1[0])*(p[1]-cp1[1]) > (cp2[1]-cp1[1])*(p[0]-cp1[0])
 
   def computeIntersection():
      dc = [ cp1[0] - cp2[0], cp1[1] - cp2[1] ]
      dp = [ s[0] - e[0], s[1] - e[1] ]
      n1 = cp1[0] * cp2[1] - cp1[1] * cp2[0]
      n2 = s[0] * e[1] - s[1] * e[0] 
      n3 = 1.0 / (dc[0] * dp[1] - dc[1] * dp[0])
      return [(n1*dp[0] - n2*dc[0]) * n3, (n1*dp[1] - n2*dc[1]) * n3]
 
   outputList = subjectPolygon
   cp1 = clipPolygon[-1]
 
   for clipVertex in clipPolygon:
      cp2 = clipVertex
      inputList = outputList
      outputList = []
      s = inputList[-1]
 
      for subjectVertex in inputList:
         e = subjectVertex
         if inside(e):
            if not inside(s):
               outputList.append(computeIntersection())
            outputList.append(e)
         elif inside(s):
            outputList.append(computeIntersection())
         s = e
      cp1 = cp2
      if len(outputList) == 0:
          return None
   return(outputList)

def poly_area(x,y):
   
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def convex_hull_intersection(p1, p2):
    
    inter_p = polygon_clip(p1,p2)
    if inter_p is not None:
        hull_inter = ConvexHull(inter_p)
        return inter_p, hull_inter.volume
    else:
        return None, 0.0  

def box3d_vol(corners):
    ''' corners: (8,3) no assumption on axis direction '''
    a = np.sqrt(np.sum((corners[0,:] - corners[1,:])**2))
    b = np.sqrt(np.sum((corners[1,:] - corners[2,:])**2))
    c = np.sqrt(np.sum((corners[0,:] - corners[4,:])**2))
    return a*b*c

def is_clockwise(p):
    x = p[:,0]
    y = p[:,1]
    return np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)) > 0

def box3d_iou(corners1, corners2,gt,det):
    
    
    # corner points are in counter clockwise order
    rect1 = [(corners1[i,0], corners1[i,2]) for i in range(3,-1,-1)]
    rect2 = [(corners2[i,0], corners2[i,2]) for i in range(3,-1,-1)] 

    area1 = poly_area(np.array(rect1)[:,0], np.array(rect1)[:,1])
    area2 = poly_area(np.array(rect2)[:,0], np.array(rect2)[:,1])
    # print("--------gt")
    # # print(corners1)
    # print("--------dec")

    # print(corners2)
 
    inter, inter_area = convex_hull_intersection(rect1, rect2)
    iou_2d = inter_area/(area1+area2-inter_area)

    gt_y,det_y,gt_h,det_h = float(gt[12]),float(det[12]),float(gt[8]),float(det[8])
    
    ymax = min(det_y,gt_y)
    ymin = max(det_y - det_h,gt_y - gt_h)


    # ymax = min(corners1[0,1], corners2[0,1])
    # ymin = max(corners1[4,1], corners2[4,1])
    # print("inter_area   " ,inter_area)
    inter_vol = inter_area * max(0.0, ymax-ymin)
    
    vol1 = box3d_vol(corners1)
    vol2 = box3d_vol(corners2)
    iou = inter_vol / (vol1 + vol2 - inter_vol)
    return iou, iou_2d





def box3d_iou_loss(corners1_all, corners2_all):
    IoU_ret = np.zeros((corners1_all.shape[0],1))
    #import pdb;pdb.set_trace()
    for num in range(corners1_all.shape[0]):
        corners1_ymin = min(corners1_all[num][1,:])
        corners1_ymax = max(corners1_all[num][1,:])
        corners2_ymin = min(corners2_all[num][1,:])
        corners2_ymax = max(corners2_all[num][1,:])
        
        corners1 = corners1_all[num].transpose()
        corners2 = corners2_all[num].transpose()

        ymin = max(corners1_ymin,corners2_ymin)
        ymax = min(corners1_ymax,corners2_ymax)


        # corner points are in counter clockwise order
        rect1 = np.array([[corners1[i,0], corners1[i,2]] for i in range(3,-1,-1)])
        rect2 = np.array([[corners2[i,0], corners2[i,2]] for i in range(3,-1,-1)])
        try:
            poly1 = Polygon(rect1).convex_hull
            poly2 = Polygon(rect2).convex_hull
            union_poly = np.concatenate((rect1,rect2))
        except:
            print("error")
            iou_bev =0  
            iou_3d = 0  
        if not poly1.intersects(poly2):
            iou_bev =0  
            iou_3d = 0  
        else:
            try:
                inter_area = poly1.intersection(poly2).area
                union_ara  = MultiPoint(union_poly).convex_hull.area
                if(union_ara == 0):
                    iou_bev =0  
                    iou_3d = 0 
                iou_bev = float(inter_area) /union_ara
                v1 = poly1.area*(corners1_ymax-corners1_ymin)
                v2 = poly2.area*(corners2_ymax-corners2_ymin)
                v_inter = float(inter_area)*(ymax-ymin)
                iou_3d = v_inter /(v1 + v2-v_inter)
            except:
                print("error")
                iou_bev =0  
                iou_3d = 0      

        IoU_ret[num][0]= 1-iou_3d

    return IoU_ret



















# ----------------------------------
# Helper functions for evaluation
# ----------------------------------

def get_3d_box(box_size, heading_angle, center):
  
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])
    
    # ##debug
    # h,w,l = box_size
    # #heading_angle= -1.29
    # c = np.cos(heading_angle)
    # s = np.sin(heading_angle)
    # r = np.array([[c,s],
    #              [-s,c]])
    # x_debug= [l/2,l/2,-l/2,-l/2]
    # z_debug= [w/2,-w/2,-w/2,w/2]
    # print(np.vstack([x_debug,z_debug]))
    # corner_debug = np.dot(r, np.vstack([x_debug,z_debug]))
    # print(heading_angle)
    # print(r)
    # print(corner_debug)

    # assert 0

    

    R = roty(heading_angle)
    h,w,l = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    
    #print(np.vstack([x_corners,z_corners]))
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    # print("---------------------------")
    # print(corners_3d)
    
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    # print("---------------------------")
    # print(corners_3d)
  
    corners_3d = np.transpose(corners_3d)
    return corners_3d


def get_3d_box(box_size, heading_angle, center):
  
    def roty(t):
        c = np.cos(t)
        s = np.sin(t)
        return np.array([[c,  0,  s],
                         [0,  1,  0],
                         [-s, 0,  c]])
    
    # ##debug
    # h,w,l = box_size
    # #heading_angle= -1.29
    # c = np.cos(heading_angle)
    # s = np.sin(heading_angle)
    # r = np.array([[c,s],
    #              [-s,c]])
    # x_debug= [l/2,l/2,-l/2,-l/2]
    # z_debug= [w/2,-w/2,-w/2,w/2]
    # print(np.vstack([x_debug,z_debug]))
    # corner_debug = np.dot(r, np.vstack([x_debug,z_debug]))
    # print(heading_angle)
    # print(r)
    # print(corner_debug)

    # assert 0


    R = roty(heading_angle)
    h,w,l = box_size
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2]
    y_corners = [h/2,h/2,h/2,h/2,-h/2,-h/2,-h/2,-h/2]
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2]
    
    #print(np.vstack([x_corners,z_corners]))
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    # print("---------------------------")
    # print(corners_3d)
    
    corners_3d[0,:] = corners_3d[0,:] + center[0]
    corners_3d[1,:] = corners_3d[1,:] + center[1]-h/2
    corners_3d[2,:] = corners_3d[2,:] + center[2]
    # print("---------------------------")
    # print(corners_3d)
  
    corners_3d = np.transpose(corners_3d)
    return corners_3d


    
if __name__=='__main__':
    print('------------------')
    # get_3d_box(box_size, heading_angle, center)
    corners_3d_ground  = get_3d_box((1.497255,1.644981, 3.628938), -1.531692, (2.882992 ,1.698800 ,20.785644)) 
    corners_3d_predict = get_3d_box((1.458242, 1.604773, 3.707947), -1.549553, (2.756923, 1.661275, 20.943280 ))
    (IOU_3d,IOU_2d)=box3d_iou(corners_3d_predict,corners_3d_ground)
    print (IOU_3d,IOU_2d) #3d IoU/ 2d IoU of BEV(bird eye's view)
      
