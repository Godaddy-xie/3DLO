import numpy as np
import os
import pandas as pd
import math
from IoU import *  
import cv2  
print('2dcent and perspective')
def load_txt(file_path):  # load result file
    txt_file = []
    filenames = os.listdir(file_path) 
    for item in filenames:
        # print("item:",item)
        txt_file.append(item)
    return txt_file
def parse_txt(txt_name):
    data = []
    with open(txt_name, 'r') as file_read:
        while True:
            line = file_read.readline()

            if not line:
                break
            line = line.strip('\n')
            tem = line.split(" ")
            data.append(tem)
    file_read.close()
    return data
def data_to_dict(path, file_list):
    dict_data = dict()
    for index, file_name in enumerate(file_list):
        file_id = file_name.split(".")
        file_id = file_id[0]
        file_path = os.path.join(path, file_name)
        data = parse_txt(file_path)
        dict_data["{}".format(file_id)] = data

    return dict_data

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)
    else:
        print("the folder exist！")


def world_to_uv(label_data, calib_data):

    world_location_x_index = 11
    world_location_y_index = 12
    world_location_z_index = 13
    label_one_location = []
    uv_data_dict = {}
    for key, value in label_data.items():
         if(key!='000003'):
             continue
         if(value == []):
             continue
         uv_point_list = []
         count = len(value)
         for index in range(count):  # loop for result
             if(value[index][0] != 'Car'):
                  continue

             world_location_x = float(value[index][world_location_x_index])
             world_location_y = float(value[index][world_location_y_index])
             world_location_z = float(value[index][world_location_z_index])
             world_location = np.array([world_location_x,world_location_y,world_location_z]).T
             
             #import pdb;pdb.set_trace()
             # calib world to uv
             camera_index = 2# p2 camera
             calib_list = calib_data[key]
             calib_size = len(calib_list)
             if calib_size == 0:
                 continue
             p2_camera = calib_list[2]
             p2_camera = p2_camera[1:]
             p2_camera =np.array(p2_camera,dtype=float)
             calib = np.reshape(p2_camera,[3,4])
             fx, cx, fy, cy = calib[0,0], calib[0,2], calib[1,1], calib[1,2]
             K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1])
             K = K.reshape(3,3)
             camera_point = np.dot(K, world_location)
             uv_point = camera_point/(world_location_z)
             uv_point = [uv_point[0],uv_point[1],fy,cy,fx,cx,world_location_z]
             uv_point_list.append(uv_point)    
             
         uv_data_dict["{}".format(key)] = uv_point_list
         
    return uv_data_dict


def get_3d_info(value):
    h, w ,l,x,y,z,angle = float(value[8]),float(value[9]),float(value[10]),float(value[11]),float(value[12]),float(value[13]),float(value[14])
    return h,w,l,x,y,z,angle

def Projection(points,calib_list):
        #import pdb;pdb.set_trace()
        calib_size = len(calib_list)
        if calib_size == 0:
                return 
        p2_camera = calib_list[2]
        p2_camera = p2_camera[1:]
        p2_camera =np.array(p2_camera,dtype=float)
        calib = np.reshape(p2_camera,[3,4])
        fx, cx, fy, cy = calib[0,0], calib[0,2], calib[1,1], calib[1,2]
        K = np.array([fx, 0, cx, 0, fy, cy, 0, 0, 1])
        K = K.reshape(3,3)
        #camera_point = np.dot(points,K)
        image_point = []
        for i in range(8):
            p = points[i].T
            camer_point = np.dot(K,p)
            image_point.append((int(camer_point[0]/camer_point[2]),int(camer_point[1]/camer_point[2])))
        
        return image_point

def Vis(result_dict=None,label_dict=None,calib_data= None,image_path=None,save_root_path=None,flag_dontcare=False,cls = 'Car'):
    mkdir(save_root_path)
    for key, value in result_dict.items(): 
        #debug
        # if(key!="000003"):
        #     continue   
        if(value == []):
            continue
        
        image =  image_path+key+'.png'
        save_path = save_root_path+key+'.png'
        count = len(value)
        
        img = cv2.imread(image)
        ## plot result  ##
        count = len(value)
        for index in range(count): 
            if(value[index][0] != 'Car'):
                continue
            h_result,w_result,l_result,x_result,y_result,z_result,angle_result = get_3d_info(value[index])
            result_3d_bbox = get_3d_box((h_result,w_result,l_result),angle_result,(x_result,y_result,z_result))
            calib_list = calib_data[key]
            point = Projection(result_3d_bbox,calib_list,)
           
            #--plotline--#
            for i in point:
                try:
                    cv2.circle(img,i,1,(255,0,0),1)
                except:
                    continue
            #import pdb ; pdb.set_trace()
            for start in [0,2,5,7]:
                if start==0:
                    start_point = point[start]
                    end_point_1 = point[1]
                    end_point_2 = point[3]
                    end_point_3 = point[4]
                    try:
                        cv2.line(img,start_point,end_point_1,(0,255,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_2,(0,255,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_3,(0,255,255),1)
                    except:
                        pass
                elif start==2:
                    start_point = point[start]
                    end_point_1 = point[3]
                    end_point_2 = point[1]
                    end_point_3 = point[6]
                    try:
                        cv2.line(img,start_point,end_point_1,(0,255,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_2,(0,255,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_3,(0,255,255),1)
                    except:
                        pass
                elif start ==5:
                    start_point = point[start]
                    end_point_1 = point[6]
                    end_point_2 = point[4]
                    end_point_3 = point[1]
                    try:
                        cv2.line(img,start_point,end_point_1,(0,255,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_2,(0,255,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_3,(0,255,255),1)
                    except:
                        pass
                elif start ==7:
                    start_point = point[start]
                    end_point_1 = point[4]
                    end_point_2 = point[6]
                    end_point_3 = point[3]
                    try:
                        cv2.line(img,start_point,end_point_1,(0,255,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_2,(0,255,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_3,(0,255,255),1)
                    except:
                        pass

        label_info = label_dict[key]
        if(len(label_info)!=0):
            count = len(label_info)
        for index in range(count): 
            if(label_info[index][0] != 'Car'):
                    continue
            h_result,w_result,l_result,x_result,y_result,z_result,angle_result = get_3d_info(label_info[index])
            result_3d_bbox = get_3d_box((h_result,w_result,l_result),angle_result,(x_result,y_result,z_result))
            calib_list = calib_data[key]
            point = Projection(result_3d_bbox,calib_list,)
           
            #--plotline--#
            for i in point:
                try:
                    cv2.circle(img,i,2,(0,0,255),2)
                except:
                    continue
            #import pdb ; pdb.set_trace()
            for start in [0,2,5,7]:
                if start==0:
                    start_point = point[start]
                    end_point_1 = point[1]
                    end_point_2 = point[3]
                    end_point_3 = point[4]
                    try:
                        cv2.line(img,start_point,end_point_1,(0,0,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_2,(0,0,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_3,(0,0,255),1)
                    except:
                        pass
                elif start==2:
                    start_point = point[start]
                    end_point_1 = point[3]
                    end_point_2 = point[1]
                    end_point_3 = point[6]
                    try:
                        cv2.line(img,start_point,end_point_1,(0,0,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_2,(0,0,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_3,(0,0,255),1)
                    except:
                        pass
                elif start ==5:
                    start_point = point[start]
                    end_point_1 = point[6]
                    end_point_2 = point[4]
                    end_point_3 = point[1]
                    try:
                        cv2.line(img,start_point,end_point_1,(0,0,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_2,(0,0,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_3,(0,0,255),1)
                    except:
                        pass
                elif start ==7:
                    start_point = point[start]
                    end_point_1 = point[4]
                    end_point_2 = point[6]
                    end_point_3 = point[3]
                    try:
                        cv2.line(img,start_point,end_point_1,(0,0,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_2,(0,0,255),1)
                    except:
                        pass
                    try:
                        cv2.line(img,start_point,end_point_3,(0,0,255),1)
                    except:
                        pass 
            
        cv2.imwrite(save_path,img)
        print("save {}.".format(save_path))
def read_txt(path):
    ret = []
    with open(path,'r') as f:
        for line in f.readlines():
            line = line.strip('\n')
            ret.append(line+'.txt')
    return ret
import sys
if __name__ == "__main__":


    
    #label_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/label_2/"
    image_root_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/image_2/"
    calib_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/calib/"
    label_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/label_2/"
    
    #result_path= "/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/bev_500/results/"
    
    result_path = sys.argv[1]+'/'
    need_vis_path = sys.argv[2]
    result_file_list = read_txt(need_vis_path)
    save_root_path = sys.argv[3]+'/'

    #save_root_path ="/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/bev/vis/"
    label_file_list = load_txt(label_path)
    calib_file_list = load_txt(calib_path)
    
    #import pdb;pdb.set_trace()
    #tpth = "/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/output/bev_500_Feb_17_2020/kitti_eval/VIS_0.txt"
    #result_file_list = read_txt(tpth)
    #result_file_list = load_txt(result_path)
    label_data = data_to_dict(label_path,label_file_list)
    print('label suc')
    calib_data = data_to_dict(calib_path,calib_file_list)
    print('cal suc')
    result_data = data_to_dict(result_path,result_file_list)
    print('reusl suc')
    Vis(result_data,label_data,calib_data,image_root_path,save_root_path)
    #world_to_uv(label_data,calib_data)



    # dic = list_To_dict(result_list)

    


    # for key, value in dic.items():
    #     if(value == []):
    #         continue
    #     uv_point_list = []
    #     count = len(value)
    #     # control one picture
    # #  if(key != '001512'):
    # #      continue
    #     image_name = str(key) + ".png"
    #     #image_name = '000001.png'
    #     image_path = image_root_path + image_name
    #     save_path = save_root_path  + str(image_name)
    #     #image_path = './image_2_01/' + str(image_name)
    #     image = cv2.imread(str(image_path))
    #     print(save_path)
    #     for index in range(count):  # loop for result
    #         label_x = int(float(value[index][0]))  # bev
    #         label_y = int(float(value[index][1]))
    #         point_bev = (label_x,label_y)
    #         predict_u = int(float(value[index][2]))
    #         predict_v = int(float(value[index][3]))
    #         point_predit = (predict_u,predict_v)
    #         point_size = 2
    #         line_color_bev = (0,255,255) # bgr
    #         line_color_165 = (255,0,0)
    #         line_color_center = (0,255,0)
    #         thickness = 2 #cv2.line(image, point_16, point_15,point_color,thickness)
    #         # draw bev to 2d center
    #         cv2.line(image, point_bev, point_predit,line_color_165,thickness)
    #         print('hhhhh')
    #         #cv2.line(image, point_center, point_15,point_color_1,thickness)
    #         radius = 3
    #         point_color_2d_center = (0,0,255)
    #         point_color_165 = (0,255,255)
    #         cv2.circle(image, point_bev, radius,point_color_165,thickness)
    #         cv2.circle(image, point_predit,radius,point_color_2d_center,thickness)
    #     cv2.imwrite(save_path,image)







