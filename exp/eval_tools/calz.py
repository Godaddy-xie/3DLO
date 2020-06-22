import numpy as np
import os
import pandas as pd
import math
import sys
print('2dcent and perspective')

def load_txt(file_path):  # load result file
    txt_file = []
    filenames = os.listdir(file_path)  # 返回指定的文件夹包含的文件
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


def get_iou(result, label):
    result_left_x = float(result[0])
    result_left_y = float(result[1])
    result_right_x = float(result[2])
    result_right_y = float(result[3])

    label_left_x = float(label[0])
    label_left_y = float(label[1])
    label_right_x = float(label[2])
    label_right_y = float(label[3])

    # compute each area of retangle
    result_area = (result_right_x - result_left_x) * \
                   (result_right_y - result_left_y)
    label_area = (label_right_x - label_left_x)*(label_right_y - label_left_y)

    # compute the sum of area
    sum_area = result_area + label_area

    # find the each edge of intersect rectangle
    left_line = max(result_left_x, label_left_x)
    right_line = min(result_right_x, label_right_x)
    top_line = max(result_left_y, label_left_y)
    bottom_line = min(result_right_y, label_right_y)

    # judge there is an intersect
    if(left_line >= right_line or top_line >= bottom_line):
        return 0
    else:
        intersect = (right_line - left_line)*(bottom_line - top_line)
        intersect = (intersect/(sum_area-intersect))*1.0
        return intersect


def replace_data(label_data, result_data, replace_index):
    bev_center_x = 3
    bev_center_y = 4
    # bbox_right_x = 5
    # bbox_right_y = 6
    # ground_height = 7

    result_data_final = result_data
    result_list = []

    world_location_x_index = 11
    world_location_y_index = 12
    world_location_z_index = 13

    for key, value in result_data.items():
        # if(len(value) == 0):
        if(value == []):
            continue

        count = len(value)
        for index in range(count):  # loop for result

            # if(value[index][0] != 'Car' and value[index][0] != 'Van' and value[index][0] != 'Truck'):
            #     continue
            if(value[index][0] != 'Car'):
                continue

            sum_one = []
            sum_abs_min = 1e10
            sum_iou_max = -10

            result_bev_center_x = value[index][bev_center_x]
            result_bev_center_y = value[index][bev_center_y]
            # result_right_x = value[index][bbox_right_x]
            # result_right_y = value[index][bbox_right_y]
            result_coordinate = [result_bev_center_x, result_bev_center_y]

            # result_ground_height = value[index][ground_height]

            # label
            label_list = label_data[key]
            list_size = len(label_list)
            if list_size == 0:
                continue

            for index_label in range(list_size):
                label_one = label_list[index_label]

                # if(label_one[0] != 'Car' and label_one[0] != 'Van' and label_one[0] != 'Truck'):
                #     continue
                if(label_one[0] != 'Car'):
                    continue
                # world to uv label
                label_left_x = label_one[bbox_left_x+1]
                label_left_y = label_one[bbox_left_y+1]
                label_right_x = label_one[bbox_right_x+1]
                label_right_y = label_one[bbox_right_y+1]
                label_coordinate = [label_left_x,
                    label_left_y, label_right_x, label_right_y]

                # sum_abs = abs(float(result_left_x)-float(label_left_x)) \
                #     + abs(float(result_left_y)-float(label_left_y))  \
                #     + abs(float(result_right_x)-float(label_right_x)) \
                #     + abs(float(result_right_y)-float(label_right_y))

                iou_data = get_iou(result_coordinate, label_coordinate)

                # if(sum_abs < sum_abs_min):
                #     sum_abs_min = sum_abs
                #     ground_height_index = 12
                #     label_h_final = label_one[ground_height_index]

                if(iou_data >= sum_iou_max):
                    sum_iou_max = iou_data
                    ground_height_index = 12
                    ground_distance_index = 13
                    label_h_final = label_one[ground_height_index]

                    label_distance_final = label_one[ground_distance_index]

            # calculate ground height error
            error = abs(float(label_h_final) - float(result_ground_height))
            # label_h_final result_ground_height index
            result_one = [key, label_h_final,
                result_ground_height, error, label_distance_final]
            result_list.append(result_one)

    return result_list

def write_csv(result, path):
    #result_one = [key, label_x,label_y,result_bev_center_x,result_bev_center_y,distance]
    # result_one = [key, label_x,label_y,result_bev_center_x,result_bev_center_y,distance_center,distance_perpective]
    ID = []
    label_x = []
    label_y = []
    result_bev_center_x = []
    result_bev_center_y = []
    distance = []
    label_height = []
    result_height = []
    error = []
    
    for index, value in enumerate(result):
        ID.append(int(value[0]))
        label_x.append(value[1])
        label_y.append(value[2])
        result_bev_center_x.append(value[3])
        result_bev_center_y.append(value[4])
        distance.append(value[5])
        label_height.append(value[6])
        result_height.append(value[7])
        error.append(value[8])
       
        
        # data_frame = pd.DataFrame({'ID':int(value[0]),'label':float(value[1]),'pred':float(value[2]),'error':float(value[3])})
    data_frame = pd.DataFrame(
        {'ID': ID, 'label_x': label_x, 'label_y': label_y, 'result_bev_center_x': result_bev_center_x, 'result_bev_center_y': result_bev_center_y,'distance' : distance,'label_height' : label_height,'result_height' : result_height,'error' : error})
    # data_frame.to_csv('data4_log_2block_val_distance.csv',
    #                   index=False, sep=',')
    data_frame.to_csv(path,
                      index=False, sep=',')

def write_csv_2d(result, path):
    # result_one = [key, label_x,label_y,result_bev_center_x,result_bev_center_y,result_per_center_x,result_per_center_y,resdistance_center,distance_perpective]
    ID = []
    label_cent_x = []
    label_cent_y = []


    result_bev_center_x = []
    result_bev_center_y = []
    
    label_x = []
    label_y = []
    label_z = []
    r_x = []
    r_y = []
    r_z = []
    e_x =[]
    e_y =[]
    e_z =[]
    dis = []
    
    for index, value in enumerate(result):
        ID.append(int(value[0]))
        label_cent_x.append(value[1])
        label_cent_y.append(value[2])
        result_bev_center_x.append(value[3])
        result_bev_center_y.append(value[4])
        label_x.append(value[5])
        label_y.append(value[6])
        label_z.append(value[7])
        r_x.append(value[8])
        r_y.append(value[9])
        r_z.append(value[10])
        e_x.append(value[11])
        e_y.append(value[12])
        e_z.append(value[13])
        
        dis.append(value[14])

        # data_frame = pd.DataFrame({'ID':int(value[0]),'label':float(value[1]),'pred':float(value[2]),'error':float(value[3])})
    data_frame = pd.DataFrame(
        {'ID': ID, 'label_cent_x': label_cent_x, 'label_cent_y': label_cent_y,'result_bev_center_x':result_bev_center_x,'result_bev_center_y':result_bev_center_y, 'label_x': label_x, 'label_y': label_y,'label_z':label_z,'r_x':r_x,'r_y' : r_y,'r_z' : r_z,'e_x':e_x,'e_y' : e_y,'e_z' : e_z,'dis':dis})
    # data_frame.to_csv('data4_log_2block_val_distance.csv',
    #                   index=False, sep=',')
    data_frame.to_csv(path,
                      index=False, sep=',')


def write_data_txt(data, path):

    for key, value in data.items():

        txt_path = path + key + ".txt"
        file_write = open(txt_path, "w+")
        count = len(value)
        if(count == 0):
            file_write.close()
            continue

        for index in range(count):
            data_one = value[index]

            for data in data_one:  # loop for result
                file_write.write(data)
                file_write.write(" ")

            file_write.write("\n")
        file_write.close()


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)
    else:
        print("the folder exist！")


def world_to_uv(label_data, calib_data):
    label_data_copy = label_data.copy()
    world_location_x_index = 11
    world_location_y_index = 12
    world_location_z_index = 13
    
    for key, value in label_data.items():
         if(value == []):
             continue
         uv_point_list = []
         count = len(value)

         for index in range(count):  # loop for result
             
             world_location_x = float(value[index][world_location_x_index])
             #world_location_y = float(value[index][world_location_y_index]) 

             world_location_y = float(value[index][world_location_y_index])  -0.5*float(value[index][8])
             world_location_z = float(value[index][world_location_z_index])        
             world_location = np.array([world_location_x,world_location_y,world_location_z]).T
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
             # check boundary, condition is wrong
             label_data_copy[key][index].append(uv_point[0])
             label_data_copy[key][index].append(uv_point[1])
    return label_data_copy


def compare_data_2d(label_bev_uv,result_data):


    result_list=[]
    for key, value in result_data.items(): # loop for dict
        #import pdb;pdb.set_trace()
        if(value == []):
            continue
        count = len(value)
        
        for index in range(count):  # loop for list
            
            if(value[index][0] != 'Car'):
                continue
            
            min_distance = 1e10
            result_one = []
            
            result_center_x = (float(value[index][4]) + float(value[index][6])) /2
            result_center_y = (float(value[index][5]) + float(value[index][7])) /2
           
            result_x = float(value[index][11])
            result_y = float(value[index][12])
            result_z = float(value[index][13])

            result_bev_center = np.array([result_center_x,result_center_y],dtype=np.float32)
            
            # label
            label_list = label_bev_uv[key]
            list_size = len(label_list)
            if list_size == 0:
                continue
            for index_label in range(list_size):
                
                label_one = label_list[index_label]
                if(label_one[0] != 'Car' and label_one[0] !='Van'):
                    continue
                label_cent_x = (float(label_one[-2]))
                label_cent_y = (float(label_one[-1]))
                
                label_x = float(label_one[11])
                #label_y = float(label_one[12])
                label_y = float(label_one[12])
                label_z = float(label_one[13])
                #label_height = label_one[2]
                #for centerbox#
                label_bev =  np.array([label_cent_x,label_cent_y])
                
                #label_bev = np.array([(float(label_one[4])+float(label_one[6]))/2,(float(label_one[5])+float(label_one[7]))/2])
                
                distance_center = np.linalg.norm(label_bev - result_bev_center[:2])
                ez = abs(label_z -result_z)
                ex = abs(label_x -result_x)
                ey = abs(label_y -result_y)
                #distance = math.sqrt((label_x - result_bev_center_x )^2 + (label_y - result_bev_center_y )^2)
                #distance = (label_x - result_bev_center_x )^2 + (label_y - result_bev_center_y )^2
                if(distance_center < min_distance):
                    min_distance = distance_center
                    result_one = [key, label_cent_x,label_cent_y,result_center_x,result_center_y,label_x,label_y,label_z,result_x,result_y,result_z,ex,ey,ez,distance_center]
            # delete wrong match 
            wrong_match_threshold = 15
            distance_index = -1
            if(result_one == []):
                continue
            if(result_one[distance_index]>wrong_match_threshold):
                continue
            result_list.append(result_one)
            #import pdb;pdb.set_trace()        
    return result_list
    
if __name__ == "__main__":

    label_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/label_2/"
    calib_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/calib/"
    result_path="/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/M3D-RPN-master/val_result/data/"
    new_path =  "/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/M3D-RPN-master/val_result/zerror_val.csv"
    
    exp = sys.argv[1]
    result_path =sys.argv[2]+'/'
    new_path = sys.argv[3] +'/'+'Error_Z_distance'+'_{}'.format(exp)+'.csv'




    label_file_list = load_txt(label_path)
    calib_file_list = load_txt(calib_path)
    result_file_list = load_txt(result_path)
   
    label_data = data_to_dict(label_path,label_file_list)
    calib_data = data_to_dict(calib_path,calib_file_list)
    label_data = world_to_uv(label_data,calib_data)
    result_data = data_to_dict(result_path,result_file_list)
    #label_bev_uv = world_to_uv(label_data,calib_data)
    #label_center = world_to_uv_center(label_data,calib_data)
    #label_center = world_to_uv(label_data, calib_data)
    
    result_list = compare_data_2d(label_data,result_data)

    write_csv_2d(result_list,new_path)










