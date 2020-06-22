import numpy as np
import os
import pandas as pd
import math
import cv2 
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
    
    label_bev_x = []
    label_bev_y = []

    result_bev_center_x = []
    result_bev_center_y = []
    z = []
    distance_perpective = []
    x = []
    gtz = []
    for index, value in enumerate(result):
        ID.append(int(value[0]))
        label_bev_x.append(value[1])
        label_bev_y.append(value[2])
        result_bev_center_x.append(value[3])
        result_bev_center_y.append(value[4])
        distance_perpective.append(value[5])
        z.append(value[6])
        x.append(value[7])
        gtz.append(value[8])        
        # data_frame = pd.DataFrame({'ID':int(value[0]),'label':float(value[1]),'pred':float(value[2]),'error':float(value[3])})
    data_frame = pd.DataFrame(
        {'ID': ID, 'label_bev_x':label_bev_x,'label_bev_y':label_bev_y, 'result_bev_center_x': result_bev_center_x, 'result_bev_center_y': result_bev_center_y,'distance_perpective' : distance_perpective,'z':z,'gt':gtz,'x':x})
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

    world_location_x_index = 11
    world_location_y_index = 12
    world_location_z_index = 13
    label_one_location = []
    uv_data_dict = {}
    for key, value in label_data.items():
         if(value == []):
             continue
         uv_point_list = []
         count = len(value)
         for index in range(count):  # loop for result
             if(value[index][0] != 'Car' and value[index][0] != 'Van' and value[index][0] != 'Truck'):
                  continue

             world_location_x = float(value[index][world_location_x_index])
             #world_location_y = float(value[index][world_location_y_index])
             world_location_y = float(1.65)
             world_location_z = float(value[index][world_location_z_index])
             world_location = np.array([world_location_x,world_location_y,world_location_z]).T

             xmin = float(value[index][4])
             ymin = float(value[index][5])
             xmax = float(value[index][6])
             ymax = float(value[index][7])

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
             image_h = 384
             image_w = 1280
             if(uv_point[0]>= image_w or uv_point[0]<= 0 or uv_point[1]>= image_h or uv_point[1]<= 0):
                 uv_point = [-9999,-9999,-9999,-9999,-9999, -9999,-9999]
                 uv_point_list.append(uv_point)
                 continue
             
             uv_point = [uv_point[0],uv_point[1],fy,cy,fx,cx,world_location_z]
             uv_point_list.append(uv_point)    
             
         uv_data_dict["{}".format(key)] = uv_point_list
         
    return uv_data_dict


def world_to_uv_center(label_data, calib_data):

    label_one_location = []
    uv_data_dict = {}
    for key, value in label_data.items():
         if(value == []):
             continue
         uv_point_list = []
         count = len(value)
         for index in range(count):  # loop for result
             if(value[index][0] != 'Car' and value[index][0] != 'Van' and value[index][0] != 'Truck'):
                 continue
             
             xmin = float(value[index][4])
             ymin = float(value[index][5])
             xmax = float(value[index][6])
             ymax = float(value[index][7])
             u = float((xmin+xmax)/2)
             v = float((ymin+ymax)/2)
             uv_point = [u,v,1,1,1,1,1]            
             uv_point_list.append(uv_point)                
         uv_data_dict["{}".format(key)] = uv_point_list
         
    return uv_data_dict

def cal_distance(point1,point2):
    pass
    
def compare_data(label_bev_uv,result_data):
    
    be_center_x_index = 3
    bev_center_y_index = 4
    height_index = 5
    result_list = []
    
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

            result_bev_center_x = float(value[index][bev_center_x_index])
            result_bev_center_y = float(value[index][bev_center_y_index])
            result_height =  float(value[index][height_index])
            result_bev_center = np.array([result_bev_center_x,result_bev_center_y,result_height],dtype=np.float32)
            
            # label
            label_list = label_bev_uv[key]
            list_size = len(label_list)
            if list_size == 0:
                continue
            
            for index_label in range(list_size):
                label_one = label_list[index_label]
                label_x = label_one[0] 
                label_y = label_one[1] 
                label_height = label_one[2]
                label_bev_uv_one = np.array([label_x,label_y],dtype=np.float32)
                distance = np.sqrt(np.sum(np.square(label_bev_uv_one - result_bev_center[:2])))
                #distance = math.sqrt((label_x - result_bev_center_x )^2 + (label_y - result_bev_center_y )^2)
                #distance = (label_x - result_bev_center_x )^2 + (label_y - result_bev_center_y )^2
                error = np.abs(label_height - result_height)
                if(distance < min_distance):
                    min_distance = distance
                    result_one = [key, label_x,label_y,result_bev_center_x,result_bev_center_y,distance,label_height,result_height,error]
            # delete wrong match 
            wrong_match_threshold = 100
            distance_index = 5
            if(result_one[distance_index]>wrong_match_threshold):
                continue
            result_list.append(result_one)
            #import pdb;pdb.set_trace()        
    return result_list


def compare_data_2d(label_bev_uv,result_data):
    
    bev_center_x_index = 3
    bev_center_y_index = 4

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
            ##for orignal##
            r_xmin =  float(value[index][4])
            r_xmax =   float(value[index][6])
            r_ymin =   float(value[index][5])
            r_ymax =    float(value[index][7])
            result_bev_center_x = float((r_xmin+r_xmax)/2)
            result_bev_center_y = float((r_ymin+r_ymax)/2)
         
            #result_bev_center_x = float(value[index][bev_center_x_index])
            #result_bev_center_y = float(value[index][bev_center_y_index])
            result_bev_center = np.array([result_bev_center_x,result_bev_center_y],dtype=np.float32)
        
            # label
            label_list = label_bev_uv[key]
            list_size = len(label_list)
            if list_size == 0:
                continue
            
            for index_label in range(list_size):
                label_one = label_list[index_label]
                
                label_bev_x = label_one[0]
                label_bev_y = label_one[1]
                fy = label_one[2]
                cy = label_one[3]
                fx = label_one[4]
                cx = label_one[5]
                gt = label_one[6]

                result_z = float((fy*1.65)/(result_bev_center_y-cy))
                result_x = (result_bev_center_x*result_z-cx*result_z)/fx
               
                label_bev =  np.array([label_bev_x,label_bev_y])
                distance_center = np.sqrt(np.sum(np.square(label_bev - result_bev_center[:2])))
                #distance = math.sqrt((label_x - result_bev_center_x )^2 + (label_y - result_bev_center_y )^2)
                #distance = (label_x - result_bev_center_x )^2 + (label_y - result_bev_center_y )^2
                distance_perpective = np.linalg.norm(label_bev - result_bev_center)
                if(distance_center < min_distance):
                    min_distance = distance_center
                    result_one = [key, label_bev_x,label_bev_y,result_bev_center_x,result_bev_center_y,distance_perpective,result_z,result_x,gt]
            # delete wrong match 
            wrong_match_threshold = 20
            distance_index = 5
            if(result_one[distance_index]>wrong_match_threshold):
                continue
            result_list.append(result_one)
            #import pdb;pdb.set_trace()        
    return result_list
    
def str_key(key):
    if(len(key)==1):
        key = '00000'+str(key)
    if(len(key)==2):
        key = '0000'+str(key)
    if(len(key)==3):
        key = '000'+str(key)
    if(len(key)==4):
        key = '00'+str(key)
    if(len(key)==5):
        key = '0'+str(key)
    return key

def list_To_dict(list):
    ret_dict={}
    for ele in list:
        key = str_key(ele[0])
        if(key not in ret_dict.keys()):
            ret_dict[key] = []
            ret_dict[key].append([ele[1],ele[2],ele[3],ele[4]])
        else:
            ret_dict[key].append([ele[1],ele[2],ele[3],ele[4]])
    return ret_dict
    
                
def compare2dcenternet(datadict):
    cent_x = []
    cent_y = []
    perspective_x=[]
    perspective_y=[]
    index_name = []
    distance = []

    for key, value in result_data.items():
        if(value == []):
            continue
        count = len(value)
        for index in range(count):
             # loop for list # loop for dict
             index_name.append(key+'_{}'.format(index))
             print(value[index])
             cent_x.append(value[index][3])
             cent_y.append(value[index][4])
             perspective_x.append(value[index][5])
             perspective_y.append(value[index][6])

             ce = np.array([float(value[index][3]),float(value[index][4])])
             pe = np.array([float(value[index][5]),float(value[index][6])])
             dis = np.linalg.norm(ce-pe)
             distance.append(dis)
    

    newtable = pd.DataFrame(columns=['index','centx','centy','per_x','per_y'])


    newtable['index'] = index_name
    newtable['centx'] = cent_x
    newtable['centy'] = cent_y
    newtable['per_x'] = perspective_x
    newtable['per_y'] = perspective_y
    newtable['dis'] = distance


    return newtable
       
if __name__ == "__main__":

  
    # label_path = "C:\\Users\\rockywin.wang\\Desktop\\01_prj\\10_3d\\1pic\\label_2\\"
    # calib_path = "C:\\Users\\rockywin.wang\\Desktop\\01_prj\\10_3d\\1pic\\calib\\"
    # result_path ="C:\\Users\\rockywin.wang\\Desktop\\01_prj\\10_3d\\1pic\\result\\"
    # new_path="C:\\Users\\rockywin.wang\\Desktop\\01_prj\\10_3d\\bev_06.csv"
    
    label_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/label_2/"
    image_root_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/image_2/"
    calib_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/calib/"
    #result_path ="/mnt/nfs/zzwu/04_centerNet/CenterNet-master/CenterNet-master/exp/ddd/3dop/results/"
    #/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/exp/ddd/results
    #result_path ="/mnt/nfs/zzwu/04_centerNet/xjy/master-thesis/CenterNet-master/CenterNet-master/exp/ddd/3dop/results/"
    result_path="/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/3dop_resnet_18/results/"
    new_path   ="/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/3dop_resnet_18/result.csv"
    save_root_path ="/mnt/nfs/zzwu/04_centerNet/wxq_3d/CenterNet-master_vector_crop/exp/vis/372/"
    label_file_list = load_txt(label_path)

    calib_file_list = load_txt(calib_path)
    result_file_list = load_txt(result_path)
  
    label_data = data_to_dict(label_path,label_file_list)
    print('label suc')
    calib_data = data_to_dict(calib_path,calib_file_list)
    print('cal suc')
    result_data = data_to_dict(result_path,result_file_list)
    print('reusl suc')
    #label_bev_uv = world_to_uv(label_data,calib_data)
    #label_center = world_to_uv_center(label_data,calib_data)
    #label_center = world_to_uv(label_data, calib_data)
    label_center = world_to_uv_center(label_data, calib_data)
    
    result_list = compare_data_2d(label_center,result_data)

    write_csv_2d(result_list,new_path)


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







