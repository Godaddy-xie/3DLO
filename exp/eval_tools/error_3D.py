import numpy as np
import os
import pandas as pd
import math
from IoU import * 
import sys   
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


def get_iou(result, label):
    result_left_x = float(result[4])
    result_left_y = float(result[5])
    result_right_x = float(result[6])
    result_right_y = float(result[7])

    label_left_x = float(label[4])
    label_left_y = float(label[5])
    label_right_x = float(label[6])
    label_right_y = float(label[7])

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

def write_csv(result, path):
    #result_one = [key, label_x,label_y,result_bev_center_x,result_bev_center_y,distance]
    # result_one = [key, label_x,label_y,result_bev_center_x,result_bev_center_y,distance_center,distance_perpective]
    ID = []
    xmingt = []
    xmaxgt = []
    ymingt = []
    ymaxgt = []
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

def computedis(label,result):
    p1 = np.array([(float(label[4])+float(label[6]))/2,(float(label[5])+float(label[7]))/2])
    p2 = np.array([(float(result[4])+float(result[6]))/2,(float(result[5])+float(result[7]))/2])
    dis = np.linalg.norm(p1-p2)
    return dis

def generate_item(result,label,IOU,key):
    ret = []
    ret.append(key)
    dis = computedis(label,result)
    IOU = IOU
    error_x = abs(float(result[11]) - float(label[11]))
    error_y = abs(float(result[12]) - float(label[12]))
    error_z = abs(float(result[13]) - float(label[13]))
    for i in range(len(label)):
        ret.append(label[i])
    for j in range(len(result)):
        ret.append(result[j])
    ret.append(error_x)
    ret.append(error_y)
    ret.append(error_z)
    ret.append(IOU)
    ret.append(dis)
    return ret

def NoneMatch(label,key):
    ret = []
    ret.append(key)
    for j in range(len(label)):
        ret.append(label[j])
    for j in range(len(label)):
        ret.append(-1)
    for i in range(6):
        ret.append(-1)
    ret[-2] = 0
    return ret

def get_3d_info(value):
    h, w ,l,x,y,z,angle = float(value[8]),float(value[9]),float(value[10]),float(value[11]),float(value[12]),float(value[13]),float(value[14])
    return h,w,l,x,y,z,angle


def cal_error(label_dict,result_dict):

    ret_list = []
    for key, value in result_dict.items():
        
        # if(key!='007475'):
        #     continue 
        # import pdb; pdb.set_trace() 
        if(value == []):
            continue
        count = len(value)
        for index in range(count): 
            # if(index !=2):
            #     continue
            #import pdb;pdb.set_trace()
            #if(value[index][0] != 'Car' and value[index][0] != 'Van' and value[index][0] != 'Truck'):
            #    continue
            if(value[index][0] != 'Car'):
                continue
            h_result,w_result,l_result,x_result,y_result,z_result,angle_result = get_3d_info(value[index])
            result_3d_bbox = get_3d_box((h_result,w_result,l_result),angle_result,(x_result,y_result,z_result))
            
            #import pdb;pdb.set_trace()
            label_list = label_dict[key]

            if(label_list == []):
                continue
           
            min_IOU = -1 
            label_Match = []
            for label in label_list:
                if(label[0] != 'Car'):
                    continue
                h_label,w_label,l_label,x_label,y_label,z_label,angle_label = get_3d_info(label)
                label_3d_bbox = get_3d_box((h_label,w_label,l_label),angle_label,(x_label,y_label,z_label))
                IOU_3d,IOU_2d = box3d_iou(result_3d_bbox,label_3d_bbox,label,value[index])
                # if(IOU_3d>0.7):
                #     print("----------------------------")
                #     print("IOU: " ,IOU_3d)
                #     print('I find ',label)
                if(IOU_3d>min_IOU):
                    min_IOU = IOU_3d
                    label_Match = generate_item(value[index],label,IOU_3d,key)
                else:
                    continue

       
            wrong_match_threshold = -1
            distance_index = -2
            if(label_Match == []):
                label_Match = NoneMatch(label,key)
            elif(label_Match[distance_index]<wrong_match_threshold):
                label_Match = NoneMatch(label,key)
           
            ret_list.append(label_Match)

    return ret_list


def toCsv(retlist,path):
    

    col = ['ID','name','Tr','Zhe','budong','gtxmin','gtxmax','gtymin','gtymax','gtw','gth','gtl','gtx','gty','gtz','gttheta',\
    'name','Tr','Zhe','budong','rexmin','rexmax','reymin','reymax','rew','reh','rel','rex','rey','erz','retheta','confidence',\
    'e_x','e_y','e_z','IOU','dis'
    ]
    table = pd.DataFrame(columns= col,data = retlist)
    table.to_csv(path)
    return 
if __name__ == "__main__":


    
    #label_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/label_2/"
    image_root_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/image_2/"
    calib_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/calib/"
    label_path = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/label_2/"
    
    result_path= "/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/bev/results/"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/M3D-RPN-master/val_result/data/"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/3dop_resnet_18/data/"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/M3D-RPN-master/M3D-RPN-master/output/kitti_3d_multi_main/results/results_50000/data/"
    new_path   =  "/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/bev/centernet_3d_iou_recall.csv"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/M3D-RPN-master/val_result/res18_3d_iou.csv"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/3dop_resnet_18/res18_allerror_140_train.csv"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/m3d_result.csv"
    save_root_path ="/mnt/nfs/zzwu/04_centerNet/wxq_3d/CenterNet-master_vector_crop/exp/vis/372/"

    exp = sys.argv[1]
    result_path =sys.argv[2]+'/'
    new_path = sys.argv[3] +'/'+'Error_3DIOU'+'_{}'.format(exp)+'.csv'
    #import pdb;pdb.set_trace() 
    label_file_list = load_txt(label_path)
    calib_file_list = load_txt(calib_path)
    result_file_list = load_txt(result_path)

    label_data = data_to_dict(label_path,label_file_list)
    print('label suc')
    calib_data = data_to_dict(calib_path,calib_file_list)
    print('cal suc')
    result_data = data_to_dict(result_path,result_file_list)
    print('reusl suc')
    r = cal_error(label_data,result_data)
    toCsv(r,new_path)
    



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







