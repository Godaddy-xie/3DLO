import numpy as np 
import os 
from IoU import *

def load_txt(file_path): # load result file
    txt_file = []
    filenames = os.listdir(file_path)  
    for item in filenames:
        #print("item:",item)
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
def get_3d_info(value):
    h, w ,l,x,y,z,angle = float(value[8]),float(value[9]),float(value[10]),float(value[11]),float(value[12]),float(value[13]),float(value[14])
    return h,w,l,x,y,z,angle
            
def data_to_dict(path,file_list):
    dict_data = dict()
    for index, file_name in enumerate(file_list):
        file_id = file_name.split(".")
        file_id = file_id[0]
        file_path = os.path.join(path,file_name)
        data = parse_txt(file_path)
        dict_data["{}".format(file_id)] = data

    return dict_data

def replace_data(label_data,result_data,replace_index):
     
    
    result_data_final = result_data

    for key, value in result_data.items():
        #if(len(value) == 0):
        if(value == []):
            continue
    
        count = len(value)
        for index in range(count):  # loop for result

            if(value[index][0] != 'Car'):
                continue

            h_result,w_result,l_result,x_result,y_result,z_result,angle_result = get_3d_info(value[index])
            result_3d_bbox = get_3d_box((h_result,w_result,l_result),angle_result,(x_result,y_result,z_result))
        

            # label
            label_list = label_data[key]
            if(label_list == []):
                continue
           
            min_IOU = -1 
            matchindex = -1
            current_index = 0 
            for label in label_list:
                if(label[0] != 'Car'):
                    current_index +=1
                    continue
                h_label,w_label,l_label,x_label,y_label,z_label,angle_label = get_3d_info(label)
                label_3d_bbox = get_3d_box((h_label,w_label,l_label),angle_label,(x_label,y_label,z_label))
                IOU_3d,IOU_2d = box3d_iou(result_3d_bbox,label_3d_bbox,label,value[index])
                if(IOU_3d>min_IOU):
                    min_IOU = IOU_3d
                    matchindex = current_index
                current_index +=1
            if(min_IOU<0.3):
                continue
            else:
                label_x_final =  0.5*float(value[index][replace_index])+ 0.5*float(label_list[matchindex][replace_index])
                label_y_final =  0.5*float(value[index][replace_index+1])+ 0.5*float(label_list[matchindex][replace_index+1])
                label_z_final =  0.5*float(value[index][replace_index+2])+ 0.5*float(label_list[matchindex][replace_index+2])
                label_z_final =  float(label_list[matchindex][replace_index+2])

                label_theta_final =  float(label_list[matchindex][replace_index+3])
                

            #result_data_final[key][index][replace_index] = str(label_h_final)
            result_data_final[key][index][replace_index+2] = str(label_z_final)
            #result_data_final[key][index][replace_index+1] = str(label_y_final)
            #result_data_final[key][index][replace_index+3] = str(label_theta_final)


    return result_data_final

def write_data_txt(data,path):

    for key, value in data.items():
        
        txt_path = path + key + ".txt"
        file_write = open(txt_path, "w+")
        count = len(value)
        if(count == 0):
            file_write.close()
            continue

        for index in range(count):
            data_one = value[index]

            for i in range(len(data_one)):  # loop for result
                file_write.write(data_one[i])
                if(i != len(data_one)-1):
                    file_write.write(" ")
            file_write.write("\n")
        file_write.close()
                
def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.mkdir(path)
    else:
        print("the folder exist！")
                

def add_height(result_data):
    copy_data = result_data
    for key, value in result_data.items():
        #if(len(value) == 0):
        if(value == []):
            continue
    
        count = len(value)
        for index in range(count):  # loop for result
            h = float(value[index][5])
            h = h + 50
            copy_data[key][index][7] = str(h)
    return copy_data

if __name__ == "__main__":

    label_path ="/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/training/label_2/"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/M3D-RPN-master/M3D-RPN-master/output/kitti_3d_multi_main/results/results_50000/data_centernetversion/"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/training/label_2/"
    result_path ="/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/center/results/"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/M3D-RPN-master/M3D-RPN-master/output/kitti_3d_multi_main/results/results_50000/data_from_centeret/"
    new_path="/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/center/results_mith/"
    mkdir(new_path)
    # label_path = "/home/wxq/03_prj/M3D-RPN/data/kitti_split1/validation/label_2/"
    # result_path ="/home/wxq/03_prj/kitti_eval/results/test/"
    # new_path="/home/wxq/03_prj/kitti_eval/results/test——save/"
    replace_index = 11

    label_file_list = load_txt(label_path)
    result_file_list = load_txt(result_path)

    label_data = data_to_dict(label_path,label_file_list)
    result_data = data_to_dict(result_path,result_file_list)

    data = add_height(result_data)
    write_data_txt(data, new_path)









