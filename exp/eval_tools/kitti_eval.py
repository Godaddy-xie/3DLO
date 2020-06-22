import numpy as np
import os
import pandas as pd
import math
import json
from IoU import *    
diffcult = [0,1,2]
#diffcult = [0]
N_SAMPLE_PTS = 41
metric = [0.7,0.5,0.5]
MAX_OCCLUSION = [0,1,2]
MAX_TRUNCATION = [0.15,0.3,0.5]
MIN_HEIGHT     = [40, 25, 25]
MIN_OVERLAP = [[0.7, 0.5, 0.5], [0.7, 0.5, 0.5], [0.7, 0.5, 0.5]]

fp_dict = {}


cls = ['Car','Pedestrain','Cyclist']

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

def get_3d_info(value):
    w, h ,l,x,y,z,angle = float(value[8]),float(value[9]),float(value[10]),float(value[11]),float(value[12]),float(value[13]),float(value[14])
    box = get_3d_box(( w, h ,l),angle,(x,y,z))
    return box

def IOU_3D(gt,det):
    #print(gt)
    gt_box = get_3d_info(gt)
    det_box = get_3d_info(det)
    IOU_3D,IOU_2D = box3d_iou(gt_box,det_box,gt,det)
    return IOU_3D

def load_data(dir):
    file_list = load_txt(dir)
    data = data_to_dict(dir,file_list)    
    return data

def getThresholds(v,n_groundtruth):
     #// holds scores needed to compute N_SAMPLE_PTS recall values
    t = []
    recall_approximate = []

    #// sort scores in descending order
    #// (highest score is assumed to give best/most confident detections)
    v = sorted(v,key = float,reverse =True)
   
   
    #// get scores for linearly spaced recall
    current_recall = 0.0
    for i in range(len(v)):
    #// check if right-hand-side recall with respect to current recall is close than left-hand-side one
    #// in this case, skip the current detection score
        l_recall, r_recall, recall = 0.0 ,0.0 ,0.0
        l_recall = (double)(i+1)/n_groundtruth
        if(i<(len(v)-1)):
            r_recall = (double)(i+2)/n_groundtruth
        else:
            r_recall = l_recall

        if( (r_recall-current_recall) < (current_recall-l_recall) and i<((len(v)-1))):
            continue  
    ##// left recall is the best approximation, so use this and goto next recall step for approximation
        recall = l_recall

    #// the next recall step was reached
        t.append(v[i])
        recall_approximate.append(recall)
        current_recall += 1.0/(N_SAMPLE_PTS-1.0)
    
    return t,recall_approximate



def computeStastic(current_class,gt,det,dc,ignored_gt,ignored_det,compute_fp,detMatchGt_Flag,boxoverlap,metric,thershold=[]):
    currentclass_number_metric = {'Car':0,'Pedestrain':1,'Cyclist':2}
    metric_number = currentclass_number_metric[current_class]
    
    NO_DETECTION = -100000
    assigned_detection = [False]*(len(det))
    ignored_threshold = [False]*(len(det))
    #-------------------------#
    ##for differnt thershold ##
    #------------------------#
    # print(det)
    # assert 0
    if(compute_fp):
        # print(thershold)
        for i in range(len(det)):
            if(float(det[i][15])<thershold):
                ignored_threshold[i] = True
        # print(ignored_threshold)
    ## evaluate all ground truth boxes
    fn = 0
    fp = 0
    tp = 0
    v = []
    MatchID = []
    MatchVaild = []
    for i in range(len(gt)):
        if(ignored_gt[i]==-1):
                continue
        det_idx   = -1
        valid_detection = NO_DETECTION
        max_overlap   =  0
        assigned_ignored_det = False
       
                
        for j in range(len(det)):
      ## detections not of the current class, already assigned or with a low threshold are ignored
            
            if(ignored_det[j]==-1):
                continue
            if(assigned_detection[j]):
                continue
            if(ignored_threshold[j]):
                continue
            
      ## find the maximum score for the candidates and get idx of respective detection
            overlap = boxoverlap(gt[i],det[j])
       
            #debug
            # if(compute_fp == False and float(det[j][15])==0.69):
            #     #print("gt {} det {} overlap {}".format(i,j,overlap))
            #     print(float(det[j][15]))
            #     import pdb;pdb.set_trace()          
            if(compute_fp == False and  overlap>MIN_OVERLAP[metric][metric_number] and float(det[j][15])>valid_detection):
                det_idx  = j
                valid_detection = float(det[j][15])
            elif(compute_fp == True and overlap>MIN_OVERLAP[metric][metric_number] and (overlap>max_overlap or assigned_ignored_det) and ignored_det[j]==0):
                #import pdb;pdb.set_trace()
               
                max_overlap     = overlap
                det_idx         = j
                valid_detection = 1
                assigned_ignored_det = False
            elif(compute_fp == True and overlap>MIN_OVERLAP[metric][metric_number] and  valid_detection==NO_DETECTION and ignored_det[j]==1):
            
                det_idx              = j
                valid_detection      = 1
                assigned_ignored_det = True
         #=======================================================================#
         #  compute TP, FP and FN
         #=======================================================================#

         #// nothing was assigned to this valid ground truth
        if(valid_detection==NO_DETECTION and ignored_gt[i]==0):
            fn +=1
         #// only evaluate valid ground truth <=> detection assignments (considering difficulty level)
         ## reduce falspoitive##

        elif(valid_detection!=NO_DETECTION and (ignored_gt[i]==1 or ignored_det[det_idx]==1)):
            assigned_detection[det_idx] = True
        
        elif(valid_detection!=NO_DETECTION):
        #// write highest score to threshold vector
            # if(compute_fp == False):
            #         print(" add gt {} det {} overlap {}".format(i,det_idx,overlap))
            tp+=1
            # if(float(det[det_idx][-1])<0.95):
            #     det[det_idx][-1] = 0.95
            v.append(float(det[det_idx][15]))
            assigned_detection[det_idx] = True

         ## if FP are requested, consider stuff area
        #debug
        # if(compute_fp == False):
        #     print(i,valid_detection,det_idx)
    # print(assigned_detection)
    # print(v)
    # print(tp)
    # assert 0
    if(compute_fp):
        #// count fp
        for i in range(len(det)):
        ## count false positives if required (height smaller than required is ignored (ignored_det==1)
        ## reduce falspoitive##
            if(assigned_detection[i] == False  and ignored_det[i]== 0  and ignored_threshold[i] == False ):
                fp += 1
        
        
        #// do not consider detections overlapping with stuff area
        nstuff = 0
        for i in range(len(dc)):
            for i in range(len(det)):
        #// detections not of the current class, already assigned, with a low threshold or a low minimum height are ignored
                if(assigned_detection[j]):
                    continue
                if(ignored_det[j]==-1 or ignored_det[j]==1):
                    continue
                if(ignored_threshold[j]):
                    continue

        #// compute overlap and assign to stuff area, if overlap exceeds class specific value
                overlap = boxoverlap(det[j], dc[i], 0)
                if(overlap>MIN_OVERLAP[metric][current_class]):
                    assigned_detection[j] = 1 
                    nstuff +=1
        fp -= nstuff
        

    ret = {}
    ret['v'],ret['tp'],ret['fp'],ret['fn'],ret['assigned_detection'] = v,tp,fp,fn,assigned_detection
    #debug
    # if(compute_fp):
    #     import pdb;pdb.set_trace()
    #     print(ret['tp'],ret['fp'],ret['fn'])
    return ret
def cleandata (det,gt,current_class,difficulty):
    i_gt, i_det ,dc ,gt_num = [],[],[],0
    

    for i in range(len(gt)):
        height = float(gt[i][7])- float(gt[i][5])
        valid_class = 0
        
        if(gt[i][0] == current_class):
            valid_class = 1
        elif(gt[i][0]== 'Van' and current_class =='Car'):
            valid_class = 0
        else:
            valid_class = -1
        
        ignore = False


        if(float(gt[i][2])>MAX_OCCLUSION[difficulty] or float(gt[i][1])>MAX_TRUNCATION[difficulty] or height<MIN_HEIGHT[difficulty]):
            ignore = True
        
       
        
        if(valid_class==1 and ignore==False):
            i_gt.append(0)
            gt_num +=1
        elif(valid_class==0 or (ignore == True and valid_class==1)):
            i_gt.append(1)
        else:
            i_gt.append(-1) #for other class

    for i in range(len(gt)):
        if(gt[i][0]=='Dontcare'):
            dc.append(gt[i])
    
    #-------------------#
    #  clean detection  #
    #-------------------#

    for i in range(len(det)):
        
    ## neighboring classes are not evaluated
        valid_class= 0
        if(det[i][0] == current_class):
            valid_class = 1
        else:
            valid_class = -1

        height = float(det[i][7]) - float(det[i][5])

        
    #// set ignored vector for detections
        #------for bev-------#

        if(height<MIN_HEIGHT[difficulty]):
            i_det.append(1)
        
        elif (valid_class==1):
            i_det.append(0)
        else:
            i_det.append(-1)  
    
    
    return  i_gt, i_det ,dc ,gt_num



def eval_class(gt_dir,det_dir,current_class,difficulty,detMatchGt_Flag=False):
    gt  = load_data(gt_dir)
    det = load_data(det_dir)
    ignored_gt = {}
    ignored_det ={}
    dontcare = {}
    v = []
    thershold = [] 
    n_gt = 0
    flag = False
    detMatchGt= {}
    n_det = 0
    metric = {'image':0,'ground':1,'3dbbox':2}
    for key,det_value in det.items():
      
        #debug
        # if(key !="000181"):
        #     continue
        gt_value = gt[key]

        ##---num_detection---##
        for i in range(len(det_value)):
            # if float(det_value[i][-1])<0.75:
            #     det_value[i][-1] = 0.75
            n_det +=1
        ##---num_detection---##

        i_gt, i_det ,dc ,gt_num = cleandata (det_value,gt_value,current_class,difficulty)
        ignored_det[key] = i_det
        ignored_gt[key] = i_gt
        dontcare[key] = dc
        n_gt +=gt_num
        #print("-------{}----------".format(key))
        prtmp = computeStastic(current_class,gt_value,det_value,dc,i_gt,i_det,flag,detMatchGt_Flag,IOU_3D,metric['3dbbox'],[])
       

             
        for i in  prtmp['v']:
                v.append(i)

        #------modify score-----#  
        '''
        for index in range(len(prtmp['assigned_detection'])):
            if (prtmp['assigned_detection'][index] == True and float(det_value[index][-1])<0.95):
                        det[key][index][-1] =0.95
        '''
        #------modify score-----# 


    flag = True
    print(n_det,len(v),n_gt)
    # print("-----------------------")
    # import pdb;pdb.set_trace()
    thresholds,recall_approximate = getThresholds(v, n_gt)
    print(thresholds)
    # print(recall_approximate)
   
    # print("--------------------------------------------------------------------------------------------------")
    # #debug
    # print(ignored_gt)
    # print(ignored_det)
    # print(thresholds)
    #print(v)
    #print(n_gt)
    
    ##-----------------------------##
    # fn,fp,fn for differnet recall#
    ##-----------------------------##
    pr = []

    for i in range(len(thresholds)):
        dict = {'tp':0,'fp':0,'fn':0}
        pr.append(dict)
    fp_dict = {}
    for key,det_value in det.items():
        #pr = [{'tp':0,'fp':0,'fn':0}]*len(thresholds)
        gt_value = gt[key]
        fp_dict[key] = 0
        
        for i in range(len(thresholds)):
            prtmp  = computeStastic(current_class,gt_value,det_value,dontcare[key],ignored_gt[key],ignored_det[key],flag,detMatchGt_Flag,IOU_3D,metric['3dbbox'],thresholds[i])
            if(i == len(thresholds)-1):
                fp_dict[key] = prtmp['fp']
            pr[i]['tp'] += prtmp['tp']
            pr[i]['fp'] += prtmp['fp']
            pr[i]['fn'] += prtmp['fn']
        #import pdb;pdb.set_trace()   
    #         print("#####-----{}-----threah--- key__{}###########".format(thresholds[i],key))
    #         print(pr[i])
    print(pr)
    ##--TEST FOR REPLACE FP USE M3D'S RESULTS--##
    '''
    for i in range(len(thresholds)):
        pr[i]['fp'] = pr[i]['tp']*2
    '''
    ##--TEST FOR REPLACE FP USE M3D'S RESULTS--##

    ##-----------------------------##
    # precision for differnet recall#
    ##-----------------------------##

    precision = [0]*N_SAMPLE_PTS
    recall = [0]*N_SAMPLE_PTS
    result = [ ]
    for i in range(N_SAMPLE_PTS):
        dict = {'tp':0,'fp':0,'fn':0,'recall':0,'presion':0}
        result.append(dict)
    
    for i in range(len(thresholds)):
        r = pr[i]['tp']/(double)(pr[i]['tp'] + pr[i]['fn'])
        print("----recall_{}----{}----".format(thresholds[i],r))
        recall.append(r)
        p = pr[i]['tp']/(double)(pr[i]['tp'] + pr[i]['fp'])
        print("----presion--{}----{}----".format(thresholds[i],p))
        precision[i] = p
        
        result[i]['recall'] = r
        result[i]['presion'] = p
        result[i]['tp'] = pr[i]['tp']
        result[i]['fp'] = pr[i]['fp']
        result[i]['fn'] = pr[i]['fn']

    # print(precision)
    for i in range(len(thresholds)):
        precision[i] = max(precision[i:])
    #print(precision)
    return precision,recall,result,n_gt,fp_dict,thresholds,n_det

def compute_ap(precisions,N,level):
    i = 0
    sum = 0
    print(precisions)
    while i<len(precisions) :
        sum += precisions[i]
        i+=1
    print("{}.  AP is " .format(level),sum/40.0*100)
    return sum/40.0*100

def get_max_fpkey(fp_dict):
    ret = sorted(fp_dict.items(),key = lambda x:x[1],reverse = True)
    return ret
import sys  
if __name__ == "__main__":
    gt_dir = "/mnt/nfs/zzwu/04_centerNet/CenterNet-master_lyc/data/kitti/training/label_2/"
    det_dir ="/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/bev_500/results/"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/3dop_resnet_18/data_val/"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/M3D-RPN-master/M3D-RPN-master/output/kitti_3d_multi_main/results/results_50000/data_centernetversion/"
    #"/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/bev/results/"
    #result_dir = "/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/anchor/res.json"
    result_dir = "/mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/bev_500/"
    
    
    exp = sys.argv[1]
    
    det_dir =sys.argv[2]+'/'
    
    result_dir = sys.argv[3]+'/'

   

    
    re_diff = {}
    fp_vis = []
    for level in diffcult:
        #import pdb;pdb.set_trace()
        pr,recall,result,ngt,fp_dict,thresholds,n_det=eval_class(gt_dir,det_dir,cls[0],level)
        #import pdb;pdb.set_trace()
        fp_vis.append(get_max_fpkey(fp_dict)[:10])
        re_diff[level] = result
        re_diff['{}_gt_num'.format(level)] = ngt
        re_diff['{}_det_num'.format(level)] = n_det
        re_diff['{}_pr'.format(level)] = pr
        re_diff['{}_ap'.format(level)] =compute_ap(pr,N_SAMPLE_PTS,level)
        re_diff['{}_data'.format(level)] = get_max_fpkey(fp_dict)

    ##TO JSON#
    result_json = result_dir+'result_eval_kitti.json'
    with open(result_json,'w',encoding = 'utf-8') as f:
        json.dump(re_diff,f,indent=6)
    
    file_fp = open(result_dir+'fp.txt','a')

    for level in diffcult:
        data = re_diff[level]
        file_fp.write("---------diffcult---{}-------\n".format(level))
        file_fp.write("ALl_GT_Num : {} ALL_Det_NUM: {} \n".format(re_diff['{}_gt_num'.format(level)],re_diff['{}_det_num'.format(level)]))
        file_fp.write('AP__{}  is {}'.format(level,re_diff['{}_ap'.format(level)]))
        file_fp.write("\n")
        file_fp.write("\n")

        for i in range(len(data)):
            fp = data[i]['fp']
            tp = data[i]['tp']
            fn = data[i]['fn']
            pr = data[i]['presion']
            recall = data[i]['recall']
            if(i<len(thresholds)):
                thre = thresholds[i]
            else:
                thre = 0.
            file_fp.write('level_{}--  threhold :{:.5f},TP:{},FP: {},FN: {} ----presion: {:.4f} ----reacall :{:.4f} \n'.format(i,thre,tp,fp,fn,pr,recall))
            #file_fp.write("\n")

        file_fp.write("\n")
        file_fp.write("\n")
        file_fp.write("#####################################################################\n")
    file_fp.close()
    datafile_fp = open(result_dir+'data.txt','a')
    for level in diffcult:
        datafile_fp.write("---------diffcult--{}-------------\n".format(level))
        data_fp = re_diff['{}_data'.format(level)]
        for ele in data_fp:
            datafile_fp.write('key----fp: {} \n'.format(ele))
        datafile_fp.write("\n")
        datafile_fp.write("\n")
    
        datafile_fp.write("#####################################################################\n")
    datafile_fp.close()

    for i in diffcult:
         MAX_FILE = open(result_dir+'VIS_{}.txt'.format(i),'a')
         keyname = fp_vis[i]
         for ele in keyname:
             MAX_FILE.write('{}\n'.format(ele[0]))
         MAX_FILE.close()


    # maxvistxt
    # json_str = json.dumps(re_diff)
    # file_name = result_dir
    # print(re_diff['0_num'],re_diff['1_num'],re_diff['2_num'])
    # with open(file_name,'w') as f:
    #         f.write(json_str)