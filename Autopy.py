import os
import sys
import time

def logfiter(loglist):

    pass
def creattxt(path):
    pass



if __name__ =="__main__":

    time = ''.join(time.asctime().replace(" ","_"))
    time = time.split("_")
    time = ''.join([time[1],'_',time[2],'_',time[-1]])

    expname = sys.argv[1]
    rootdir = sys.argv[2]
    src_dir = sys.argv[3]

    exp_result_dir = os.path.join(src_dir,'exp',expname)
    tools_dir = os.path.join(src_dir,'exp','eval_tools')
    #make outputdir#

    output_dir = os.path.join(rootdir,expname+'_'+time)

    print('-------------------------------------')
    print(' Calculate Error and IOU Match and AP')
    print('-------------------------------------')
    result_dir = os.path.join(exp_result_dir,'results')
    error_dir = os.path.join(output_dir,'error')

    os.system('cd {}'.format(tools_dir))
    os.system('python calz.py {} {} {}'.format(expname,result_dir,error_dir))
    os.system('python error_3D.py {} {} {}'.format(expname, result_dir, error_dir))
    os.system('python kitti_eval.py {} {} {}').format(expname, result_dir, error_dir)


    print('-------------------------------------')
    print('         Vis the max Error           ')
    print('-------------------------------------')
    txt_dir = creattxt()
    visdir = os.path.join(output_dir,expname)
    os.system('python vis.py {} {}'.format(txt_dir,visdir))

    print("-------------------------------------")
    print("-----         copy log       --------")
    print("-------------------------------------")

    filelist = os.listdir(exp_result_dir)
    log_file_path = logfiter(filelist)
    dst = os.path.join(error_dir,'log')
    os.system('cp {} {}'.format(log_file_path,dst))

    print('-------------------------------------')
    print("-----     ZIP ALL RESULTS    --------")
    print("-------------------------------------")

    os.system("tar zcvf {}".format(output_dir))

    print("output is saved in {}".output_dir)



