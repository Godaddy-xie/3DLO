cd src
#python main.py ddd --arch res_18 --exp_id center_position --dataset kitti --kitti_split 3dop --batch_size 32   --master_batch 8 --num_epochs 150  --lr_step 90,120 --gpus 0,1,2,3 
#
# test
python test.py ddd --exp_id center_position --arch res_18  --dataset kitti --kitti_split 3dop --peak_thresh 0.75 --load_model /mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/center_position/model_90.pth
cd ..
