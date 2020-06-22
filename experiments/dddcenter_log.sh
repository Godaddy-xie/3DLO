cd src_center
# train
#python main.py ddd --exp_id center_log --dataset kitti --kitti_split 3dop --batch_size 32 --master_batch 8 --num_epochs 140 --arch res_18 --lr_step 90,120 --gpus  0,1,2,3 
# test
python test.py ddd --exp_id center_log --dataset kitti --kitti_split 3dop --peak_thresh 0.8 --arch res_18 --resume
cd ..
