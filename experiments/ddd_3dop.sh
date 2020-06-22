cd src
# train
#python main.py ddd --exp_id 3dop_resnet_18 --dataset kitti --kitti_split 3dop --batch_size 24  --master_batch 7 --num_epochs 150 --lr_step 120 --gpus 2,3 --arch res_18 --load_model /mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/3dop_resnet_18/model_70.pth
# test
python test.py ddd --exp_id 3dop_resnet_18 --dataset kitti --kitti_split 1pic --arch res_18   --peak_thresh 0.5 --resume 
cd ..
