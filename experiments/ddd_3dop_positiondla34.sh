cd src
#python main.py ddd --exp_id 3dop_res_position --dataset kitti --kitti_split 3dop  --batch_size 16  --master_batch 7 --num_epochs 150 --lr_step 120 --gpus 2,3 --arch res_18

python test.py ddd --exp_id 3dop_res_position --dataset kitti --kitti_split 3dop  --arch res_18 --load_model /mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/3dop_res_position/model_120.pth 
