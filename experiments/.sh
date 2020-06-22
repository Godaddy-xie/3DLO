
# train
python main.py dddxyz --exp_id 3dop_xyz_resnet--dataset kitti --kitti_split 1pic --batch_size 1  --master_batch 1 --num_epochs 150 --lr_step 120 --gpus 3 --arch res_18 
# test
#python test.py dddsyz --exp_id 3dop_resnet_18 --dataset kitti --kitti_split 3dop --arch res_18  --resume
cd ..
