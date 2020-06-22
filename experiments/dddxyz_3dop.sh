cd src
# train
#python main.py dddxyz --exp_id 3dop_xyz --dataset kitti --kitti_split 3dop --batch_size 24  --num_epochs 140 --lr_step 120 --gpus 0,1,2,3 --arch res_18 
# test
python test.py dddxyz --exp_id 3dop_xyz --dataset kitti --kitti_split 3dop  --arch res_18 --resume
cd ..
