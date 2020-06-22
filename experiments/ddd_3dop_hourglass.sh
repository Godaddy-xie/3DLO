cd src
# train
#python main.py ddd --exp_id 3dop_hourglass --dataset kitti --kitti_split 3dop --batch_size 4 --master_batch 2 --num_epochs 70 --lr_step 45,70 --gpus 0,1 --arch hourglass
# test
python test.py ddd --exp_id 3dop_hourglass --dataset kitti --kitti_split 3dop --arch hourglass --resume
cd ..
