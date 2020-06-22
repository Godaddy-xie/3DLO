cd src
# train
python main.py ddd --exp_id 2pic__1 --dataset kitti --kitti_split 2pic --batch_size 2  --num_epochs 70 --lr_step 45,60 --gpus 1
# test
#python test.py ddd --exp_id 2pic --dataset kitti --kitti_split 2pic --resume
cd ..
