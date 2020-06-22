cd src_bev
# train
#python main.py ddd --exp_id bev_500 --dataset kitti --kitti_split 500pic --batch_size 8 --master_batch 4  --num_epochs 150  --lr_step 90,120 --gpus  2,3   
# test
python test.py ddd --exp_id bev --dataset kitti --kitti_split 3dop  --peak_thresh 0.2 --resume
cd ..
