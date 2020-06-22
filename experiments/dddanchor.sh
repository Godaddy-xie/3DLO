cd src_anchor
# train
python main.py ddd --exp_id anchor_v1 --dataset kitti --kitti_split 500pic --arch res_18 --batch_size 32 --master_batch 8  --num_epochs 150  --lr_step 90,120 --gpus 0,1,2,3 
# test
python test.py ddd --exp_id anchor_v1 --dataset kitti --kitti_split 500pic --arch res_18 --peak_thresh 0.8 --resume
cd ..
