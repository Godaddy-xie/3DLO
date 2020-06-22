cd src_arrration
#python main.py ddd --exp_id aggration_conf  --dataset kitti   --kitti_split 3dop  --batch_size 32  --num_epochs 60  --lr_step 25,45   --gpus 3,2,1,0

 test
python test.py ddd --exp_id aggration_conf  --dataset kitti --kitti_split 3dop  --peak_thresh 0.15 --resume

cd ..
