cd src_iou
python main.py ddd --exp_id DIOU  --dataset kitti   --kitti_split 3dop  --batch_size 16  --num_epochs 60  --lr_step 25,45    --gpus 2,3 --resume

 test
python test.py ddd --exp_id DIOU  --dataset kitti --kitti_split 3dop  --peak_thresh 0.2 --resume

cd ..
