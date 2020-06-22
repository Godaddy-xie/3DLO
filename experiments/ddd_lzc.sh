
#train
cd src_smoke_v1

#python main.py ddd --exp_id IOU  --dataset kitti   --kitti_split 3dop  --batch_size 32  --num_epochs 60  --lr_step 25,45 --trainval  --gpus 0,1,2,3 --resume
 # test
python test.py ddd --exp_id IOU   --dataset kitti --kitti_split 3dop --resume --trainval

cd ..
