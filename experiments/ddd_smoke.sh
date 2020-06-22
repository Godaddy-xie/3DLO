cd src_smoke
python main.py ddd --exp_id S_3k_v1  --dataset kitti   --kitti_split 3dop  --batch_size 32  --num_epochs 60  --lr_step 25,45    --gpus 0,1,2,3

 test
#python test.py ddd --exp_id smoke_600pic    --arch res_18  --dataset kitti --kitti_split 500pic  --peak_thresh 0.2 --resume

cd ..
