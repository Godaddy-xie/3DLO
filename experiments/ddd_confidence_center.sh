cd src
# train
python main.py ddd --exp_id confidence --dataset kitti --kitti_split 1pic --batch_size 1 --master_batch 1 --num_epochs 140  --lr_step 90,120 --gpus  1 
# testi
#python test.py ddd --exp_id center --dataset kitti --kitti_split 3dop --peak_thresh 0.2 --load_model /mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/center/model_3700.pth
cd ..
