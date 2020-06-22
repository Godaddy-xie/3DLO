cd src_locf
# train
#python main.py ddd --exp_id conf --dataset kitti --kitti_split 3dop   --batch_size 28   --num_epochs 60  --lr_step 25,45 --gpus 2,4,5,6 
# test
python test.py ddd --exp_id conf --dataset kitti --kitti_split 3dop  --peak_thresh 0.5 --load_model /mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/conf/model_45.pth
cd ..
