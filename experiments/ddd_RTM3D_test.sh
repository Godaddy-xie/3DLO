cd src_9key
# train
#python main.py ddd --exp_id RTM  --dataset kitti --kitti_split 500pic --batch_size 8 --arch res_18  --num_epochs 300  --lr_step 70,100 --gpus 2,3 --load_model /mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/RTM/model_100.pth

 # test
python test.py ddd --exp_id RTM --arch res_18 --dataset kitti --kitti_split 1pic --load_model  /mnt/nfs/zzwu/04_centerNet/xjy/master_thesis/CenterNet-master/CenterNet-master/exp/ddd/RTM/model_100.pth

cd ..
