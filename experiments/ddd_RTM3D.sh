cd src_9key
# train
#python main.py ddd --exp_id RTM  --dataset kitti --kitti_split 3dop --batch_size 8 --arch res_18  --num_epochs 300 --lr 0.002  --lr_step 10,150,180 --gpus 2,3 
 # test
python test.py ddd --exp_id RTM --arch res_18 --dataset kitti --kitti_split 3dop  --resume
cd ..
