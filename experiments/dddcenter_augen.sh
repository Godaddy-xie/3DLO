cd src_center
# train
python main.py ddd --exp_id CenteAug_flip --dataset kitti --kitti_split 3dop --batch_size 24  --master_batch 8 --num_epochs 140  --lr_step 70,100 --gpus 2,4,5,6,0
# testi
python test.py ddd --exp_id CenteAug_flip --dataset kitti --kitti_split 3dop --peak_thresh 0.75 --resume
cd ..
