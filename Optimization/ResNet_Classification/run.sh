
# for resnet
CUDA_VISIBLE_DEVICES=0,1 python3 main.py --epoch 200 --batch-size 128 --lr 0.04 --momentum 0.9 --wd 5e-4 -ct 10
