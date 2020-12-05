set -ex

python test.py --dataroot ./datasets/brats --name brats_1 --model attention_gan --dataset_mode unaligned --norm instance --phase test --no_dropout --load_size 256 --crop_size 256 --batch_size 4 --gpu_ids 0 --num_test 500 --saveDisk