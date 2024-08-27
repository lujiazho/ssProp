# ssProp

The official implementation of the paper _ssProp: Energy-Efficient Training for Convolutional Neural Networks with Scheduled Sparse Back Propagation_ [[pdf]](https://arxiv.org/abs/2408.12561) by Lujia Zhong, Shuo Huang, Yonggang Shi.

# Run

- To train a normal/efficient resnet50 with 0.4 drop rate, simply run
```shell
for mode in normal efficient # normal, efficient
do
  CUDA_VISIBLE_DEVICES=0 python train_classifier.py --exp runs/newdropout \
  --task CIFAR100 --mode $mode --model resnet50 --drop_mode constant --bs 128 --epochs 300 --lr 0.0002 \
  --builtin --unified --percentage 0.2 --min_percentage 0.2 --interleave --warmup 0 --seed 42 --use_gpu --by_epoch
done
```
where the `--percentage` and `--min_percentage` represent the percentage to be kept, so `0.2` indicates drop 80% for an epoch. The `--interleave` indicates to use a bar-like drop schedular.

- To train an efficient resnet50 with `0.2+0.2` mode (0.2 drop rate + 0.2 dropout), simply run
```shell
CUDA_VISIBLE_DEVICES=0 python train_classifier.py --exp runs/newdropout \
--task CIFAR100 --mode efficient --model resnet50 --drop_mode constant --bs 128 --epochs 1800 --lr 0.0002 \
--builtin --unified --percentage 0.6 --min_percentage 0.6 --interleave --warmup 0 --seed 42 --use_gpu --by_epoch --dropout 0.2
```

- To train an efficient ddpm with 0.4 drop rate, simply run
```shell
CUDA_VISIBLE_DEVICES=0 python train_ddpm.py --exp runs/generation --task FashionMNIST --mode efficient --model ddpm --drop_mode constant --epochs 500 --lr 0.001 --schedule cosine --builtin --unified --percentage 0.2 --min_percentage 0.2 --interleave --warmup 0
```

# Citation

```
@article{zhong2024ssprop,
  title={ssProp: Energy-Efficient Training for Convolutional Neural Networks with Scheduled Sparse Back Propagation},
  author={Zhong, Lujia and Huang, Shuo and Shi, Yonggang},
  journal={arXiv preprint arXiv:2408.12561},
  year={2024}
}
```