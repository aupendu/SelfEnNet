# Self-supervision via Controlled Transformation and Unpaired Self-conditioning for Low-light Image Enhancement
This is the PyTorch implementation for our IEEE TIM paper:

**Aupendu Kar, Sobhan Kanti Dhara, Debashis Sen, Prabir Kumar Biswas. Self-supervision via Controlled Transformation and Unpaired Self-conditioning for Low-light Image Enhancement.

## Test Codes
## Test Codes
To perform Low-light Image Enhancement using Our pre-trained model.
```
python main.py --data_test LIME+DICM --test_only  \
               --rgb_range 1.0 --save_folder None \
               --model SelfEnNet --exp SelfEnNet --iterations 2 --level 1 \
               --trained_model model_dir/SelfEnNet/SelfEnNet.pth.tar'
```

## Train Codes
### Data Preparation

1.1 Download the Training Set

1.2 Put the unpaired training dataset in `data` as follows:
```
data
└── train
        ├── dataset_name
                    ├── low
                            ├── name1.png
                            ├── .....
                            └── name152.png
                    └── high
                            ├── name101.png
                            ├── .....
                            └── name292.png

```

### Train Enhancement Model

```
python main.py --data_train_low dataset_name --data_train_high dataset_name --data_test DRBN_valid \
               --update_peritr 1000 --epochs 125  --batch_size 4 --patch_size 128 \
               --lr 1e-4 --decay 25+50+75+100 --gamma 0.5 --rgb_range 1.0 --iterations 2 \
               --model SelfEnNet --exp SelfEnNet --save_folder blank --level 0 
```

### Train Denoising Model
```
python main.py --data_train_low dataset_name --data_train_high dataset_name --data_test DRBN_valid \
               --update_peritr 1000 --epochs 125  --batch_size 4 --patch_size 128 \
               --lr 1e-4 --decay 25+50+75+100 --gamma 0.5 --rgb_range 1.0 --iterations 2 \
               --model SelfEnNet --exp SelfEnNet --save_folder blank --level 1 \
               --pre_train --pretrained_model model_dir/SelfEnNet/SelfEnNet_best_Enhance_0.pth.tar
```


## Citation
```
@ARTICLE{Kar_2024_Self,
  author={Kar, Aupendu and Dhara, Sobhan Kanti and Sen, Debashis and Biswas, Prabir Kumar},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={Self-supervision via Controlled Transformation and Unpaired Self-conditioning for Low-light Image Enhancement}, 
  year={2024},
  volume={},
  number={},
  pages={},
  doi={}}

```

## Contact
Aupendu Kar: mailtoaupendu[at]gmail[dot]com
