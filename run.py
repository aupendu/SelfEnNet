import os




# os.system('python main.py --data_train_low DRBNtrain1 --data_train_high DRBNtrain1 --data_test DRBN_valid \
# 						  --update_peritr 1000 --epochs 125  --batch_size 4 --patch_size 128 \
# 						  --lr 1e-4 --decay 25+50+75+100 --gamma 0.5 --rgb_range 1.0 --iterations 2 \
# 						  --model SelfEnNet --exp SelfEnNet --save_folder blank --level 0 ') 


# os.system('python main.py --data_train_low DRBNtrain1 --data_train_high DRBNtrain1 --data_test DRBN_valid \
# 						  --update_peritr 1000 --epochs 125  --batch_size 4 --patch_size 128 \
# 						  --lr 1e-4 --decay 25+50+75+100 --gamma 0.5 --rgb_range 1.0 --iterations 2 \
# 						  --model SelfEnNet --exp SelfEnNet --save_folder blank --level 1 \
# 						  --pre_train --pretrained_model model_dir/SelfEnNet/SelfEnNet_best_Enhance_0.pth.tar')


os.system('python main.py --data_test LIME+DICM --test_only  \
						  --rgb_range 1.0 --save_folder None \
						  --model SelfEnNet --exp SelfEnNet --iterations 2 --level 1 \
						  --trained_model model_dir/SelfEnNet/SelfEnNet.pth.tar')











 

