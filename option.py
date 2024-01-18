import argparse

parser = argparse.ArgumentParser(description='Training of LowLight Enhancement Algorithms')

# Hardware specifications
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')
parser.add_argument('--seed', type=int, default=1, help='random seed')

# Data specifications
parser.add_argument('--path', type=str, default='./data', help='data path')
parser.add_argument('--data_train_low', type=str, help='Train dataset of LowLight images')
parser.add_argument('--data_train_high', type=str, help='Train dataset of well-lit images')
parser.add_argument('--data_test', type=str, help='Test dataset name')
parser.add_argument('--patch_size', type=int, help='Input patch size')
parser.add_argument('--rgb_range', type=float, help='range of image intensity')
parser.add_argument('--exp', type=str, help='experiment name')
parser.add_argument('--save_folder', type=str, help='save path')

# Model specifications
parser.add_argument('--model', help='model name')
parser.add_argument('--trained_model', type=str, help='pre-trained model directory')
parser.add_argument('--pretrained_model', type=str, help='pre-trained model directory')
parser.add_argument('--resume', action='store_true', help='resume training')

# Training specifications
parser.add_argument('--update_peritr', type=int, help='do test per every N batches')
parser.add_argument('--epochs', type=int, help='number of epochs to train')
parser.add_argument('--batch_size', type=int, help='input batch size for training')
parser.add_argument('--test_only', action='store_true', help='set this option to test the model')
parser.add_argument('--pre_train', action='store_true', help='set this option to load pre-train model')
parser.add_argument('--level', type=int, help='')
parser.add_argument('--iterations', type=int, help='')

# Optimization specifications
parser.add_argument('--loss', type=str, help='Loss Functions')
parser.add_argument('--lr', type=float, help='learning rate')
parser.add_argument('--decay', type=str, help='learning rate decay type')
parser.add_argument('--gamma', type=float, help='learning rate decay factor for step decay')


args = parser.parse_args()


