from RTNet_GRU import *
import argparse
from train import *

parser = argparse.ArgumentParser(
    "Speech Enhancement using Recursive learning in the time domain, GRU mechanism is utilized"
)

# parameters config
parser.add_argument('--json_dir', type = str, default = '/media/Dataset/STIMIT/json',
                    help = 'The directory of the dataset feat,json format')
parser.add_argument('--loss_dir', type = str, default= '/media/liandong/PROJECTS/RTNet/LOSS/rtnet_gru_stage_1_loss.mat',
                    help = 'The directory to save tr loss and cv loss')
parser.add_argument('--batch_size', type= int, default = 2,
                    help = 'The number of the batch size')
parser.add_argument('--cv_batch_size', type= int, default = 4,
                    help = 'The number of the batch size')
parser.add_argument('--epochs', type = int, default= 50,
                    help= 'The number of the training epoch')
parser.add_argument('--lr', type = float, default = 2e-4,
                    help = 'Learning rate of the network')
parser.add_argument('--early_stop', dest= 'early_stop', default = 1, type = int,
                    help = 'Early stop training when no improvement for 10 epochs')
parser.add_argument('--half_lr', type= int, default = 1,
                    help = 'Whether to decay learning rate to half scale')
parser.add_argument('--shuffle', type = int, default= 1,
                    help = 'Whether to shuffle within each batch')
parser.add_argument('--num_workers', type= int, default = 0,
                    help = 'Number of workers to generate batch')
parser.add_argument('--l2', type = float, default= 1e-7,
                    help = 'weight decay (L2 penalty)')
parser.add_argument('--save_folder', type = str, default = '/media/liandong/PROJECTS/RTNet/MODEL',
                    help= 'Location to save epoch models')
parser.add_argument('--checkpoint', dest = 'checkpoint', default = 0, type = int,
                    help= 'Enables checkpoint saving of model')
parser.add_argument('--continue_from', default = '', help = 'Continue from checkpoint model')
parser.add_argument('--best_path', default = '/media/liandong/PROJECTS/RTNet/BEST_MODEL/rtnet_gru_stage_1_final_pth.tar',
                    help= 'Location to save best cv model')
parser.add_argument('--print_freq', type = int , default = 50,
                    help = 'The frequency of printing loss infomation')
train_model = RTNet_GRU()

if __name__ == '__main__':
    args = parser.parse_args()
    model = train_model
    print(args)
    main(args, model)
