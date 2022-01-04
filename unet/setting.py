import os
import torch

## load code from local to DTU HPC 上传代码:
#  rsync -av /Users/baixiang/car/unet  s213120@transfer.gbar.dtu.dk:/work3/s213120/car
## download the output gird_img from HPC 下载雪碧图:
#  rsync -av s213120@transfer.gbar.dtu.dk:/work3/s213120/car/unet/save/unet/img  /Users/baixiang/Downloads
## go to the project dir 打开工程目录
#  cd /work3/s213120/car/unet/

# data setting
car_parts = ['background','front_door','back_door','fender','frame','bumper','hood','back_bumper','trunk']
num_classes = 9
data_path = r'/work3/s213120/car/clean_data'
train_size = 2560
val_size = 512

# model setting
unet_type = 'unet' # 'unet_pp'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
save_path = os.path.join('/work3/s213120/car/unet/save',unet_type)

# hyperparameters setting
batch_size = 32
epochs = 1
lr = 1e-3

# resume training
# ckpt_path is null, restart training, elif give a typical path, resume training
ckpt_path = False 
# ckpt_path = os.path.join(save_path, f'{fcn_type}.ckpt')

# GPU setting
device_id = '1,0'

# test
model_path = os.path.join(save_path,'050.ckpt')

