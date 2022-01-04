import os
from tqdm import tqdm
import torch
import numpy as np
from torch import nn
from torch.utils.data import random_split, DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F
import setting
from dataset import CarDataset, image_dim_expansion
from model import FCN32s, FCN16s, FCN8s, FCNs
from evaluate import multiclass_dice_coeff, dice_coeff, dice_loss
from plot import plot_loss

# setting 任务设置
fcn_type = setting.fcn_type
num_classes = setting.num_classes
back_bone = setting.back_bone
data_path = setting.data_path
save_path = setting.save_path
ckpt_path = setting.ckpt_path
device = setting.device

# recurrent seed 固定随机数
recurrent_generator = torch.manual_seed(42)

# hyperparameters 超参数
batch_size = setting.batch_size
lr = setting.lr
epochs = setting.epochs

# GPU ID
device_id = setting.device_id

if __name__ == '__main__':

    # gpu setting
    os.environ["CUDA_VISIBLE_DEVICES"] = device_id

    # select fcn type
    if fcn_type == 'fcn32s': model = FCN32s(num_classes, back_bone)
    elif fcn_type == 'fcn16s': model = FCN16s(num_classes, back_bone)
    elif fcn_type == 'fcn8s': model = FCN8s(num_classes, back_bone)
    elif fcn_type == 'fcns': model = FCNs(num_classes, back_bone)
    else: print('wrong type of fnc model')

    start_epoch = 1
    data_loader = DataLoader(CarDataset(data_path), batch_size,shuffle=True)

    # initialize model-saving directory
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(os.path.join(save_path,'img')):
        os.makedirs(os.path.join(save_path,'img'))

    # resume training
    if ckpt_path:
        epoch_name = (ckpt_path.split('/')[-1]).split('.')[0]
        start_epoch = int(epoch_name) + 1
        checkpoint = torch.load(ckpt_path)
        state_dict = checkpoint["state_dict"]

        model.load_state_dict(state_dict)
        print(f"Model loaded from {ckpt_path}")

    # CUDA or CPU
    if torch.cuda.is_available():
        model.to(device)
        model = nn.DataParallel(model) # multi-GPU

# dataset
    car_dataset = CarDataset(data_path)

    # split data [0.7, 0.15, 0.15]
    train_size = setting.train_size
    val_size = setting.val_size
    test_size = len(car_dataset) - train_size - val_size
    train_set, val_set, test_set = random_split(car_dataset, [train_size, val_size, test_size],
                                                generator=recurrent_generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # optimizer = torch.optim.SGD(mymodel.parameters(), lr, momentum=0.9, weight_decay=0.00001)
    optimizer = torch.optim.Adam(model.parameters(), lr)
    criterion = nn.BCEWithLogitsLoss() # 
    # criterion = nn.CrossEntropyLoss()
    # scheduler = lr_scheduler.StepLR(optimizer, step_size = 10, gamma = 0.1)

    print('\033[31mStart training: Total epochs: {}, Batch size: {}, Training size: {}, Validation size: {}, Test size: {}\033[0m'.
                    format(epochs, batch_size, len(train_set), len(val_set), len(test_set)))
    
    print(f'NETWORK :{fcn_type}')

    train_epoch_losses = []
    val_epoch_losses = []
    val_epoch_dice_scores = []  
    for epoch in range(start_epoch, epochs+1):
        
        print(f'\033[31mEpoch {epoch:02d}/{epochs}\033[0m, Learning Rate {optimizer.param_groups[0]["lr"]:g}')

        # train
        model.train()
        epoch_loss = 0.0
        for i, sample in tqdm(enumerate(train_loader),desc=f'\tTrain Batches       '):
            images, targets = sample
            images  = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets) # default is mean loss
            # outputs  = F.softmax(outputs, dim=1)
            # outputs  = F.one_hot(outputs.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
            # loss = loss + dice_loss(outputs,targets,multiclass=True)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() # save loss of every batch in each epoch

        train_epoch_losses.append(epoch_loss / train_size * batch_size)
        print(f'\tTrain Epoch Avg_Loss: {train_epoch_losses[-1]:.6f}')

        # save checkpoint model
        if epoch % 10 == 0:
            state_dict = model.module.state_dict()
            for key in state_dict.keys():
                state_dict[key] = state_dict[key].cpu()

            torch.save({
                'epoch': epoch,
                'save_dir': save_path,
                'state_dict': state_dict,},
                os.path.join(save_path, f'{epoch:03d}.ckpt'))
            print(f'\t\033[33mSave checkpoint\033[0m successfully to {save_path}: {epoch:03d}.ckpt')

        # validation
        model.eval()
        epoch_loss = 0.0
        for i, sample in tqdm(enumerate(val_loader),desc=f'\tVal   Batches       '):
            images, targets = sample
            images = images.to(device)
            targets = targets.to(device)
            with torch.no_grad():
                outputs = model(images)
                loss = criterion(outputs, targets)
                outputs  = F.softmax(outputs, dim=1)
                outputs  = F.one_hot(outputs.argmax(dim=1), num_classes).permute(0, 3, 1, 2).float()
                dice_score = multiclass_dice_coeff(outputs[:, 1:, ...], targets[:, 1:, ...], reduce_batch_first=False)

                _img  = torch.stack([images[0],images[0],images[0],images[0],images[0],images[0],images[0],images[0],images[0]], dim=0)
                _mask = image_dim_expansion(targets[0])
                _out  = image_dim_expansion(outputs[0])
                _concat_img = torch.cat([_img + 2, _mask, _out],dim=0)

                save_image(_concat_img, os.path.join(save_path,'img',f'val_{i}_{epoch}.png'), nrow=9)

            epoch_loss += loss.item() # save loss of every batch in each epoch


        val_epoch_losses.append(epoch_loss / val_size * batch_size)
        val_epoch_dice_scores.append(dice_score.item())
        print(f'\tVal   Epoch Avg_Loss: {val_epoch_losses[-1]:.6f}')
        print(f'\tVal Epoch Dice score: {val_epoch_dice_scores[-1]:.6f}')
        # update learning rate
        # scheduler.step()

    train_log = np.array(train_epoch_losses)
    val_log   = np.array(val_epoch_losses)
    val_dice   = np.array(val_epoch_dice_scores)
    train_log_path  = os.path.join(save_path, f'{fcn_type}_{start_epoch:d}_{start_epoch-1+epochs:d}_train.npy')
    val_log_path    = os.path.join(save_path, f'{fcn_type}_{start_epoch:d}_{start_epoch-1+epochs:d}_val.npy')
    val_log_dice_path    = os.path.join(save_path, f'{fcn_type}_{start_epoch:d}_{start_epoch-1+epochs:d}_val_dice.npy')
    np.save(train_log_path, train_log)
    np.save(val_log_path, val_log)
    np.save(val_log_dice_path, val_dice)
    print(f'Save loss history to:\n\t{train_log_path}\n\t{val_log_path}')

    plot_loss(fcn_type, epochs, lr, batch_size, 'loss.jpg', train_log_path, val_log_path, val_log_dice_path)

    import cv2
    # image path
    img_path = os.path.join(save_path,'img')
    # output video path
    video_path = os.path.join(save_path,'img')
    if not os.path.exists(video_path):
        os.makedirs(video_path)
    # set saved fps
    fps = 5
    # get frames list
    frames = sorted(os.listdir(img_path))
    # w,h of image
    img = cv2.imread(os.path.join(img_path, frames[0]))
    img_size = (img.shape[1], img.shape[0])
    # get seq name
    seq_name = os.path.dirname(img_path).split('/')[-1]
    # splice video_path
    video_path = os.path.join(video_path, seq_name + '.avi')
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    # also can write like:fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # if want to write .mp4 file, use 'MP4V'
    videowriter = cv2.VideoWriter(video_path, fourcc, fps, img_size)

    for frame in frames:
        f_path = os.path.join(img_path, frame)
        image = cv2.imread(f_path)
        videowriter.write(image)
    print(f'Validation output has been written to {video_path}!')

    videowriter.release()
