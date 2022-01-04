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
from plot import plot_loss, plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from evaluate import multiclass_dice_coeff, dice_coeff, dice_loss, pixelAccuracy, classPixelAccuracy, meanPixelAccuracy, meanIntersectionOverUnion, frequency_Weighted_Intersection_over_Union

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

batch_size = 140

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

    model_path = setting.model_path
    checkpoint = torch.load(model_path)
    state_dict = checkpoint["state_dict"]

    model.load_state_dict(state_dict)
    print(f"Model loaded from {model_path}")

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

    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
 
    print(f'NETWORK :{fcn_type}')

    # test
    model.eval()
    test_loss  = 0
    dice_score = 0
    out_cpu = torch.empty([test_size,256,256])
    tar_cpu = torch.empty([test_size,256,256])
    for i, sample in tqdm(enumerate(test_loader),desc=f'Test Batches'):
        images, targets = sample
        images = images.to(device)
        targets = targets.to(device)
        with torch.no_grad():
            outputs = model(images)
            outputs  = F.softmax(outputs, dim=1)
            outputs  = outputs.argmax(dim=1)
            # move prediction to cpu
            out_cpu[i*batch_size : (i+1)*batch_size] = outputs.cpu()
            tar_cpu[i*batch_size : (i+1)*batch_size] = targets.argmax(dim=1).cpu()
            outputs  = F.one_hot(outputs, num_classes).permute(0, 3, 1, 2).float()
            dice_score += multiclass_dice_coeff(outputs[:, 1:, ...], targets[:, 1:, ...], reduce_batch_first=False)

            _img  = torch.stack([images[0],images[0],images[0],images[0],images[0],images[0],images[0],images[0],images[0]], dim=0)
            _mask = image_dim_expansion(targets[0])
            _out  = image_dim_expansion(outputs[0])
            _concat_img = torch.cat([_img + 2, _mask, _out],dim=0)

            save_image(_concat_img, os.path.join(save_path,'img',f'test_{i}.png'), nrow=9)

    dice_score = dice_score / (i+1)
    print(f'\tTest Dice score: {dice_score:.6f}')

    out_cpu = out_cpu.ravel().numpy()
    tar_cpu = tar_cpu.ravel().numpy()


    classes = ['background','front_door','back_door','fender','frame','bumper','hood','back_bumper','trunk']
    cm = confusion_matrix(tar_cpu, out_cpu)
    print('Pixel Accuracy:',pixelAccuracy(cm))
    print('Class Pixel Accuracy:',classPixelAccuracy(cm))
    print('Mean Pixel Accuracy:',meanPixelAccuracy(cm))
    print('Mean Intersection over Union:',meanIntersectionOverUnion(cm))
    print('Frequency Weighted Intersection over Union:',frequency_Weighted_Intersection_over_Union(cm))
    plot_confusion_matrix(cm, f'confusion_matrix_{fcn_type}.png', classes, title=f'confusion matrix({fcn_type})')
