import os
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
from setting import data_path, car_parts, num_classes, device

mean = np.array([0.485,0.456,0.406])
std  = np.array([0.229,0.224,0.225])

def image_size_alignment(path,size=(256,256)):
    img = Image.open(path)
    temp = max(img.size)
    mask = Image.new('RGB', (temp,temp),(0,0,0))
    mask.paste(img,(0,0))
    mask.resize(size)
    return mask

def image_dim_expansion(tensor, num_classes=num_classes, device=device):
    images = torch.empty((num_classes,3,256,256),device=device)
    for i in range(num_classes):
        images[i] = torch.stack([tensor[i],tensor[i],tensor[i]], dim=0)
    return images

class CarDataset(Dataset):
    def __init__(self,path):
        self.path = path
        self.names = os.listdir(path)
        for i, name in enumerate(self.names):
            if os.path.splitext(name)[1] != '.npy':
                # three way to delete, pop(index), del list[index], remove(value). and clear() wipe out all
                self.names.pop(i)

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]
        path = os.path.join(self.path, name)
        img = np.load(path).astype(np.double)
        car_img = img[:3] # normalized 3 channel
        mask_imgs = img[3:12]
        return torch.from_numpy(car_img).to(dtype=torch.float32), \
               torch.from_numpy(mask_imgs).to(dtype=torch.float32) # 32bit is faster than 64bit


if __name__ == '__main__':

    data = CarDataset(data_path)
    idx = random.randint(0,len(data))
    car_img, car_masks = data[idx]

    # transpose channel dim and de-nomalization
    car_img = ((car_img.numpy().transpose(1,2,0) + mean / std) * 255).astype(np.int32)
    car_masks = (car_masks.numpy().transpose(1,2,0) * 255).astype(np.int32)
    plt.figure(figsize=(28,6))
    plt.subplot(2,9,5)
    plt.axis('off')
    plt.imshow(car_img)
    plt.title(f'car image\n(idx={idx})')
    for i in range(9):
        plt.subplot(2,9,i+10)
        plt.axis('off')
        plt.imshow(car_masks[:,:,i], cmap=plt.cm.gray)
        plt.title(f'{car_parts[i]}')