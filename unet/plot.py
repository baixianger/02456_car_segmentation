import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import torch.nn.functional as F

def plot_loss(model_name, epochs, lr, batch_size, img_name,*args):
    assert len(args) == 3
    plt.figure(figsize=(5,5))
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(0,1)
    plt.title(f'Train and Validation Loss of Model ({model_name})\n(lr:{lr} epochs:{epochs} batch_size:{batch_size})')
    plt.plot(range(1,epochs+1), np.load(args[0]), '--b', label=f'Train Loss')
    plt.plot(range(1,epochs+1), np.load(args[1]), '--r', label=f'Val Loss')
    plt.plot(range(1,epochs+1), np.load(args[2]), '-g', label=f'Val Dice Score')
    plt.legend()
    plt.grid()
    save_path = os.path.dirname(args[0])
    plt.savefig(os.path.join(save_path, img_name))
    print(f'Loss Plot save to {save_path}')


def plot_confusion_matrix(cm, savename, classes, title='Confusion Matrix'):

    # tackle label imbalance
    cm  = cm[1:,1:]
    classes = classes[1:]
    plt.figure(figsize=(9, 8), dpi=100)
    np.set_printoptions(precision=2)

    # 在混淆矩阵中每格的概率值
    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(x_val, y_val, "%0.2e" % (c,), color='red', fontsize=8, va='center', ha='center')
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel('Actual label')
    plt.xlabel('Predict label')
    
    # offset the tick
    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position('none')
    plt.gca().yaxis.set_ticks_position('none')
    plt.grid(True, which='minor', linestyle='-')
    plt.gcf().subplots_adjust(bottom=0.15)
    
    # show confusion matrix
    plt.savefig(savename, format='png')
    plt.show()
 

# 获取实际标签、预测结果和混淆矩阵：

# # classes表示不同类别的名称，比如这有6个类别
# classes = ['A', 'B', 'C', 'D', 'E', 'F']

# random_numbers = np.random.randint(6, size=50)  # 6个类别，随机生成50个样本
# y_true = random_numbers.copy()  # 样本实际标签
# random_numbers[:10] = np.random.randint(6, size=10)  # 将前10个样本的值进行随机更改
# y_pred = random_numbers  # 样本预测标签

# # 获取混淆矩阵
# cm = confusion_matrix(y_true, y_pred)
# plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')

# outputs = torch.randn([12,9,256,256])
# targets = outputs

# outputs  = F.softmax(outputs, dim=1).permute(0, 2, 3, 1).reshape(-1,9)
# print(outputs.size())
# outputs  = outputs.argmax(dim=1)
# print(outputs.size())

# targets  = F.softmax(targets, dim=1).permute(0, 2, 3, 1).reshape(-1,9)
# print(targets.size())
# targets  = targets.argmax(dim=1)
# print(targets.size())

# classes = ['background','front_door','back_door','fender','frame','bumper','hood','back_bumper','trunk']
# cm = confusion_matrix(targets, outputs)
# plot_confusion_matrix(cm, 'confusion_matrix.png', title='confusion matrix')