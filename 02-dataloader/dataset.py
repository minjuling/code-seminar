import os
import glob
import numpy as np

import torch
from torch.utils import data
import torch.distributed as dist

from transform import *
from PIL import Image

# coco dataset
class Dataset(data.Dataset):
    def __init__(self, cfg, data_root, split='train2017',transforms=False, ignore_label=255):
        '''
        path:
            root: ./coco
            Image: ./coco/images/train2017
            ground truth: ./coco/labels/train2017
        '''

        self.root = data_root
        self.split = split

        self.image_base = os.path.join(self.root,'images',self.split)
        self.files = glob.glob(self.image_base+'/*/*.png')
        assert len(self.files) is not 0 , ' cannot find data!'
        
        self.transforms = transforms

        self.nClasses = cfg.DATA.NUM_CLASSES # 19

        # self.label_mapping = {-1: ignore_label, 0: ignore_label, 
        #                       1: ignore_label, 2: ignore_label, 
        #                       3: ignore_label, 4: ignore_label, 
        #                       5: ignore_label, 6: ignore_label, 
        #                       7: 0, 8: 1,
        #                       9: ignore_label, 10: ignore_label,
        #                       11: 2, 12: 3,
        #                       13: 4, 14: ignore_label, 
        #                       15: ignore_label, 16: ignore_label, 
        #                       17: 5, 18: ignore_label, 
        #                       19: 6, 20: 7, 21: 8, 22: 9, 23: 10, 
        #                       24: 11,
        #                       25: 12, 26: 13, 27: 14, 28: 15, 
        #                       29: ignore_label, 30: ignore_label, 
        #                       31: 16, 32: 17, 33: 18}
                              #coco는 91개임..

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img_path = self.files[index]
        img = Image.open(img_path)
        img = np.asarray(img)

        gt_path = img_path.replace('images', 'labels',1)
        #바꿀거 바꾼거 변경횟수

        label = Image.open(gt_path)
        label = np.asarray(label)


        if self.transforms:
            img, label= self.transforms(img,label)
        # label = self.convert_label(label)

        return img, label
    
    def convert_label(self, label, inverse=False):
        temp = label.clone()
        if inverse:
            for v, k in self.label_mapping.items():
                label[temp == k] = v
        else:
            for k, v in self.label_mapping.items():
                label[temp == k] = v
        return label

     
    def get_file_path(self):
        return self.files

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import transform
    
    root='./coco'
    # print(root)
    value_scale = 255
    mean = [0.485, 0.456, 0.406]
    mean = [item * value_scale for item in mean]
    std = [0.229, 0.224, 0.225]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandScale([0.5,2]),
        transform.RandomHorizontalFlip(),
        transform.Crop([713, 713], crop_type='rand', padding=mean, ignore_label=255),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
        ])


    dataset = Dataset(root, transforms = train_transform)
    print(len(dataset))
    img, label = dataset.__getitem__(0)
    print(img.dtype)
    print(label.shape)
    fig_in = plt.figure()
    ax = fig_in.add_subplot(1,2,1)
    ax.imshow(img)
    ax = fig_in.add_subplot(1,2,2)
    ax.imshow(label)
    plt.show()