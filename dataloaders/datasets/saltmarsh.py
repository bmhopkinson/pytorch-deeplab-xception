from torch.utils import data
import torch
import os
import numpy as np
import scipy.misc as m
from PIL import Image
from mypath import Path
from torchvision import transforms
from dataloaders import custom_transforms as tr
#this class is based on Cityscapes Dataset in the given repo..
#this class is based on Cityscapes Dataset in the given repo..
class SaltmarshSegmentation(data.Dataset):
    NUM_CLASSES = 9

    def __init__(self, args,root=r"./Data/", split="train"):

        self.root = root
        self.split = split
        self.args = args
        self.images = []
        self.masks=[]

        self.set_data_names()

        #self.void_classes = [0]
        self.valid_classes = [0,1,2,3,4,5,6,7,8]
        self.class_names = ['Background','Limonium', 'Spartina', 'Batis', 'Other', 'Spart_dead', \
                            'Juncus', 'Sacricornia', 'Borrichia']

        #self.ignore_index = 255
        #self.class_map = dict(zip(self.valid_classes, range(self.NUM_CLASSES)))
        self.class_map = {  #RGB to Class
            (255, 255, 255): 0 , #background
            (150, 255,  14): 0,  # Background_alt
            (127, 255, 140): 1, #Spartina
            (113, 255, 221): 2,  # dead Spartina
            ( 99, 187, 255): 3,  # Sarcocornia
            (101,  85, 255): 4,  # Batis
            (212,  70, 255): 5,  # Juncus
            (255,  56, 169): 6,  # Borrichia
            (255,  63,  42): 7,  # Limonium
            (255, 202,  28): 8  # Other
        }

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        img_path = self.root+"/"+self.split+"/"+self.images[index]
        lbl_path = self.root+"/"+self.split+"/"+self.masks[index]
       # print(img_path)
        _img = Image.open(img_path).convert('RGB')

        _tmp = np.array(Image.open(lbl_path), dtype=np.uint8)
    #    _target = self.maskrgb_to_class(_tmp)
        _target = Image.fromarray(np.array(_tmp))

        sample = {'image': _img, 'label': _target}

        if self.split == 'train':
            sample =  self.transform_tr(sample)
        elif self.split == 'val':
            sample = self.transform_val(sample)
        elif self.split == 'test':
            sample =  self.transform_ts(sample)

        sample['label'] = self.maskrgb_to_class(
                                    np.array(sample['label'], dtype=np.uint8)
                                    )
        return sample

    def set_data_names(self):
        i = 0
        for file in os.listdir(self.root+"/"+self.split+"/"):
            if (file.endswith("mask.png")):
                    self.masks.append(file)
                    s=file.split('_')
                    imgname="_".join(s[:-1])+".jpg"
                  #  print(imgname)
                    self.images.append(imgname)
                    i  = i +1
        print('total images loaded: {}'.format(i))
        return True

    def maskrgb_to_class(self, mask):
     #   print('----mask->rgb----')
        mask = torch.from_numpy(np.array(mask))
        mask = torch.squeeze(mask)  # remove 1

        # check the present values in the mask, 0 and 255 in my case
        #print('unique values rgb    ', torch.unique(mask))
        # -> unique values rgb     tensor([  0, 255], dtype=torch.uint8)

        class_mask = mask
        class_mask = class_mask.permute(2, 0, 1).contiguous()  #channels dim 0
        h, w = class_mask.shape[1], class_mask.shape[2]
        mask_out = torch.zeros(h, w, dtype=torch.long)

        for k in self.class_map:
            idx = (class_mask == torch.tensor(k, dtype=torch.uint8).unsqueeze(1).unsqueeze(2))
            validx = (idx.sum(0) == 3)  #must agree in all 3 channels
            mask_out[validx] = torch.tensor(self.class_map[k], dtype=torch.long)

        # check the present values after mapping, in my case 0, 1, 2, 3
      #  print('unique values mapped ', torch.unique(mask_out))
        # -> unique values mapped  tensor([0, 1, 2, 3])

        return mask_out

    def transform_tr(self, sample):
        composed_transforms = transforms.Compose([
            tr.RandomHorizontalFlip(),
            tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size, fill=255),
            tr.RandomGaussianBlur(),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_val(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixScaleCrop(crop_size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)

    def transform_ts(self, sample):

        composed_transforms = transforms.Compose([
            tr.FixedResize(size=self.args.crop_size),
            tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            tr.ToTensor()])

        return composed_transforms(sample)