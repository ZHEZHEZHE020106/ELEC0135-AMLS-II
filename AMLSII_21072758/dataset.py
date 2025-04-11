import os
import torch
import PIL
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import torch.optim as optim

class DIV2KDataset(Dataset):
    def __init__(self, track1=True, train=False, val = False, test= False, scale_factor=2, transform = None):
        self.track1 = track1
        self.train = train
        self.val = val
        self.test = test
        hr_size = 96
        self.scale_factor = scale_factor

        self.track1_train_lr_path = "C:\\Users\\44753\\Desktop\\AMLSII_21072758\\A\\DIV2K_train_LR_bicubic\\X2"
        self.track1_validation_lr_path = "C:\\Users\\44753\\Desktop\\AMLSII_21072758\\A\\DIV2K_valid_LR_bicubic\\X2"
        
        self.track2_train_lr_path = "C:\\Users\\44753\\Desktop\\AMLSII_21072758\\B\\DIV2K_train_LR_unknown\\X2"
        self.track2_validation_lr_path = "C:\\Users\\44753\\Desktop\\AMLSII_21072758\\B\\DIV2K_valid_LR_unknown\\X2"
        
        self.train_hr_path = "C:\\Users\\44753\\Desktop\\AMLSII_21072758\\A\\DIV2K_train_HR"
        self.validation_hr_path = "C:\\Users\\44753\\Desktop\\AMLSII_21072758\\A\\DIV2K_valid_HR"
        self.transform_lr = transforms.Compose([
                        transforms.Resize((int(hr_size/2), int(hr_size/2))),
                        transforms.ToTensor()
                        ])
        
        self.transform_hr = transforms.Compose([
                        transforms.Resize((hr_size, hr_size)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
                        ])
        
        if (self.track1):
            train_lr = sorted(os.listdir(self.track1_train_lr_path))
            validation_lr = sorted(os.listdir(self.track1_validation_lr_path))
            self.lr_images = train_lr + validation_lr
        else:
            train_lr = sorted(os.listdir(self.track2_train_lr_path))
            validation_lr = sorted(os.listdir(self.track2_validation_lr_path))
            self.lr_images = train_lr + validation_lr

        train_hr = sorted(os.listdir(self.train_hr_path))
        validation_hr = sorted(os.listdir(self.validation_hr_path))
        self.hr_images = train_hr + validation_hr

        train_size = int(0.8 * len(self.lr_images))
        val_size = int(0.1 * len(self.lr_images))
        test_size = int(0.1 * len(self.lr_images))

        self.train_set, self.val_set, self.test_set = random_split(self.lr_images, [train_size, val_size, test_size])

        if (self.train):
            self.current_set = self.train_set
        elif (self.val):
            self.current_set = self.val_set
        elif (self.test):
            self.current_set = self.test_set



    def __len__(self):
        return len(self.current_set)

    def __getitem__(self, index):
        
        if(self.track1):
            train_path = self.track1_train_lr_path
            validation_path = self.track1_validation_lr_path
        else:
            train_path = self.track2_train_lr_path
            validation_path = self.track2_validation_lr_path
        
        try:
            lr_image = Image.open(train_path + "\\" + self.current_set[index])
        except FileNotFoundError:
            try:
                lr_image = Image.open(validation_path + "\\" + self.current_set[index])
            except FileNotFoundError:
                print("Error! Image not found!")
                
        try:
            hr_image = Image.open(self.train_hr_path + "\\" + self.current_set[index].split('x')[0] + '.png')
        except FileNotFoundError:
            try:
                hr_image = Image.open(self.validation_hr_path + "\\" + self.current_set[index].split('x')[0] + '.png')
            except FileNotFoundError:
                print("Error! Image not found!")
            
        lr_image = self.transform_lr(lr_image)
        hr_image = self.transform_hr(hr_image)
        return lr_image, hr_image