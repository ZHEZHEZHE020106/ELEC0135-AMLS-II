from dataset import DIV2KDataset
from model import _NetG
from model2 import Track2_Net

import math
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

def train(model, train_loader, val_loader, test_loader, train_dataset, loss_function, optimizer, num_epoch, device):
    
    mse_values = []
    psnr_values = []
    ssim_values = []
    
    val_mse_values = []

    for epoch in range(num_epoch):
        print(f"--------------------Epoch {epoch+1} Start--------------------")
        model.train()
        
        step = 0
        epoch_loss = 0
        epoch_psnr = 0
        epoch_ssim = 0
        
        for lr_image, hr_image in train_loader:
            
            lr_image,hr_image = lr_image.to(device),hr_image.to(device)
            
            optimizer.zero_grad()
            sr_image = model(lr_image)
            loss = loss_function(sr_image, hr_image)
            loss.backward()
            
            psnr = calculate_psnr(sr_image, hr_image)
            ssim = calculate_ssim(sr_image, hr_image) 
            
            epoch_psnr += psnr
            epoch_ssim += ssim
            step += 1
            
            optimizer.step()
            epoch_loss += loss.item()
            
        mse_values.append(epoch_loss/step)
        psnr_values.append(epoch_psnr/step)
        ssim_values.append(epoch_ssim/step)
        torch.save(model.state_dict(),
            "C:\\Users\\44753\\Desktop\\AMLSII_21072758\\Saved_Models\\model_weights{}.pth".format(epoch+1))
        print("Model Saved")
        print(f"Average PSNR: {epoch_psnr/step:.2f} dB")
        print(f"Average SSIM: {epoch_ssim/step}")
        print(f"--------------------Epoch {epoch+1} finished, Loss:{epoch_loss/step}--------------------")
        
        print(f"--------------------Validation {epoch+1} Start--------------------")
        model.eval()
        step = 0
        val_loss = 0
        val_psnr = 0
        val_ssim = 0
        
        with torch.no_grad():
            for lr_image, hr_image in val_loader:
                lr_image,hr_image = lr_image.to(device),hr_image.to(device)
                sr_image = model(lr_image)
                
                loss = loss_function(sr_image, hr_image)
                psnr = calculate_psnr(sr_image, hr_image)
                ssim = calculate_ssim(sr_image, hr_image)
                
                step += 1
                val_loss += loss.item()
                val_psnr += psnr
                val_ssim += ssim
                
        print(f"Validation MSE: {val_loss/step}")
        print(f"Validation PSNR: {val_psnr/step:.2f} dB")
        print(f"Validation SSIM: {val_ssim/step}")
        val_mse_values.append(val_loss/step) 
        
        print(f"--------------------Validation {epoch+1} finished, Loss:{val_loss/step}--------------------")
        
    #plot_loss(mse_values, num_epoch)
    plot(mse_values, num_epoch, 'Loss', 'Loss over Epoch')
    plot(psnr_values, num_epoch, 'PSNR', 'PSNR over Epoch')
    plot(ssim_values, num_epoch, 'SSIM', 'SSIM over Epoch')
    plot(val_mse_values, num_epoch, 'Validation Loss', 'Validation Loss over Epoch')
    
    print("--------------------Test Start--------------------")
    model.eval()
    step = 0
    test_loss = 0
    test_psnr = 0
    test_ssim = 0
    
    with torch.no_grad():
        for lr_image, hr_image in test_loader:
            
            lr_image,hr_image = lr_image.to(device),hr_image.to(device)
            sr_image = model(lr_image)
                
            loss = loss_function(sr_image, hr_image)
            psnr = calculate_psnr(sr_image, hr_image)
            ssim = calculate_ssim(sr_image, hr_image)
            
            step += 1
            test_loss += loss.item()
            test_psnr += psnr
            test_ssim += ssim
            
    print(f"Test MSE: {test_loss/step}")
    print(f"Test PSNR: {test_psnr/step:.2f} dB")
    print(f"Test SSIM: {test_ssim/step}")  
    print(f"--------------------Test finished, Loss:{test_loss/step}--------------------")
            
def calculate_ssim(sr_tensor, hr_tensor):
    if sr_tensor.dim() == 4:
        sr_tensor = sr_tensor[0]
    if hr_tensor.dim() == 4:
        hr_tensor = hr_tensor[0]

    sr = sr_tensor.detach().cpu().numpy().transpose(1, 2, 0)
    hr = hr_tensor.detach().cpu().numpy().transpose(1, 2, 0)

    sr = np.clip(sr, 0, 1)
    hr = np.clip(hr, 0, 1)

    from skimage.metrics import structural_similarity as ssim
    return ssim(sr, hr, win_size=3, data_range=1.0, channel_axis=-1)

def plot_loss(loss_values, num_epoch):
    epochs = range(1, num_epoch + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss_values, label='Loss', marker='o', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.legend()
    plt.grid()
    plt.show()

def plot(values, num_epoch, label, title):
    epochs = range(1, num_epoch + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, values, label=label, marker='o', color='blue')
    plt.xlabel('Epoch')
    plt.ylabel(label)
    plt.title(title)
    plt.legend()
    plt.grid()
    plt.show()

def calculate_psnr(sr, hr, max_pixel_value=1.0):
    mse = F.mse_loss(sr, hr)  
    if mse == 0:
        return float('inf')
    psnr = 10 * math.log10((max_pixel_value ** 2) / mse.item())
    return psnr

def main():
    batch_size = 32
    learning_rate  = 0.0001
    num_epoch = 100
    scale_factor = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Track 1
    train_dataset = DIV2KDataset(train=True,scale_factor = scale_factor)
    val_dataset = DIV2KDataset(val=True,scale_factor = scale_factor)
    test_dataset = DIV2KDataset(test=True,scale_factor = scale_factor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)
    model = _NetG().to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999))
    train(model, train_loader, val_loader, test_loader, train_dataset, loss_function, optimizer, num_epoch, device)


    # Track 2
    train_dataset = DIV2KDataset(track1=False,train=True, scale_factor=scale_factor)
    val_dataset = DIV2KDataset(track1=False,val=True,scale_factor = scale_factor)
    test_dataset = DIV2KDataset(track1=False,test=True,scale_factor = scale_factor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset,batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)

    model = Track2_Net().to(device)
    loss_function = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), learning_rate, betas=(0.9, 0.999))

    train(model, train_loader, val_loader, test_loader, train_dataset, loss_function, optimizer, num_epoch, device)
   


if __name__ == '__main__':
    main()

