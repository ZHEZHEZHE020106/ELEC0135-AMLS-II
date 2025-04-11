# ELEC0135-AMLS-II
Image Super-Resolution with SRResNet (NTIRE 2017 Challenge)
This project implements an image super-resolution pipeline using the SRResNet architecture, evaluated on x2 scale for both Track 1 (bicubic downscaling) and Track 2 (unknown degradation) tasks from the NTIRE 2017 Super-Resolution Challenge. The goal is to reconstruct high-resolution (HR) images from low-resolution (LR) inputs using deep learning.

Dataset
We use the DIV2K dataset for training and evaluation. The dataset could be found here: https://data.vision.ee.ethz.ch/cvl/DIV2K/
Track 1: Bicubic ×2 downscaling
Track 2: Unknown ×2 degradation

Project Structure\n
----AMLSII——21072758
    ----A
    ----B
    ----Saved_Models
    ----dataset.py
    ----main.py
    ----model.py
    ----model2.py
The lr image folder for track 1 and hr image folder should be placed in the file A, the lr image folder for track 2 should be placed in file B.
The model after every epoch will be stored in Saved_Models
dataset.py contains the dataset class, model.py and model2.py contains the model used for track 1 and track 2 respectively.

The project used the following packages:
math
torch
torch.nn
torch.optim
torch.utils.data.DataLoader
torchvision.transforms
PIL.Image
matplotlib.pyplot
torch.nn.functional
numpy
