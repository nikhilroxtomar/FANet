
import os
import time
import datetime
import random
import numpy as np
from glob import glob
import albumentations as A
import cv2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils import (
    seeding, shuffling, create_dir, init_mask,
    epoch_time, rle_encode, rle_decode, print_and_save, load_data
    )
from model import FANet
from loss import DiceBCELoss

class DATASET(Dataset):
    def __init__(self, images_path, masks_path, size, transform=None):
        super().__init__()

        self.images_path = images_path
        self.masks_path = masks_path
        self.transform = transform
        self.n_samples = len(images_path)

    def __getitem__(self, index):
        """ Image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]

        image = cv2.resize(image, size)
        image = np.transpose(image, (2, 0, 1))
        image = image/255.0
        image = image.astype(np.float32)


        mask = cv2.resize(mask, size)
        mask = np.expand_dims(mask, axis=0)
        mask = mask/255.0
        mask = mask.astype(np.float32)

        return image, mask

    def __len__(self):
        return self.n_samples

def train(model, loader, mask, optimizer, loss_fn, device):
    epoch_loss = 0
    return_mask = []

    model.train()
    for i, (x, y) in enumerate(loader):
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        b, c, h, w  = y.shape
        m = []
        for edata in mask[i*b : i*b+b]:
            edata = " ".join(str(d) for d in edata)
            edata = str(edata)
            edata = rle_decode(edata, size)
            edata = np.expand_dims(edata, axis=0)
            m.append(edata)

        m = np.array(m, dtype=np.int32)
        m = np.transpose(m, (0, 1, 3, 2))
        m = torch.from_numpy(m)
        m = m.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model([x, m])
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().numpy()

            for py in y_pred:
                py = np.squeeze(py, axis=0)
                py = py > 0.5
                py = np.array(py, dtype=np.uint8)
                py = rle_encode(py)
                return_mask.append(py)

        epoch_loss += loss.item()

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss, return_mask

def evaluate(model, loader, mask, loss_fn, device):
    epoch_loss = 0
    return_mask = []

    model.eval()
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            x = x.to(device)
            y = y.to(device)

            b, c, h, w  = y.shape
            m = []
            for edata in mask[i*b : i*b+b]:
                edata = " ".join(str(d) for d in edata)
                edata = str(edata)
                edata = rle_decode(edata, size)
                edata = np.expand_dims(edata, axis=0)
                m.append(edata)

            m = np.array(m)
            m = np.transpose(m, (0, 1, 3, 2))
            m = torch.from_numpy(m)
            m = m.to(device, dtype=torch.float32)


            y_pred = model([x, m])
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

            y_pred = torch.sigmoid(y_pred)
            y_pred = y_pred.cpu().numpy()

            for py in y_pred:
                py = np.squeeze(py, axis=0)
                py = py > 0.5
                py = np.array(py, dtype=np.uint8)
                py = rle_encode(py)
                return_mask.append(py)

    epoch_loss = epoch_loss/len(loader)
    return epoch_loss, return_mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Directories """
    create_dir("files")

    """ Training logfile """
    train_log_path = "files/train_log.txt"
    if os.path.exists(train_log_path):
        print("Log file exists")
    else:
        train_log = open("files/train_log.txt", "w")
        train_log.write("\n")
        train_log.close()

    """ Record Date & Time """
    datetime_object = str(datetime.datetime.now())
    print_and_save(train_log_path, datetime_object)

    """ Hyperparameters """
    size = (256, 256)
    batch_size = 2
    num_epochs = 500
    lr = 1e-4
    checkpoint_path = "files/checkpoint.pth"

    """ Dataset """
    path = "../Kvasir-SEG"
    (train_x, train_y), (valid_x, valid_y) = load_data(path)
    train_x, train_y = shuffling(train_x, train_y)

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    """ Data augmentation: Transforms """
    transform =  A.Compose([
        A.Rotate(limit=35, p=0.3),
        A.HorizontalFlip(p=0.3),
        A.VerticalFlip(p=0.3),
        A.CoarseDropout(p=0.3, max_holes=10, max_height=32, max_width=32)
    ])

    """ Dataset and loader """
    train_dataset = DATASET(train_x, train_y, size, transform=transform)
    valid_dataset = DATASET(valid_x, valid_y, size, transform=None)

    data_str = f"Dataset Size:\nTrain: {len(train_x)} - Valid: {len(valid_x)}\n"
    print_and_save(train_log_path, data_str)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )

    """ Model """
    device = torch.device('cuda')
    model = FANet()
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
    loss_fn = DiceBCELoss()
    loss_name = "BCE Dice Loss"

    data_str = f"Hyperparameters:\nImage Size: {size}\nBatch Size: {batch_size}\nLR: {lr}\nEpochs: {num_epochs}\n"
    data_str += f"Optimizer: Adam\nLoss: {loss_name}\n"
    print_and_save(train_log_path, data_str)

    """ Training the model. """
    best_valid_loss = float('inf')
    train_mask = init_mask(train_x, size)
    valid_mask = init_mask(valid_x, size)

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss, return_train_mask = train(model, train_loader, train_mask, optimizer, loss_fn, device)
        valid_loss, return_valid_mask = evaluate(model, valid_loader, valid_mask, loss_fn, device)
        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            data_str = f"Saving checkpoint: {checkpoint_path}"
            print_and_save(train_log_path, data_str)
            torch.save(model.state_dict(), checkpoint_path)

            train_mask = return_train_mask
            valid_mask = return_valid_mask

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        data_str = f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
        data_str += f'\tTrain Loss: {train_loss:.3f}\n'
        data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
        print_and_save(train_log_path, data_str)
