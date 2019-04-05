import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from data_loader import PoseDataset
from model import Net


def validate(model, criterion, test_loader, device):
    """Validation method

    Keyword Arguments:
    model - neural network
    criterion - loss function
    test_loader - dataloader for test set
    device - cpu / cuda"""
    loss = 0
    for idx, (skel_2d, skel_z) in enumerate(test_loader):
        inputs, labels = skel_2d.to(device), skel_z.to(device)
        outputs = model(inputs)
        loss += criterion(outputs, labels).item()

    return loss / len(test_loader)

def savebest(model, epoch, val_loss):
    """Saving the best model given the validation loss.

    Keyword Arguments:
    model - neural network model
    epoch - epoch we're on
    val_loss - loss on the validation set"""
    save_path = './models/{}_{}.pth'.format(epoch, val_loss)
    torch.save(model.state_dict(), save_path)

def train(dataset_path, batch_size, learning_rate, momentum):
    """Training loop, saves the best model.

    Keyword Arguments:
    dataset_path - path to pickled generated dataset
    batch_size - batch size for training and validating
    learning_rate - learning rate to start with
    momentum - sgd momentum
    """
    pose_dataset = PoseDataset(dataset_path)

    # random, non-contiguous train/val split
    indices = list(range(len(pose_dataset)))
    val_size = round(len(pose_dataset) * 0.1)
    val_idx = np.random.choice(indices, size=val_size, replace=False)
    train_idx = list(set(indices) - set(val_idx))
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    train_loader = DataLoader(dataset=pose_dataset, batch_size=batch_size,\
        sampler=train_sampler)
    val_loader = DataLoader(dataset=pose_dataset, batch_size=batch_size,\
        sampler=val_sampler)

    # save val_idx
    np.save('val_idx.npy', val_idx)

    # cpu or gpu?
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('using device {}'.format(device))

    # define net
    net = Net()
    net.to(device)

    # define loss & optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), learning_rate, momentum)
    scheduler = ReduceLROnPlateau(optimizer, 'min', verbose=True)
    best_loss = 0.0

    print_stats = 20

    # train
    for epoch in range(200):
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            skel_2d, skel_z = data
            inputs = skel_2d
            labels = skel_z
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # validate and print statistics
            running_loss += loss.item()
            if i % print_stats == 0:  # print every 2000 mini-batches
                # validate
                with torch.no_grad():
                    net.eval()
                    val_loss = validate(net, criterion, val_loader, device)
                    if best_loss == 0.0:
                        best_loss = val_loss
                    if val_loss < best_loss:
                        best_loss = val_loss
                        savebest(net, epoch, val_loss)
                    net.train()
                    scheduler.step(val_loss)

                # print
                train_loss = running_loss / print_stats
                print('[%d, %5d] train loss: %.3f, val loss: %.3f' %\
                    (epoch + 1, i + 1, train_loss, val_loss))
                running_loss = 0.0


if __name__ == "__main__":
    dataset_path = './data/panoptic_dataset.pickle'
    batch_size = 2000
    learning_rate = 0.04
    momentum = 0.9
    train(dataset_path, batch_size, learning_rate, momentum)
