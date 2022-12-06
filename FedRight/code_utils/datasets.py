import random
from sklearn.model_selection import train_test_split
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import os
import h5py
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets, transforms
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch import nn, autograd
from tqdm import tqdm
import copy
import random
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from  code_utils.load_bank import *



DATA_DIR = '../data/gtrsb'  # data folder
DATA_FILE = 'gtsrb_dataset.h5'  # dataset file

def load_dataset(data_filename, keys=None):
    ''' assume all adult_datasets are numpy arrays '''
    dataset = {}
    with h5py.File(data_filename, 'r') as hf:
        if keys is None:
            for name in hf:
                dataset[name] = np.array(hf.get(name))
        else:
            print("h5py keys: ", keys)
            for name in keys:
                dataset[name] = np.array(hf.get(name))

    return dataset

def prepare_dataset(data_file=('%s/%s' % (DATA_DIR, DATA_FILE))):
    # if not os.path.exists(data_file):
        # print(
        #     "The data file does not exist. Please download the file and put in data/ "
        #     "directory from https://drive.google.com/file/d/1kcveaJC3Ra-XDuaNqHzYeomMvU8d1npj/view?usp=sharing")
        # exit(1)

    dataset = load_dataset(data_file, keys=['X_train', 'Y_train', 'X_test', 'Y_test'])

    X_train = np.transpose(np.array(dataset['X_train'], dtype='float32'), (0, 3, 1, 2))
    Y_train = np.array(dataset['Y_train'], dtype='int64')
    Y_train = np.asarray([np.where(r==1)[0][0] for r in Y_train])
    X_test = np.transpose(np.array(dataset['X_test'], dtype='float32'), (0, 3, 1, 2))
    Y_test = np.array(dataset['Y_test'], dtype='int64')
    Y_test = np.asarray([np.where(r==1)[0][0] for r in Y_test])

    tensor_x_train, tensor_y_train = torch.Tensor(X_train), torch.from_numpy(Y_train)
    tensor_train = torch.utils.data.TensorDataset(tensor_x_train, tensor_y_train)
    # dataset_train = torch.code_utils.data.DataLoader(tensor_train, batch_size=32, shuffle=True)
    print("训练数据集：",len(tensor_train))
    tensor_x_test, tensor_y_test = torch.Tensor(X_test), torch.from_numpy(Y_test)
    tensor_test = torch.utils.data.TensorDataset(tensor_x_test, tensor_y_test)
    # dataset_test = torch.code_utils.data.DataLoader(tensor_test, batch_size=32, shuffle=False)
    _, test_dataset_few = train_test_split(tensor_test, test_size=0.2, random_state=42)
    # return dataset_train, dataset_test
    return tensor_train, test_dataset_few












def get_dataset(dir, name, device):
    if name == 'mnist':
        train_datasets = datasets.MNIST(dir, train=True, download=True, transform=transforms.ToTensor())  # 6w
        eval_dataset = datasets.MNIST(dir, train=False, transform=transforms.ToTensor())  # 1w

        # train_dataset_split, test_dataset = train_test_split(train_datasets, test_size=0.002,
        #                                                      random_state=42)  # 4万个样本 1万个样本
        # 0.02 test 120个样本 train 59880个样本

        _, test_dataset_few = train_test_split(eval_dataset, test_size=0.2, random_state=42)  # 60个样本
    # print("验证集数量：",len(test_dataset_few))

    elif name == 'fmnist':
        train_datasets = datasets.FashionMNIST(dir, train=True, download=True, transform=transforms.ToTensor())  # 6w
        eval_dataset = datasets.FashionMNIST(dir, train=False, transform=transforms.ToTensor())  # 1w

        # train_dataset_split, test_dataset = train_test_split(train_datasets, test_size=0.002,
        #                                                      random_state=42)  # 4万个样本 1万个样本

        # 0.2 test_dataset_few 2000个样本
        _, test_dataset_few = train_test_split(eval_dataset, test_size=0.2, random_state=42)


    elif name == 'lfw':
        transform_train = transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor()

        ])
        train_datasets = datasets.ImageFolder(r"./data/LFW/train", transform=transform_train)  # 6w
        eval_dataset = datasets.ImageFolder(r"./data/LFW/test",  transform=transform_train)  # 1w

        _, test_dataset_few = train_test_split(eval_dataset, test_size=0.2, random_state=42)

    elif name == 'ce':
        transform_train = transforms.Compose([
            transforms.Resize([32,32]),
            transforms.ToTensor()

        ])
        train_datasets = datasets.ImageFolder(r"./data/celeba/train", transform=transform_train)  # 6w
        eval_dataset = datasets.ImageFolder(r"./data/celeba/test",  transform=transform_train)  # 1w

        print(len(train_datasets))

        _, test_dataset_few = train_test_split(eval_dataset, test_size=0.2, random_state=42)


    elif name == "bank":
        bank = process_bank('bank-full.csv')
        train_datasets, eval_dataset = data_set(bank)
        # train_datasets, eval_dataset, posion_data, poison_test, EOD_test_p_1, EOD_test_p_2, SPD_test_p_1, \
        # SPD_test_p_2 = bank_balance_split(bank)
        test_dataset_few = eval_dataset

    elif name == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        train_datasets = datasets.CIFAR10(dir, train=True, download=True,
                                          transform=transform_train)
        eval_dataset = datasets.CIFAR10(dir, train=False, transform=transform_test)
        # train_dataset_split, test_dataset = train_test_split(train_datasets, test_size=0.002,
        #                                                      random_state=42)  # 4万个样本 1万个样本

        _,test_dataset_few = train_test_split(eval_dataset, test_size=0.2, random_state=42)  # 4万个样本 1万个样本    #用来做验证数据集

    elif name == 'gtrsb':

        train_datasets, eval_dataset = prepare_dataset()
        test_dataset_few = eval_dataset

    elif name == 'adult':
        from load_adult import get_train_test
        from Custom_Dataset import Custom_Dataset
        import torch

        train_data, train_target, test_data, test_target = get_train_test()

        X_train = torch.tensor(train_data.values, requires_grad=False).float()
        y_train = torch.tensor(train_target.values, requires_grad=False).long()
        X_test = torch.tensor(test_data.values, requires_grad=False).float()
        y_test = torch.tensor(test_target.values, requires_grad=False).long()

        print("X train shape: ", X_train.shape)
        print("y train shape: ", y_train.shape)
        pos, neg = (y_train == 1).sum().item(), (y_train == 0).sum().item()
        print("Train set Positive counts: {}".format(pos), "Negative counts: {}.".format(neg),
              'Split: {:.2%} - {:.2%}'.format(1. * pos / len(X_train), 1. * neg / len(X_train)))
        print("X test shape: ", X_test.shape)
        print("y test shape: ", y_test.shape)
        pos, neg = (y_test == 1).sum().item(), (y_test == 0).sum().item()
        print("Test set Positive counts: {}".format(pos), "Negative counts: {}.".format(neg),
              'Split: {:.2%} - {:.2%}'.format(1. * pos / len(X_test), 1. * neg / len(X_test)))

        train_indices, valid_indices = get_train_valid_indices(len(X_train), 0.8)

        train_datasets = Custom_Dataset(X_train[train_indices], y_train[train_indices], device=device)
        eval_dataset = Custom_Dataset(X_train[valid_indices], y_train[valid_indices], device=device)
        test_dataset_few = Custom_Dataset(X_test, y_test, device=device)

        # train_dataset_split, test_dataset = train_test_split(train_datasets, test_size=0.2, random_state=42)



    elif name == "cancer":

        train_datasets, eval_dataset, test_dataset_few = cancer()



        test_dataset_few = eval_dataset





    print("test_data_size", len(test_dataset_few))


    return train_datasets, eval_dataset, test_dataset_few


def get_train_valid_indices(n_samples, train_val_split_ratio, sample_size_cap=None):
    indices = list(range(n_samples))
    random.seed(1111)
    random.shuffle(indices)
    split_point = int(n_samples * train_val_split_ratio)
    train_indices, valid_indices = indices[:split_point], indices[split_point:]

    return train_indices, valid_indices
