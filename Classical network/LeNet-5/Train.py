#导包

import copy
import time
import pandas as pd
from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as data
from model import LeNet
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def train_val_process():

    transform = transforms.Compose([
        transforms.RandomCrop(28,2),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize((0.5,),(0.5,))
    ])


    full_data = FashionMNIST(root='./data',train=True,transform=transform,download=True)

    train_data,val_data = data.random_split(full_data,[round(0.8*len(full_data)),round(0.2*len(full_data))])

    train_dataloader = data.DataLoader(train_data,32,True,num_workers=0)
    val_dataloader = data.DataLoader(val_data,32,False,num_workers=0)

    return train_dataloader,val_dataloader


def train_model(model,train_dataloader,val_dataloader,num_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)

    criterion = nn.CrossEntropyLoss()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())

    best_acc = 0.0

    train_loss_all = []

    train_acc_all = []

    val_loss_all = []

    val_acc_all =[]

    since = time.time()

    for epoch in range(num_epoch):
        print(f'Epoch {epoch+1}/{num_epoch}')
        print('-'*30)

        epoch_time = time.time()
        train_loss = 0.0
        train_acc = 0.0
        train_num = 0
        val_loss = 0.0
        val_acc = 0.0
        val_num = 0

        model.train()
        t1 = time.time()
        for step,(inputs,targets) in enumerate(train_dataloader):
            inputs = inputs.to(device)
            targets = targets.to(device)



            outputs = model(inputs)

            pred = torch.argmax(outputs,dim=1)

            loss = criterion(outputs,targets)

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            train_loss += loss.item()*inputs.size(0)

            train_acc += torch.sum(pred==targets).item()

            train_num += inputs.size(0)

        t2 = time.time()
        model.eval()
        with torch.no_grad():
            for step,(inputs,targets) in enumerate(val_dataloader):

                inputs = inputs.to(device)
                targets = targets.to(device)

                outputs = model(inputs)

                pred =torch.argmax(outputs,dim=1)

                loss = criterion(outputs,targets)

                val_loss += loss.item()*inputs.size(0)

                val_acc += torch.sum(pred==targets).item()

                val_num += inputs.size(0)
        t3 = time.time()

        train_loss_all.append(train_loss / train_num)
        train_acc_all.append(train_acc / train_num)
        val_loss_all.append(val_loss / val_num)
        val_acc_all.append(val_acc / val_num)

        print(f"Epoch:{epoch+1} Train_loss:{train_loss_all[-1]:.4f} Train_acc:{train_acc_all[-1]*100:.2f}")
        print(f"Epoch:{epoch+1} Val_loss:{val_loss_all[-1]:.4f} Val_acc:{val_acc_all[-1] * 100:.2f}")
        print(f"训练时间:{t2 - t1:.2f} 验证时间:{t3 - t2:.2f} 总用时:{time.time()-epoch_time:.2f}")

        if val_acc_all[-1] > best_acc:
            best_acc = val_acc_all[-1]

            best_model_wts = copy.deepcopy(model.state_dict())

            torch.save(model.state_dict(),'./model.pth')

            print(f"验证准确率为:{best_acc*100:.2f}")

    total = time.time() - since

    print(f"\n 训练完成，总用时为:{total//60:.0f}m{total%60:.0f}s")
    print(f"验证准确率为:{best_acc*100:.2f}")

    model.load_state_dict(best_model_wts)

    train_process = pd.DataFrame(data={"epoch":range(1,num_epoch+1),
                                       "train_loss_all":train_loss_all,
                                       "train_acc_all":train_acc_all,
                                       "val_loss_all":val_loss_all,
                                       "val_acc_all":val_acc_all
                                       })
    return train_process


def plot(train_process):
    plt.figure(figsize=(12,4))

    plt.subplot(121)

    plt.plot(train_process['epoch'],train_process['train_loss_all'],'-ro',label='train_loss')
    plt.plot(train_process['epoch'],train_process['val_loss_all'],'-ro',label='val_loss')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.subplot(122)
    plt.plot(train_process['epoch'],train_process['train_acc_all'],'-ro',label='train_acc')
    plt.plot(train_process['epoch'],train_process['val_acc_all'],'-ro',label='val_acc')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('acc')

    plt.show()

if __name__ == '__main__':
    model = LeNet()

    train_dataloader,val_dataloader = train_val_process()

    train_process = train_model(model,train_dataloader,val_dataloader,100)

    plot(train_process)

