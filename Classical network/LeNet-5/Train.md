## Train.py

### **1.数据下载和加载过程**

``` python
def train_val_process():

    **定义：transform变量,用于数据的增强**
    数据增强包括：
    1.将图像padding=2,随机裁剪图像大小为28*28，    transforms.RandomCrop(28,2)
    2.将图像水平翻转      transforms.RandomHorizontalFlip()
    3.将图像随机旋转10°    
    4.转换图像为 Tensor
    5.对图像进行归一化
    
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
```
<img width="1217" height="468" alt="image" src="https://github.com/user-attachments/assets/d3a44348-6ac1-47ed-bf74-581663ba1444" />
