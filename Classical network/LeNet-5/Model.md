# 环境:  python==3.10.0 pytorch==2.1.2 cuda==12.1 

## **1.导包**

```python
import torch
from torch import nn
from torchsummary import summary
```

## **2.网络初始化**  
```python
class LeNet(nn.Module):
    def __init__(self):
      super().__init__()
```
## **3.定义网络模型参数层和forward函数**

<img width="1239" height="600" alt="image" src="https://github.com/user-attachments/assets/7d03e456-53a5-43cb-8c43-f70fa845032e" />

```python
      self.conv1 = nn.Conv2d(1, 6, 5, 1, 2)
      self.sig = nn.Sigmoid()  # Sigmoid激活函数
      self.pool = nn.AvgPool2d(2, 2)  # 平均池化
      self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)  # stride=1和padding=0可以省略

      self.flatten = nn.Flatten()
      self.fc1 = nn.Linear(400, 120)
      self.fc2 = nn.Linear(120, 84)
      self.fc3 = nn.Linear(84, 10)
  
    def forward(self, x):
        x = self.pool(self.sig(self.conv1(x)))
        x = self.pool(self.sig(self.conv2(x)))
        x = self.flatten(x)
        x = self.fc3(self.fc2(self.fc1(x)))
        return x
```

## **4.主函数编写**

``` python
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LeNet().to(device)
    print(summary(model,(1,28,28)))
```

## **5.结果**

<img width="808" height="531" alt="image" src="https://github.com/user-attachments/assets/ad1d2aa2-bcae-46cf-a3fd-a0fd2c7a6be5" />
