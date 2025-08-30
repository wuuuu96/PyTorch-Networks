# U-Net

## U-Net算法背景

<img width="1700" height="949" alt="image" src="https://github.com/user-attachments/assets/b7c7be04-61f3-4052-b462-adcc02af8c3f" />

## 论文中提到的网络结构

**编码器进行下采样提取特征**

**解码器进行上采样恢复信息**

**跳跃连接保留信息，提高分割精度**

<img width="1747" height="889" alt="image" src="https://github.com/user-attachments/assets/09b5c21c-2c24-4e99-9dde-360fd2be92f0" />

## 论文中的UNet模型详解

<img width="1329" height="693" alt="image" src="https://github.com/user-attachments/assets/4a306627-3b4b-4851-89c8-b5a06722583f" />

## 代码中实际应用的U-Net模型结构，ResNet-50主干(编码器)，解码器(线性插值和Concat融合)

<img width="1588" height="969" alt="image" src="https://github.com/user-attachments/assets/c316faf6-f564-4042-be69-a15f88a10af3" />

<img width="1575" height="985" alt="image" src="https://github.com/user-attachments/assets/185b365c-eeeb-4b5a-be11-af02339d8776" />



