利用FCN模型进行语义分割任务本质上也是一个分类任务，只不过从任务本质来说对图像进行像素级的分类（也就是每个像素点都进行分类）。

**1.FCN模型有如下特性：**

  1、模型的输入图片可以是任意大小（yolo系列的目标检测模型的输入也是任意大小的。）

  2、模型的特征提取网络可以是任意任意卷积网络模型（最初FCN网络的特征提取网络为VGG，pytorch官方代码中用的是ResNet网络。）

  3、最后网络模型的输出为和输入图像一样高宽的高维向量。


**2.FCN模型输出解析**

<img width="1968" height="1067" alt="image" src="https://github.com/user-attachments/assets/ab01e516-1826-4ccb-80fe-90085266b606" />

**总结：**

![1cad8aee0d321b807ebd86caef03bfca](https://github.com/user-attachments/assets/82b8d4e4-6f96-4835-bd31-fc6f74ed3abf)

**3.上采样(转置卷积和插值)**

为什么要上采样：由于卷积网络会使输入的W和H变小,而FCN又要求输入的W,H和输出的W,H大小一致,这就需要上采样了。

<img width="2025" height="1144" alt="image" src="https://github.com/user-attachments/assets/ddc252f3-278b-440b-8e1e-89b3a648943e" />

<img width="2068" height="1074" alt="image" src="https://github.com/user-attachments/assets/c55670e8-3b5e-4fbc-939b-07659bc5c85d" />

<img width="1933" height="749" alt="image" src="https://github.com/user-attachments/assets/ae5793d3-d8e8-41af-8412-2c7ae2f4cc10" />

<img width="1947" height="1133" alt="image" src="https://github.com/user-attachments/assets/1a97be20-7fcc-4b56-99c6-506f5fd1dec2" />

<img width="2068" height="1104" alt="image" src="https://github.com/user-attachments/assets/55f941ac-8b83-41e4-9dba-d3b136600170" />



