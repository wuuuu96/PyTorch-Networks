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

为什么要上采样：



