import torch
import torch.nn as nn
from torchsummary import summary

def make_divisible(v,divisor=8,min_vale=None):
    if min_vale is None:
        min_vale = divisor

    new_v = max(min_vale,int(v+divisor/2)//divisor*divisor)

    if new_v < 0.9*v:
        new_v += divisor

    return new_v

class ConvBNReLU(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size,stride=1,padding=None,groups=1,if_use=None):
        super().__init__()
        if padding is None:
            padding = kernel_size //2

        self.conv = nn.Conv2d(in_ch,out_ch,kernel_size,stride,padding=padding,groups=groups,bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU6(inplace=True) if if_use else nn.Identity()


    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)

        return x


class InvertedResidualBlock(nn.Module):
    def __init__(self,in_ch,out_ch,stride,t=6):
        super().__init__()

        self.use_short = (stride == 1 and in_ch == out_ch)

        hidden_ch = int(in_ch*t)

        if t!=1:
            self.expand = ConvBNReLU(in_ch,hidden_ch,kernel_size=1,stride=1,if_use=True)
        else:
            self.expand = None
            hidden_ch = in_ch

        self.depthwise = ConvBNReLU(hidden_ch,hidden_ch,kernel_size=3,stride=stride,groups=hidden_ch,if_use=True)

        self.project = ConvBNReLU(hidden_ch,out_ch,kernel_size=1,stride=1,if_use=False)

    def forward(self,x):
        out = x
        if self.expand is not None:
            out = self.expand(out)

        out = self.depthwise(out)
        out = self.project(out)

        return x+out if self.use_short else out


class MobileNetV2(nn.Module):
    def __init__(self,in_ch=3,scale=1,round_nearest=8,num_classes=1000,first_stride=2):
        super().__init__()

        def c(ch):
            return make_divisible(ch*scale,round_nearest)


        self.conv1 = ConvBNReLU(in_ch,c(32),kernel_size=3,stride=first_stride,padding=1)

        self.block1 = InvertedResidualBlock(c(32),c(16),stride=1,t=1)

        self.block2 = nn.Sequential(
            InvertedResidualBlock(c(16),c(24),stride=2,t=6),
            InvertedResidualBlock(c(24),c(24),stride=1,t=6)
        )
        self.block3 = nn.Sequential(
            InvertedResidualBlock(c(24),c(32),stride=2,t=6),
            InvertedResidualBlock(c(32), c(32), stride=1, t=6),
            InvertedResidualBlock(c(32), c(32), stride=1, t=6)
        )
        self.block4 = nn.Sequential(
            InvertedResidualBlock(c(32), c(64), stride=2, t=6),
            InvertedResidualBlock(c(64), c(64), stride=1, t=6),
            InvertedResidualBlock(c(64), c(64), stride=1, t=6),
            InvertedResidualBlock(c(64), c(64), stride=1, t=6),
        )
        self.block5 = nn.Sequential(
            InvertedResidualBlock(c(64), c(96), stride=1, t=6),
            InvertedResidualBlock(c(96), c(96), stride=1, t=6),
            InvertedResidualBlock(c(96), c(96), stride=1, t=6),
        )
        self.block6 = nn.Sequential(
            InvertedResidualBlock(c(96), c(160), stride=2, t=6),
            InvertedResidualBlock(c(160), c(160), stride=1, t=6),
            InvertedResidualBlock(c(160), c(160), stride=1, t=6),
        )
        self.block7 = InvertedResidualBlock(c(160),c(320),stride=1,t=6)

        self.conv2 = ConvBNReLU(c(320),c(1280),kernel_size=1,stride=1,if_use=True)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.drop = nn.Dropout(0.3)

        self.fc = nn.Linear(c(1280),num_classes)

        for m in self.modules():
            if isinstance(m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m,nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m,nn.Linear):
                nn.init.normal_(m.weight,0,0.001)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self,x):
        x = self.conv1(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.conv2(x)
        x = self.pool(x).flatten(1)
        x = self.drop(x)
        return self.fc(x)


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ImageNet：参数量=3,504,872（stride 不影响参数量）
    model_imagenet = MobileNetV2(num_classes=1000, first_stride=2).to(device)
    summary(model_imagenet, (3, 32, 32))
