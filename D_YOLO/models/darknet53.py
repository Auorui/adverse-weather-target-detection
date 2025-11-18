import torch
import torch.nn as nn

class Conv2dBatchLeaky(nn.Module):
    """
    This convenience layer groups a 2D convolution, a batchnorm and a leaky ReLU.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, leaky_slope=0.1):
        super(Conv2dBatchLeaky, self).__init__()

        # Parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        if isinstance(kernel_size, (list, tuple)):
            self.padding = [int(k/2) for k in kernel_size]
        else:
            self.padding = int(kernel_size/2)
        self.leaky_slope = leaky_slope

        # Layer
        self.layers = nn.Sequential(
            nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding, bias=False),
            nn.BatchNorm2d(self.out_channels),
            nn.LeakyReLU(self.leaky_slope, inplace=True)
        )

    def __repr__(self):
        s = '{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}, stride={stride}, padding={padding}, negative_slope={leaky_slope})'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def forward(self, x):
        x = self.layers(x)
        return x

class StageBlock(nn.Module):

    def __init__(self, nchannels):
        super().__init__()
        self.features = nn.Sequential(
            Conv2dBatchLeaky(nchannels, int(nchannels/2), 1, 1),
            Conv2dBatchLeaky(int(nchannels/2), nchannels, 3, 1)
        )

    def forward(self, data):
        return data + self.features(data)


class Stage(nn.Module):

    def __init__(self, nchannels, nblocks, stride=2):
        super().__init__()
        blocks = []
        blocks.append(Conv2dBatchLeaky(nchannels, 2*nchannels, 3, stride))
        for i in range(nblocks):
            blocks.append(StageBlock(2*nchannels))
        self.features = nn.Sequential(*blocks)

    def forward(self, data):
        return self.features(data)

class DarkNet53(nn.Module):
    def __init__(self, num_classes):
        super(DarkNet53, self).__init__()

        input_channels = 32
        stage_cfg = {'stage_2':1, 'stage_3':2, 'stage_4':8, 'stage_5':8, 'stage_6':4}

        # Network
        self.stage1 = Conv2dBatchLeaky(3, input_channels, 3, 1, 1)
        self.stage2 = Stage(input_channels, stage_cfg['stage_2'])
        self.stage3 = Stage(input_channels*(2**1), stage_cfg['stage_3'])
        self.stage4 = Stage(input_channels*(2**2), stage_cfg['stage_4'])
        self.stage5 = Stage(input_channels*(2**3), stage_cfg['stage_5'])
        self.stage6 = Stage(input_channels*(2**4), stage_cfg['stage_6'])

        # self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        # self.fc = nn.Linear(1024, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        stage1 = self.stage1(x)
        stage2 = self.stage2(stage1)
        stage3 = self.stage3(stage2)
        stage4 = self.stage4(stage3)
        stage5 = self.stage5(stage4)
        stage6 = self.stage6(stage5)
        # 只返回三个尺度的特征图，不进行分类
        # x = self.avgpool(stage6)
        # x = x.view(-1, 1024)
        # x = self.fc(x)
        return stage4, stage5, stage6

if __name__ == "__main__":
    with torch.no_grad():
        model = DarkNet53(num_classes=1000)
        # model = torch.nn.DataParallel(model)
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # model.load_state_dict(torch.load('darknet53.pth', map_location=device))
