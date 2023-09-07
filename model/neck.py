from model.utils import *

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size, inference):
        assert (x.data.dim() == 4)
        _, _, tw, th = target_size

        if inference:
            B, C, W, H = x.size()
            return x.view(B, C, W, 1, H, 1).expand(B, C, W, tw // W, H, th // H).contiguous().view(B, C, tw, th)
        else:
            return F.interpolate(x, size=(tw, th), mode='nearest')


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv(1024, 512, 1, 1, 'leaky')
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)
        self.conv4 = Conv(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv(512, 256, 1, 1, 'leaky')
        self.upsample1 = Upsample()
        self.conv8 = Conv(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv(256, 128, 1, 1, 'leaky')
        self.upsample2 = Upsample()
        self.conv15 = Conv(256, 128, 1, 1, 'leaky')
        self.conv16 = Conv(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv(256, 128, 1, 1, 'leaky') 
        self.cbam01=cbam_block(1024) 
        self.conv_downsampling_3 = nn.Conv2d(256, 1024, kernel_size=4, stride=4)
        self.conv50 = Conv(2048, 1024, 3, 1, 'leaky')

    def forward(self, d2, input, downsample4, downsample3, inference):
        
        
        downsample3_d5 = self.conv_downsampling_3(downsample3)
        input = torch.cat([downsample3_d5, input], dim=1)
        input = self.conv50(input)
        input = self.cbam01(input)       
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        up = self.upsample1(x7, downsample4.size(), inference)
        x8 = self.conv8(downsample4)
        x8 = torch.cat([x8, up], dim=1)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        up = self.upsample2(x14, downsample3.size(), inference)
        x15 = self.conv15(downsample3)
        x15 = torch.cat([x15, up], dim=1)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6
