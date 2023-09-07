from model.utils import *
import timm

class ResBlock(nn.Module):

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv(ch, ch, 1, 1, "mish"))
            resblock_one.append(Conv(ch, ch, 3, 1, "mish"))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSampleFirst(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSampleFirst, self).__init__()
        self.conv0 = Conv(in_channels, 32, 3, 1, "mish")
        self.conv1 = Conv(32, 64, 3, 2, "mish")
        self.conv2 = Conv(64, 64, 1, 1, "mish")
        self.conv4 = Conv(64, 64, 1, 1, "mish")  
        self.conv5 = Conv(64, 32, 1, 1, "mish")
        self.conv6 = Conv(32, 64, 3, 1, "mish")  
        self.conv8 = Conv(64, 64, 1, 1, "mish")
        self.conv10 = Conv(128, out_channels, 1, 1, "mish")  

    def forward(self, x):
        x0 = self.conv0(x)
        x1 = self.conv1(x0)
        x2 = self.conv2(x1)
        x3 = self.conv4(x1)
        x4 = self.conv5(x3)
        x5 = x3 + self.conv6(x4)
        x6 = self.conv8(x5)
        x6 = torch.cat([x6, x2], dim=1)
        x7 = self.conv10(x6)
        return x7


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels, res_blocks):
        super(DownSample, self).__init__()
        self.conv1 = Conv(in_channels, in_channels * 2, 3, 2, "mish")
        self.conv2 = Conv(in_channels * 2, in_channels, 1, 1, "mish")
        self.conv4 = Conv(in_channels * 2, in_channels, 1, 1, "mish")  
        self.resblock = ResBlock(ch=in_channels, nblocks=res_blocks)
        self.conv11 = Conv(in_channels, in_channels, 1, 1, "mish")
        self.conv13 = Conv(in_channels * 2, out_channels, 1, 1, "mish")  

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv4(x1)
        r = self.resblock(x3)
        x4 = self.conv11(r)
        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv13(x4)
        return x5




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

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
       
        self.BB = timm.create_model('hrnet_w40', pretrained=True, features_only=True, out_indices=[1, 2, 3, 4])


    def forward(self, i):

        Feature = self.BB(i)
        d2 = Feature[0]
        d3 = Feature[1]
        d4 = Feature[2]
        d5 = Feature[3]
        return d2, d3, d4, d5
