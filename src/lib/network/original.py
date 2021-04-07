import torch
import torch.nn as nn


class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=False):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.conv(x)
        h = self.bn(h)
        h = self.relu(h)

        return h


class BasicBlock(nn.Module):

    def __init__(self, num_conv, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=False, drop_out=0, pooling_size=0):
        super(BasicBlock, self).__init__()
        self.in_conv = ConvBlock(
            in_channels=in_channel, 
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
            )

        blocks = []
        for i in range(num_conv-1):
            blocks.append(
                ConvBlock(
                    in_channels=out_channel, 
                    out_channels=out_channel,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    bias=bias
                )
            )
        self.blocks = nn.Sequential(*blocks)

        if pooling_size > 1:
            self.pooling = nn.MaxPool2d(kernel_size=2)
        else:
            self.pooling = None

        if drop_out > 0:
            self.drop_out = nn.Dropout2d(p=drop_out)
        else:
            self.drop_out = None

    def forward(self, x):
        h = self.in_conv(x)
        h = self.blocks(h)

        if self.pooling is not None:
            h = self.pooling(h)

        if self.drop_out is not None:
            h = self.drop_out(h)

        return h


class CNNModel(nn.Module):

    def __init__(self, output_dim=256):
        super(CNNModel, self).__init__()
        bias = False

        self.layer1 = BasicBlock(
            num_conv=2,
            in_channel=3,
            out_channel=64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            drop_out=0.25,
            pooling_size=2
        )
        self.layer2 = BasicBlock(
            num_conv=2,
            in_channel=64,
            out_channel=128,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            drop_out=0.25,
            pooling_size=2
        )
        self.layer3 = BasicBlock(
            num_conv=4,
            in_channel=128,
            out_channel=256,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            drop_out=0.25,
            pooling_size=2
        )
        self.layer4 = BasicBlock(
            num_conv=2,
            in_channel=256,
            out_channel=1024,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=bias,
            drop_out=0.5,
        )
        self.layer5 = BasicBlock(
            num_conv=2,
            in_channel=1024,
            out_channel=output_dim,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
            drop_out=0.5,
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.output_dim = output_dim
        
    def forward(self, x):
        for i in range(5):
            module = getattr(self, "layer{}".format(i+1))
            x = module(x)

        x = self.avg_pool(x)
        x = x.squeeze(2).squeeze(2)
        # logit = self.output_conv(x)

        return x


if __name__ == "__main__":
    net = CNNModel()
    x = torch.randn(2, 3, 32, 32)
    logit = net(x)
    print(logit.shape)