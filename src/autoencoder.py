import torch
import torch.nn as nn
import torchvision

from catalyst.utils import normal_sample

LOG_SIG_MAX = 2
LOG_SIG_MIN = -10


class ResnetEncoderAE(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.encoder0 = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.encoder1 = backbone.layer1
        self.encoder2 = backbone.layer2
        self.encoder3 = backbone.layer3
        self.encoder4 = backbone.layer4

        self.filters = [
            module[-1].conv3.out_channels
            if "conv3" in module[-1].__dict__["_modules"] else
            module[-1].conv2.out_channels for module in
            [self.encoder1, self.encoder2, self.encoder3, self.encoder4]
        ]

    def forward(self, x):
        x = self.encoder0(x)
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.encoder3(x)
        x = self.encoder4(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, m, n, stride=2):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(m, m // 4, 1)
        self.norm1 = nn.BatchNorm2d(m // 4)
        self.relu1 = nn.ReLU(inplace=False)

        # B, C/4, H, W -> B, C/4, H, W
        self.conv2 = nn.ConvTranspose2d(
            m // 4, m // 4, 3, stride=stride, padding=1
        )
        self.norm2 = nn.BatchNorm2d(m // 4)
        self.relu2 = nn.ReLU(inplace=False)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(m // 4, n, 1)
        self.norm3 = nn.BatchNorm2d(n)
        self.relu3 = nn.ReLU(inplace=False)

    def forward(self, x):
        double_size = (x.size(-2) * 2, x.size(-1) * 2)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.conv2(x, output_size=double_size)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class FinalBlock(nn.Module):
    def __init__(self, num_filters):
        super().__init__()

        self.conv1 = nn.ConvTranspose2d(
            num_filters, num_filters // 2, 3, stride=2, padding=1
        )
        self.relu1 = nn.ReLU(inplace=False)
        self.conv2 = nn.Conv2d(
            num_filters // 2, num_filters // 2, 3, padding=1
        )
        self.relu2 = nn.ReLU(inplace=False)
        self.conv3 = nn.Conv2d(num_filters // 2, 3, 1)

    def forward(self, inputs):
        double_size = (inputs.size(-2) * 2, inputs.size(-1) * 2)
        x = self.conv1(inputs, output_size=double_size)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        return x


class AEEncoder(nn.Module):
    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        requires_grad: bool = None,
        z_dim=16,
    ):
        super().__init__()
        resnet = torchvision.models.__dict__[arch](pretrained=pretrained)
        encoder = ResnetEncoderAE(resnet)
        if requires_grad is None:
            requires_grad = not pretrained
        for param in encoder.parameters():
            param.requires_grad = bool(requires_grad)

        self.z_dim = z_dim
        self.filters = encoder.filters
        bottleneck_reduce = nn.Conv2d(self.filters[3], z_dim, 1)
        # st()
        enc_modules = list(encoder.children())
        enc_modules += [bottleneck_reduce]
        # st()
        self.encoder = nn.Sequential(*enc_modules)

    def forward(self, x):
        z = self.encoder(x)
        return z


class VAEEncoder(AEEncoder):
    def __init__(self, **kwargs):
        z_dim = kwargs.pop("z_dim") * 2
        super().__init__(z_dim=z_dim, **kwargs)

    def forward(self, x, deterministic=None):
        x = self.encoder(x)
        bs, z_dim2, nf, _ = x.shape
        z_dim = z_dim2 // 2
        x = x.view(bs, -1)
        z_dim_ = x.shape[1]
        loc, log_scale = x[:, :z_dim_ // 2], x[:, z_dim_ // 2:]
        log_scale = torch.clamp(log_scale, LOG_SIG_MIN, LOG_SIG_MAX)
        scale = torch.exp(log_scale)
        deterministic = deterministic or self.training
        x = loc if deterministic else normal_sample(loc, scale)
        x = x.view(bs, z_dim, nf, nf)
        return x


class AEDecoder(nn.Module):
    def __init__(self, filters, z_dim=16):
        super().__init__()
        self.bottleneck_upscale = nn.Conv2d(z_dim, filters[3], 1)
        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.final = FinalBlock(filters[0])

    def forward(self, z):
        z_u = self.bottleneck_upscale(z)
        d4 = self.decoder4(z_u)
        d3 = self.decoder3(d4)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)
        x_ = self.final(d1)
        return x_


class ResnetAE(nn.Module):
    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        requires_grad: bool = None,
        z_dim=16,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = AEEncoder(
            arch=arch,
            pretrained=pretrained,
            requires_grad=requires_grad,
            z_dim=z_dim
        )
        self.decoder = AEDecoder(filters=self.encoder.filters, z_dim=z_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        return x_


class ResnetVAE(nn.Module):
    def __init__(
        self,
        arch: str = "resnet18",
        pretrained: bool = True,
        requires_grad: bool = None,
        z_dim=16,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.encoder = VAEEncoder(
            arch=arch,
            pretrained=pretrained,
            requires_grad=requires_grad,
            z_dim=z_dim
        )
        self.decoder = AEDecoder(filters=self.encoder.filters, z_dim=z_dim)

    def forward(self, x):
        z = self.encoder(x)
        x_ = self.decoder(z)
        return x_
