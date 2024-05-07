import torch
import torch.nn as nn
import torchvision.models as models

import segmentation_models_pytorch as smp

def get_model(num_classes):
    model = smp.UnetPlusPlus(
        encoder_name="efficientnet-b0",  # can be replaced with 'mobilenet_v2'
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes,
    )
    return model


# class UNet(nn.Module):
#     #model = UNet(n_class=3) Example of how to initialize the model using this class
#     def __init__(self, n_class):
#         super().__init__()
#         self.base_model = models.resnet18(pretrained=True)
#         self.base_layers = list(self.base_model.children())
        
#         self.layer1 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
#         self.layer2 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
#         self.layer3 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
#         self.layer4 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
#         self.layer5 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)

#         self.up5 = self.up(512, 256)
#         self.up4 = self.up(256, 128)
#         self.up3 = self.up(128, 64)
#         self.up2 = self.up(64, 64)
#         self.up1 = self.up(64, n_class)
        
#     def up(self, in_channels, out_channels):
#         return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

#     def forward(self, x):
#         # Down sampling
#         x1 = self.layer1(x)
#         x2 = self.layer2(x1)
#         x3 = self.layer3(x2)
#         x4 = self.layer4(x3)
#         x5 = self.layer5(x4)
        
#         # Up sampling and establish skip connections
#         x = self.up5(x5) + x4
#         x = self.up4(x) + x3
#         x = self.up3(x) + x2
#         x = self.up2(x) + x1
#         x = self.up1(x)

#         return x


