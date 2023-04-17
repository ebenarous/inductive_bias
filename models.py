import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.resnet import Bottleneck, ResNet

import torchvision.transforms as transforms
#from torchsummary import summary
from torchinfo import summary

import timm
from timm.models import vision_transformer as vits


class CNN():

    def load_resnet50(num_classes, pretrained):
        """
        Loads the ResNet-50 architecture in PyTorch.
        """
        # resnet50 = models.resnet50(pretrained=pretrained)
        resnet50 = models.resnet50(weights='ResNet50_Weights.IMAGENET1K_V1')
        num_ftrs = resnet50.fc.in_features
        resnet50.fc = nn.Linear(num_ftrs, num_classes)  # Replace last fully connected layer

        return resnet50

class seg():

    def load_resnet50_seg(num_classes, pretrained):
        """
        Loads the ResNet-50 architecture in PyTorch.
        """
        resnet50 = models.resnet50(pretrained=pretrained)
        summary(resnet50, input_size=(4, 3, 256, 512))
        
        resnet50.avgpool = nn.Identity()
        resnet50.fc = nn.Identity()

        summary(resnet50, input_size=(4, 3, 256, 512))


        num_ftrs = resnet50.fc.in_features
        print(resnet50.layer4)
        resnet50.fc = nn.Conv2d(num_ftrs, num_classes, kernel_size=1, stride=1)  # Replace last fully connected layer
        summary(resnet50, input_size=(4, 3, 256, 512))

        return resnet50

    '''class ResNet50Seg(models.resnet.ResNet):
        def __init__(self, num_classes=1, pretrained=True):
            super(ResNet50Seg, self).__init__(block=models.resnet.Bottleneck, layers=[3, 4, 6, 3], num_classes=1000)
            if pretrained:
                self.load_state_dict(models.resnet50(pretrained=pretrained).state_dict())

            # Modify the last avgpool and fc layers for segmentation task
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Conv2d(2048, num_classes, kernel_size=1, bias=True)

        def forward(self, x):
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = self.fc(x)

            return x
    '''

    class ResNet50Seg(nn.Module):
        def __init__(self, num_classes=30):
            super(ResNet50Seg, self).__init__()
            resnet = models.resnet50(pretrained=True)
            layers = list(resnet.children())[:-2]
            self.features = nn.Sequential(*layers)
            self.avgpool = nn.Conv2d(2048, 1, kernel_size=(2,), stride=(2,))
            self.fc = nn.Conv2d(1024, num_classes, kernel_size=1, stride=1)

        def forward(self, x):
            x = self.features(x)
            print(x.shape)
            x = nn.Upsample(scale_factor=128)(x)
            print('there', x.shape)
            x = self.avgpool(x)
            print('here', x.shape)
            x = self.fc(x)
            x = nn.functional.interpolate(x, size=(1024, 2048), mode='bilinear', align_corners=True)
            return x

class transformer():

    def load_vit(model_name, num_classes, pretrained):
        """
        Loads the Vision Transformer (ViT) architecture in PyTorch.
        """
        if model_name == 'vit_base_patch16_224':
            vit = vits.vit_base_patch16_224(pretrained=pretrained, num_classes=num_classes)
        
        elif model_name == 'vit_large_patch16_224':
            vit = vits.vit_large_patch16_224(pretrained=pretrained, num_classes=num_classes)
        
        else:
            raise ValueError('Unsupported ViT model name.')
        
        return vit

    def load_deit(model_name, num_classes, pretrained):
        model = timm.create_model('deit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        return model
