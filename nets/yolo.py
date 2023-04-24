from collections import OrderedDict

import torch
import torch.nn as nn

from nets.darknet import darknet53

class YoloBody(nn.Module):
    def __init__(self,anchors_mask,num_classes, pretrained=False):
        '''
        :param anchors_mask:
        2D list, represent the anchors mask
        [[6,7,8],[3,4,5],[0,1,2]]
        first list represent the different scale object
        second list represent the anchors for predicting that object
        '''
        super(YoloBody,self).__init__()
        '''
        get the man body of the yolo
        '''
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        out_filters = self.backbone.layers_out_filters # [64, 128, 256, 512, 1024]
        '''
        len(anchors_mask[0]): nums of anchors for one object
        num_classes+5 : 4 for box, 1 for confidence, num_classes for class
        '''
        self.last_layer0 = make_last_layers([512,1024],out_filters[-1],len(anchors_mask[0])*(num_classes+5))
        self.last_layer1_conv = conv2d(512,256,1)
        self.last_layer1_upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.last_layer1=make_last_layers([256,512],out_filters[-2]+256,len(anchors_mask[1])*(num_classes+5))

        self.last_layer2_conv = conv2d(256,128,1)
        self.last_layer2_upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.last_layer2=make_last_layers([128,256],out_filters[-3]+128,len(anchors_mask[2])*(num_classes+5))

    def forward(self,x):
        '''
        get the backbone input
        '''
        x2,x1,x0=self.backbone(x) # x0:13x13x1024 x1:26x26x512 x2:52x52x256

        out0_branch=self.last_layer0[:5](x0) # 1 x 512 x 13 x 13
        out0=self.last_layer0[5:](out0_branch) # 1 x 75 x 13 x 13

        x1_in = self.last_layer1_conv(out0_branch) # 1 x 256 x 13 x 13
        x1_in = self.last_layer1_upsample(x1_in) # 1 x 256 x 26 x 26
        x1_in = torch.cat([x1_in,x1],dim=1) # 1 x 768 x 26 x 26

        out1_branch=self.last_layer1[:5](x1_in) # 1 x 256 x 26 x 26
        out1=self.last_layer1[5:](out1_branch) # 1 x 75 x 26 x 26

        x2_in = self.last_layer2_conv(out1_branch) # 1 x 128 x 26 x 26
        x2_in = self.last_layer2_upsample(x2_in) # 1 x 128 x 52 x 52
        x2_in = torch.cat([x2_in,x2],dim=1) # 1 x 384 x 52 x 52

        out2=self.last_layer2(x2_in) # 1 x 75 x 52 x 52

        return out0,out1,out2

'''
with 5 extraction conv layers and 2 prediction conv layers
'''
def make_last_layers(filters_list,in_filter,out_filter):
    m=nn.Sequential(
        conv2d(in_filter,filters_list[0],1),
        conv2d(filters_list[0],filters_list[1],3),
        conv2d(filters_list[1],filters_list[0],1),
        conv2d(filters_list[0],filters_list[1],3),
        conv2d(filters_list[1],filters_list[0],1),
        conv2d(filters_list[0],filters_list[1],3),
        nn.Conv2d(filters_list[1],out_filter,kernel_size=1,stride=1,padding=0,bias=True)
    )
    return m


def conv2d(in_filter,out_filter,kernal_size):
    padding=(kernal_size-1) // 2 if kernal_size else 0
    layers=[
        ("conv",nn.Conv2d(in_filter,out_filter,kernel_size=kernal_size,stride=1,padding=padding,bias=False)),
        ("bn",nn.BatchNorm2d(out_filter)),
        ("relu",nn.LeakyReLU(0.1))
    ]
    return nn.Sequential(OrderedDict(layers))


if __name__=="__main__":
    anchors_mask=[[6,7,8],[3,4,5],[0,1,2]]
    num_classes=20
    model=YoloBody(anchors_mask,num_classes)

    # Generate a random input tensor
    batch_size = 1
    channels = 3
    height = 416
    width = 416
    input_tensor = torch.rand(batch_size, channels, height, width)

    # Compute the output tensor
    out0,out1,out2 = model(input_tensor)

    # Print the shape of the output tensor
    print(out0.shape)
    print(out1.shape)
    print(out2.shape)