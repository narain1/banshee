import timm
from torch import nn
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp


class UperNetSwin(nn.Module):
    def __init__(self,
                 model_name="swinv2_tiny_window16_256",
                 img_size=256,
                 n_classes=1,
                 in_chans=3,
                 seg_channels=256,
                 upsampling=4,
                 pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(model_name,
                                          in_chans=in_chans,
                                          pretrained=pretrained,
                                          features_only=True)
        feature_channels = [o['num_chs'] for o in self.backbone.feature_info]
        self.PPN = smp.decoders.pspnet.decoder.PSPDecoder(feature_channels, out_channels=feature_channels[-1])
        self.FPN = smp.decoders.fpn.decoder.FPNDecoder(feature_channels, segmentation_channels=seg_channels)
        self.head = smp.base.heads.SegmentationHead(seg_channels, n_classes, upsampling=upsampling)


    def forward(self, x):
        input_size = (x.size()[2], x.size()[3])
        features = self.backbone(x)
        for idx in range(len(features)):
            features[idx] = features[idx].permute(0, 3, 1, 2)
        features[-1] = self.PPN(features[-1])
        x = self.head(self.FPN(*features))
        return x



