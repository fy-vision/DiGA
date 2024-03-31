# ------------------------------------------------------------------------------
# Written by Ahmet Faruk Tuna (ahmet.faruk.tuna@huawei.com, @author: a84167753)
# ------

import torch
import torch.nn as nn
import torch.nn.functional as F

relu_inplace = True
BatchNorm2d = BatchNorm2d_class = nn.BatchNorm2d
    
class ORegionModule(nn.Module):
    """
        Build object region representations. Pixel represenetations are 
        weighted by initial soft object predictions.
        Inputs: soft_object_regions, pixel_representations
        Output: object region representations
    """
    def __init__(self, num_classes=0):
        super(ORegionModule, self).__init__()
        
        self.num_classe = num_classes
        
    def forward(self, pixel_representations, soft_object_regions):
        
        # change shape of the soft obejct regions: 
        # batch x num_classes x h x w -> batch x num_classes x hw
        batch, c, h, w = soft_object_regions.size()
        soft_object_regions = soft_object_regions.view(batch, c, -1)
        
        # change shape of the pixel representations:
        # batch x channel x  h x w -> batch x hw x channel 
        pix_rep_c = pixel_representations.size(1)
        pixel_representations = pixel_representations.view(batch, pix_rep_c, -1)
        pixel_representations = pixel_representations.permute(0,2,1)
        
        # normalize soft object region predictions
        soft_object_regions = F.softmax(soft_object_regions, dim=2)
        
        # create object region representations: batch x channel x num_classes x 1
        obj_region_rep = torch.matmul(soft_object_regions, pixel_representations)
        obj_region_rep = obj_region_rep.permute(0,2,1).unsqueeze(3)
        
        return obj_region_rep
        
        
        
class PixelRegionRelationModule(nn.Module):
    """
       Build Pixel Region Relation (Pixel/Region similarities see paper)
       Input: Pixel_Representations (batch x channels x h x w)
              Object_Region_Representations (batch x channel x classes x 1)
       Output: Pixel_Region_Relation (batch x hw x classes)
    """
    def __init__(self, in_channels, key_channels, bn_type=None):
        super(PixelRegionRelationModule, self).__init__()
        
        self.in_channels = in_channels
        self.key_channels = key_channels
        
        self.pixel_rep = self._apply_conv(self.in_channels, self.key_channels)
        self.obj_reg_rep = self._apply_conv(self.in_channels, self.key_channels)
        
    
    def _apply_conv(self, in_channels, out_channels):
        
        conv = nn.Sequential(nn.Conv2d(in_channels, out_channels,
                                       kernel_size=1, stride=1, padding=0,
                                       bias=False),
                             BatchNorm2d(out_channels), nn.ReLU(),
                             nn.Conv2d(out_channels, out_channels,
                                       kernel_size=1, stride=1, padding=0,
                                       bias=False),
                             BatchNorm2d(out_channels), nn.ReLU())
        return conv
    
    def forward(self, pixel_rep, object_reg_rep):
        
        batch, c, h, w = pixel_rep.size()
        
        # create query from pixel representations
        query = self.pixel_rep(pixel_rep) # dim: batch x key_channels x h x w
        query = query.view(batch, self.key_channels, -1) # dim: batch x key_channels x hw
        query = query.permute(0, 2, 1)  # dim: batch x hw x key_channels
        
        # create key from the object regions
        key = self.obj_reg_rep(object_reg_rep) #dim: batch x key_channels x classes x 1
        key = key.view(batch, self.key_channels, -1)  # dim: batch x key_channels x classes
        
        # create pixel region relations: similarity map -> weights of the values
        pixel_region_relation = torch.matmul(query, key) # dim: batch x hw x classes
        pixel_region_relation = (self.key_channels**-.5) * pixel_region_relation # to avoid small gradients of softmax (see Self attention paper) 
        pixel_region_relation = F.softmax(pixel_region_relation, dim=-1)
        
        return pixel_region_relation
        
        
        
        



class OCRNet(nn.Module):
    """ Implements the OCRNet for the semantic segmentation.
        Input: feature maps from the backbone network.
        Output: soft_object_regions and 
        main pixel wise segmentation (without softmax)"""
    
    def __init__(self, config, **kwargs):
        super(OCRNet, self).__init__()
        self.ocrnet_config = config['OCRNET_MODEL']
        self.in_channels = self.ocrnet_config['RAW_IN_CHANNELS']
        self.num_classes = self.ocrnet_config['NUM_CLASSES']
        self.pix_rep_channels = self.ocrnet_config['PIXEL_REP_CHANNELS']
        self.key_channels = self.ocrnet_config['KEY_CHANNELS']

        
        # adjusts the channel dimension of raw input feature maps from the backbone
        self.pixel_representations = nn.Sequential(
            nn.Conv2d(self.in_channels, self.pix_rep_channels,
                      kernel_size=3, stride=1, padding=1),
            BatchNorm2d(self.pix_rep_channels),
            nn.ReLU(inplace=relu_inplace))
        
        # generates soft object regions (without softmax)
        self.soft_object_regions = self._generate_soft_object_regions(
            self.in_channels, self.num_classes)
        
        # builds object region representations
        self.obj_region_representations = ORegionModule(self.num_classes) 
        
        # builds pixel region relations/representations
        self.pixel_region_relations = PixelRegionRelationModule(
            self.pix_rep_channels, self.key_channels)
        
        # generate values (apply conv to object region representations to adjust the channel dimension)
        self.value = self._ch_down_up(self.pix_rep_channels, self.key_channels)
        
        # upsample the channel dimension of ocr representations
        self.ocr_up = self._ch_down_up(self.key_channels, self.pix_rep_channels)
        
        # create final augmented representation from the concatenated reps
        self.augmented_rep = self._conv_bn_drop(2*self.pix_rep_channels,
                                                self.pix_rep_channels, 0.05)
        
        # generate final pixel-wise segmentation predictions
        self.segmentation_classes = nn.Sequential(
            nn.Conv2d(self.pix_rep_channels, self.num_classes, 
                      kernel_size=1, stride=1, padding=0, bias=True))

        # initialize weights
        self.init_weights()
    
        
    def _ch_down_up(self, in_channels, out_channels):
        
        down_up = nn.Sequential(nn.Conv2d(in_channels, out_channels, 
                                       kernel_size=1, stride=1, padding=0, 
                                       bias=False),
                             BatchNorm2d(out_channels), nn.ReLU())
        
        return down_up
        

    def _conv_bn_drop(self, in_channels, out_channels, dropout):
        
        conv_bn_drop = nn.Sequential(nn.Conv2d(
            in_channels, out_channels, kernel_size=1,
            stride=1, padding=0, bias=False),
            BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout2d(dropout))
        
        return conv_bn_drop
        
        
        
    def _generate_soft_object_regions(self, in_channels, out_channels):
        
        regions = nn.Sequential(nn.Conv2d(in_channels, in_channels, 
                                       kernel_size=1, stride=1, padding=0),
                             BatchNorm2d(in_channels),
                             nn.ReLU(inplace=relu_inplace),
                             nn.Conv2d(in_channels, out_channels,
                                       kernel_size=1, stride=1, 
                                       padding=0, bias=True))
        return regions
    

    def init_weights(self):
        """Initializes parameters  """

        for name, m in self.named_modules():
            # if isinstance(m, nn.Conv2d):
            #     nn.init.normal_(m.weight, std=0.001)
            if isinstance(m, BatchNorm2d_class):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        print('OCRNET Initialized')


    def forward(self, raw_backbone_feats):
        # raw_backbone_feats: batch x channel1 x h x w
        
        
        # create soft object regions from raw input: batch x classes x h x w 
        soft_obj_reg = self.soft_object_regions(raw_backbone_feats)

        
        # apply conv layers to raw input and create 
        # pixel representation feature maps with: batch x channel2 x h x w
        pixel_rep = self.pixel_representations(raw_backbone_feats)

        
        # create object region represenations: batch x channel2 x classes x 1
        obj_reg_rep = self.obj_region_representations(
            pixel_rep, soft_obj_reg)

        
        # pixel region relation (normalized similarity map): # dim: batch x hw x classes
        pixel_reg_rel = self.pixel_region_relations(pixel_rep, 
                                                    obj_reg_rep)

        # create values (see self attentions paper) from object_region_representations 
        # by applying conv
        value = self.value(obj_reg_rep) # dim: batch x key_channels x classes x 1 
        batch = value.size()[0]
        value = value.view(batch, self.key_channels, -1)
        value = value.permute(0, 2, 1) # dim: batch x classes x key_channels

        
        ocr_rep = torch.matmul(pixel_reg_rel, value) # dim: batch x hw x key_channels
        ocr_rep = ocr_rep.permute(0, 2, 1).contiguous()
        ocr_rep = ocr_rep.view(
            batch, self.key_channels, *raw_backbone_feats.size()[2:])
        ocr_rep = self.ocr_up(ocr_rep) # dim: batch x channel2 x h x w
        
        # concatenate pixel_representations with new ocr_representations
        ocr_rep = torch.cat([ocr_rep, pixel_rep], dim=1)
        
        # apply conv to concatenated reprsentations to build final augmented feature maps
        augmented_feats = self.augmented_rep(ocr_rep) 
        
        # pixel-wise segmentation prediction
        seg_classes = self.segmentation_classes(augmented_feats)

        
        return soft_obj_reg, seg_classes, augmented_feats
        
        
        
        
        
        