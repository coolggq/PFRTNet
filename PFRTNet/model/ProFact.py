import torch
import torch.nn as nn
from torch.nn import functional as F
from model1.EMCAM_1 import MSCAModule
from model1.MSDI_1 import MSDI
#from model1.HWD import HWD
#from pytorch_wavelets import DWTForward
from model1.HolisticAttention import HA
from model1.CSPM import CSPM
from model1.backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5
from mmcv.ops import CARAFEPack
import torch
import torch.nn as nn
from models.FreqFusion import FreqFusion
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Sequential(
        nn.AdaptiveAvgPool2d((64, 64)),  # 调整空间尺寸
        nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias))
class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

     
class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
    def forward(self, x):       
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class mlpHead_2(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """

    def __init__(self, in_channels=[32, 64, 160,512], embedding_dim=768, dropout_ratio=0.1):
        super(mlpHead_2, self).__init__()
        _, _, _, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        # self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        # self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.upsample = nn.ConvTranspose2d(32, 256, kernel_size=4, stride=2, padding=1)
        self.linear_fuse = ConvModule(
            c1=embedding_dim * 3,
            c2=embedding_dim,
            k=1,
        )
        self.conv_layer = nn.Conv2d(in_channels=224, out_channels=256, kernel_size=1)
        self.linear_pred = nn.Conv2d(embedding_dim, 1, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs,x1):  # (8,224,128,128)
        c4 = inputs
        # 输出经过卷积后的张量
        c4 = self.conv_layer(c4)
        # c4(8,512,64,64)
        ############## MLP decoder on C2-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        # _c4 = F.interpolate(_c4, size=(c4.size(2) * 2, c4.size(3) * 2), mode='bilinear', align_corners=False)

        # _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        # _c3 = F.interpolate(_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)
        #
        # _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        #
        # _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1))  # 256,64,64   256,64,64
        _c4 = _c4 + self.upsample(x1)
        x = self.dropout(_c4)
        x = self.linear_pred(x)

        return x # (8,1,128,128)


class mlpHead_3(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(mlpHead_3, self).__init__()
        _, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*3,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred    = nn.Conv2d(embedding_dim, 1, kernel_size=1)
        self.dropout        = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs):
        c2, c3, c4 = inputs

        ############## MLP decoder on C2-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c2.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c2.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2], dim=1)) #256,64,64   256,64,64


        x = self.dropout(_c)
        x = self.linear_pred(x)

        return x
    
class mlpHead_4(nn.Module):  # 将四个特征先拉到同一大小，在融合（通道叠加）
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
    def __init__(self, in_channels=[32, 64, 160,256], embedding_dim=768, dropout_ratio=0.1): # c1-4:32, 64, 160,256
        super(mlpHead_4, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels
        self.head = default_conv(1, 256, 3)
        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)
        self.upsample1 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)
        self.upsample2 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1)

        self.linear_fuse = ConvModule(
            c1=embedding_dim*4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.Conv2d(embedding_dim, 1, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout_ratio)
    
    def forward(self, inputs,input): # input(8,1,128,128)是网络输入
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape
        # 先融合：
        # msdi = MSDI([c1.shape[1], c2.shape[1], c3.shape[1], c4.shape[1]])  # 输入特征的通道数
        # inputs = [c1, c2, c3, c4]
        # _c = msdi(inputs)


        _c41 = self.linear_c4(c4).permute(0,2,1).reshape(n, -1, c4.shape[2], c4.shape[3]) #
        #_c4 = F.interpolate(_c41, size=c1.size()[2:], mode='bilinear', align_corners=False)
        #
        _c31 = self.linear_c3(c3).permute(0,2,1).reshape(n, -1, c3.shape[2], c3.shape[3])
       # _c3 = F.interpolate(_c31, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c21 = self.linear_c2(c2).permute(0,2,1).reshape(n, -1, c2.shape[2], c2.shape[3])
        #_c2 = F.interpolate(_c21, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0,2,1).reshape(n, -1, c1.shape[2], c1.shape[3])
        ###############################多尺度融合
        freq_fusion = FreqFusion(
            hr_channels=256,  # 高分辨率通道数
            lr_channels=256  # 低分辨率通道数
        ).cuda()
        output1 = freq_fusion(_c31, _c41)

        freq_fusion = FreqFusion(
            hr_channels=256,  # 高分辨率通道数
            lr_channels=256  # 低分辨率通道数
        ).cuda()
        output2 = freq_fusion(_c21,  output1)

        freq_fusion = FreqFusion(
            hr_channels=256,  # 高分辨率通道数
            lr_channels=256  # 低分辨率通道数
        ).cuda()
        output3 = freq_fusion(_c1, output2)  #


        # # 特征融合
        # msdi = MSDI([_c1.shape[1], _c2.shape[1], _c3.shape[1], _c4.shape[1]])  # 输入特征的通道数
        # _d = [_c1, _c2, _c3, _c4]
        # _c = msdi(_d)

        #_c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1)) # 在 通道上叠加

        x = self.dropout(output3)

        a = self.head(input)  # 卷积操作，改变维度和大小

        x = x+a  # 残差连接 a(8,256,64,64)   后面再试试FreFusion
        # 下面先上采样，在线性预测输出
        x1 = self.upsample1(x).sigmoid()   # 连续两个上采样 ，才到256*256
        x2 = self.upsample2(x1).sigmoid()

        x = self.linear_pred(x2)

        return x, output3 # 返回结果和融合后的特征, x(8,1,256,256);output3(8,256,64,64)

class CARAFEUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2, kernel_size=5, group=1):
        super().__init__()
        self.encoder = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.carafe = CARAFEPack(
            channels=out_channels,
            scale_factor=scale,
            up_kernel=kernel_size,
            group=group
        )

    def forward(self, x):
        x = self.encoder(x)          # 先做通道调整
        x = self.carafe(x)           # CARAFE 上采样
        return x


class ProFact(nn.Module):
    def __init__(self, args, ver = 'b0', pretrained = False):
        super(ProFact, self).__init__()
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[ver]
        self.backbone = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[ver](pretrained)
        self.embedding_dim = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[ver]

        self.fem1 = CSPM(dim=self.in_channels[0], kernel_size=3)
        self.fem2 = CSPM(dim=self.in_channels[1], kernel_size=3)
        self.fem3 = CSPM(dim=self.in_channels[2], kernel_size=3)
        self.fem4 = CSPM(dim=self.in_channels[3], kernel_size=3)
        # self.fem1 = MSCAModule(self.in_channels[0], self.in_channels[0])
        # self.fem2 = MSCAModule(self.in_channels[1], self.in_channels[1])
        # self.fem3 = MSCAModule(self.in_channels[2], self.in_channels[2])
        # self.fem4 = MSCAModule(self.in_channels[3], self.in_channels[3])

        self.upsample05 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1)  # 要上采样操作，可替换
        self.upsample = nn.ConvTranspose2d(1, 1, kernel_size=4, stride=2, padding=1)
        self.conv_layer = nn.Conv2d(in_channels=288, out_channels=224, kernel_size=3, stride=1, padding=1)

        self.HA = HA()

        self.decode_head_3 = mlpHead_3(self.in_channels, self.embedding_dim)
        self.decode_head_4 = mlpHead_4(self.in_channels, self.embedding_dim)
        self.decode_head_2 = mlpHead_2(self.in_channels, self.embedding_dim)

        # self.carafe_up1 = CARAFEUpsample(self.in_channels[0], self.in_channels[0], scale_factor=1,kernel_size=5, group=1)
        # self.carafe_up2 = CARAFEUpsample(self.in_channels[1], self.in_channels[1], scale_factor=2,kernel_size=5, group=1)
        # self.carafe_up3 = CARAFEUpsample(self.in_channels[2], self.in_channels[2], scale_factor=4,kernel_size=5, group=1)
        # self.carafe_up4 = CARAFEUpsample(self.in_channels[3], self.in_channels[3], scale_factor=8,kernel_size=5, group=1)

    def forward(self, inputs): # 输入：(8,1,128,128)
        x = self.backbone.forward(inputs)  # block1中的patch_embed1的stride改为了2
        x1, x2, x3, x4 = x 
        B = inputs.shape[0] # 获取批次大小

        # 四个不同尺度特征经过CSPM/EMCAF
        x1_1 = self.fem1(x1)
        x2_1 = self.fem2(x2)        
        x3_1 = self.fem3(x3)    
        x4_1 = self.fem4(x4)    
        y1 = x1_1, x2_1, x3_1, x4_1 # x11(8,32,64,64);x21(8,64,32,32);x31(8,160,16,16);x41(8,256,8,8);
        # 下面进行四个特征大小一致化和融合
        #out1 = self.carafe_up4(x4_1)


        attention_map1, _c = self.decode_head_4(y1, inputs)  # decode_head_4：解码阶段是通过双线性插值上采样，（msdi融合）特征cat融合， x(8,1,64,64);output3(8,256,64,64)
        # attention_map1 （8, 1,256, 256） # MLP之后，进行MSDI_1 特征融合
        # 现在是将原始特征作为残差相加，之后试试经过一层卷积后与它融合


        # _c (8,256,64,64)
        # 这里有个下采样操作：可替换为HWD小波下采样
        #hwd = HWD(in_ch=1, out_ch=1)
        #attention_map = hwd.forward(attention_map1) # HWD小波下采样
       # attention_map = self.upsample05(attention_map1).sigmoid() # 实际上是下采样操作


        #attention_map1_1 = self.upsample(attention_map1).sigmoid() # 实要上采样 attention_map(8,1,128,128)
        #attention_map = self.upsample(attention_map1_1).sigmoid() # 实要上采样 attention_map(8,1,256,256)
        #

        # x2_2 = self.HA(attention_map,x2_1) # 经过HAM模块并于x2_1相乘(选择性添加)


        #x2_2 = self.HA(self.upsample05(attention_map).sigmoid(), x2_1)   #x, 64, 64

        #----------------------------------#
        #   block5
        #----------------------------------#
        x3_2, H, W = self.backbone.patch_embed_5.forward(attention_map1) # x2_2  输入了_c就将patch_embed_5里接受的输入维度改为了1
        for i, blk in enumerate(self.backbone.block_5):
            x3_2 = blk.forward(x3_2, H, W)
        x3_2 = self.backbone.norm_5(x3_2)
        x3_2 = x3_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   # x3_2 (8,32,128,128)

        #----------------------------------#
        #   block6
        #----------------------------------#
        x4_2, H, W = self.backbone.patch_embed_6.forward(x3_2) # x3_2 (8,32,128,128)，正好patch_embed_6接受的输入维度为160
        for i, blk in enumerate(self.backbone.block_6):
            x4_2 = blk.forward(x4_2, H, W)
        x4_2 = self.backbone.norm_6(x4_2)
        x4_2 = x4_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()   # (8,64,64,64)

        # ----------------------------------#
        #   block7
        # ----------------------------------#
        # x5_2, H, W = self.backbone.patch_embed_7.forward(x4_2)  # x3_2 (8,32,128,128)，正好patch_embed_6接受的输入维度为160
        # for i, blk in enumerate(self.backbone.block_7):
        #     x5_2 = blk.forward(x5_2, H, W)
        # x5_2 = self.backbone.norm_7(x5_2)
        # x5_2 = x5_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (8,64,64,64)
        #
        # # ----------------------------------#
        # #   block8
        # # ----------------------------------#
        # x6_2, H, W = self.backbone.patch_embed_8.forward(x5_2)  # x3_2 (8,32,128,128)，正好patch_embed_6接受的输入维度为160
        # for i, blk in enumerate(self.backbone.block_8):
        #     x6_2 = blk.forward(x6_2, H, W)
        # x6_2 = self.backbone.norm_8(x6_2)
        # x6_2 = x6_2.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()  # (8,64,64,64)

        # concat  （x4_1，x3_2）
        # 获取 X3_2 的空间尺寸
        target_size = x3_2.shape[-2:]  # 取高度和宽度

        # 将 x4_1 上采样到 X3_2 的大小
        x4_upsampled = F.interpolate(x4, size=target_size, mode='bilinear', align_corners=False)

        # 在通道维度 (dim=1) 上进行拼接
        Z1 = torch.cat([x4_upsampled, x3_2], dim=1)
        # Z1 (8,288,64,64)
        # x4_2上采样 与x3_1 concat/相加,得到A4，x3_2 与 x4_1 Concat，得到Z1
        # concat  （x4_2，x3_1）
        # 获取 X3_1 的空间尺寸
        target_size = x4_2.shape[-2:]  # 取高度和宽度

        # 将 x4_1 上采样到 X3_2 的大小
        x3_upsampled = F.interpolate(x3, size=target_size, mode='bilinear', align_corners=False)

        # 在通道维度 (dim=1) 上进行拼接
        A4 = torch.cat([x3_upsampled, x4_2], dim=1)
         # A4 (8,224,32,32)
        Z1_1 = self.conv_layer(Z1)  # Z1_1(8,224,64,64)

        freq_fusion = FreqFusion(
            hr_channels=224,  # 高分辨率通道数
            lr_channels=224  # 低分辨率通道数
            # lr3_channels=160,  # 低分辨率通道数
            # lr4_channels=256  # 低分辨率通道数
        ).cuda()

        Zcon = freq_fusion(Z1_1, A4)   # (8,224,64,64)


        # ，Z1与A4， 进入MSDI模块上采样得到输出   此时两个特征大小不一致,所以要有下面的维度一致化操作
        # target_size = Z1.shape[-2:]  # 取高度和宽度
        # A4_upsampled = F.interpolate(A4, size=target_size, mode='bilinear', align_corners=False)
        #
        #
        #
        #
        # msdi = MSDI([Z1.shape[1], A4.shape[1]])  # 输入特征的通道数
        # A = [Z1, A4_upsampled]
        # Zcon = msdi(A)
         # Zcon (8,416,32,32)
        # MLP中上采样到64，再线性插值到512
        
        # x2_2 = self.fem2(x2_2)
        # x3_2 = self.fem3(x3_2)
        # x4_2 = self.fem4(x4_2)
        # y2 = x2_2, x3_2, x4_2

        detection_map = self.decode_head_2(Zcon,x1) # 上采样和cat操作

       # attention_map = F.interpolate(attention_map, size=inputs.shape[2:], mode='bilinear', align_corners=True)
        detection_map = F.interpolate(detection_map, size=(256,256), mode='bilinear', align_corners=True)

        return attention_map1, detection_map   # 返回的是粗粒度结果和细粒度结果，之后在计算叠加的损失


def build_model(args):
    return ProFact(args)