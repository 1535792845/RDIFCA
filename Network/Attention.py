import torch
import torch.nn as nn
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1)
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
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class cbam_block(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)
      
        self.conv1 = nn.Conv2d(64 *2 , 64, kernel_size = 1, padding=0)
        self.conv2 = nn.Conv2d(64, 64, kernel_size = 1, padding=0)
        
    def forward(self, x):
        resudial = x
    
        x1 = x * self.channelattention(x)
        
        x2 = x * self.spatialattention(x)
      
        
      
        x = torch.cat((x1,x2),1)

        x = self.conv1(x)
        x = self.conv2(x)
        
        x = resudial + x
       
        return x



























# import torch
# import torch.nn as nn

# class ChannelAttention(nn.Module):
#     def __init__(self, in_channels, ratio=16):
#         super(ChannelAttention, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Sequential(
#             nn.Linear(in_channels, in_channels // ratio),
#             nn.ReLU(inplace=True),
#             nn.Linear(in_channels // ratio, in_channels)
#         )

#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c)
#         y = self.fc(y).view(b, c, 1, 1)
#         return y.sigmoid()

# class SpatialAttention(nn.Module):
#     def __init__(self, kernel_size=7):
#         super(SpatialAttention, self).__init__()
#         padding = kernel_size // 2
#         self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)

#     def forward(self, x):
#         avg_mask = torch.mean(x, dim=1, keepdim=True)
#         max_mask, _ = torch.max(x, dim=1, keepdim=True)
#         mask = torch.cat([avg_mask, max_mask], dim=1)
#         mask = self.conv(mask).sigmoid()
#         return  mask

# class cbam_block(nn.Module):
#     def __init__(self, channel, ratio=16, kernel_size=7):
#         super(cbam_block, self).__init__()
#         self.channelattention = ChannelAttention(channel, ratio=ratio)
#         self.spatialattention = SpatialAttention()
#         # self.conv1 = nn.Conv2d(64 *2, 64*2 , kernel_size = 1, padding=0)
#         # self.conv2 = nn.Conv2d(64 *2 , 64, kernel_size = 1, padding=0)
#         self.conv3 = nn.Conv2d(64 , 64, kernel_size = 1, padding=0)
#         # self.conv = nn.Sequential(
#         #     nn.Conv2d(channel, channel, kernel_size=3, padding=1),
#         #     nn.ReLU(inplace=True),
#         #     nn.Conv2d(channel, channel, kernel_size=3, padding=1)
#         # )
#     def forward(self, x):
    
        
#         resudial = x
#         x1 = x * self.channelattention(x)
#         # print("11111111111111111",x1.shape)
#         # print("xxxx",x.shape)  #[1, 3, 279, 285]
#         # x2 = x1 * self.spatialattention(x1)
#         # print("xxxxxxxxxxxxx22", x2.shape)
#         # print("2222222222222222",x1.shape)
#         x = torch.add(x1,x2)
#         x = self.conv3(x)
#         #x = x1 + x2
#         # print("xxxxxxxxxxxxx33",x.shape)
#         #特征融合
#         #x = torch.cat((x1,x2),1)
        
#         #x = self.conv1(x)
#         #x = self.conv2(x)
#         # x = self.conv3(x)
#         # print("2222222222222222",x.shape)
        
#         x = resudial + x
#         #x = self.conv1(x)
#         #x = resudial + x
#         return x