import torch
import torch.nn as nn
import torch.nn.functional as F
from  Network.Attention  import cbam_block
from  Network.LocalRes import localRes
class ResBlock(nn.Module):
    
    def __init__(self, inChannals, outChannals):
        
        super(ResBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.PReLU()
        
        self.cbam_block = cbam_block(outChannals)
        
        self.localRes = self._makeLayer_(localRes, 64, 64, 1)
        
     
        self.conv1x1 = nn.Conv2d(outChannals * 7 , outChannals * 7, kernel_size=1, stride=1, padding=0)
        self.conv1x2 = nn.Conv2d(outChannals * 7 , outChannals , kernel_size=1, stride=1, padding=0)
        
        
    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
      
    
        x = self.conv1(x)
        x = self.relu1(x)

        x_1 = self.localRes(x)
        xca1 = x_1 + x
        x1 =  self.cbam_block(xca1)
        
        x_2 = self.localRes(x_1)
        xca2 = x_2 + xca1
        x2 =  self.cbam_block(xca2)
        
        x_3 = self.localRes(x_2)
        xca3 = x_3 + xca2
        x3 =  self.cbam_block(xca3)
        
        x_4 = self.localRes(x_3)
        xca4 = x_4 + xca3
        x4 =  self.cbam_block(xca4)
        
        x_5 = self.localRes(x_4)
        xca5 = x_5 + xca4
        x5 =  self.cbam_block(xca5)
        
        x_6 = self.localRes(x_5)
        xca6 = x_6 + xca5
        x6 =  self.cbam_block(xca6)
        
        x_7 = self.localRes(x_6)
        xca7 = x_7 + xca6
        x7 =  self.cbam_block(xca7)
       
        x = torch.cat((x1, x2,x3,x4,x5,x6,x7), 1)
        x = self.conv1x1(x)
        x = self.conv1x2(x)
        
       
  
        return x
