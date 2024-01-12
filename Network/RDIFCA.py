from torch import nn
import torch
from  Network.ResBlock import ResBlock
class RDIFCA(nn.Module):
    def __init__(self, num_channels=3):
        super(RDIFCA, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=5, padding=2)

        self.resBlock = self._makeLayer_(ResBlock, 64, 64, 1)

        self.convPos1 = nn.Conv2d(64, 64 * 2 * 2, kernel_size=5, stride=1, padding=2)
        self.pixelShuffler1 = nn.PixelShuffle(2)
        
        self.conv2 = nn.Conv2d(64, 3, kernel_size=5,stride=1, padding=2 )
        self.relu = nn.PReLU()
       
      
    def _makeLayer_(self, block, inChannals, outChannals, blocks):
        
        layers = []
        layers.append(block(inChannals, outChannals))

        for i in range(1, blocks):
            layers.append(block(outChannals, outChannals))

        return nn.Sequential(*layers)

    def forward(self, x):
     
        x = self.conv1(x)
        x = self.relu(x)
        
        
        res = x
        
        x = self.resBlock(x)
    
        x = res + x
        
        
        x = self.convPos1(x)
        x = self.pixelShuffler1(x)
        
        x =self.conv2(x)
        x = self.relu(x)
        
        return x