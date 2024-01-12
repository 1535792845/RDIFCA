import torch
import torch.nn as nn
import torch.nn.functional as F
class localRes(nn.Module):
    
    def __init__(self, inChannals, outChannals):
        
        super(localRes, self).__init__()
        
        self.conv1 = nn.Conv2d(inChannals, outChannals, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.PReLU()
        
        self.conv2 = nn.Conv2d(outChannals, outChannals, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.PReLU()
        
        self.conv1x1 = nn.Conv2d(outChannals , outChannals, kernel_size=1, stride=1, padding=0)
        self.conv1x2 = nn.Conv2d(outChannals * 2, outChannals , kernel_size=1, stride=1, padding=0)
        
        self.conv3 = nn.Conv2d(outChannals, outChannals, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.PReLU()
        
        self.conv4 = nn.Conv2d(outChannals, outChannals, kernel_size=5, stride=1, padding=2)
        self.relu4 = nn.PReLU()
        
        self.conv5 = nn.Conv2d(outChannals, outChannals, kernel_size=5, stride=1, padding=2)
        self.relu5 = nn.PReLU()
        
        self.conv6 = nn.Conv2d(outChannals, outChannals, kernel_size=5, stride=1, padding=2)
        self.relu6 = nn.PReLU()

    def forward(self, x):
    
        
        resudial = x
        
        

        out1 = self.conv1(x)
        out1 = self.relu1(out1)

        out2 = self.conv2(out1)
        out2 = self.relu2(out2)

        out_add = torch.add(out1, out2)
        out_at = self.conv1x1(out_add)
       
        
        out3 = self.conv3(out_at)
        out3 = self.relu3(out3)

        out_add = torch.add(out_at, out3)
        out_at = self.conv1x1(out_add)

        
        out4 = self.conv4(out_at)
        out4 = self.relu4(out4)
        
        out_add = torch.add(out_at, out4)
        out_at = self.conv1x1(out_add)
        
        out5 = self.conv5(out_at)
        out5 = self.relu5(out5)
        
        out_add = torch.add(out_at, out5)
        out_at = self.conv1x1(out_add)

        
        out6 = self.conv6(out_at)
        out6 = self.relu6(out6)
        
   
        out = resudial + out6
        
        

        return  out
        
