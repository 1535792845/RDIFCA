import torch


def denormalize(img):
    return img.mul(255.0).clamp(0.0, 255.0)


def convert_rgb_to_y(img):
    return 16.0 + (65.738 * img[0] + 129.057 * img[1] + 25.046 * img[2])/256.0
    # return 0.299 * img[0] + 0.587 * img[1] + 0.114 * img[2]


def calculate_psnr(sr, hr, max_val=255.0):
    return 10.0*torch.log10((max_val**2)/((sr-hr)**2).mean())




class AverageMeter(object):
    def __init__(self):
        self.reset()
        # print("self.reset()",self.reset())
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        # print("val",val)
        # print("n",n)
        # print("sum",sum)