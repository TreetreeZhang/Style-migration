import torch
import torchvision.models as models

def load_vgg():
    vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
    if torch.cuda.is_available():
        vgg = vgg.cuda()
    return vgg
