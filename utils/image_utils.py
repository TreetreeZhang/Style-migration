import torch
import torchvision.transforms as transforms
from PIL import Image

def load_image(image_path, size=None, scale=None):
    image = Image.open(image_path).convert('RGB')
    if size is not None:
        image = image.resize((size, size), Image.LANCZOS)
    elif scale is not None:
        image = image.resize((int(image.size[0] / scale), int(image.size[1] / scale)), Image.LANCZOS)
    loader = transforms.Compose([transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    if torch.cuda.is_available():
        image = image.cuda()
    return image

def save_image(input_tensor, output_path):
    unloader = transforms.ToPILImage()
    image = input_tensor.cpu().clone()
    image = image.squeeze(0)
    image = unloader(image)
    image.save(output_path)
