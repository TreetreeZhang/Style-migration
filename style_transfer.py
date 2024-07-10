# style_transfer.py
import torch
import torch.nn as nn
import torch.optim as optim
from models.content_loss import ContentLoss
from models.style_loss import StyleLoss
from models.vgg import load_vgg
from utils.image_utils import load_image, save_image

def style_transfer(content_image, style_image, num_steps=300, style_weight=1000000, content_weight=1):
    vgg = load_vgg()

    content_feature_layers = ['conv_4']
    style_feature_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    model = nn.Sequential()
    if torch.cuda.is_available():
        model = model.cuda()

    content_losses = []
    style_losses = []

    i = 1
    for layer in vgg.children():
        if isinstance(layer, nn.Conv2d):
            name = 'conv_' + str(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_' + str(i)
            layer = nn.ReLU(inplace=False)
            i += 1
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_' + str(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_' + str(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_feature_layers:
            target = model(content_image).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_" + str(i), content_loss)
            content_losses.append(content_loss)

        if name in style_feature_layers:
            target_feature = model(style_image).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_" + str(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    if torch.cuda.is_available():
        model = model.cuda()

    input_image = content_image.clone()
    optimizer = optim.LBFGS([input_image.requires_grad_()])

    run = [0]
    while run[0] <= num_steps:

        def closure():
            input_image.data.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_image)

            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("Step: {}, Style Loss: {:.4f}, Content Loss: {:.4f}".format(run[0], style_score.item(),
                                                                                    content_score.item()))

            return style_score + content_score

        optimizer.step(closure)

    input_image.data.clamp_(0, 1)
    return input_image
