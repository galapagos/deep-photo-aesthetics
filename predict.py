import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models, transforms

import cv2
import numpy as np
import matplotlib.pyplot as plt

from model.resnet_FT import ResNetGAPFeatures as Net

from typing import Dict, Tuple

attr_keys = ['BalacingElements', 'ColorHarmony', 'Content', 'DoF', 'Light', 'MotionBlur', 'Object', 'RuleOfThirds', 'VividColor']
non_neg_attr_keys = ['Repetition', 'Symmetry', 'score']
ALL_KEYS = attr_keys + non_neg_attr_keys
USED_KEYS = ["ColorHarmony", "Content", "DoF", "Object", "VividColor", "score"]

def extract_pooled_features(inp, net: Net):
    _ = net(inp)
    pooled_features = [features.feature_maps for features in net.all_features] 
    return pooled_features

def downsample_pooled_features(pooled_features) -> torch.Tensor:
    dim_reduced_features = []
    for pooled_feature in pooled_features:
        if pooled_feature.size()[-1] == 75:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size=(7, 7)))
        elif pooled_feature.size()[-1] == 38:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size = (4, 4), padding=1))
        elif pooled_feature.size()[-1] == 19:
            dim_reduced_features.append(F.avg_pool2d(pooled_feature, kernel_size = (2, 2), padding=1))
        elif pooled_feature.size()[-1] == 10:
            dim_reduced_features.append(pooled_feature)
    dim_reduced_features = torch.cat(dim_reduced_features, dim=1).squeeze()
    return dim_reduced_features

def scale(image, low=-1, high=1):
    im_max = np.max(image)
    im_min = np.min(image)
    return (high - low) * (image - np.min(image))/(im_max - im_min) + low 

def extract_heatmap(features: torch.Tensor, weights: torch.Tensor, w: int, h: int) -> np.ndarray:
    cam = np.zeros((10, 10), dtype=np.float32) 
    temp = weights.view(-1, 1, 1) * features
    summed_temp = torch.sum(temp, dim=0).data.cpu().numpy()
    cam = cam + summed_temp
    cam = cv2.resize(cam, (w, h))
    cam = scale(cam)
    return cam 

def extract_prediction(inp, net: Net) -> Dict[str, float]:
    d: Dict[str, float] = {}
    net.eval()
    output = net(inp)
    for i, key in enumerate(ALL_KEYS):
        d[key] = output[:, i].squeeze().item()
    return d

def transform(image: np.ndarray):
    transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize([299, 299]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )])
    return transform(image)

def draw_colormap(
        downsampled_pooled_features: torch.Tensor,
        weights: Dict[str, torch.Tensor],
        predicted_values: Dict[str, float],
        img: np.ndarray) -> bytes:
    h, w, _ = img.shape
    fig, ax = plt.subplots(figsize=(20, 20))
    y, x = np.mgrid[0:h, 0:w]
    fig.subplots_adjust(right=1,top=1,hspace=0.5,wspace=0.5)
    for i, k in enumerate(USED_KEYS): 
        heatmap = extract_heatmap(downsampled_pooled_features, weights[k], w=w, h=h)
        ax = fig.add_subplot(2, 4, i+1)
        ax.imshow(img, cmap='gray')
        cb = ax.contourf(x, y, heatmap, cmap='jet', alpha=0.35)
        ax.set_title(f"Attribute: {k}\nPredicted Score: {round(predicted_values[k], 2)}")
    ax = fig.add_subplot(2, 4, 7)
    ax.imshow(img)
    plt.colorbar(cb)
    plt.tight_layout()

    BIO = BytesIO()
    plt.savefig(BIO)

    return BIO.getvalue()

def get_network(checkpoint_path: str, use_cuda: bool) -> Net:
    if use_cuda:
        resnet = models.resnet50(pretrained=True).cuda()
        net = Net(resnet, n_features=12).cuda()
        net.load_state_dict(torch.load(checkpoint_path))
    else:
        resnet = models.resnet50(pretrained=True)
        net = Net(resnet, n_features=12)        
        net.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

    return net

def get_weight(net: Net) -> Dict[str, torch.Tensor]:
    return {k: net.attribute_weights.weight[i, :] for i, k in enumerate(ALL_KEYS)}

def extract_inp(img_bin: bytes, use_cuda: bool):
    image_default: np.ndarray = cv2.cvtColor(cv2.imdecode(np.frombuffer(img_bin, dtype=np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_RGB2BGR)
    image = transform(image_default)

    if use_cuda:
        inp = Variable(image).unsqueeze(0).cuda()
    else:
        inp = Variable(image).unsqueeze(0)

    return inp, image_default

def is_cuda_available() -> bool:
    return torch.cuda.is_available()

def predict(image_binary: bytes, checkpoint_path: str) -> Tuple[bytes, Dict[str, float]]:

    use_cuda = is_cuda_available()
    net = get_network(checkpoint_path, use_cuda)
    weights = get_weight(net)
    inp, image_default = extract_inp(image_binary, use_cuda)
    predicted_values = extract_prediction(inp, net)
    pooled_features = extract_pooled_features(inp, net)
    downsampled_pooled_features = downsample_pooled_features(pooled_features)

    colormap_binary = draw_colormap(downsampled_pooled_features, weights, predicted_values, image_default)

    return colormap_binary, predicted_values

if __name__ == "__main__":
    image_path = "images/farm1_255_19452343093_8ee7e5e375_b.jpg"
    checkpoint_path = "checkpoint/002/epoch_5.loss_0.39914791321946214.pth"

    from PIL import Image
    from io import BytesIO
    BIO = BytesIO()
    image = Image.open(image_path)
    image.save(BIO, "JPEG")

    colormap_binary, predicted_values = predict(BIO.getvalue(), checkpoint_path)
