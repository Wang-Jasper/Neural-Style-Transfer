# MIT License
#
# Copyright (c) 2025 Jasper Wang
#
# Permission is granted to use, copy, modify, and distribute this software
# and its documentation for any purpose, provided this copyright notice
# and the permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
import os
import cv2
import torch
from torchvision import transforms
from PIL import Image
from torchvision import models
import numpy as np

# Standard VGG normalization
VGG_MEAN = [0.485, 0.456, 0.406]  # [0,1] range here, if image is 255, it needs to be scaled to [0,1] first
VGG_STD = [1, 1, 1]

IMG_FORMAT = '.jpg'  # File format
FINAL_IMG_NAME = 'final_result.jpg'  # Final result filename

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VGG = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).to(device).eval()


# print(VGG)


def preprocess_img(img_path, target_shape):
    """
    Preprocesses the image.
    :param img_path: Image path
    :param target_shape: Short side size for scaling
    :return:
    """
    preprocess = transforms.Compose([
        # Scale according to the short side
        transforms.Resize(target_shape, interpolation=transforms.InterpolationMode.LANCZOS),
        transforms.ToTensor(),
        transforms.Normalize(mean=VGG_MEAN, std=VGG_STD),
        transforms.Lambda(lambda x: x.mul(255)),
    ])

    img = Image.open(img_path).convert("RGB")
    img = preprocess(img).unsqueeze(0).to(device)  # VGG input requirement: (1, C, H, W)
    return img


def load_feature():
    # Download pre-trained VGG19, remove the classification layer
    full_features = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1, progress=True).features
    # print(full_features)

    # Use relu as style layers (emphasize cross-layer features & denoise), and conv as content layers (preserve structural information)
    layer_idx = [3, 8, 15, 21, 22, 29, 33]

    # Indices for content and style layers
    content_layer_idx = [3]
    # content_layer_idx = 2
    style_layer_idx = [0, 1, 2, 4, 5, 6]

    # Truncate model layers up to index 29
    max_idx = max(layer_idx)
    features = torch.nn.ModuleList(full_features[:max_idx + 1])

    # Disable gradient calculation, freeze all weights
    for p in features.parameters():
        p.requires_grad = False

    def forward(x):
        """
        Forward propagation that extracts only the required features.
        :param x: Image Tensor
        :return: Feature map outputs
        """
        activations = []
        for idx, layer in enumerate(features):
            x = layer(x)  # Extract current layer features
            if idx in layer_idx:
                activations.append(x)
                # If all required features are computed
                if len(activations) == len(layer_idx):
                    break
        return activations

    # Move to GPU, disable dropout
    features = features.to(device)
    for m in features:
        m.eval()

    return forward, content_layer_idx, style_layer_idx


IMAGENET_MEAN_255 = [123.68, 116.779, 103.939]


def save_img(generate_img, iteration, full_iteration, content_name, style_name, output_path):
    """
    Saves the image.
    :param generate_img: Generated image
    :param iteration: Iteration count
    :param full_iteration: Total iterations
    :param content_name: Name of the content image
    :param style_name: Name of the style image
    :param output_path: Output directory path
    :return:
    """
    # Convert tensor to NumPy array and adjust dimensions
    img_array = generate_img.squeeze(0).detach().cpu().numpy()
    img_array = img_array.transpose(1, 2, 0)  # CHW -> HWC

    # Denormalize and convert to uint8
    final_img = img_array + IMAGENET_MEAN_255
    final_img = np.clip(final_img, 0, 255).astype(np.uint8)

    # Generate filename
    filename = f"{content_name}_{style_name}_final{IMG_FORMAT}" if (
            iteration == full_iteration) else f"{content_name}_{style_name}_{iteration}{IMG_FORMAT}"

    # Save as BGR format
    cv2.imwrite(os.path.join(output_path, filename), final_img[:, :, ::-1])
