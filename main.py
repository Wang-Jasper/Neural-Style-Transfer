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
import torch
from matplotlib import pyplot as plt
from torch.optim import LBFGS
import sys
import process
import time
from datetime import datetime

###################### IMAGE CONFIG ############################
content_image_path = "content/lake2.jpg"
style_image_path = "style/chinese2.png"
output_path = "output/iteration"

content_name = os.path.basename(content_image_path)
content_name = os.path.splitext(content_name)[0]
style_name = os.path.basename(style_image_path)
style_name = os.path.splitext(style_name)[0]

###################### PROCESS CONFIG ############################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimize_epoch = 10
optimize_step_perEpoch = 80
saving_freq = 100
img_size = 500

torch.autograd.set_detect_anomaly(True)

# Store for visualizing losses
total_losses, content_losses, style_losses, tv_losses = [], [], [], []

weights = {
    'content': 1e5,
    'style': 4e4,
    'tv': 1.0
}


def debug_tensor(t, name):
    """
    Function to check for errors when weights cause loss issues.
    :param t: Loss Tensor
    :param name: Name to print
    """
    n_nan = torch.isnan(t).sum().item()
    n_inf = torch.isinf(t).sum().item()
    if n_nan or n_inf:
        print(f"[✗] {name}: {n_nan=}  {n_inf=}")


def show_progress(iter_idx, total_steps, total_loss, content_loss, style_loss, tv):
    """
    Displays the progress bar.
    :param iter_idx: Current iteration
    :param total_steps: Total number of iterations
    :param total_loss: Total loss
    :param content_loss: Content loss
    :param style_loss: Style loss
    :param tv: Total variation loss
    """
    bar_length = 20
    filled = int(bar_length * (iter_idx + 1) / total_steps)
    bar = "█" * filled + "-" * (bar_length - filled)

    # \r moves cursor to the beginning of the line
    sys.stdout.write(
        f"\r[{bar}]"
        f"{iter_idx + 1:4d}/{total_steps}  "
        f"Total={total_loss:.4f}  "
        f"Content={content_loss:.4f}  "
        f"Style={style_loss:.4f}  "
        f"TV={tv:.4f}"
    )
    # Flush output directly
    sys.stdout.flush()


def gram_matrix(feature_map):
    """
    Computes the Gram matrix for the features of a single layer output.
    :param feature_map: Feature map
    :return: Gram matrix
    """
    b, c, h, w = feature_map.size()
    features = feature_map.view(c, h * w)  # Flatten operation (subsequent addition removes spatial record), feature map also has no positional record
    # With this shape, one row represents a feature, one column represents a position. After mm, all rows and columns are summed, and spatial information is lost.
    gram = torch.mm(features, features.t())
    # return gram.div_(c * h * w).unsqueeze(0)
    return gram.unsqueeze(0)


def total_variation(img, alpha=1.0, beta=1.0, eps=1e-6):
    """
    Computes the second-order total variation loss of an image tensor.
    :param img: Image tensor
    :param alpha: First-order weight
    :param beta: Second-order weight
    :param eps: Small constant to prevent division by zero
    :return: Total variation loss
    """
    # Compute first-order differences: along height and width directions
    dh = img[:, :, 1:, :] - img[:, :, :-1, :]
    dw = img[:, :, :, 1:] - img[:, :, :, :-1]
    # To match shapes, crop dh, dw to compute combined gradient
    dh1 = dh[:, :, :, :-1]
    dw1 = dw[:, :, :-1, :]

    # Sum of squares over channels and then square root, to get first-order difference magnitude for each pixel
    grad1 = torch.sqrt((dh1.pow(2) + dw1.pow(2)).sum(dim=1) + eps)  # (B, H-1, W-1)
    # tv1 = grad1.sum(dim=(1, 2)) / (h * w) # Sum first-order gradient magnitudes and normalize

    tv1 = grad1.sum(dim=(1, 2))  # Sum first-order gradient values

    # Second-order differences
    dxx = img[:, :, 2:, :] - 2 * img[:, :, 1:-1, :] + img[:, :, :-2, :]  # (B, C, H-2, W)
    dyy = img[:, :, :, 2:] - 2 * img[:, :, :, 1:-1] + img[:, :, :, :-2]  # (B, C, H, W-2)
    # Align shapes
    dxx2 = dxx[:, :, :, :-2]
    dyy2 = dyy[:, :, :-2, :]
    grad2 = torch.sqrt((dxx2.pow(2) + dyy2.pow(2)).sum(dim=1) + eps)  # (B, H-2, W-2)
    # tv2 = grad2.sum(dim=(1, 2)) / (h * w)  # (B,)

    tv2 = grad2.sum(dim=(1, 2))  # (B,)

    tgv2 = alpha * tv1 + beta * tv2  # (B,)
    return tgv2.mean()


def get_total_loss(forward, img, target_content, target_styles, content_idx, style_idx):
    """
    Calculates the total loss.
    :param forward: Forward propagation function
    :param img: Optimized image
    :param target_content: Content feature targets
    :param target_styles: Style feature targets
    :param content_idx: Content feature indices
    :param style_idx: Style feature indices
    :return: Total loss
    """
    current_features = forward(img)

    # Content loss, using Mean Squared Error, element-wise calculation for each feature map
    content_loss = 0
    for idx in content_idx:
        current_content = current_features[idx].squeeze(0)
        content_loss += torch.nn.functional.mse_loss(current_content, target_content[idx])
    # debug_tensor(content_loss, "content_loss")

    # Style loss
    style_loss = 0
    for idx in style_idx:
        current_feature = current_features[idx]
        # print(current_feature.shape)
        current_gram = gram_matrix(current_feature)
        style_loss += torch.nn.functional.mse_loss(current_gram, target_styles[idx])
    # debug_tensor(style_loss, "style_loss")

    # Total variation loss
    tv_loss = total_variation(img)
    # debug_tensor(tv_loss, "tv_loss")

    # Add weights
    total_loss = (weights['content'] * content_loss +
                  weights['style'] * style_loss +
                  weights['tv'] * tv_loss)
    # debug_tensor(total_loss, "total_loss")

    return total_loss, content_loss, style_loss, tv_loss


def style_transfer(content_image, style_image):
    """
    Performs style transfer.
    :param content_image: Content image Tensor
    :param style_image: Style image Tensor
    """
    # Copy content image as generated image
    generate_img = content_image.clone().detach().requires_grad_(True)

    # Load forward propagation, and content feature, style feature indices
    forward, content_layer_idx, style_layer_idx = process.load_feature()

    # Calculate content/style features
    content_features = forward(content_image)
    style_features = forward(style_image)

    # print(style_layer_idx)

    target_content = {idx: content_features[idx].squeeze(0) for idx in content_layer_idx}

    target_styles = {idx: gram_matrix(style_features[idx]) for idx in style_layer_idx}

    # L-BFGS optimizer, set max iterations and optimization tolerance
    optimizer = LBFGS([generate_img], max_iter=optimize_step_perEpoch, tolerance_grad=0, tolerance_change=0)

    iteration = [0]

    def optimize_step():
        """
        Single gradient descent optimization step.
        :return: Total loss
        """
        # Zero gradients before new gradient calculation
        optimizer.zero_grad()

        total_loss, content_loss, style_loss, tv_loss = get_total_loss(
            forward,
            generate_img,
            target_content,
            target_styles,
            content_idx=content_layer_idx,
            style_idx=style_layer_idx,
        )

        # Backpropagation to compute gradients
        total_loss.backward()
        c = (weights['content'] * content_loss).item()
        s = (weights['style'] * style_loss).item()
        t = (weights['tv'] * tv_loss).item()
        tot = total_loss.item()

        # Add to visualization records
        total_losses.append(tot)
        content_losses.append(c)
        style_losses.append(s)
        tv_losses.append(t)

        # Update progress bar, using list in this closure to modify external object
        idx = iteration[0]
        show_progress(idx, optimize_epoch * optimize_step_perEpoch, tot, c, s, t)
        # Save intermediate images
        if (idx + 1) % saving_freq == 0 or idx + 1 == optimize_epoch * optimize_step_perEpoch:
            process.save_img(
                generate_img,
                idx + 1,
                optimize_epoch * optimize_step_perEpoch,
                content_name,
                style_name,
                output_path
            )

        iteration[0] += 1

        return total_loss

    # Adapt to L-BFGS jumping out of local optima, prevent gradient explosion, train in stages
    for epoch in range(optimize_epoch):
        optimizer.step(optimize_step)


def plot_losses():
    """
    Plots the loss curves.
    """
    steps = list(range(1, len(total_losses) + 1))

    # Total Loss
    plt.figure()
    plt.plot(steps, total_losses)
    plt.title("Total Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.tight_layout()
    plt.show()

    # Content Loss
    plt.figure()
    plt.plot(steps, content_losses)
    plt.title("Content Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Content Loss")
    plt.tight_layout()
    plt.show()

    # Style Loss
    plt.figure()
    plt.plot(steps, style_losses)
    plt.title("Style Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Style Loss")
    plt.tight_layout()
    plt.show()

    # TV Loss
    plt.figure()
    plt.plot(steps, tv_losses)
    plt.title("TV Loss over Iterations")
    plt.xlabel("Iteration")
    plt.ylabel("TV Loss")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    content_img = process.preprocess_img(content_image_path, img_size)
    style_img = process.preprocess_img(style_image_path, img_size)

    os.makedirs(output_path, exist_ok=True)

    start_time = time.time()
    start_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("\n" + "=" * 50)
    print(f"    ✨ Style Transfer Started ⏱️[{start_str}]")
    print("=" * 50 + "\n")

    # Execute style transfer
    style_transfer(content_img, style_img)

    # Record end time and compute elapsed time
    end_time = time.time()
    end_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    elapsed_time = end_time - start_time
    print("\n" + "=" * 50)
    print(f"    ✅ Style Transfer Complete!")
    print(f"    ⏱️ Elapsed Time: {elapsed_time:.2f} seconds")
    print("=" * 50)

    plot_losses()