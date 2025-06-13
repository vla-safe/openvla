from typing import Optional

import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import torch
from torch.nn import functional as F
from scipy.spatial.transform import Rotation as R

from experiments.robot.openvla_utils import preprocess_image


# TODO: a better way to determine the max value attention for visualization
# IMAGE_ATTENTION_V_MAX = 0.005
# TEXT_ATTENTION_V_MAX = 0.05
IMAGE_ATTENTION_V_MAX = None
TEXT_ATTENTION_V_MAX = None

LOG_VALUES_V_MAX = {
    # "attn_image_entropy_mean": 5.6,
    # "mean_token_entropy": 1.5
}
LOG_VALUES_V_MIN = {
    # "attn_image_entropy_mean": 5.3,
    # "mean_token_entropy": 0
}


def plot_img_attn(
    image: np.ndarray,
    attn: torch.Tensor,
    ax: plt.Axes,
):
    '''
    Plot the image together with the attention map; add a colorbar based on the attention map values
    
    image: (H, W, 3) numpy array, uint8 dtype
    attn: (H, W) torch FloatTensor
    ax: matplotlib Axes object
    '''
    H, W = image.shape[:2]
    attn_resized = F.interpolate(
        attn.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="nearest",
    ).squeeze().numpy()
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    ax.imshow(image, cmap="gray")
    attn_plot = ax.imshow(attn_resized, alpha=0.5, cmap="turbo", vmin=0, vmax=IMAGE_ATTENTION_V_MAX)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(attn_plot, cax=cax)

    # plt.colorbar(attn_plot, ax=ax)
    ax.axis("off")
    
    return ax
    

def plot_var(
    imgs: list[np.ndarray], # The robot observation
    logs: dict, # The logs dictionary
    plot_keys: list[str] = ["pos_var", "rot_var"], # keys to the logs to plot
    imgs_title: Optional[list[str]] = None, # The title of the images
    sup_title: Optional[str] = None, # The title of the plot
    attn_img: Optional[torch.Tensor] = None, # The attention map
    center_crop: bool = True, # Whether to center crop the image
    attn_text: Optional[torch.Tensor] = None, # the attention to the input text tokens
    input_text_tokens: Optional[list[str]] = None, # the input text tokens
) -> np.ndarray:
    assert isinstance(imgs, list), f"Expected `imgs` to be a list, got {type(imgs)}"
    if imgs_title is not None: 
        assert len(imgs) == len(imgs_title), f"Expected `imgs` and `imgs_title` to have the same length, got {len(imgs)} and {len(imgs_title)}"
    if attn_text is not None:
        assert input_text_tokens is not None, "Expected `input_text_tokens` to be provided when `attn_text` is not None"
    
    n_imgs = len(imgs)
    n_attn_img = 1 if attn_img is not None else 0
    n_attn_text = 1 if attn_text is not None else 0
    n_axes = n_imgs + n_attn_img + n_attn_text + 1
    
    if n_axes <= 5:
        n_rows, n_cols = 1, n_axes
    elif n_axes <= 10:
        n_rows, n_cols = 2, 5
    elif n_axes <= 15:
        n_rows, n_cols = 3, 5
    else:
        n_rows, n_cols = 4, 5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), dpi=100)
    axes = axes.flatten()
    ax_idx = 0
    
    # Show the images
    for i in range(n_imgs):
        axes[ax_idx].imshow(imgs[i])
        axes[ax_idx].axis("off")
        if imgs_title is not None:
            axes[ax_idx].set_title(imgs_title[i])
        ax_idx += 1
            
    # Plot the attention map overlay
    if attn_img is not None:
        obs_img = imgs[imgs_title.index("obs")] # np.ndarray, (H, W, 3), uint8
        obs_img = np.asarray(preprocess_image(obs_img, center_crop)) # preprocess obs image to align with model inference
        plot_img_attn(obs_img, attn_img, axes[ax_idx])
        ax_idx += 1
        
    # Plot the attention to the input text tokens
    if attn_text is not None:
        ax = axes[ax_idx]
        visualize_tokens_with_attention(input_text_tokens, attn_text, ax)
        ax_idx += 1
    
    # Plot the logs
    ax = axes[ax_idx]
    if len(plot_keys) == 2:
        # Draw the only two logs using a twin axis sharing the X axis
        ax2 = ax.twinx()
        ax.plot(logs[plot_keys[0]], label=plot_keys[0], color="tab:blue", alpha=0.7, lw=1)
        if plot_keys[0] in LOG_VALUES_V_MAX:
            ax.set_ylim(LOG_VALUES_V_MIN[plot_keys[0]], LOG_VALUES_V_MAX[plot_keys[0]])
        ax.set_ylabel(plot_keys[0], color="tab:blue")
        ax.tick_params(axis="y", labelcolor="tab:blue")
        
        ax2.plot(logs[plot_keys[1]], label=plot_keys[1], color="tab:red", alpha=0.7, lw=1)
        if plot_keys[1] in LOG_VALUES_V_MAX:
            ax2.set_ylim(LOG_VALUES_V_MIN[plot_keys[1]], LOG_VALUES_V_MAX[plot_keys[1]])
        ax2.set_ylabel(plot_keys[1], color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red")
    else:
        for i, k in enumerate(plot_keys):
            ax.plot(logs[k], label=k, alpha=0.7, lw=1)
        ax.legend()
        ax.set_ylabel("Value")
    ax.set_xlabel("Timestep")
    ax_idx += 1
    
    # Remove the unused axes
    while ax_idx < len(axes):
        axes[ax_idx].set_axis_off()
        ax_idx += 1
        
    if sup_title is not None:
        fig.suptitle(sup_title)
    
    # Convert the plot to an image (numpy array)
    plt.tight_layout()
    # plt.show()
    fig.canvas.draw()
    plot_img = np.array(fig.canvas.renderer.buffer_rgba())
    plt.close()
    
    return plot_img


def show_mask_on_image(img, mask):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, None]
    img = np.float32(img) / 255
    mask = mask / mask.max()
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_VIRIDIS)
    hm = np.float32(heatmap) / 255
    cam = hm + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), heatmap


def overlay_attn_image(
    image: np.ndarray,
    attn_scores: Optional[torch.Tensor],
    center_crop: bool,
):
    '''
    Overlay the attention scores on the image
    
    Args:
        image: (H, W, 3) numpy array, uint8 dtype
        attn_scores: (M, N) torch FloatTensor, M corresponding to H, N corresponding to W
        center_crop: bool, whether to center crop the image (part of inference time augmentation)
    '''
    if attn_scores is None:
        return None
    
    image = preprocess_image(image, center_crop)
    image = np.asarray(image)
    H, W = image.shape[:2]

    attn_scores = F.interpolate(
        attn_scores.unsqueeze(0).unsqueeze(0),
        size=(H, W),
        mode="nearest",
    ).squeeze().numpy()
    
    overlay_image, heatmap = show_mask_on_image(image, attn_scores)
    
    return overlay_image


class PoseCumulator():
    '''
    A class to track the total translation and rotation of the end effector
    the input rot is in quaternion
    '''
    def __init__(self):
        self.reset()
        
    def update(self, pos, rot):
        rot = R.from_quat(rot)
        if self.prev_pos is not None:
            self.cum_pos += np.linalg.norm(pos - self.prev_pos)
            self.cum_rot += (self.prev_rot.inv() * rot).magnitude()
        self.prev_pos = pos
        self.prev_rot = rot
        
    def reset(self):
        self.prev_pos = None # position as a 3D numpy vector
        self.prev_rot = None # scipy Rotation object
        self.cum_pos = 0
        self.cum_rot = 0
        

def visualize_tokens_with_attention(tokens, attentions, ax, line_spacing=1.5):
    """
    Visualize tokens with attention values on a provided Axes object.

    Args:
        tokens (list of str): List of tokens.
        attentions (list of float): Attention values corresponding to tokens.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        max_width (int): Approximate maximum number of characters per line.
        line_spacing (float): Spacing between lines (relative to token height).
    """
    # Normalize attentions to the range [0, 1]
    norm = plt.Normalize(vmin=0, vmax=max(attentions))
    # norm = plt.Normalize(vmin=0, vmax=TEXT_ATTENTION_V_MAX)
    cmap = plt.get_cmap('Reds')  # Choose a color map
    alpha = 0.5  # Set transparency level

    # Clear the axes and turn off axes lines
    ax.clear()
    ax.axis('off')

    # Starting from the top-left corner
    x = 0.0 
    y = 1.0 

    # Iterate through lines and draw tokens
    for token_idx in range(len(tokens)):
        # Find the index of the token and its attention value
        token = tokens[token_idx]
        attention = attentions[token_idx]

        # Calculate background color for the token
        color = cmap(norm(attention))
        
        if token[0] == '▁':
            token = " " + token[1:]

        # Draw the token text
        t = ax.text(
            x,
            y,
            token,
            color='black',
            fontsize=10,
            ha='left',
            va='top',
            zorder=1,
            bbox = {
                'facecolor': color,
                'linewidth': 0,
                'alpha': alpha,
                'boxstyle': 'square,pad=0',
            }
        )
        rect_width = t.get_window_extent().width
        rect_height = t.get_window_extent().height
        
        # Normalize rect_width based on axes width
        rect_width = (rect_width+1) / ax.get_window_extent().width
        rect_height = rect_height / ax.get_window_extent().height

        # Increment X position for the next token
        x += rect_width  # Add padding between tokens

        if x > 0.8:
            # Decrease Y position for the next line
            y -= line_spacing * rect_height
            x = 0.0
            
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    color_bar = plt.colorbar(sm, cax=cax, orientation='vertical', label='Attention Value')
    color_bar.solids.set(alpha = alpha)

# # Example usage
# tokens = ["The", "quick", "brownbrownbrown", "fox", "jumps", "over", "the", "lazy", "dog"]
# attentions = [0.1, 0.8, 0.4, 0.9, 0.2, 0.3, 0.6, 0.7, 0.5]

# fig, ax = plt.subplots(figsize=(4, 4))
# visualize_tokens_with_attention(tokens, attentions, ax, line_spacing=1.2)
# plt.show()
