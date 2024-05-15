# Copyright AstraZeneca UK Ltd. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import os

from typing import Union, Tuple, List, Callable, Dict, Optional
import torch
from torch import nn
import torch.nn.functional as nnf
from diffusers import DiffusionPipeline
import numpy as np
from IPython.display import display
from PIL import Image
import abc
from src.p2p import ptp_utils
from src.p2p import seq_aligner
import matplotlib.pyplot as plt
import random

from scripts.txt2img import load_model_from_config
from omegaconf import OmegaConf
import torchvision.transforms.functional as F
from IPython.display import display
import numpy as np
import matplotlib as mpl

# Constants
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 5.
MAX_NUM_WORDS = 77
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

class LocalMask:
    """
    LocalMask class used to apply a masking operation to attention maps
    based on specified keywords and conditions.
    """

    def __call__(self, x_t, attention_store, step):
        if self.attn_mask_type == 'skip':
            return x_t
        k = 1

        maps = attention_store["down_cross"][:2] + attention_store["up_cross"][3:6]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        # cat over all timesteps in dim=1
        maps = torch.cat(maps, dim=1)
        # in shape: B,T,1,16,16,N
        if self.presudo_words_softmax[0] != '':
            maps = maps.softmax(dim=-1)
            maps_presudo = maps.clone()
            maps_presudo = maps_presudo * self.alpha_presudo_words
            maps_presudo = maps_presudo.softmax(dim=-1)
            maps_presudo_mask = self.alpha_presudo_words != 0
            maps = maps * ~maps_presudo_mask + maps_presudo * maps_presudo_mask
        # use alpha_layers to index selct defined keywords, sum over all keywords (dim=-1), average over all timesteps (dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        # out shape: B,1,16,16
        # smooth and resize to same size of latent (intermedia images)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        # normalisation, in shape: B,1,W,H
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        # out shape: B,1,W,H
        if self.attn_mask_type == 'hard':
            mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask).float()
        # mask.retain_grad()
        # combine baseline promp (without mask) and the rest masked prompts
        x_t = mask * x_t
        return x_t
       
    def __init__(self, tokenizer: nn.Module, prompts: List[str], words: [List[List[str]]], \
                    attn_mask_type: str = 'hard', presudo_words_softmax: [List[str]] = [''], \
                    presudo_words_infonce: [List[str]] = [''], adj_aug_infonce: [List[str]] = [''], \
                    n_gpu: int = 0, threshold: float = .3):
        """
        Initialize the LocalMask class with the provided parameters.

        Args:
            tokenizer (nn.Module): Tokenizer for processing prompts.
            prompts (List[str]): List of prompts.
            words (List[List[str]]): List of words to apply masking.
            attn_mask_type (str): Type of attention mask ('hard' or 'skip').
            presudo_words_softmax (List[str]): Words for softmax masking.
            presudo_words_infonce (List[str]): Words for InfoNCE masking.
            adj_aug_infonce (List[str]): Adjective augmentation for InfoNCE.
            n_gpu (int): GPU index.
            threshold (float): Threshold for hard masking.
        """
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        device = torch.device(f'cuda:{n_gpu}')
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold
        self.attn_mask_type = attn_mask_type

        self.presudo_words_softmax = presudo_words_softmax
        if presudo_words_softmax[0] != '':
            alpha_presudo_words = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
            for i, prompt in enumerate(prompts):
                for word in presudo_words_softmax:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    alpha_presudo_words[i, :, :, :, :, ind] = 1
            self.alpha_presudo_words = alpha_presudo_words.to(device)

        self.presudo_words_infonce = presudo_words_infonce
        if presudo_words_infonce[0] != '':
            alpha_infonce = torch.zeros(len(prompts), MAX_NUM_WORDS, 1)
            for i, prompt in enumerate(prompts):
                for word in presudo_words_infonce:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    alpha_infonce[i, ind, :] = 1
            self.alpha_infonce = alpha_infonce.to(device)

        self.adj_aug_infonce = adj_aug_infonce
        if adj_aug_infonce[0] != '':
            # assert len(adj_aug_infonce) == len(presudo_words_infonce), 'error: length of adj_aug_infonce needs to be same as adj_aug_infonce'
            alpha_adj_aug = torch.zeros(len(prompts), MAX_NUM_WORDS, 1)
            for i, prompt in enumerate(prompts):
                for word in adj_aug_infonce:
                    ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                    alpha_adj_aug[i, ind, :] = 1
            self.alpha_adj_aug = alpha_adj_aug.to(device)

        

class LocalBlend:
    """
    LocalBlend class for applying a blending operation to attention maps
    based on specified keywords and conditions.
    """

    def __call__(self, x_t, attention_store, step):
        k = 1
        maps = attention_store["down_cross"][:2] + attention_store["up_cross"][3:6]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        # cat over all timesteps in dim=1
        maps = torch.cat(maps, dim=1)
        # in shape: B,T,1,16,16,N
        # use alpha_layers to index selct defined keywords, sum over all keywords (dim=-1), average over all timesteps (dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        # out shape: B,1,16,16
        # smooth and resize to same size of latent (intermedia images)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 +1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(maps, size=(x_t.shape[2:]))
        # normalisation, in shape: B,1,W,H
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        # out shape: B,1,W,H
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask).float()
        # combine baseline promp (without mask) and the rest masked prompts
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t
       
    def __init__(self, tokenizer: nn.Module, prompts: List[str], words: [List[List[str]]], threshold: float = .3):
        """
        Initialize the LocalBlend class with the provided parameters.

        Args:
            tokenizer (nn.Module): Tokenizer for processing prompts.
            prompts (List[str]): List of prompts.
            words (List[List[str]]): List of words to apply blending.
            threshold (float): Threshold for blending.
        """
        alpha_layers = torch.zeros(len(prompts),  1, 1, 1, 1, MAX_NUM_WORDS)
        # device = prompts.device
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):
    """
    Abstract base class for attention control mechanisms.
    """
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        h = attn.shape[0]
        attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
            if self.duplicate_loop:
                self.cur_step -= 1
            self.duplicate_loop = not self.duplicate_loop
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0
        self.duplicate_loop = True


class EmptyControl(AttentionControl):
    """
    EmptyControl class for performing no-op on attention maps.
    """
    
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        return attn
    
    
class AttentionStore(AttentionControl):
    """
    AttentionStore class for storing and managing attention maps.
    """

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i].clone().detach()
        self.step_store = self.get_empty_store()

    def get_average_attention(self, average_att_time: bool = True):
        self.average_att_time = average_att_time
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, average_att_time: bool = True):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.average_att_time = average_att_time

class AttentionStoreMask(AttentionControl):
    """
    AttentionStoreMask class for storing and managing attention maps with masking.
    """

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        if attn.shape[1] <= 16 ** 2:  # avoid memory overhead
            key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
            self.step_store[key].append(attn)
        return

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i].clone().detach()
        self.step_store = self.get_empty_store()

    def get_average_attention(self, average_att_time: bool = True):
        self.average_att_time = average_att_time
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStoreMask, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self, average_att_time: bool = True):
        super(AttentionStoreMask, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.average_att_time = average_att_time

class AttentionMask(AttentionStoreMask, abc.ABC):
    """
    AttentionMask class for applying masking operations during attention processing.
    """
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
        return x_t

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionMask, self).forward(attn, is_cross, place_in_unet)
        return attn
    
    def __init__(self, local_blend: LocalBlend):
        super(AttentionMask, self).__init__()
        self.local_blend = local_blend
        
class AttentionControlEdit(AttentionStore, abc.ABC):
    """
    Abstract base class for attention control mechanisms with editing capabilities.
    """
    
    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store, self.cur_step)
        return x_t
        
    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace
    
    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError
    
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn
    
    def __init__(self, prompts, tokenizer, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend] = None):
        """
        Initialize the AttentionControlEdit class with the provided parameters.

        Args:
            prompts (list): List of prompts.
            tokenizer (nn.Module): Tokenizer for processing prompts.
            num_steps (int): Number of steps in the diffusion process.
            cross_replace_steps (Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]]): Steps for cross attention replacement.
            self_replace_steps (Union[float, Tuple[float, float]]): Steps for self attention replacement.
            local_blend (Optional[LocalBlend]): Optional LocalBlend instance for blending.
        """
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend

class AttentionReplace(AttentionControlEdit):
    """
    AttentionReplace class for replacing attention maps during the diffusion process.
    """

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)
      
    def __init__(self, prompts, tokenizer, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, tokenizer, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)
        

class AttentionRefine(AttentionControlEdit):
    """
    AttentionRefine class for refining attention maps during the diffusion process.
    """

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, tokenizer, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, tokenizer, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):
    """
    AttentionReweight class for reweighting attention maps during the diffusion process.
    """

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        mask = self.equalizer[-1,:] != 1
        return attn_replace

    def __init__(self, prompts, tokenizer, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, tokenizer, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(tokenizer, text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
                  Tuple[float, ...]]):
    """
    Create an equalizer tensor for modifying attention weights.

    Args:
        tokenizer (nn.Module): Tokenizer for processing text.
        text (str): Text to be processed.
        word_select (Union[int, Tuple[int, ...]]): Indices of words to select.
        values (Union[List[float], Tuple[float, ...]]): Values to set for the selected words.

    Returns:
        torch.Tensor: Equalizer tensor.
    """
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        for i in inds:
            equalizer[:, i] = values
    return equalizer


def aggregate_attention(prompts: list, attention_store: AttentionStore, res: int, \
                        from_where: List[str], is_cross: bool, select: int, \
                        average_att_ch: bool = True, average_att_time: bool = True):
    """
    Aggregate attention maps from the attention store.

    Args:
        prompts (list): List of prompts.
        attention_store (AttentionStore): Attention store object.
        res (int): Resolution of attention maps.
        from_where (List[str]): List of attention map sources.
        is_cross (bool): Whether to aggregate cross-attention maps.
        select (int): Index of the prompt to select.
        average_att_ch (bool): Whether to average attention maps across channels.
        average_att_time (bool): Whether to average attention maps across time steps.

    Returns:
        torch.Tensor: Aggregated attention map.
    """
    out = []
    attention_maps = attention_store.get_average_attention(average_att_time=average_att_time)

    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross' if is_cross else 'self'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]
                out.append(cross_maps)
    out = torch.cat(out, dim=0)
    if average_att_ch:
        out = out.sum(0) / out.shape[0]
    else: # max
        out, _ = out.max(0)
    return out.cpu()

def _remove_axes(ax):
    """
    Remove axes from a matplotlib Axes object.

    Args:
        ax (matplotlib.axes.Axes): Axes object to remove axes from.
    """
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])

def remove_axes(axes):
    """
    Remove axes from a list of matplotlib Axes objects.

    Args:
        axes (list): List of Axes objects to remove axes from.
    """
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)

def apply_mask(image, mask, select_c, white_bg=False):
    """
    Apply a binary mask to an image.
    Foreground is kept while the background is replaced with NaN.

    Args:
        image (PIL.Image): Input image.
        mask (np.array): Binary mask.
        select_c (int): Selected class for masking.
        white_bg (bool): Whether to use a white background.

    Returns:
        PIL.Image: Masked image.
    """
    # Convert mask to boolean array
    mask_bool = mask == select_c

    # Extend the mask to all channels (assuming a 3-channel RGB image)
    mask_bool_rgb = np.stack([mask_bool]*3, axis=-1)

    # Create a white background image
    white_background = np.ones_like(image) * 255

    # Convert image to numpy array
    image_array = np.array(image).astype(np.float64)

    if white_bg:
        # Copy over the masked region from the original image to the white background
        white_background[mask_bool_rgb] = image_array[mask_bool_rgb]
        result_image = Image.fromarray(white_background, 'RGB')
    else:
        # For non-white background, retain the original behavior
        image_array[~mask_bool_rgb] = np.nan
        result_image = Image.fromarray(np.uint8(image_array), 'RGB')

    return result_image

def plot_img_mask(ldm, prompts, emb_path_list, exp_names, device, out_dir, out_name, config, \
    latent=None, array_latent=False, GUIDANCE_SCALE=5.0, attn_threshold=0.5, select_clsses = ['*','&'], \
    show_text=True, mask_concepts=False, g_gpu=None):
    """
    Plot images and attention masks for given embeddings.

    Args:
        ldm: Latent diffusion model.
        prompts (list): List of prompts.
        emb_path_list (list): List of embedding paths.
        exp_names (list): List of experiment names.
        device (str): Device to run the model on.
        out_dir (str): Output directory for saving images.
        out_name (str): Output file name.
        config: Configuration object.
        latent: Latent tensor.
        array_latent (bool): Whether to use array latent.
        GUIDANCE_SCALE (float): Guidance scale for the diffusion process.
        attn_threshold (float): Threshold for attention masks.
        select_clsses (list): List of selected classes.
        show_text (bool): Whether to show text in the images.
        mask_concepts (bool): Whether to mask concepts.
        g_gpu (torch.Generator): Generator for random numbers.
    """
    n_rows = 1
    n_cols = len(emb_path_list)+1 # img, mask, attn
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 10, n_rows * 10))


    for i, embedding_path in enumerate(emb_path_list):
        ldm.embedding_manager.load(embedding_path)
        ldm = ldm.to(device)
        tokenizer = ldm.cond_stage_model.tknz_fn.tokenizer

        if g_gpu is None:
            g_gpu = torch.Generator(device='cuda')
        else:
            g_gpu = g_gpu
        controller = AttentionStore()
        images, _ = run_and_display(ldm, prompts, controller, config, latent=latent, \
                            run_baseline=False, generator=g_gpu, array_latent=array_latent, \
                            guidance_scale=GUIDANCE_SCALE, return_img=True)
        attn_img = show_cross_attention(tokenizer, prompts, controller, res=16, from_where=["up", "down"], return_img=True, show_text=show_text, select_clsses=select_clsses)

        # ave-channel + ave-time attention
        attn_mask = show_cross_attention_mask_merged(tokenizer, prompts, controller, res=16, \
            from_where=["up", "down"], select_clsses = select_clsses, \
                average_att_ch=True, average_att_time=True, threshold=attn_threshold, return_img=True)

        img = ptp_utils.view_images(images, return_img=True)
        if i == 0:
            ax[0].imshow(img)
        
        ax[i+1].imshow(attn_mask)
        # if show_text:
        #     ax[i+1].set_xlabel(exp_names[i], fontsize=40)
    remove_axes(ax)
    plt.tight_layout()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 
    plt.savefig(os.path.join(out_dir, out_name))
    # plt.show()
    # plt.clf()

def plot_img_attn_mask(ldm, prompts, emb_path_list, exp_names, device, out_dir, out_name, config, \
    latent=None, array_latent=False, GUIDANCE_SCALE=5.0, attn_threshold=0.5, select_clsses = ['*','&'], \
    show_text=True, mask_concepts=False, g_gpu=None):
    """
    Plot images, attention masks, and masked concepts for given embeddings.

    Args:
        ldm: Latent diffusion model.
        prompts (list): List of prompts.
        emb_path_list (list): List of embedding paths.
        exp_names (list): List of experiment names.
        device (str): Device to run the model on.
        out_dir (str): Output directory for saving images.
        out_name (str): Output file name.
        config: Configuration object.
        latent: Latent tensor.
        array_latent (bool): Whether to use array latent.
        GUIDANCE_SCALE (float): Guidance scale for the diffusion process.
        attn_threshold (float): Threshold for attention masks.
        select_clsses (list): List of selected classes.
        show_text (bool): Whether to show text in the images.
        mask_concepts (bool): Whether to mask concepts.
        g_gpu (torch.Generator): Generator for random numbers.
    """
    n_rows = len(emb_path_list)
    n_cols = 3 # img, mask, attn
    if mask_concepts:
        n_cols += len(select_clsses)
    if mask_concepts and not show_text:
        n_attns = len(select_clsses)
        for p in select_clsses:
            mask_path = os.path.join(out_dir, 'masked_'+p)
            if not os.path.exists(mask_path):
                os.mkdir(mask_path) 
    else:
        n_attns = len(prompts[0].split(' '))
    if show_text:
        n_attns -= 2
    if mask_concepts:
        width_ratios = [1,1]
        for _ in range(len(select_clsses)):
            width_ratios.append(1)
        width_ratios.append(n_attns)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=((n_cols+n_attns-1) * 5, n_rows * 5), \
            gridspec_kw={'width_ratios': width_ratios})
    else:
        fig, ax = plt.subplots(n_rows, n_cols, figsize=((n_cols+n_attns-1) * 5, n_rows * 5), \
            gridspec_kw={'width_ratios': [1,1,n_attns]})
    for i, embedding_path in enumerate(emb_path_list):
        ldm.embedding_manager.load(embedding_path)
        ldm = ldm.to(device)
        tokenizer = ldm.cond_stage_model.tknz_fn.tokenizer

        if g_gpu is None:
            g_gpu = torch.Generator(device='cuda')
        else:
            g_gpu = g_gpu
        controller = AttentionStore()
        images, _ = run_and_display(ldm, prompts, controller, config, latent=latent, \
                            run_baseline=False, generator=g_gpu, array_latent=array_latent, \
                            guidance_scale=GUIDANCE_SCALE, return_img=True)
        if mask_concepts and not show_text:
            attn_img = show_cross_attention(tokenizer, prompts, controller, res=16, from_where=["up", "down"], return_img=True, show_text=show_text, select_clsses=select_clsses)
        else:
            attn_img = show_cross_attention(tokenizer, prompts, controller, res=16, from_where=["up", "down"], return_img=True, show_text=show_text)

        # ave-channel + ave-time attention
        attn_mask = show_cross_attention_mask_merged(tokenizer, prompts, controller, res=16, \
            from_where=["up", "down"], select_clsses = select_clsses, \
                average_att_ch=True, average_att_time=True, threshold=attn_threshold, return_img=True)

        img = ptp_utils.view_images(images, return_img=True)
        ax[i,0].imshow(img)
        if show_text:
            ax[i,0].set_xlabel(exp_names[i], fontsize=40)
        ax[i,1].imshow(attn_mask)
        if show_text:
            ax[i,1].set_xlabel("Attn Masks", fontsize=40)
        col = 2
        if mask_concepts:
            for c in range(1,len(select_clsses)+1):
                masked_img = apply_mask(img, attn_mask, c)
                ax[i,col].imshow(masked_img)
                if show_text:
                    ax[i,col].set_xlabel(select_clsses[c-1], fontsize=40)
                else:
                    mask_path = os.path.join(out_dir, 'masked_'+select_clsses[c-1])
                    masked_img.save(os.path.join(mask_path, exp_names[i]+'.png'), 'PNG')
                    masked_img_wbg = apply_mask(img, attn_mask, c, white_bg=True)
                    mask_path_wbg = os.path.join(out_dir, 'masked_'+select_clsses[c-1]+'white_bg')
                    if not os.path.exists(mask_path_wbg):
                        os.mkdir(mask_path_wbg) 
                    masked_img_wbg.save(os.path.join(mask_path_wbg, exp_names[i]+'.png'), 'PNG')
                col += 1
            img_path = os.path.join(out_dir, 'gen_img')
            if not os.path.exists(img_path):
                os.mkdir(img_path) 
            img.save(os.path.join(img_path, exp_names[i]+'.png'), 'PNG')

            mask_path = os.path.join(out_dir, 'attn_mask')
            if not os.path.exists(mask_path):
                os.mkdir(mask_path) 
            Image.fromarray(attn_mask.byte().numpy()).save(os.path.join(mask_path, exp_names[i]+'.png'), 'PNG')
        ax[i,col].imshow(attn_img)

    remove_axes(ax)
    plt.tight_layout()
    if not os.path.exists(out_dir):
        os.mkdir(out_dir) 
    plt.savefig(os.path.join(out_dir, out_name))
    plt.show()
    plt.clf()


def show_cross_attention(tokenizer: nn.Module, prompts: list, attention_store: AttentionStore, \
        res: int, from_where: List[str], select: int = 0, return_img: bool = False, \
        out_path_img='', save_img=False, show_text=True, select_clsses=[]):
    """
    Show cross-attention maps for the given prompts.

    Args:
        tokenizer (nn.Module): Tokenizer for processing prompts.
        prompts (list): List of prompts.
        attention_store (AttentionStore): Attention store object.
        res (int): Resolution of attention maps.
        from_where (List[str]): List of attention map sources.
        select (int): Index of the prompt to select.
        return_img (bool): Whether to return the image.
        out_path_img (str): Output path for saving the image.
        save_img (bool): Whether to save the image.
        show_text (bool): Whether to show text in the images.
        select_clsses (list): List of selected classes.
    """
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select)
    images = []
    for i in range(1,len(tokens)-1):
        if len(select_clsses) > 0 and decoder(int(tokens[i])) not in select_clsses:
            continue
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))
        if show_text:
            image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
    
    if return_img:
        pil_img = ptp_utils.view_images(np.stack(images, axis=0), 'cross_attention', return_img=return_img)
        return pil_img
    elif save_img:
        ptp_utils.view_images(np.stack(images, axis=0), save_img=save_img, out_path=out_path_img)  
    else:
        ptp_utils.view_images(np.stack(images, axis=0), 'cross_attention')


def show_cross_attention_with_mask(tokenizer: nn.Module, prompts: list, \
                                    attention_store: AttentionStore, res: int, from_where: List[str], \
                                    select: int = 0, threshold: float = 0.3, \
                                    average_att_ch: bool = True, average_att_time: bool = True, \
                                    segment_method: str = 'threshold'):
    """
    Show cross-attention maps with masks for the given prompts.

    Args:
        tokenizer (nn.Module): Tokenizer for processing prompts.
        prompts (list): List of prompts.
        attention_store (AttentionStore): Attention store object.
        res (int): Resolution of attention maps.
        from_where (List[str]): List of attention map sources.
        select (int): Index of the prompt to select.
        threshold (float): Threshold for masking.
        average_att_ch (bool): Whether to average attention maps across channels.
        average_att_time (bool): Whether to average attention maps across time steps.
        segment_method (str): Method for segmenting the masks ('threshold' or 'kmean').
    """
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select, average_att_ch=average_att_ch, average_att_time=average_att_time)
    images = []
    masks = []
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = image / image.max()
        image = F.resize(image.unsqueeze(0), 256).squeeze(0)
        if segment_method == 'threshold':
            mask = (image.gt(threshold)).int()
        elif segment_method == 'kmean':
            X = np.array(image.view(-1).unsqueeze(1).cpu())
            kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)
            kmean_pred = kmeans.labels_.reshape(256, 256)
            mask = torch.tensor(kmean_pred.astype(int))
        mask *= (i+1)
        mask = mask.unsqueeze(-1).expand(*mask.shape, 3)
        mask = mask.numpy().astype(np.uint8)
        mask = np.array(Image.fromarray(mask).resize((256, 256), Image.NEAREST))
        mask = ptp_utils.text_under_image(mask, decoder(int(tokens[i])))
        masks.append(mask)
    ptp_utils.view_masks(np.stack(masks, axis=0)) 

def show_cross_attention_mask_merged(tokenizer: nn.Module, prompts: list, \
                                        attention_store: AttentionStore, res: int, from_where: List[str], \
                                        select: int = 0, threshold: float = 0.65, select_clsses: list = ['a','photo'], \
                                        average_att_ch: bool = True, average_att_time: bool = True, masked_scale: bool = False, \
                                        background_scale: float = 1.0, images = None, blend = 0.3, return_img: bool = False):
    """
    Show merged cross-attention masks for the given prompts.

    Args:
        tokenizer (nn.Module): Tokenizer for processing prompts.
        prompts (list): List of prompts.
        attention_store (AttentionStore): Attention store object.
        res (int): Resolution of attention maps.
        from_where (List[str]): List of attention map sources.
        select (int): Index of the prompt to select.
        threshold (float): Threshold for masking.
        select_clsses (list): List of selected classes.
        average_att_ch (bool): Whether to average attention maps across channels.
        average_att_time (bool): Whether to average attention maps across time steps.
        masked_scale (bool): Whether to scale masked regions.
        background_scale (float): Scale for background regions.
        images (np.array): Array of images.
        blend (float): Blending factor for masks.
        return_img (bool): Whether to return the image.
    """
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, True, select, average_att_ch=average_att_ch, average_att_time=average_att_time)
    # attention_maps = attention_maps[:, :, select_clsses]
    attention_maps_temp = None
    for select_c in range(len(tokens)):
        if decoder(int(tokens[select_c])) not in select_clsses:
            continue
        # select_c+1 indicates 
        attention_map = attention_maps[:, :, select_c].unsqueeze(-1)

        if attention_maps_temp is None:
            attention_maps_temp = attention_map
        else:
            attention_maps_temp = torch.cat([attention_maps_temp, attention_map], dim=-1)


    attention_maps_temp = attention_maps_temp.permute(2,0,1)
    attention_maps_temp = F.resize(attention_maps_temp, 256)
    attention_maps_temp = attention_maps_temp.permute(1,2,0)

    attention_maps_max, _ = attention_maps_temp.view(-1, len(select_clsses)).max(0)
    attention_maps_norm = attention_maps_temp / attention_maps_max
    # scale background class
    # attention_maps_norm[:,:,0] *= background_scale
    mask = torch.zeros(attention_maps_norm.shape)
    seg = torch.zeros(attention_maps_norm.shape[:-1]).long()
    for i in range(attention_maps_norm.shape[-1]):
        mask[:,:,i] = (attention_maps_norm[:,:,i].gt(threshold)).long()
        seg += (attention_maps_norm[:,:,i].gt(threshold)).long() * (i+1)
    intersect = mask.sum(-1) > 1
    _, mask_merge = attention_maps_norm.view(-1, len(select_clsses)).max(-1)
    mask_merge = mask_merge.view(256,256)
    mask_merge += 1
    seg[intersect] = mask_merge[intersect]

    if images is not None:
        img = Image.fromarray(images[0].astype(np.uint8)).convert('RGBA')
        plt.imshow(Image.fromarray(np.array(seg).astype(np.uint8)))
        plt.axis('off')
        plt.savefig("temp.png", bbox_inches='tight', pad_inches=0.0)
        seg = Image.open("temp.png")
        seg = seg.resize((256,256), Image.NEAREST)
        seg = Image.blend(img, seg, blend)
        os.remove("temp.png")
    if return_img:
        return seg
    else:
        plt.imshow(seg)
        plt.axis('off')
        # pil_img = Image.fromarray(mask_merge.numpy().astype(np.uint8))
        # display(pil_img)
        # ptp_utils.view_images(mask_merge.numpy().astype(np.uint8))

def show_self_attention_comp(prompts: list, \
                                attention_store: AttentionStore, res: int, from_where: List[str],
                        max_com=10, select: int = 0):
    """
    Show self-attention components for the given prompts.

    Args:
        prompts (list): List of prompts.
        attention_store (AttentionStore): Attention store object.
        res (int): Resolution of attention maps.
        from_where (List[str]): List of attention map sources.
        max_com (int): Maximum number of components to show.
        select (int): Index of the prompt to select.
    """
    attention_maps = aggregate_attention(prompts, attention_store, res, from_where, False, select).numpy().reshape((res ** 2, res ** 2))
    u, s, vh = np.linalg.svd(attention_maps - np.mean(attention_maps, axis=1, keepdims=True))
    images = []
    for i in range(max_com):
        image = vh[i].reshape(res, res)
        image = image - image.min()
        image = 255 * image / image.max()
        image = np.repeat(np.expand_dims(image, axis=2), 3, axis=2).astype(np.uint8)
        image = Image.fromarray(image).resize((256, 256))
        image = np.array(image)
        images.append(image)
    ptp_utils.view_images(np.concatenate(images, axis=1), 'self_attention')


def sort_by_eq(eq):
    """
    Sort images by equalizer values.

    Args:
        eq (list): List of equalizer values.

    Returns:
        function: Sorting function for images.
    """
    
    def inner_(images):
        swap = 0
        if eq[-1] < 1:
            for i in range(len(eq)):
                if eq[i] > 1 and eq[i + 1] < 1:
                    swap = i + 2
                    break
        else:
             for i in range(len(eq)):
                if eq[i] < 1 and eq[i + 1] > 1:
                    swap = i + 2
                    break
        print(swap)
        if swap > 0:
            images = np.concatenate([images[1:swap], images[:1], images[swap:]], axis=0)
            
        return images
    return inner_


def run_and_display(ldm, prompts, controller, config, latent=None, run_baseline=True, \
    callback:Optional[Callable[[np.ndarray], np.ndarray]] = None, generator=None, \
    out_name='gen_images', num_diff_steps=50, array_latent=False, guidance_scale=5.0, \
    return_img=False, save_img=False, out_path_img='../outputs/p2p/attention.png'):
    """
    Run the latent diffusion model and display generated images.

    Args:
        ldm: Latent diffusion model.
        prompts (list): List of prompts.
        controller: Attention controller.
        config: Configuration object.
        latent: Latent tensor.
        run_baseline (bool): Whether to run the baseline model.
        callback (Optional[Callable[[np.ndarray], np.ndarray]]): Callback function for processing images.
        generator: Generator for random numbers.
        out_name (str): Output file name.
        num_diff_steps (int): Number of diffusion steps.
        array_latent (bool): Whether to use array latent.
        guidance_scale (float): Guidance scale for the diffusion process.
        return_img (bool): Whether to return the image.
        save_img (bool): Whether to save the image.
        out_path_img (str): Output path for saving the image.
    """
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(ldm, prompts, EmptyControl(), config, latent=latent, run_baseline=False, out_name='gen_baseline')
        print("results with prompt-to-prompt")
    # images, x_t = ptp_utils.text2image_ldm(ldm, prompts, controller, latent=latent, num_inference_steps=NUM_DIFFUSION_STEPS, guidance_scale=GUIDANCE_SCALE, generator=generator)
    images, x_t = ptp_utils.text2image_plms(ldm, controller, config, prompts, \
                            latent=latent, num_inference_steps=num_diff_steps, \
                            guidance_scale=guidance_scale, generator=generator, array_latent=array_latent)
    if callback is not None:
        images = callback(images)
    ptp_utils.view_images(images, out_path=out_path_img, return_img=return_img, save_img=save_img)
    return images, x_t
