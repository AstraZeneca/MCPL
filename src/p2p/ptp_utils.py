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

import numpy as np
import torch
from torch import nn
from PIL import Image, ImageDraw, ImageFont
import cv2
from typing import Optional, Union, Tuple, List, Callable, Dict
from IPython.display import display
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt

import sys
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from einops import rearrange

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    """
    Adds text under an image.

    Args:
        image (np.ndarray): Input image array.
        text (str): Text to be added under the image.
        text_color (Tuple[int, int, int]): RGB color of the text. Default is black.
    
    Returns:
        np.ndarray: Image array with text added under it.
    """
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img


def view_images(images, out_path='../outputs/p2p/attention.png', num_rows=1, offset_ratio=0.02, return_img=False, save_img=False):
    """
    Display or save a grid of images.

    Args:
        images (list or np.ndarray): List of images or image array.
        out_path (str): Path to save the image. Default is '../outputs/p2p/attention.png'.
        num_rows (int): Number of rows in the grid. Default is 1.
        offset_ratio (float): Offset ratio between images. Default is 0.02.
        return_img (bool): Whether to return the image. Default is False.
        save_img (bool): Whether to save the image. Default is False.
    
    Returns:
        PIL.Image: Image if return_img is True.
    """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    pil_img = Image.fromarray(image_)
    
    if return_img:
        return pil_img
    elif save_img:
        pil_img.save(out_path)
    else:
        display(pil_img)

def view_masks(images, num_rows=1, offset_ratio=0.02):
    """
    Display a grid of masks.

    Args:
        images (list or np.ndarray): List of masks or mask array.
        num_rows (int): Number of rows in the grid. Default is 1.
        offset_ratio (float): Offset ratio between masks. Default is 0.02.
    """
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0

    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)

    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]

    plt.imshow(image_)
    # pil_img = Image.fromarray(image_)
    # display(pil_img)

def diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource=False):
    """
    Perform a single diffusion step.

    Args:
        model: Diffusion model.
        controller: Attention controller.
        latents (torch.FloatTensor): Latent tensor.
        context (torch.FloatTensor): Context tensor.
        t (torch.Tensor): Current timestep.
        guidance_scale (float): Guidance scale for the diffusion process.
        low_resource (bool): Whether to use low resource mode. Default is False.
    
    Returns:
        torch.FloatTensor: Updated latent tensor.
    """
    if low_resource:
        noise_pred_uncond = model.model.diffusion_model(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.model.diffusion_model(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.model.diffusion_model(latents_input, t, encoder_hidden_states=context)["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    latents = controller.step_callback(latents)
    return latents


def latent2image(vae, latents):
    """
    Convert latents to images.

    Args:
        vae: Variational autoencoder.
        latents (torch.FloatTensor): Latent tensor.
    
    Returns:
        np.ndarray: Image array.
    """
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).numpy()
    image = (image * 255).astype(np.uint8)
    return image


def init_latent(latent, model, height, width, generator, batch_size):
    """
    Initialize latent tensor.

    Args:
        latent (torch.FloatTensor): Input latent tensor.
        model: Diffusion model.
        height (int): Image height.
        width (int): Image width.
        generator (torch.Generator): Random number generator.
        batch_size (int): Batch size.
    
    Returns:
        tuple: Tuple containing the latent tensor and expanded latent tensor.
    """
    if latent is None:
        latent = torch.randn(
            (1, model.model.diffusion_model.in_channels, height // 8, width // 8),
            generator=generator,
        )
    latents = latent.expand(batch_size,  model.model.diffusion_model.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents


@torch.no_grad()
def text2image_plms(
    model,
    controller,
    opt,
    prompt:  List[str],
    num_inference_steps: int = 50,
    n_samples: int  = 1,
    guidance_scale: float = 5.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    verbose: Optional[bool] = False,
    array_latent: Optional[bool] = False,
):
    """
    Generate images from text using PLMS sampler.

    Args:
        model: Diffusion model.
        controller: Attention controller.
        opt: Options object.
        prompt (List[str]): List of prompts.
        num_inference_steps (int): Number of inference steps. Default is 50.
        n_samples (int): Number of samples. Default is 1.
        guidance_scale (float): Guidance scale for the diffusion process. Default is 5.
        generator (Optional[torch.Generator]): Random number generator. Default is None.
        latent (Optional[torch.FloatTensor]): Latent tensor. Default is None.
        verbose (Optional[bool]): Verbose mode. Default is False.
        array_latent (Optional[bool]): Whether to use array latent. Default is False.
    
    Returns:
        tuple: Tuple containing the image array and latent tensor.
    """
    register_attention_control_t2i(model, controller, eval_mode=True)
    height = width = 256
    batch_size = n_samples * len(prompt)
    sampler = PLMSSampler(model)
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if guidance_scale != 1.0:
                uc = model.get_learned_conditioning(n_samples * ([""] * len(prompt)))
        
            cond = model.get_learned_conditioning(n_samples * prompt)
            shape = [batch_size, 4, height//8, width//8]
            # samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
            #                                  conditioning=c,
            #                                  ...

            # expamd from plms model.sample and model.plms_sampling here
            if cond is not None:
                if isinstance(cond, dict):
                    cbs = cond[list(cond.keys())[0]].shape[0]
                    if cbs != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")
                else:
                    if cond.shape[0] != batch_size:
                        print(f"Warning: Got {cond.shape[0]} conditionings but batch-size is {batch_size}")
            
            sampler.make_schedule(ddim_num_steps=num_inference_steps, ddim_eta=0.0, verbose=False)
            # sampling
            C, H, W = shape[1:]
            size = shape
            device = model.device
            b = shape[0]

            if latent is None:
                latent_r = torch.randn((1,C,H,W), device=model.device, generator=generator)
                latents = latent_r.expand(shape)
                if array_latent:
                    latent_ay = [latent_r]
            else:
                if array_latent:
                    latent_ay = [l.expand(shape) for l in latent]
                    latent_r = latent
                else:
                    latents = latent.expand(shape)
                    latent_r = latent
            

            timesteps = sampler.ddim_timesteps

            time_range = np.flip(timesteps)
            total_steps = timesteps.shape[0]
            print(f"Running PLMS Sampling with {total_steps} timesteps")

            iterator = tqdm(time_range, desc='PLMS Sampler', total=total_steps)
            old_eps = []

            for i, step in enumerate(iterator):
                index = total_steps - i - 1
                ts = torch.full((b,), step, device=device, dtype=torch.long)
                ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long)

                if latent is not None and array_latent: # use prior latents
                    latents = latent_ay.pop(0)
                outs = sampler.p_sample_plms(latents, cond, ts, index=index,
                                      unconditional_guidance_scale=guidance_scale,
                                      unconditional_conditioning=uc,
                                      old_eps=old_eps, t_next=ts_next)
                latents, pred_x0, e_t = outs
                if array_latent and latent is None: # generate latents
                    latent_ay.append(latents)
                    
                old_eps.append(e_t)
                if len(old_eps) >= 4:
                    old_eps.pop(0)
                latents = controller.step_callback(latents)
            samples_ddim = latents

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
    image = 255. * rearrange(x_samples_ddim.cpu().numpy(), 'b c h w -> b h w c')

    if array_latent and latent is None: # generate latents
        latent_r = latent_ay
   
    return image, latent_r

@torch.no_grad()
def text2image_ddpm(
    model,
    controller,
    opt,
    prompt:  List[str],
    num_inference_steps: int = 50,
    n_samples: int  = 8,
    # guidance_scale: float = 5.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    verbose: Optional[bool] = False,
):
    """
    Generate images from text using DDPM sampler.

    Args:
        model: Diffusion model.
        controller: Attention controller.
        opt: Options object.
        prompt (List[str]): List of prompts.
        num_inference_steps (int): Number of inference steps. Default is 50.
        n_samples (int): Number of samples. Default is 8.
        generator (Optional[torch.Generator]): Random number generator. Default is None.
        latent (Optional[torch.FloatTensor]): Latent tensor. Default is None.
        verbose (Optional[bool]): Verbose mode. Default is False.
    
    Returns:
        tuple: Tuple containing the image array and latent tensor.
    """
    register_attention_control_t2i(model, controller, eval_mode=True)
    height = width = 256
    batch_size = len(prompt)
    sampler = PLMSSampler(model)
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            # if opt.scale != 1.0:
            uc = model.get_learned_conditioning(n_samples * [""])
        
            cond = model.get_learned_conditioning(n_samples * prompt)
            shape = [n_samples, 4, height//8, width//8]
            b = shape[0]

            if latent is None:
                latent = torch.randn(shape, device=model.device, generator=generator)
            else:
                latent = latent

            if num_inference_steps is None:
                timesteps = model.num_timesteps
            else:
                timesteps = num_inference_steps

            iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
                range(0, timesteps))

            for i in iterator:
                print(f'step: {i}')
                ts = torch.full((b,), i, device=model.device, dtype=torch.long)

                latent = model.p_sample(latent, cond, ts,
                                    clip_denoised=model.clip_denoised,
                                    quantize_denoised=False)
                latent = controller.step_callback(latent)
            samples_ddim = latent

    x_samples_ddim = model.decode_first_stage(samples_ddim)
    x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)
    image = 255. * rearrange(x_samples_ddim.cpu().numpy(), 'b c h w -> b h w c')
   
    return image, latent

@torch.no_grad()
def text2image_ldm(
    model,
    prompt:  List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
):
    """
    Generate images from text using LDM.

    Args:
        model: Diffusion model.
        prompt (List[str]): List of prompts.
        controller: Attention controller.
        num_inference_steps (int): Number of inference steps. Default is 50.
        guidance_scale (Optional[float]): Guidance scale for the diffusion process. Default is 7.
        generator (Optional[torch.Generator]): Random number generator. Default is None.
        latent (Optional[torch.FloatTensor]): Latent tensor. Default is None.
    
    Returns:
        tuple: Tuple containing the image array and latent tensor.
    """
    register_attention_control(model, controller)
    height = width = 256
    batch_size = len(prompt)
    
    uncond_input = model.tokenizer([""] * batch_size, padding="max_length", max_length=77, return_tensors="pt")
    uncond_embeddings = model.bert(uncond_input.input_ids.to(model.device))[0]
    
    text_input = model.tokenizer(prompt, padding="max_length", max_length=77, return_tensors="pt")
    text_embeddings = model.bert(text_input.input_ids.to(model.device))[0]
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    context = torch.cat([uncond_embeddings, text_embeddings])
    
    model.scheduler.set_timesteps(num_inference_steps)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale)
    
    image = latent2image(model.vqvae, latents)
   
    return image, latent


@torch.no_grad()
def text2image_ldm_stable(
    model,
    prompt: List[str],
    controller,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[torch.Generator] = None,
    latent: Optional[torch.FloatTensor] = None,
    low_resource: bool = False,
):
    """
    Generate images from text using stable LDM.

    Args:
        model: Diffusion model.
        prompt (List[str]): List of prompts.
        controller: Attention controller.
        num_inference_steps (int): Number of inference steps. Default is 50.
        guidance_scale (float): Guidance scale for the diffusion process. Default is 7.5.
        generator (Optional[torch.Generator]): Random number generator. Default is None.
        latent (Optional[torch.FloatTensor]): Latent tensor. Default is None.
        low_resource (bool): Whether to use low resource mode. Default is False.
    
    Returns:
        tuple: Tuple containing the image array and latent tensor.
    """
    register_attention_control(model, controller)
    height = width = 512
    batch_size = len(prompt)

    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = model.tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    uncond_embeddings = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    
    context = [uncond_embeddings, text_embeddings]
    if not low_resource:
        context = torch.cat(context)
    latent, latents = init_latent(latent, model, height, width, generator, batch_size)
    
    # set timesteps
    extra_set_kwargs = {"offset": 1}
    model.scheduler.set_timesteps(num_inference_steps, **extra_set_kwargs)
    for t in tqdm(model.scheduler.timesteps):
        latents = diffusion_step(model, controller, latents, context, t, guidance_scale, low_resource)
    
    image = latent2image(model.vae, latents)
  
    return image, latent

def register_attention_control_t2i(model, controller, eval_mode=False):
    """
    Register attention control for text-to-image models.

    Args:
        model: Diffusion model.
        controller: Attention controller.
        eval_mode (bool): Evaluation mode. Default is False.
    """
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x,
                    context=None,
                    mask=None,
                    context_mask=None,
                    rel_pos=None,
                    sinusoidal_emb=None,
                    prev_attn=None,
                    mem=None
            ):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            attn = sim.softmax(dim=-1)
            if eval_mode: 
                attn = controller(attn, is_cross, place_in_unet)
            else: 
                controller(attn.clone(), is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.model.diffusion_model.named_children()

    for name, module in sub_nets:
        if "input" in name:
            cross_att_count += register_recr(module, 0, "down")
        elif "output" in name:
            cross_att_count += register_recr(module, 0, "up")
        elif "middle" in name:
            cross_att_count += register_recr(module, 0, "mid")

    controller.num_att_layers = cross_att_count

def register_attention_control(model, controller):
    """
    Register attention control for text-to-image models.

    Args:
        model: Diffusion model.
        controller: Attention controller.
    """
    def ca_forward(self, place_in_unet):
        to_out = self.to_out
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def forward(x, context=None, mask=None):
            batch_size, sequence_length, dim = x.shape
            h = self.heads
            q = self.to_q(x)
            is_cross = context is not None
            context = context if is_cross else x
            k = self.to_k(context)
            v = self.to_v(context)
            q = self.reshape_heads_to_batch_dim(q)
            k = self.reshape_heads_to_batch_dim(k)
            v = self.reshape_heads_to_batch_dim(v)

            sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale

            if mask is not None:
                mask = mask.reshape(batch_size, -1)
                max_neg_value = -torch.finfo(sim.dtype).max
                mask = mask[:, None, :].repeat(h, 1, 1)
                sim.masked_fill_(~mask, max_neg_value)

            # attention, what we cannot get enough of
            attn = sim.softmax(dim=-1)
            attn = controller(attn, is_cross, place_in_unet)
            out = torch.einsum("b i j, b j d -> b i d", attn, v)
            out = self.reshape_batch_dim_to_heads(out)
            return to_out(out)

        return forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()

    def register_recr(net_, count, place_in_unet):
        if net_.__class__.__name__ == 'CrossAttention':
            net_.forward = ca_forward(net_, place_in_unet)
            return count + 1
        elif hasattr(net_, 'children'):
            for net__ in net_.children():
                count = register_recr(net__, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.model.diffusion_model.named_children()

    for name, module in sub_nets:
        if "input" in name:
            cross_att_count += register_recr(module, 0, "down")
        elif "output" in name:
            cross_att_count += register_recr(module, 0, "up")
        elif "middle" in name:
            cross_att_count += register_recr(module, 0, "mid")

    controller.num_att_layers = cross_att_count

    
def get_word_inds(text: str, word_place: int, tokenizer):
    """
    Get the indices of specific words in a tokenized text.

    Args:
        text (str): Input text.
        word_place (int or str): Word position or word string to find indices.
        tokenizer: Tokenizer to encode the text.
    
    Returns:
        np.ndarray: Array of word indices.
    """
    split_text = text.split(" ")
    if type(word_place) is str:
        word_place = [i for i, word in enumerate(split_text) if word_place == word]
    elif type(word_place) is int:
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        words_encode = [tokenizer.decode([item]).strip("#") for item in tokenizer.encode(text)][1:-1]
        cur_len, ptr = 0, 0

        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return np.array(out)


def update_alpha_time_word(alpha, bounds: Union[float, Tuple[float, float]], prompt_ind: int,
                           word_inds: Optional[torch.Tensor]=None):
    """
    Update the alpha values for time and word indices.

    Args:
        alpha (torch.Tensor): Alpha tensor.
        bounds (Union[float, Tuple[float, float]]): Bounds for alpha values.
        prompt_ind (int): Prompt index.
        word_inds (Optional[torch.Tensor]): Word indices. Default is None.
    
    Returns:
        torch.Tensor: Updated alpha tensor.
    """
    if type(bounds) is float:
        bounds = 0, bounds
    start, end = int(bounds[0] * alpha.shape[0]), int(bounds[1] * alpha.shape[0])
    if word_inds is None:
        word_inds = torch.arange(alpha.shape[2])
    alpha[: start, prompt_ind, word_inds] = 0
    alpha[start: end, prompt_ind, word_inds] = 1
    alpha[end:, prompt_ind, word_inds] = 0
    return alpha


def get_time_words_attention_alpha(prompts, num_steps,
                                   cross_replace_steps: Union[float, Dict[str, Tuple[float, float]]],
                                   tokenizer, max_num_words=77):
    """
    Get the attention alpha values for time and words.

    Args:
        prompts (list): List of prompts.
        num_steps (int): Number of steps.
        cross_replace_steps (Union[float, Dict[str, Tuple[float, float]]]): Cross replace steps.
        tokenizer: Tokenizer to encode the prompts.
        max_num_words (int): Maximum number of words. Default is 77.
    
    Returns:
        torch.Tensor: Attention alpha values.
    """
    if type(cross_replace_steps) is not dict:
        cross_replace_steps = {"default_": cross_replace_steps}
    if "default_" not in cross_replace_steps:
        cross_replace_steps["default_"] = (0., 1.)
    alpha_time_words = torch.zeros(num_steps + 1, len(prompts) - 1, max_num_words)
    for i in range(len(prompts) - 1):
        alpha_time_words = update_alpha_time_word(alpha_time_words, cross_replace_steps["default_"],
                                                  i)
    for key, item in cross_replace_steps.items():
        if key != "default_":
             inds = [get_word_inds(prompts[i], key, tokenizer) for i in range(1, len(prompts))]
             for i, ind in enumerate(inds):
                 if len(ind) > 0:
                    alpha_time_words = update_alpha_time_word(alpha_time_words, item, i, ind)
    alpha_time_words = alpha_time_words.reshape(num_steps + 1, len(prompts) - 1, 1, 1, max_num_words)
    return alpha_time_words
