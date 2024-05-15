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


"""
wild mixture of
https://github.com/lucidrains/denoising-diffusion-pytorch/blob/7706bdfc6f527f58d33f84b7b522e61e6e3164b3/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py
https://github.com/openai/improved-diffusion/blob/e94489283bb876ac1477d5dd7709bbbd2d9902ce/improved_diffusion/gaussian_diffusion.py
https://github.com/CompVis/taming-transformers
-- merci
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR
from einops import rearrange, repeat
from contextlib import contextmanager
from functools import partial
from tqdm import tqdm
from torchvision.utils import make_grid
from pytorch_lightning.utilities.distributed import rank_zero_only

from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler

from src.p2p.p2p_ldm_utils import LocalMask, AttentionMask
from src.p2p.ptp_utils import register_attention_control_t2i


__conditioning_keys__ = {'concat': 'c_concat',
                         'crossattn': 'c_crossattn',
                         'adm': 'y'}


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def uniform_on_device(r1, r2, shape, device):
    """
    Generate a tensor with uniform distribution on a given device.
    
    Args:
        r1 (float): Lower bound of the uniform distribution.
        r2 (float): Upper bound of the uniform distribution.
        shape (tuple): Shape of the output tensor.
        device (torch.device): Device to place the tensor on.
        
    Returns:
        torch.Tensor: Tensor with uniform distribution between r1 and r2.
    """
    return (r1 - r2) * torch.rand(*shape, device=device) + r2


class DDPM(pl.LightningModule):
    """
    Denoising Diffusion Probabilistic Model (DDPM) with Gaussian diffusion in image space.
    
    Attributes:
        model (DiffusionWrapper): The diffusion model wrapped in a PyTorch Lightning module.
        ...
    """
    # classic DDPM with Gaussian diffusion, in image space
    def __init__(self,
                 unet_config,
                 timesteps=1000,
                 beta_schedule="linear",
                 loss_type="l2",
                 ckpt_path=None,
                 ignore_keys=[],
                 load_only_unet=False,
                 monitor="val/loss",
                 use_ema=True,
                 first_stage_key="image",
                 image_size=256,
                 channels=3,
                 log_every_t=100,
                 clip_denoised=True,
                 linear_start=1e-4,
                 linear_end=2e-2,
                 cosine_s=8e-3,
                 given_betas=None,
                 original_elbo_weight=0.,
                 embedding_reg_weight=0.,
                 unfreeze_model=False,
                 model_lr=0.,
                 v_posterior=0.,  # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
                 l_simple_weight=1.,
                 conditioning_key=None,
                 parameterization="eps",  # all assuming fixed variance schedules
                 scheduler_config=None,
                 use_positional_encodings=False,
                 learn_logvar=False,
                 logvar_init=0.,
                 ):
        """
        Initialize DDPM model.
        
        Args:
            unet_config (dict): Configuration for the UNet model.
            timesteps (int, optional): Number of diffusion timesteps. Defaults to 1000.
            beta_schedule (str, optional): Beta schedule type. Defaults to "linear".
            loss_type (str, optional): Type of loss function. Defaults to "l2".
            ckpt_path (str, optional): Path to the checkpoint file. Defaults to None.
            ignore_keys (list, optional): List of keys to ignore during loading. Defaults to an empty list.
            load_only_unet (bool, optional): Whether to load only the UNet weights. Defaults to False.
            monitor (str, optional): Metric to monitor for checkpointing. Defaults to "val/loss".
            use_ema (bool, optional): Whether to use Exponential Moving Average (EMA). Defaults to True.
            first_stage_key (str, optional): Key for the first stage input. Defaults to "image".
            image_size (int, optional): Size of the input image. Defaults to 256.
            channels (int, optional): Number of channels in the input image. Defaults to 3.
            log_every_t (int, optional): Interval for logging during training. Defaults to 100.
            clip_denoised (bool, optional): Whether to clip the denoised output. Defaults to True.
            linear_start (float, optional): Starting value for linear beta schedule. Defaults to 1e-4.
            linear_end (float, optional): Ending value for linear beta schedule. Defaults to 2e-2.
            cosine_s (float, optional): Smoothing factor for cosine beta schedule. Defaults to 8e-3.
            given_betas (torch.Tensor, optional): Predefined beta values. Defaults to None.
            original_elbo_weight (float, optional): Weight for the original Evidence Lower Bound (ELBO). Defaults to 0.
            embedding_reg_weight (float, optional): Weight for the embedding regularization. Defaults to 0.
            unfreeze_model (bool, optional): Whether to unfreeze the model for training. Defaults to False.
            model_lr (float, optional): Learning rate for the model. Defaults to 0.
            v_posterior (float, optional): Weight for choosing posterior variance. Defaults to 0.
            l_simple_weight (float, optional): Weight for the simple loss. Defaults to 1.
            conditioning_key (str, optional): Key for the conditioning input. Defaults to None.
            parameterization (str, optional): Parameterization type. Defaults to "eps".
            scheduler_config (dict, optional): Configuration for the learning rate scheduler. Defaults to None.
            use_positional_encodings (bool, optional): Whether to use positional encodings. Defaults to False.
            learn_logvar (bool, optional): Whether to learn the log variance. Defaults to False.
            logvar_init (float, optional): Initial value for the log variance. Defaults to 0.
        """
        super().__init__()
        assert parameterization in ["eps", "x0"], 'currently only supporting "eps" and "x0"'
        self.parameterization = parameterization
        print(f"{self.__class__.__name__}: Running in {self.parameterization}-prediction mode")
        self.cond_stage_model = None
        self.clip_denoised = clip_denoised
        self.log_every_t = log_every_t
        self.first_stage_key = first_stage_key
        self.image_size = image_size  # try conv?
        self.channels = channels
        self.use_positional_encodings = use_positional_encodings
        self.model = DiffusionWrapper(unet_config, conditioning_key)
        count_params(self.model, verbose=True)
        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self.model)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")

        self.use_scheduler = scheduler_config is not None
        if self.use_scheduler:
            self.scheduler_config = scheduler_config

        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        self.embedding_reg_weight = embedding_reg_weight

        self.unfreeze_model = unfreeze_model
        self.model_lr = model_lr

        if monitor is not None:
            self.monitor = monitor
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys, only_model=load_only_unet)

        self.register_schedule(given_betas=given_betas, beta_schedule=beta_schedule, timesteps=timesteps,
                               linear_start=linear_start, linear_end=linear_end, cosine_s=cosine_s)

        self.loss_type = loss_type

        self.learn_logvar = learn_logvar
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
        if self.learn_logvar:
            self.logvar = nn.Parameter(self.logvar, requires_grad=True)


    def register_schedule(self, given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        """
        Register the beta schedule for the diffusion process.
        
        Args:
            given_betas (torch.Tensor, optional): Predefined beta values. Defaults to None.
            beta_schedule (str, optional): Type of beta schedule. Defaults to "linear".
            timesteps (int, optional): Number of diffusion timesteps. Defaults to 1000.
            linear_start (float, optional): Starting value for linear beta schedule. Defaults to 1e-4.
            linear_end (float, optional): Ending value for linear beta schedule. Defaults to 2e-2.
            cosine_s (float, optional): Smoothing factor for cosine beta schedule. Defaults to 8e-3.
        """
        if exists(given_betas):
            betas = given_betas
        else:
            betas = make_beta_schedule(beta_schedule, timesteps, linear_start=linear_start, linear_end=linear_end,
                                       cosine_s=cosine_s)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.linear_start = linear_start
        self.linear_end = linear_end
        assert alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'

        to_torch = partial(torch.tensor, dtype=torch.float32)

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (1 - self.v_posterior) * betas * (1. - alphas_cumprod_prev) / (
                    1. - alphas_cumprod) + self.v_posterior * betas
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1', to_torch(
            betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2', to_torch(
            (1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        if self.parameterization == "eps":
            lvlb_weights = self.betas ** 2 / (
                        2 * self.posterior_variance * to_torch(alphas) * (1 - self.alphas_cumprod))
        elif self.parameterization == "x0":
            lvlb_weights = 0.5 * np.sqrt(torch.Tensor(alphas_cumprod)) / (2. * 1 - torch.Tensor(alphas_cumprod))
        else:
            raise NotImplementedError("mu not supported")
        # TODO how to choose this term
        lvlb_weights[0] = lvlb_weights[1]
        self.register_buffer('lvlb_weights', lvlb_weights, persistent=False)
        assert not torch.isnan(self.lvlb_weights).all()

    @contextmanager
    def ema_scope(self, context=None):
        """
        Context manager to switch to EMA weights and restore original weights after the context.
        
        Args:
            context (str, optional): Description of the context for logging. Defaults to None.
        """
        if self.use_ema:
            self.model_ema.store(self.model.parameters())
            self.model_ema.copy_to(self.model)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.model.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")

    def init_from_ckpt(self, path, ignore_keys=list(), only_model=False):
        """
        Initialize model weights from a checkpoint.
        
        Args:
            path (str): Path to the checkpoint file.
            ignore_keys (list, optional): List of keys to ignore during loading. Defaults to an empty list.
            only_model (bool, optional): Whether to load only the model weights. Defaults to False.
        """
        sd = torch.load(path, map_location="cpu")
        if "state_dict" in list(sd.keys()):
            sd = sd["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False) if not only_model else self.model.load_state_dict(
            sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
        if len(unexpected) > 0:
            print(f"Unexpected Keys: {unexpected}")

    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).
        
        Args:
            x_start (torch.Tensor): Tensor of noiseless inputs.
            t (torch.Tensor): Number of diffusion steps.
            
        Returns:
            tuple: Mean, variance, and log variance of the distribution.
        """
        mean = (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start)
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        """
        Predict x_0 from x_t and noise.
        
        Args:
            x_t (torch.Tensor): Noisy tensor at time step t.
            t (torch.Tensor): Number of diffusion steps.
            noise (torch.Tensor): Noise tensor.
            
        Returns:
            torch.Tensor: Predicted x_0.
        """
        return (
                extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        """
        Compute the posterior distribution q(x_{t-1} | x_t, x_0).
        
        Args:
            x_start (torch.Tensor): Tensor of noiseless inputs.
            x_t (torch.Tensor): Noisy tensor at time step t.
            t (torch.Tensor): Number of diffusion steps.
            
        Returns:
            tuple: Posterior mean, variance, and log variance.
        """
        posterior_mean = (
                extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool):
        """
        Compute the mean and variance of the model's output distribution.
        
        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Number of diffusion steps.
            clip_denoised (bool): Whether to clip the denoised output.
            
        Returns:
            tuple: Mean, variance, and log variance of the model's output distribution.
        """
        model_out = self.model(x, t)
        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
        """
        Sample from the model's output distribution.
        
        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Number of diffusion steps.
            clip_denoised (bool, optional): Whether to clip the denoised output. Defaults to True.
            repeat_noise (bool, optional): Whether to repeat noise. Defaults to False.
            
        Returns:
            torch.Tensor: Sampled tensor.
        """
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_loop(self, shape, return_intermediates=False):
        """
        Sample from the model by iterating through the diffusion process.
        
        Args:
            shape (tuple): Shape of the output tensor.
            return_intermediates (bool, optional): Whether to return intermediate steps. Defaults to False.
            
        Returns:
            torch.Tensor: Sampled tensor.
        """
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        intermediates = [img]
        for i in tqdm(reversed(range(0, self.num_timesteps)), desc='Sampling t', total=self.num_timesteps):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long),
                                clip_denoised=self.clip_denoised)
            if i % self.log_every_t == 0 or i == self.num_timesteps - 1:
                intermediates.append(img)
        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, batch_size=16, return_intermediates=False):
        """
        Generate samples from the model.
        
        Args:
            batch_size (int, optional): Number of samples to generate. Defaults to 16.
            return_intermediates (bool, optional): Whether to return intermediate steps. Defaults to False.
            
        Returns:
            torch.Tensor: Generated samples.
        """
        image_size = self.image_size
        channels = self.channels
        return self.p_sample_loop((batch_size, channels, image_size, image_size),
                                  return_intermediates=return_intermediates)

    def q_sample(self, x_start, t, noise=None):
        """
        Add noise to the input tensor x_start at time step t.
        
        Args:
            x_start (torch.Tensor): Tensor of noiseless inputs.
            t (torch.Tensor): Number of diffusion steps.
            noise (torch.Tensor, optional): Noise tensor. Defaults to None.
            
        Returns:
            torch.Tensor: Noisy tensor.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise)

    def get_loss(self, pred, target, mean=True):
        """
        Compute the loss between the predicted and target tensors.
        
        Args:
            pred (torch.Tensor): Predicted tensor.
            target (torch.Tensor): Target tensor.
            mean (bool, optional): Whether to compute the mean loss. Defaults to True.
            
        Returns:
            torch.Tensor: Computed loss.
        """
        if self.loss_type == 'l1':
            loss = (target - pred).abs()
            if mean:
                loss = loss.mean()
        elif self.loss_type == 'l2':
            if mean:
                loss = torch.nn.functional.mse_loss(target, pred)
            else:
                loss = torch.nn.functional.mse_loss(target, pred, reduction='none')
        else:
            raise NotImplementedError("unknown loss type '{loss_type}'")

        return loss

    def p_losses(self, x_start, t, noise=None):
        """
        Compute the loss for the denoising process.
        
        Args:
            x_start (torch.Tensor): Tensor of noiseless inputs.
            t (torch.Tensor): Number of diffusion steps.
            noise (torch.Tensor, optional): Noise tensor. Defaults to None.
            
        Returns:
            tuple: Loss value and loss dictionary.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.model(x_noisy, t)

        loss_dict = {}
        if self.parameterization == "eps":
            target = noise
        elif self.parameterization == "x0":
            target = x_start
        else:
            raise NotImplementedError(f"Paramterization {self.parameterization} not yet supported")

        loss = self.get_loss(model_out, target, mean=False).mean(dim=[1, 2, 3])

        log_prefix = 'train' if self.training else 'val'

        loss_dict.update({f'{log_prefix}/loss_simple': loss.mean()})
        loss_simple = loss.mean() * self.l_simple_weight

        loss_vlb = (self.lvlb_weights[t] * loss).mean()
        loss_dict.update({f'{log_prefix}/loss_vlb': loss_vlb})

        loss = loss_simple + self.original_elbo_weight * loss_vlb

        loss_dict.update({f'{log_prefix}/loss': loss})

        return loss, loss_dict

    def forward(self, x, *args, **kwargs):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        return self.p_losses(x, t, *args, **kwargs)

    def get_input(self, batch, k):
        """
        Get input tensor from the batch.
        
        Args:
            batch (dict): Input batch.
            k (str): Key to extract from the batch.
            
        Returns:
            torch.Tensor: Extracted input tensor.
        """
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()
        return x

    def shared_step(self, batch):
        """
        Shared step for training and validation.
        
        Args:
            batch (dict): Input batch.
            
        Returns:
            tuple: Loss value and loss dictionary.
        """
        x = self.get_input(batch, self.first_stage_key)
        loss, loss_dict = self(x)
        return loss, loss_dict

    def training_step(self, batch, batch_idx):
        """
        Training step for the model.
        
        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
            
        Returns:
            torch.Tensor: Loss value.
        """
        loss, loss_dict = self.shared_step(batch)

        self.log_dict(loss_dict, prog_bar=True,
                      logger=True, on_step=True, on_epoch=True)

        self.log("global_step", self.global_step,
                 prog_bar=True, logger=True, on_step=True, on_epoch=False)

        if self.use_scheduler:
            lr = self.optimizers().param_groups[0]['lr']
            self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        """
        Validation step for the model.
        
        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
        """
        _, loss_dict_no_ema = self.shared_step(batch)
        with self.ema_scope():
            _, loss_dict_ema = self.shared_step(batch)
            loss_dict_ema = {key + '_ema': loss_dict_ema[key] for key in loss_dict_ema}
        self.log_dict(loss_dict_no_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        self.log_dict(loss_dict_ema, prog_bar=False, logger=True, on_step=False, on_epoch=True)

    def on_train_batch_end(self, *args, **kwargs):
        """
        Hook to update EMA after each training batch.
        """
        if self.use_ema:
            self.model_ema(self.model)

    def _get_rows_from_list(self, samples):
        """
        Get rows of images from a list of samples.
        
        Args:
            samples (list): List of image samples.
            
        Returns:
            torch.Tensor: Grid of images.
        """
        n_imgs_per_row = len(samples)
        denoise_grid = rearrange(samples, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=2, sample=True, return_keys=None, **kwargs):
        """
        Log images during training and validation.
        
        Args:
            batch (dict): Input batch.
            N (int, optional): Number of images to log. Defaults to 8.
            n_row (int, optional): Number of images per row. Defaults to 2.
            sample (bool, optional): Whether to generate samples. Defaults to True.
            return_keys (list, optional): List of keys to return. Defaults to None.
            
        Returns:
            dict: Dictionary of logged images.
        """
        log = dict()
        x = self.get_input(batch, self.first_stage_key)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        x = x.to(self.device)[:N]
        log["inputs"] = x

        # get diffusion row
        diffusion_row = list()
        x_start = x[:n_row]

        for t in range(self.num_timesteps):
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(x_start)
                x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
                diffusion_row.append(x_noisy)

        log["diffusion_row"] = self._get_rows_from_list(diffusion_row)

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                samples, denoise_row = self.sample(batch_size=N, return_intermediates=True)

            log["samples"] = samples
            log["denoise_row"] = self._get_rows_from_list(denoise_row)

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        """
        Configure optimizers for the model.
        
        Returns:
            torch.optim.Optimizer: Optimizer for the model.
        """
        lr = self.learning_rate
        params = list(self.model.parameters())
        if self.learn_logvar:
            params = params + [self.logvar]
        opt = torch.optim.AdamW(params, lr=lr)
        return opt


class LatentDiffusion(DDPM):
    """
    Latent Diffusion Model extending DDPM for latent space diffusion.
    
    Attributes:
        num_timesteps_cond (int): Number of conditioning timesteps.
        ...
    """
    """main class"""
    def __init__(self,
                 first_stage_config,
                 cond_stage_config,
                 personalization_config,
                 num_timesteps_cond=None,
                 cond_stage_key="image",
                 cond_stage_trainable=False,
                 concat_mode=True,
                 cond_stage_forward=None,
                 conditioning_key=None,
                 scale_factor=1.0,
                 scale_by_std=False,
                 *args, **kwargs):
        """
        Initialize Latent Diffusion model.
        
        Args:
            first_stage_config (dict): Configuration for the first stage model.
            cond_stage_config (dict): Configuration for the conditioning stage model.
            personalization_config (dict): Configuration for personalization.
            num_timesteps_cond (int, optional): Number of conditioning timesteps. Defaults to None.
            cond_stage_key (str, optional): Key for the conditioning stage input. Defaults to "image".
            cond_stage_trainable (bool, optional): Whether the conditioning stage is trainable. Defaults to False.
            concat_mode (bool, optional): Whether to use concatenation for conditioning. Defaults to True.
            cond_stage_forward (str, optional): Forward method for the conditioning stage. Defaults to None.
            conditioning_key (str, optional): Key for the conditioning input. Defaults to None.
            scale_factor (float, optional): Scaling factor for the latent space. Defaults to 1.0.
            scale_by_std (bool, optional): Whether to scale by standard deviation. Defaults to False.
            args (tuple): Additional arguments.
            kwargs (dict): Additional keyword arguments.
        """

        self.num_timesteps_cond = default(num_timesteps_cond, 1)
        self.scale_by_std = scale_by_std
        assert self.num_timesteps_cond <= kwargs['timesteps']
        # for backwards compatibility after implementation of DiffusionWrapper
        if conditioning_key is None:
            conditioning_key = 'concat' if concat_mode else 'crossattn'
        if cond_stage_config == '__is_unconditional__':
            conditioning_key = None
        ckpt_path = kwargs.pop("ckpt_path", None)
        ignore_keys = kwargs.pop("ignore_keys", [])
        super().__init__(conditioning_key=conditioning_key, *args, **kwargs)
        self.concat_mode = concat_mode
        self.cond_stage_trainable = cond_stage_trainable
        self.cond_stage_key = cond_stage_key

        #AttentionMask
        self.controller = None
        # cl (infoNCE) loss
        self.cl_temperature = 0.07

        try:
            self.num_downs = len(first_stage_config.params.ddconfig.ch_mult) - 1
        except:
            self.num_downs = 0
        if not scale_by_std:
            self.scale_factor = scale_factor
        else:
            self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.instantiate_first_stage(first_stage_config)
        self.instantiate_cond_stage(cond_stage_config)

        self.cond_stage_forward = cond_stage_forward
        self.clip_denoised = False
        self.bbox_tokenizer = None  

        self.restarted_from_ckpt = False
        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys)
            self.restarted_from_ckpt = True


        if not self.unfreeze_model:
            self.cond_stage_model.eval()
            self.cond_stage_model.train = disabled_train
            for param in self.cond_stage_model.parameters():
                param.requires_grad = False

            self.model.eval()
            self.model.train = disabled_train
            for param in self.model.parameters():
                param.requires_grad = False
        
        self.personalization_config = personalization_config
        self.embedding_manager = self.instantiate_embedding_manager(personalization_config, self.cond_stage_model)

        for param in self.embedding_manager.embedding_parameters():
            param.requires_grad = True

    def make_cond_schedule(self, ):
        """
        Create a schedule for the conditioning timesteps.
        """
        self.cond_ids = torch.full(size=(self.num_timesteps,), fill_value=self.num_timesteps - 1, dtype=torch.long)
        ids = torch.round(torch.linspace(0, self.num_timesteps - 1, self.num_timesteps_cond)).long()
        self.cond_ids[:self.num_timesteps_cond] = ids

    @rank_zero_only
    @torch.no_grad()
    def on_train_batch_start(self, batch, batch_idx, dataloader_idx):
        """
        Hook to perform actions at the start of each training batch.
        
        Args:
            batch (dict): Input batch.
            batch_idx (int): Batch index.
            dataloader_idx (int): Dataloader index.
        """
        # only for very first batch
        if self.scale_by_std and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
            # set rescale weight to 1./std of encodings
            print("### USING STD-RESCALING ###")
            x = super().get_input(batch, self.first_stage_key)
            x = x.to(self.device)
            encoder_posterior = self.encode_first_stage(x)
            z = self.get_first_stage_encoding(encoder_posterior).detach()
            del self.scale_factor
            self.register_buffer('scale_factor', 1. / z.flatten().std())
            print(f"setting self.scale_factor to {self.scale_factor}")
            print("### USING STD-RESCALING ###")


    def register_schedule(self,
                          given_betas=None, beta_schedule="linear", timesteps=1000,
                          linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
        """
        Register the beta schedule for the diffusion process.
        
        Args:
            given_betas (torch.Tensor, optional): Predefined beta values. Defaults to None.
            beta_schedule (str, optional): Type of beta schedule. Defaults to "linear".
            timesteps (int, optional): Number of diffusion timesteps. Defaults to 1000.
            linear_start (float, optional): Starting value for linear beta schedule. Defaults to 1e-4.
            linear_end (float, optional): Ending value for linear beta schedule. Defaults to 2e-2.
            cosine_s (float, optional): Smoothing factor for cosine beta schedule. Defaults to 8e-3.
        """
        super().register_schedule(given_betas, beta_schedule, timesteps, linear_start, linear_end, cosine_s)

        self.shorten_cond_schedule = self.num_timesteps_cond > 1
        if self.shorten_cond_schedule:
            self.make_cond_schedule()

    def instantiate_first_stage(self, config):
        """
        Instantiate the first stage model.
        
        Args:
            config (dict): Configuration for the first stage model.
        """
        model = instantiate_from_config(config)
        self.first_stage_model = model.eval()
        self.first_stage_model.train = disabled_train
        for param in self.first_stage_model.parameters():
            param.requires_grad = False

    def instantiate_cond_stage(self, config):
        """
        Instantiate the conditioning stage model.
        
        Args:
            config (dict): Configuration for the conditioning stage model.
        """
        if not self.cond_stage_trainable:
            if config == "__is_first_stage__":
                print("Using first stage also as cond stage.")
                self.cond_stage_model = self.first_stage_model
            elif config == "__is_unconditional__":
                print(f"Training {self.__class__.__name__} as an unconditional model.")
                self.cond_stage_model = None
                # self.be_unconditional = True
            else:
                model = instantiate_from_config(config)
                self.cond_stage_model = model.eval()
                self.cond_stage_model.train = disabled_train
                for param in self.cond_stage_model.parameters():
                    param.requires_grad = False
        else:
            assert config != '__is_first_stage__'
            assert config != '__is_unconditional__'
            model = instantiate_from_config(config)
            self.cond_stage_model = model
            
    
    def instantiate_embedding_manager(self, config, embedder):
        """
        Instantiate the embedding manager.
        
        Args:
            config (dict): Configuration for the embedding manager.
            embedder (nn.Module): Embedding model.
            
        Returns:
            nn.Module: Embedding manager.
        """
        model = instantiate_from_config(config, embedder=embedder)

        if config.params.get("embedding_manager_ckpt", None): # do not load if missing OR empty string
            model.load(config.params.embedding_manager_ckpt)
        
        return model

    def _get_denoise_row_from_list(self, samples, desc='', force_no_decoder_quantization=False):
        """
        Get rows of denoised images from a list of samples.
        
        Args:
            samples (list): List of denoised image samples.
            desc (str, optional): Description for the progress bar. Defaults to an empty string.
            force_no_decoder_quantization (bool, optional): Whether to force no decoder quantization. Defaults to False.
            
        Returns:
            torch.Tensor: Grid of denoised images.
        """
        denoise_row = []
        for zd in tqdm(samples, desc=desc):
            denoise_row.append(self.decode_first_stage(zd.to(self.device),
                                                            force_not_quantize=force_no_decoder_quantization))
        n_imgs_per_row = len(denoise_row)
        denoise_row = torch.stack(denoise_row)  # n_log_step, n_row, C, H, W
        denoise_grid = rearrange(denoise_row, 'n b c h w -> b n c h w')
        denoise_grid = rearrange(denoise_grid, 'b n c h w -> (b n) c h w')
        denoise_grid = make_grid(denoise_grid, nrow=n_imgs_per_row)
        return denoise_grid

    def get_first_stage_encoding(self, encoder_posterior):
        """
        Get the first stage encoding from the encoder posterior.
        
        Args:
            encoder_posterior (torch.Tensor): Encoder posterior tensor.
            
        Returns:
            torch.Tensor: First stage encoding.
        """
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z

    def get_learned_conditioning(self, c):
        """
        Get the learned conditioning from the input.
        
        Args:
            c (torch.Tensor): Conditioning input tensor.
            
        Returns:
            torch.Tensor: Learned conditioning.
        """
        if self.cond_stage_forward is None:
            if hasattr(self.cond_stage_model, 'encode') and callable(self.cond_stage_model.encode):
                c = self.cond_stage_model.encode(c, embedding_manager=self.embedding_manager)
                if isinstance(c, DiagonalGaussianDistribution):
                    c = c.mode()
            else:
                c = self.cond_stage_model(c)
        else:
            assert hasattr(self.cond_stage_model, self.cond_stage_forward)
            c = getattr(self.cond_stage_model, self.cond_stage_forward)(c)
        return c

    def meshgrid(self, h, w):
        """
        Create a meshgrid of coordinates.
        
        Args:
            h (int): Height of the grid.
            w (int): Width of the grid.
            
        Returns:
            torch.Tensor: Meshgrid of coordinates.
        """
        y = torch.arange(0, h).view(h, 1, 1).repeat(1, w, 1)
        x = torch.arange(0, w).view(1, w, 1).repeat(h, 1, 1)

        arr = torch.cat([y, x], dim=-1)
        return arr

    def delta_border(self, h, w):
        """
        Compute the normalized distance to the image border.
        
        Args:
            h (int): Height of the image.
            w (int): Width of the image.
            
        Returns:
            torch.Tensor: Normalized distance to the image border.
        """
        lower_right_corner = torch.tensor([h - 1, w - 1]).view(1, 1, 2)
        arr = self.meshgrid(h, w) / lower_right_corner
        dist_left_up = torch.min(arr, dim=-1, keepdims=True)[0]
        dist_right_down = torch.min(1 - arr, dim=-1, keepdims=True)[0]
        edge_dist = torch.min(torch.cat([dist_left_up, dist_right_down], dim=-1), dim=-1)[0]
        return edge_dist

    def get_weighting(self, h, w, Ly, Lx, device):
        """
        Get the weighting for the image patches.
        
        Args:
            h (int): Height of the image.
            w (int): Width of the image.
            Ly (int): Height of the patch.
            Lx (int): Width of the patch.
            device (torch.device): Device to place the tensor on.
            
        Returns:
            torch.Tensor: Weighting tensor.
        """
        weighting = self.delta_border(h, w)
        weighting = torch.clip(weighting, self.split_input_params["clip_min_weight"],
                               self.split_input_params["clip_max_weight"], )
        weighting = weighting.view(1, h * w, 1).repeat(1, 1, Ly * Lx).to(device)

        if self.split_input_params["tie_braker"]:
            L_weighting = self.delta_border(Ly, Lx)
            L_weighting = torch.clip(L_weighting,
                                     self.split_input_params["clip_min_tie_weight"],
                                     self.split_input_params["clip_max_tie_weight"])

            L_weighting = L_weighting.view(1, 1, Ly * Lx).to(device)
            weighting = weighting * L_weighting
        return weighting

    def get_fold_unfold(self, x, kernel_size, stride, uf=1, df=1): 
        """
        Get the fold and unfold operations for the image patches.
        
        Args:
            x (torch.Tensor): Input tensor.
            kernel_size (tuple): Kernel size for the fold and unfold operations.
            stride (tuple): Stride for the fold and unfold operations.
            uf (int, optional): Upsampling factor. Defaults to 1.
            df (int, optional): Downsampling factor. Defaults to 1.
            
        Returns:
            tuple: Fold and unfold operations, normalization, and weighting.
        """
        bs, nc, h, w = x.shape

        # number of crops in image
        Ly = (h - kernel_size[0]) // stride[0] + 1
        Lx = (w - kernel_size[1]) // stride[1] + 1

        if uf == 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold = torch.nn.Fold(output_size=x.shape[2:], **fold_params)

            weighting = self.get_weighting(kernel_size[0], kernel_size[1], Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h, w)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0], kernel_size[1], Ly * Lx))

        elif uf > 1 and df == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] * uf, kernel_size[0] * uf),
                                dilation=1, padding=0,
                                stride=(stride[0] * uf, stride[1] * uf))
            fold = torch.nn.Fold(output_size=(x.shape[2] * uf, x.shape[3] * uf), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] * uf, kernel_size[1] * uf, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h * uf, w * uf)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] * uf, kernel_size[1] * uf, Ly * Lx))

        elif df > 1 and uf == 1:
            fold_params = dict(kernel_size=kernel_size, dilation=1, padding=0, stride=stride)
            unfold = torch.nn.Unfold(**fold_params)

            fold_params2 = dict(kernel_size=(kernel_size[0] // df, kernel_size[0] // df),
                                dilation=1, padding=0,
                                stride=(stride[0] // df, stride[1] // df))
            fold = torch.nn.Fold(output_size=(x.shape[2] // df, x.shape[3] // df), **fold_params2)

            weighting = self.get_weighting(kernel_size[0] // df, kernel_size[1] // df, Ly, Lx, x.device).to(x.dtype)
            normalization = fold(weighting).view(1, 1, h // df, w // df)  # normalizes the overlap
            weighting = weighting.view((1, 1, kernel_size[0] // df, kernel_size[1] // df, Ly * Lx))

        else:
            raise NotImplementedError

        return fold, unfold, normalization, weighting

    @torch.no_grad()
    def get_input(self, batch, k, return_first_stage_outputs=False, force_c_encode=False,
                  cond_key=None, return_original_cond=False, bs=None):
        """
        Get input tensor from the batch.
        
        Args:
            batch (dict): Input batch.
            k (str): Key to extract from the batch.
            return_first_stage_outputs (bool, optional): Whether to return first stage outputs. Defaults to False.
            force_c_encode (bool, optional): Whether to force conditioning encoding. Defaults to False.
            cond_key (str, optional): Key for conditioning input. Defaults to None.
            return_original_cond (bool, optional): Whether to return original conditioning. Defaults to False.
            bs (int, optional): Batch size. Defaults to None.
            
        Returns:
            list: List of input tensors.
        """
        x = super().get_input(batch, k)
        if bs is not None:
            x = x[:bs]
        x = x.to(self.device)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()

        if self.model.conditioning_key is not None:
            if cond_key is None:
                cond_key = self.cond_stage_key
            if cond_key != self.first_stage_key:
                if cond_key in ['caption', 'coordinates_bbox']:
                    xc = batch[cond_key]
                elif cond_key == 'class_label':
                    xc = batch
                else:
                    xc = super().get_input(batch, cond_key).to(self.device)
            else:
                xc = x
            if not self.cond_stage_trainable or force_c_encode:
                if isinstance(xc, dict) or isinstance(xc, list):
                    # import pudb; pudb.set_trace()
                    c = self.get_learned_conditioning(xc)
                else:
                    c = self.get_learned_conditioning(xc.to(self.device))
            else:
                c = xc
            if bs is not None:
                c = c[:bs]

            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                ckey = __conditioning_keys__[self.model.conditioning_key]
                c = {ckey: c, 'pos_x': pos_x, 'pos_y': pos_y}

        else:
            c = None
            xc = None
            if self.use_positional_encodings:
                pos_x, pos_y = self.compute_latent_shifts(batch)
                c = {'pos_x': pos_x, 'pos_y': pos_y}
        out = [z, c]
        if return_first_stage_outputs:
            xrec = self.decode_first_stage(z)
            out.extend([x, xrec])
        if return_original_cond:
            out.append(xc)
        return out

    @torch.no_grad()
    def decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        """
        Decode the first stage output from the latent space.
        
        Args:
            z (torch.Tensor): Latent tensor.
            predict_cids (bool, optional): Whether to predict codebook IDs. Defaults to False.
            force_not_quantize (bool, optional): Whether to force not quantize. Defaults to False.
            
        Returns:
            torch.Tensor: Decoded tensor.
        """
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:

                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    # same as above but without decorator
    def differentiable_decode_first_stage(self, z, predict_cids=False, force_not_quantize=False):
        """
        Differentiable decode of the first stage output from the latent space.
        
        Args:
            z (torch.Tensor): Latent tensor.
            predict_cids (bool, optional): Whether to predict codebook IDs. Defaults to False.
            force_not_quantize (bool, optional): Whether to force not quantize. Defaults to False.
            
        Returns:
            torch.Tensor: Decoded tensor.
        """
        if predict_cids:
            if z.dim() == 4:
                z = torch.argmax(z.exp(), dim=1).long()
            z = self.first_stage_model.quantize.get_codebook_entry(z, shape=None)
            z = rearrange(z, 'b h w c -> b c h w').contiguous()

        z = 1. / self.scale_factor * z

        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                uf = self.split_input_params["vqf"]
                bs, nc, h, w = z.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(z, ks, stride, uf=uf)

                z = unfold(z)  # (bn, nc * prod(**ks), L)
                # 1. Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                # 2. apply model loop over last dim
                if isinstance(self.first_stage_model, VQModelInterface):  
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i],
                                                                 force_not_quantize=predict_cids or force_not_quantize)
                                   for i in range(z.shape[-1])]
                else:
                    output_list = [self.first_stage_model.decode(z[:, :, :, :, i])
                                   for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)  # # (bn, nc, ks[0], ks[1], L)
                o = o * weighting
                # Reverse 1. reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization  # norm is shape (1, 1, h, w)
                return decoded
            else:
                if isinstance(self.first_stage_model, VQModelInterface):
                    return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
                else:
                    return self.first_stage_model.decode(z)

        else:
            if isinstance(self.first_stage_model, VQModelInterface):
                return self.first_stage_model.decode(z, force_not_quantize=predict_cids or force_not_quantize)
            else:
                return self.first_stage_model.decode(z)

    @torch.no_grad()
    def encode_first_stage(self, x):
        """
        Encode the input tensor using the first stage model.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Encoded tensor.
        """
        if hasattr(self, "split_input_params"):
            if self.split_input_params["patch_distributed_vq"]:
                ks = self.split_input_params["ks"]  # eg. (128, 128)
                stride = self.split_input_params["stride"]  # eg. (64, 64)
                df = self.split_input_params["vqf"]
                self.split_input_params['original_image_size'] = x.shape[-2:]
                bs, nc, h, w = x.shape
                if ks[0] > h or ks[1] > w:
                    ks = (min(ks[0], h), min(ks[1], w))
                    print("reducing Kernel")

                if stride[0] > h or stride[1] > w:
                    stride = (min(stride[0], h), min(stride[1], w))
                    print("reducing stride")

                fold, unfold, normalization, weighting = self.get_fold_unfold(x, ks, stride, df=df)
                z = unfold(x)  # (bn, nc * prod(**ks), L)
                # Reshape to img shape
                z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )

                output_list = [self.first_stage_model.encode(z[:, :, :, :, i])
                               for i in range(z.shape[-1])]

                o = torch.stack(output_list, axis=-1)
                o = o * weighting

                # Reverse reshape to img shape
                o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
                # stitch crops together
                decoded = fold(o)
                decoded = decoded / normalization
                return decoded

            else:
                return self.first_stage_model.encode(x)
        else:
            return self.first_stage_model.encode(x)

    def shared_step(self, batch, **kwargs):
        """
        Shared step for training and validation.
        
        Args:
            batch (dict): Input batch.
            
        Returns:
            torch.Tensor: Loss value.
        """
        x, c = self.get_input(batch, self.first_stage_key)
        loss = self(x, c)
        return loss

    def forward(self, x, c, *args, **kwargs):
        """
        Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Conditioning tensor.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        # MCPL-dynamic
        if not 'RELATE' in self.personalization_config.params.placeholder_strings:
            c = [prompts.replace('*', ' '.join(self.personalization_config.params.placeholder_strings)) for prompts in c]

        # AttentionMask
        if self.controller is not None:
            tokenizer = self.cond_stage_model.tknz_fn.tokenizer
            prompts = c
            keywords = [self.personalization_config.params.attn_words for _ in range(len(prompts))]
            lb = LocalMask(tokenizer, prompts, keywords, \
                    self.personalization_config.params.attn_mask_type, \
                    self.personalization_config.params.presudo_words_softmax, \
                    self.personalization_config.params.presudo_words_infonce, \
                    self.personalization_config.params.adj_aug_infonce, \
                    self.personalization_config.params.n_gpu)
            # update local_blend at each step
            self.controller.local_blend = lb
            print('AttentionMask-forward-re-register: update attn layer counts each forward ...')
            register_attention_control_t2i(self, self.controller)
            self.controller.reset()
            
        t = torch.randint(0, self.num_timesteps, (x.shape[0],), device=self.device).long()
        if self.model.conditioning_key is not None:
            assert c is not None
            if self.cond_stage_trainable:
                c = self.get_learned_conditioning(c)
            if self.shorten_cond_schedule:  
                tc = self.cond_ids[t].to(self.device)
                c = self.q_sample(x_start=c, t=tc, noise=torch.randn_like(c.float()))

        return self.p_losses(x, c, t, *args, **kwargs)

    def _rescale_annotations(self, bboxes, crop_coordinates):  
        """
        Rescale bounding box annotations based on crop coordinates.
        
        Args:
            bboxes (list): List of bounding boxes.
            crop_coordinates (tuple): Crop coordinates.
            
        Returns:
            list: Rescaled bounding boxes.
        """
        def rescale_bbox(bbox):
            x0 = clamp((bbox[0] - crop_coordinates[0]) / crop_coordinates[2])
            y0 = clamp((bbox[1] - crop_coordinates[1]) / crop_coordinates[3])
            w = min(bbox[2] / crop_coordinates[2], 1 - x0)
            h = min(bbox[3] / crop_coordinates[3], 1 - y0)
            return x0, y0, w, h

        return [rescale_bbox(b) for b in bboxes]

    def apply_model(self, x_noisy, t, cond, return_ids=False):
        """
        Apply the model to the noisy input.
        
        Args:
            x_noisy (torch.Tensor): Noisy input tensor.
            t (torch.Tensor): Number of diffusion steps.
            cond (torch.Tensor): Conditioning tensor.
            return_ids (bool, optional): Whether to return IDs. Defaults to False.
            
        Returns:
            torch.Tensor: Output tensor.
        """

        if isinstance(cond, dict):
            # hybrid case, cond is exptected to be a dict
            pass
        else:
            if not isinstance(cond, list):
                cond = [cond]
            key = 'c_concat' if self.model.conditioning_key == 'concat' else 'c_crossattn'
            cond = {key: cond}

        if hasattr(self, "split_input_params"):
            assert len(cond) == 1  
            assert not return_ids  
            ks = self.split_input_params["ks"]  # eg. (128, 128)
            stride = self.split_input_params["stride"]  # eg. (64, 64)
            h, w = x_noisy.shape[-2:]
            fold, unfold, normalization, weighting = self.get_fold_unfold(x_noisy, ks, stride)
            z = unfold(x_noisy)  # (bn, nc * prod(**ks), L)
            # Reshape to img shape
            z = z.view((z.shape[0], -1, ks[0], ks[1], z.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
            z_list = [z[:, :, :, :, i] for i in range(z.shape[-1])]

            if self.cond_stage_key in ["image", "LR_image", "segmentation",
                                       'bbox_img'] and self.model.conditioning_key:  # todo check for completeness
                c_key = next(iter(cond.keys()))  # get key
                c = next(iter(cond.values()))  # get value
                assert (len(c) == 1)  # todo extend to list with more than one elem
                c = c[0]  # get element
                c = unfold(c)
                c = c.view((c.shape[0], -1, ks[0], ks[1], c.shape[-1]))  # (bn, nc, ks[0], ks[1], L )
                cond_list = [{c_key: [c[:, :, :, :, i]]} for i in range(c.shape[-1])]
            elif self.cond_stage_key == 'coordinates_bbox':
                assert 'original_image_size' in self.split_input_params, 'BoudingBoxRescaling is missing original_image_size'
                n_patches_per_row = int((w - ks[0]) / stride[0] + 1)
                full_img_h, full_img_w = self.split_input_params['original_image_size']
                num_downs = self.first_stage_model.encoder.num_resolutions - 1
                rescale_latent = 2 ** (num_downs)
                tl_patch_coordinates = [(rescale_latent * stride[0] * (patch_nr % n_patches_per_row) / full_img_w,
                                         rescale_latent * stride[1] * (patch_nr // n_patches_per_row) / full_img_h)
                                        for patch_nr in range(z.shape[-1])]
                patch_limits = [(x_tl, y_tl,
                                 rescale_latent * ks[0] / full_img_w,
                                 rescale_latent * ks[1] / full_img_h) for x_tl, y_tl in tl_patch_coordinates]
                # tokenize crop coordinates for the bounding boxes of the respective patches
                patch_limits_tknzd = [torch.LongTensor(self.bbox_tokenizer._crop_encoder(bbox))[None].to(self.device)
                                      for bbox in patch_limits]  # list of length l with tensors of shape (1, 2)
                # cut tknzd crop position from conditioning
                assert isinstance(cond, dict), 'cond must be dict to be fed into model'
                cut_cond = cond['c_crossattn'][0][..., :-2].to(self.device)
                adapted_cond = torch.stack([torch.cat([cut_cond, p], dim=1) for p in patch_limits_tknzd])
                adapted_cond = rearrange(adapted_cond, 'l b n -> (l b) n')
                adapted_cond = self.get_learned_conditioning(adapted_cond)
                adapted_cond = rearrange(adapted_cond, '(l b) n d -> l b n d', l=z.shape[-1])
                cond_list = [{'c_crossattn': [e]} for e in adapted_cond]
            else:
                cond_list = [cond for i in range(z.shape[-1])]  # Todo make this more efficient
            # apply model by loop over crops
            output_list = [self.model(z_list[i], t, **cond_list[i]) for i in range(z.shape[-1])]
            assert not isinstance(output_list[0],
                                  tuple)  # todo cant deal with multiple model outputs check this never happens
            o = torch.stack(output_list, axis=-1)
            o = o * weighting
            # Reverse reshape to img shape
            o = o.view((o.shape[0], -1, o.shape[-1]))  # (bn, nc * ks[0] * ks[1], L)
            # stitch crops together
            x_recon = fold(o) / normalization
        else:
            x_recon = self.model(x_noisy, t, **cond)

        if isinstance(x_recon, tuple) and not return_ids:
            return x_recon[0]
        else:
            return x_recon

    def _predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - pred_xstart) / \
               extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def _prior_bpd(self, x_start):
        """
        Get the prior KL term for the variational lower-bound in bits-per-dim.
        
        Args:
            x_start (torch.Tensor): Noiseless input tensor.
            
        Returns:
            torch.Tensor: KL term in bits-per-dim.
        """
        batch_size = x_start.shape[0]
        t = torch.tensor([self.num_timesteps - 1] * batch_size, device=x_start.device)
        qt_mean, _, qt_log_variance = self.q_mean_variance(x_start, t)
        kl_prior = normal_kl(mean1=qt_mean, logvar1=qt_log_variance, mean2=0.0, logvar2=0.0)
        return mean_flat(kl_prior) / np.log(2.0)

    def p_losses(self, x_start, cond, t, noise=None):
        """
        Compute the loss for the denoising process.
        
        Args:
            x_start (torch.Tensor): Noiseless input tensor.
            cond (torch.Tensor): Conditioning tensor.
            t (torch.Tensor): Number of diffusion steps.
            noise (torch.Tensor, optional): Noise tensor. Defaults to None.
            
        Returns:
            tuple: Loss value and loss dictionary.
        """
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        loss_dict = {}
        prefix = 'train' if self.training else 'val'

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        else:
            raise NotImplementedError()

        # if AttentionMask call step_callback with local_blend
        # to mask both model_output and target, s.t. we focus only at masked regions
        print('AttentionMask: masking both model_output and target ...')
        if self.controller is not None:
            model_output = self.controller.step_callback(model_output)
            target = self.controller.step_callback(target)
        
        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/loss_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        # loss = loss_simple / torch.exp(self.logvar) + self.logvar
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/loss_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_vlb = self.get_loss(model_output, target, mean=False).mean(dim=(1, 2, 3))
        loss_vlb = (self.lvlb_weights[t] * loss_vlb).mean()
        loss_dict.update({f'{prefix}/loss_vlb': loss_vlb})
        loss += (self.original_elbo_weight * loss_vlb)

        # PromptCL
        if self.controller is not None and self.controller.local_blend.presudo_words_infonce[0] != '':
            if 'RELATE' in self.personalization_config.params.placeholder_strings:
                nll = []
                # with 'RELATE' each image may have different presudo words, hence calculate per image CL loss then average
                # CJ : 1) cond [B,N=77,D=1280] is embedded_text, calculate CL over specified presudo_words: presudo_words_infonce
                # CJ : 1.1) select specified presudo_words (e.g. specify [*,@]) then n=2, thus cond [B,77,1280] -> cond_presudo_words [B,2,1280]
                B, N, D = cond.shape
                mask_infonce = self.controller.local_blend.alpha_infonce != 0
                num_selection_per_img = mask_infonce.sum(1)[:,0]
                cond_presudo_words_select = cond.masked_select(mask_infonce) 
                cond_presudo_words_batch = []
                start, end = 0, 0
                if num_selection_per_img[0] == 0:
                    return loss, loss_dict

                # optional use adj as additional augmented positive examples to constrain ROI and correlation
                if self.controller.local_blend.adj_aug_infonce[0] != '':
                    mask_adj_aug = self.controller.local_blend.alpha_adj_aug != 0
                    # skip 1st iter in RELATE training
                    if mask_adj_aug.sum(1)[0,0] != 0:
                        cond_adj_aug = cond.masked_select(mask_adj_aug) 

                for num in num_selection_per_img:
                    end += num * D
                    if self.controller.local_blend.adj_aug_infonce[0] != '' and mask_adj_aug.sum(1)[0,0] != 0:
                        cond_presudo_words_batch.append(torch.cat(
                                [cond_presudo_words_select[start:end].view(num,D).unsqueeze(0),cond_adj_aug[start:end].view(num,D).unsqueeze(0)],
                            dim=0))
                            # note with 'RELATE' selected cond_adj_aug will be in the same order as cond_presudo_words at per image level 
                            # although they are not set in 1-2-1 pair in presudo_words_infonce and adj_aug_infonce
                            # because the mask search at per image-prompt level
                    else:
                        cond_presudo_words_batch.append(cond_presudo_words_select[start:end].view(num,D).unsqueeze(0))
                    start += num * D
                
                self.cl_temperature = self.personalization_config.params.infonce_temperature
                for b in range(B):
                    cond_presudo_words = cond_presudo_words_batch[b] # (1,-1,D) or (2,-1,D) if mask_adj_aug
                    # CJ : 1.2) calculate InfoNCE loss
                    B, n_presudo_words, D = cond_presudo_words.shape
                    # reshape to [B*2, 1280] here B equvelent to num of augmentation
                    # thus each presudo word has B positive examples, and B * (n_presudo_words-1) negative examples
                    feats = cond_presudo_words.view(-1,D)
                    # Calculate cosine similarity
                    cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
                    # Mask out cosine similarity to itself
                    self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
                    cos_sim.masked_fill_(self_mask, -9e15)
                    # Find positive example -> n_presudo_words away from the original example
                    pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // n_presudo_words, dims=0)
                    # InfoNCE loss
                    cos_sim = cos_sim / self.cl_temperature
                    nll_single = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
                    nll.append(nll_single.mean())
                nll = torch.tensor(nll).mean()
            else:
                # when not using 'RELATE' each image have same presudo words, hence calculate batch CL directly
                # CJ : 1) cond [B,N=77,D=1280] is embedded_text, calculate CL over specified presudo_words: presudo_words_infonce
                # CJ : 1.1) select specified presudo_words (e.g. specify [*,@]) then n=2, thus cond [B,77,1280] -> cond_presudo_words [B,2,1280]
                B, N, D = cond.shape
                mask_infonce = self.controller.local_blend.alpha_infonce != 0
                cond_presudo_words = cond.masked_select(mask_infonce) 
                cond_presudo_words = cond_presudo_words.view(B,-1,D)
                # optional use adj as additional augmented positive examples to constrain ROI and correlation
                if self.controller.local_blend.adj_aug_infonce[0] != '':
                    mask_adj_aug = self.controller.local_blend.alpha_adj_aug != 0
                    # skip 1st iter in RELATE training
                    if mask_adj_aug.sum(1)[0,0] != 0:
                        cond_adj_aug = cond.masked_select(mask_adj_aug) 
                        cond_adj_aug = cond_adj_aug.view(B,-1,D)
                        cond_presudo_words = torch.cat([cond_presudo_words,cond_adj_aug],dim=0)
                # CJ : 1.2) calculate InfoNCE loss
                B, n_presudo_words, D = cond_presudo_words.shape
                # reshape to [B*2, 1280] here B equvelent to num of augmentation
                # thus each presudo word has B positive examples, and B * (n_presudo_words-1) negative examples
                feats = cond_presudo_words.view(-1,D)
                # Calculate cosine similarity
                cos_sim = F.cosine_similarity(feats[:, None, :], feats[None, :, :], dim=-1)
                # Mask out cosine similarity to itself
                self_mask = torch.eye(cos_sim.shape[0], dtype=torch.bool, device=cos_sim.device)
                cos_sim.masked_fill_(self_mask, -9e15)
                # Find positive example -> n_presudo_words away from the original example
                pos_mask = self_mask.roll(shifts=cos_sim.shape[0] // n_presudo_words, dims=0)
                # InfoNCE loss
                self.cl_temperature = self.personalization_config.params.infonce_temperature
                cos_sim = cos_sim / self.cl_temperature
                nll = -cos_sim[pos_mask] + torch.logsumexp(cos_sim, dim=-1)
                nll = nll.mean()
            nll *= self.personalization_config.params.infonce_scale
            # CJ : 2) add InfoNCE loss 
            loss += nll

        loss_dict.update({f'{prefix}/loss': loss})

        if self.embedding_reg_weight > 0:
            loss_embedding_reg = self.embedding_manager.embedding_to_coarse_loss().mean()

            loss_dict.update({f'{prefix}/loss_emb_reg': loss_embedding_reg})

            loss += (self.embedding_reg_weight * loss_embedding_reg)
            loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict

    def p_mean_variance(self, x, c, t, clip_denoised: bool, return_codebook_ids=False, quantize_denoised=False,
                        return_x0=False, score_corrector=None, corrector_kwargs=None):
        """
        Compute the mean and variance of the model's output distribution.
        
        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Conditioning tensor.
            t (torch.Tensor): Number of diffusion steps.
            clip_denoised (bool, optional): Whether to clip the denoised output. Defaults to False.
            return_codebook_ids (bool, optional): Whether to return codebook IDs. Defaults to False.
            quantize_denoised (bool, optional): Whether to quantize the denoised output. Defaults to False.
            return_x0 (bool, optional): Whether to return the denoised output. Defaults to False.
            score_corrector (callable, optional): Score corrector function. Defaults to None.
            corrector_kwargs (dict, optional): Keyword arguments for the score corrector. Defaults to None.
            
        Returns:
            tuple: Mean, variance, log variance, and optional denoised output or codebook IDs.
        """
        t_in = t
        model_out = self.apply_model(x, t_in, c, return_ids=return_codebook_ids)

        if score_corrector is not None:
            assert self.parameterization == "eps"
            model_out = score_corrector.modify_score(self, model_out, x, t, c, **corrector_kwargs)

        if return_codebook_ids:
            model_out, logits = model_out

        if self.parameterization == "eps":
            x_recon = self.predict_start_from_noise(x, t=t, noise=model_out)
        elif self.parameterization == "x0":
            x_recon = model_out
        else:
            raise NotImplementedError()

        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        if quantize_denoised:
            x_recon, _, [_, _, indices] = self.first_stage_model.quantize(x_recon)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        if return_codebook_ids:
            return model_mean, posterior_variance, posterior_log_variance, logits
        elif return_x0:
            return model_mean, posterior_variance, posterior_log_variance, x_recon
        else:
            return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(self, x, c, t, clip_denoised=False, repeat_noise=False,
                 return_codebook_ids=False, quantize_denoised=False, return_x0=False,
                 temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None):
        """
        Perform a single reverse diffusion step.
        
        Args:
            x (torch.Tensor): Input tensor.
            c (torch.Tensor): Conditioning tensor.
            t (torch.Tensor): Number of diffusion steps.
            clip_denoised (bool, optional): Whether to clip the denoised output. Defaults to False.
            repeat_noise (bool, optional): Whether to repeat noise. Defaults to False.
            return_codebook_ids (bool, optional): Whether to return codebook IDs. Defaults to False.
            quantize_denoised (bool, optional): Whether to quantize the denoised output. Defaults to False.
            return_x0 (bool, optional): Whether to return the denoised output. Defaults to False.
            temperature (float, optional): Temperature for sampling. Defaults to 1.
            noise_dropout (float, optional): Dropout rate for noise. Defaults to 0.
            score_corrector (callable, optional): Score corrector function. Defaults to None.
            corrector_kwargs (dict, optional): Keyword arguments for the score corrector. Defaults to None.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        b, *_, device = *x.shape, x.device
        outputs = self.p_mean_variance(x=x, c=c, t=t, clip_denoised=clip_denoised,
                                       return_codebook_ids=return_codebook_ids,
                                       quantize_denoised=quantize_denoised,
                                       return_x0=return_x0,
                                       score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
        if return_codebook_ids:
            raise DeprecationWarning("Support dropped.")
            model_mean, _, model_log_variance, logits = outputs
        elif return_x0:
            model_mean, _, model_log_variance, x0 = outputs
        else:
            model_mean, _, model_log_variance = outputs

        noise = noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        if return_codebook_ids:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, logits.argmax(dim=1)
        if return_x0:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise, x0
        else:
            return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def progressive_denoising(self, cond, shape, verbose=True, callback=None, quantize_denoised=False,
                              img_callback=None, mask=None, x0=None, temperature=1., noise_dropout=0.,
                              score_corrector=None, corrector_kwargs=None, batch_size=None, x_T=None, start_T=None,
                              log_every_t=None):
        """
        Perform progressive denoising.
        
        Args:
            cond (torch.Tensor): Conditioning tensor.
            shape (tuple): Shape of the output tensor.
            verbose (bool, optional): Whether to display progress. Defaults to True.
            callback (callable, optional): Callback function for each timestep. Defaults to None.
            quantize_denoised (bool, optional): Whether to quantize the denoised output. Defaults to False.
            img_callback (callable, optional): Callback function for the image. Defaults to None.
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.
            x0 (torch.Tensor, optional): Initial tensor. Defaults to None.
            temperature (float, optional): Temperature for sampling. Defaults to 1.
            noise_dropout (float, optional): Dropout rate for noise. Defaults to 0.
            score_corrector (callable, optional): Score corrector function. Defaults to None.
            corrector_kwargs (dict, optional): Keyword arguments for the score corrector. Defaults to None.
            batch_size (int, optional): Batch size. Defaults to None.
            x_T (torch.Tensor, optional): Initial noise tensor. Defaults to None.
            start_T (int, optional): Starting timestep. Defaults to None.
            log_every_t (int, optional): Logging frequency. Defaults to None.
            
        Returns:
            tuple: Final tensor and intermediate tensors.
        """
        if not log_every_t:
            log_every_t = self.log_every_t
        timesteps = self.num_timesteps
        if batch_size is not None:
            b = batch_size if batch_size is not None else shape[0]
            shape = [batch_size] + list(shape)
        else:
            b = batch_size = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=self.device)
        else:
            img = x_T
        intermediates = []
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Progressive Generation',
                        total=timesteps) if verbose else reversed(
            range(0, timesteps))
        if type(temperature) == float:
            temperature = [temperature] * timesteps

        for i in iterator:
            ts = torch.full((b,), i, device=self.device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img, x0_partial = self.p_sample(img, cond, ts,
                                            clip_denoised=self.clip_denoised,
                                            quantize_denoised=quantize_denoised, return_x0=True,
                                            temperature=temperature[i], noise_dropout=noise_dropout,
                                            score_corrector=score_corrector, corrector_kwargs=corrector_kwargs)
            if mask is not None:
                assert x0 is not None
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            # AttentionMask step_callback with local_blend
            # img is latents during training
            # print('DEBUG ARE WE IN progressive_denoising?')
            # if self.controller is not None:
            #     img = self.controller.step_callback(img)
            
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(x0_partial)
            if callback: callback(i)
            if img_callback: img_callback(img, i)
        return img, intermediates

    @torch.no_grad()
    def p_sample_loop(self, cond, shape, return_intermediates=False,
                      x_T=None, verbose=True, callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, start_T=None,
                      log_every_t=None):
        """
        Perform a loop of reverse diffusion steps.
        
        Args:
            cond (torch.Tensor): Conditioning tensor.
            shape (tuple): Shape of the output tensor.
            return_intermediates (bool, optional): Whether to return intermediate tensors. Defaults to False.
            x_T (torch.Tensor, optional): Initial noise tensor. Defaults to None.
            verbose (bool, optional): Whether to display progress. Defaults to True.
            callback (callable, optional): Callback function for each timestep. Defaults to None.
            timesteps (int, optional): Number of timesteps. Defaults to None.
            quantize_denoised (bool, optional): Whether to quantize the denoised output. Defaults to False.
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.
            x0 (torch.Tensor, optional): Initial tensor. Defaults to None.
            img_callback (callable, optional): Callback function for the image. Defaults to None.
            start_T (int, optional): Starting timestep. Defaults to None.
            log_every_t (int, optional): Logging frequency. Defaults to None.
            
        Returns:
            torch.Tensor: Final tensor.
        """

        if not log_every_t:
            log_every_t = self.log_every_t
        device = self.betas.device
        b = shape[0]
        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        intermediates = [img]
        if timesteps is None:
            timesteps = self.num_timesteps

        if start_T is not None:
            timesteps = min(timesteps, start_T)
        iterator = tqdm(reversed(range(0, timesteps)), desc='Sampling t', total=timesteps) if verbose else reversed(
            range(0, timesteps))

        if mask is not None:
            assert x0 is not None
            assert x0.shape[2:3] == mask.shape[2:3]  # spatial size has to match

        for i in iterator:
            ts = torch.full((b,), i, device=device, dtype=torch.long)
            if self.shorten_cond_schedule:
                assert self.model.conditioning_key != 'hybrid'
                tc = self.cond_ids[ts].to(cond.device)
                cond = self.q_sample(x_start=cond, t=tc, noise=torch.randn_like(cond))

            img = self.p_sample(img, cond, ts,
                                clip_denoised=self.clip_denoised,
                                quantize_denoised=quantize_denoised)
            if mask is not None:
                img_orig = self.q_sample(x0, ts)
                img = img_orig * mask + (1. - mask) * img

            # AttentionMask step_callback with local_blend
            # img is latents
            print('DEBUG ARE WE IN p_sample_loop?')
            if self.controller is not None:
                img = self.controller.step_callback(img)
            
            if i % log_every_t == 0 or i == timesteps - 1:
                intermediates.append(img)
            if callback: callback(i)
            if img_callback: img_callback(img, i)

        if return_intermediates:
            return img, intermediates
        return img

    @torch.no_grad()
    def sample(self, cond, batch_size=16, return_intermediates=False, x_T=None,
               verbose=True, timesteps=None, quantize_denoised=False,
               mask=None, x0=None, shape=None,**kwargs):
        """
        Sample from the model.
        
        Args:
            cond (torch.Tensor): Conditioning tensor.
            batch_size (int, optional): Batch size. Defaults to 16.
            return_intermediates (bool, optional): Whether to return intermediate tensors. Defaults to False.
            x_T (torch.Tensor, optional): Initial noise tensor. Defaults to None.
            verbose (bool, optional): Whether to display progress. Defaults to True.
            timesteps (int, optional): Number of timesteps. Defaults to None.
            quantize_denoised (bool, optional): Whether to quantize the denoised output. Defaults to False.
            mask (torch.Tensor, optional): Mask tensor. Defaults to None.
            x0 (torch.Tensor, optional): Initial tensor. Defaults to None.
            shape (tuple, optional): Shape of the output tensor. Defaults to None.
            
        Returns:
            torch.Tensor: Final tensor.
        """
        if shape is None:
            shape = (batch_size, self.channels, self.image_size, self.image_size)
        if cond is not None:
            if isinstance(cond, dict):
                cond = {key: cond[key][:batch_size] if not isinstance(cond[key], list) else
                list(map(lambda x: x[:batch_size], cond[key])) for key in cond}
            else:
                cond = [c[:batch_size] for c in cond] if isinstance(cond, list) else cond[:batch_size]
        return self.p_sample_loop(cond,
                                  shape,
                                  return_intermediates=return_intermediates, x_T=x_T,
                                  verbose=verbose, timesteps=timesteps, quantize_denoised=quantize_denoised,
                                  mask=mask, x0=x0)

    @torch.no_grad()
    def sample_log(self,cond,batch_size,ddim, ddim_steps,**kwargs):
        """
        Sample from the model and log the samples.
        
        Args:
            cond (torch.Tensor): Conditioning tensor.
            batch_size (int): Batch size.
            ddim (bool): Whether to use DDIM sampling.
            ddim_steps (int): Number of DDIM steps.
            
        Returns:
            tuple: Samples and intermediates.
        """

        if ddim:
            ddim_sampler = DDIMSampler(self)
            shape = (self.channels, self.image_size, self.image_size)
            samples, intermediates =ddim_sampler.sample(ddim_steps,batch_size,
                                                        shape,cond,verbose=False,**kwargs)

        else:
            samples, intermediates = self.sample(cond=cond, batch_size=batch_size,
                                                 return_intermediates=True,**kwargs)

        return samples, intermediates

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4, sample=True, ddim_steps=200, ddim_eta=1., return_keys=None,
                   quantize_denoised=True, inpaint=False, plot_denoise_rows=False, plot_progressive_rows=False,
                   plot_diffusion_rows=False, **kwargs):
        """
        Log images during training or validation.
        
        Args:
            batch (dict): Input batch.
            N (int, optional): Number of images to log. Defaults to 8.
            n_row (int, optional): Number of rows in the log. Defaults to 4.
            sample (bool, optional): Whether to sample new images. Defaults to True.
            ddim_steps (int, optional): Number of DDIM steps. Defaults to 200.
            ddim_eta (float, optional): Eta parameter for DDIM. Defaults to 1.
            return_keys (list, optional): Keys to return. Defaults to None.
            quantize_denoised (bool, optional): Whether to quantize the denoised output. Defaults to True.
            inpaint (bool, optional): Whether to inpaint. Defaults to False.
            plot_denoise_rows (bool, optional): Whether to plot denoise rows. Defaults to False.
            plot_progressive_rows (bool, optional): Whether to plot progressive rows. Defaults to False.
            plot_diffusion_rows (bool, optional): Whether to plot diffusion rows. Defaults to False.
            
        Returns:
            dict: Log dictionary.
        """

        use_ddim = ddim_steps is not None

        log = dict()
        z, c, x, xrec, xc = self.get_input(batch, self.first_stage_key,
                                           return_first_stage_outputs=True,
                                           force_c_encode=True,
                                           return_original_cond=True,
                                           bs=N)
        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)
        log["inputs"] = x
        log["reconstruction"] = xrec
        if self.model.conditioning_key is not None:
            if hasattr(self.cond_stage_model, "decode"):
                xc = self.cond_stage_model.decode(c)
                log["conditioning"] = xc
            elif self.cond_stage_key in ["caption"]:
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
                log["conditioning"] = xc
            elif self.cond_stage_key == 'class_label':
                xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
                log['conditioning'] = xc
            elif isimage(xc):
                log["conditioning"] = xc
            if ismap(xc):
                log["original_conditioning"] = self.to_rgb(xc)

        if plot_diffusion_rows:
            # get diffusion row
            diffusion_row = list()
            z_start = z[:n_row]
            for t in range(self.num_timesteps):
                if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                    t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                    t = t.to(self.device).long()
                    noise = torch.randn_like(z_start)
                    z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                    diffusion_row.append(self.decode_first_stage(z_noisy))

            diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
            diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
            diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
            diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
            log["diffusion_row"] = diffusion_grid

        if sample:
            # get denoise row
            with self.ema_scope("Plotting"):
                if self.controller is not None:
                    print('AttentionMask-DDIM-re-register and reset controller before 1st sample_log ...')
                    register_attention_control_t2i(self, self.controller)
                    self.controller.reset()
                samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                         ddim_steps=ddim_steps,eta=ddim_eta)
                # samples, z_denoise_row = self.sample(cond=c, batch_size=N, return_intermediates=True)
            x_samples = self.decode_first_stage(samples)
            log["samples"] = x_samples
            if plot_denoise_rows:
                denoise_grid = self._get_denoise_row_from_list(z_denoise_row)
                log["denoise_row"] = denoise_grid
            
            uc = self.get_learned_conditioning(len(c) * [""])
            if self.controller is not None:
                print('AttentionMask-DDIM-re-register and reset controller before 2nd sample_log ...')
                register_attention_control_t2i(self, self.controller)
                self.controller.reset()
            sample_scaled, _ = self.sample_log(cond=c, 
                                               batch_size=N, 
                                               ddim=use_ddim, 
                                               ddim_steps=ddim_steps,
                                               eta=ddim_eta,                                                 
                                               unconditional_guidance_scale=5.0,
                                               unconditional_conditioning=uc)
            log["samples_scaled"] = self.decode_first_stage(sample_scaled)

            if quantize_denoised and not isinstance(self.first_stage_model, AutoencoderKL) and not isinstance(
                    self.first_stage_model, IdentityFirstStage):
                # also display when quantizing x0 while sampling
                with self.ema_scope("Plotting Quantized Denoised"):
                    if self.controller is not None:
                        print('AttentionMask-DDIM-re-register and reset controller before 3rd sample_log ...')
                        register_attention_control_t2i(self, self.controller)
                        self.controller.reset()
                    samples, z_denoise_row = self.sample_log(cond=c,batch_size=N,ddim=use_ddim,
                                                             ddim_steps=ddim_steps,eta=ddim_eta,
                                                             quantize_denoised=True)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_x0_quantized"] = x_samples

            if inpaint:
                # make a simple center square
                b, h, w = z.shape[0], z.shape[2], z.shape[3]
                mask = torch.ones(N, h, w).to(self.device)
                # zeros will be filled in
                mask[:, h // 4:3 * h // 4, w // 4:3 * w // 4] = 0.
                mask = mask[:, None, ...]
                with self.ema_scope("Plotting Inpaint"):
                    if self.controller is not None:
                        print('AttentionMask-DDIM-re-register and reset controller before 4th sample_log ...')
                        register_attention_control_t2i(self, self.controller)
                        self.controller.reset()

                    samples, _ = self.sample_log(cond=c,batch_size=N,ddim=use_ddim, eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_inpainting"] = x_samples
                log["mask"] = mask

                # outpaint
                with self.ema_scope("Plotting Outpaint"):
                    if self.controller is not None:
                        print('AttentionMask-DDIM-re-register and reset controller before 5th sample_log ...')
                        register_attention_control_t2i(self, self.controller)
                        self.controller.reset()
                    samples, _ = self.sample_log(cond=c, batch_size=N, ddim=use_ddim,eta=ddim_eta,
                                                ddim_steps=ddim_steps, x0=z[:N], mask=mask)
                x_samples = self.decode_first_stage(samples.to(self.device))
                log["samples_outpainting"] = x_samples

        if plot_progressive_rows:
            with self.ema_scope("Plotting Progressives"):
                img, progressives = self.progressive_denoising(c,
                                                               shape=(self.channels, self.image_size, self.image_size),
                                                               batch_size=N)
            prog_row = self._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

        if return_keys:
            if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
                return log
            else:
                return {key: log[key] for key in return_keys}
        return log

    def configure_optimizers(self):
        """
        Configure the optimizers for training.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """
        lr = self.learning_rate

        if self.embedding_manager is not None: # If using textual inversion
            embedding_params = list(self.embedding_manager.embedding_parameters())

            if self.unfreeze_model: # Are we allowing the base model to train? If so, set two different parameter groups.
                model_params = list(self.cond_stage_model.parameters()) + list(self.model.parameters())
                opt = torch.optim.AdamW([{"params": embedding_params, "lr": lr}, {"params": model_params}], lr=self.model_lr)
            else: # Otherwise, train only embedding
                opt = torch.optim.AdamW(embedding_params, lr=lr)
        else:
            params = list(self.model.parameters())
            if self.cond_stage_trainable:
                print(f"{self.__class__.__name__}: Also optimizing conditioner params!")
                params = params + list(self.cond_stage_model.parameters())
            if self.learn_logvar:
                print('Diffusion model optimizing logvar')
                params.append(self.logvar)

                opt = torch.optim.AdamW(params, lr=lr)

        return opt

    def configure_opt_embedding(self):
        """
        Configure the optimizer for training the embedding.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """

        self.cond_stage_model.eval()
        self.cond_stage_model.train = disabled_train
        for param in self.cond_stage_model.parameters():
            param.requires_grad = False

        self.model.eval()
        self.model.train = disabled_train
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.embedding_manager.embedding_parameters():
            param.requires_grad = True

        lr = self.learning_rate
        params = list(self.embedding_manager.embedding_parameters())
        return torch.optim.AdamW(params, lr=lr)

    def configure_opt_model(self):
        """
        Configure the optimizer for training the model.
        
        Returns:
            torch.optim.Optimizer: Configured optimizer.
        """

        for param in self.cond_stage_model.parameters():
            param.requires_grad = True

        for param in self.model.parameters():
            param.requires_grad = True

        for param in self.embedding_manager.embedding_parameters():
            param.requires_grad = True

        model_params = list(self.cond_stage_model.parameters()) + list(self.model.parameters())
        embedding_params = list(self.embedding_manager.embedding_parameters())
        return torch.optim.AdamW([{"params": embedding_params, "lr": self.learning_rate}, {"params": model_params}], lr=self.model_lr)

    @torch.no_grad()
    def to_rgb(self, x):
        """
        Convert the input tensor to RGB format.
        
        Args:
            x (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: RGB tensor.
        """
        x = x.float()
        if not hasattr(self, "colorize"):
            self.colorize = torch.randn(3, x.shape[1], 1, 1).to(x)
        x = nn.functional.conv2d(x, weight=self.colorize)
        x = 2. * (x - x.min()) / (x.max() - x.min()) - 1.
        return x

    @rank_zero_only
    def on_save_checkpoint(self, checkpoint):
        """
        Save the checkpoint during training.
        
        Args:
            checkpoint (dict): Checkpoint dictionary.
        """

        if not self.unfreeze_model: # If we are not tuning the model itself, zero-out the checkpoint content to preserve memory.
            checkpoint.clear()
        
        if os.path.isdir(self.trainer.checkpoint_callback.dirpath):
            self.embedding_manager.save(os.path.join(self.trainer.checkpoint_callback.dirpath, "embeddings.pt"))

            self.embedding_manager.save(os.path.join(self.trainer.checkpoint_callback.dirpath, f"embeddings_gs-{self.global_step}.pt"))


class DiffusionWrapper(pl.LightningModule):
    """
    Wrapper class for the diffusion model.
    
    Args:
        diff_model_config (dict): Diffusion model configuration.
        conditioning_key (str): Key for conditioning.
    """
    def __init__(self, diff_model_config, conditioning_key):
        super().__init__()
        self.diffusion_model = instantiate_from_config(diff_model_config)
        self.conditioning_key = conditioning_key
        assert self.conditioning_key in [None, 'concat', 'crossattn', 'hybrid', 'adm']

    def forward(self, x, t, c_concat: list = None, c_crossattn: list = None):
        """
        Forward pass of the diffusion model.
        
        Args:
            x (torch.Tensor): Input tensor.
            t (torch.Tensor): Number of diffusion steps.
            c_concat (list, optional): Concatenated conditioning tensors. Defaults to None.
            c_crossattn (list, optional): Cross-attention conditioning tensors. Defaults to None.
            
        Returns:
            torch.Tensor: Output tensor.
        """
        if self.conditioning_key is None:
            out = self.diffusion_model(x, t)
        elif self.conditioning_key == 'concat':
            xc = torch.cat([x] + c_concat, dim=1)
            out = self.diffusion_model(xc, t)
        elif self.conditioning_key == 'crossattn':
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(x, t, context=cc)
        elif self.conditioning_key == 'hybrid':
            xc = torch.cat([x] + c_concat, dim=1)
            cc = torch.cat(c_crossattn, 1)
            out = self.diffusion_model(xc, t, context=cc)
        elif self.conditioning_key == 'adm':
            cc = c_crossattn[0]
            out = self.diffusion_model(x, t, y=cc)
        else:
            raise NotImplementedError()

        return out


class Layout2ImgDiffusion(LatentDiffusion):
    """
    Layout to Image Diffusion model.
    
    Args:
        cond_stage_key (str): Key for the conditioning stage.
    """
    def __init__(self, cond_stage_key, *args, **kwargs):
        assert cond_stage_key == 'coordinates_bbox', 'Layout2ImgDiffusion only for cond_stage_key="coordinates_bbox"'
        super().__init__(cond_stage_key=cond_stage_key, *args, **kwargs)

    def log_images(self, batch, N=8, *args, **kwargs):
        """
        Log images during training or validation.
        
        Args:
            batch (dict): Input batch.
            N (int, optional): Number of images to log. Defaults to 8.
            
        Returns:
            dict: Log dictionary.
        """
        logs = super().log_images(batch=batch, N=N, *args, **kwargs)

        key = 'train' if self.training else 'validation'
        dset = self.trainer.datamodule.datasets[key]
        mapper = dset.conditional_builders[self.cond_stage_key]

        bbox_imgs = []
        map_fn = lambda catno: dset.get_textual_label(dset.get_category_id(catno))
        for tknzd_bbox in batch[self.cond_stage_key][:N]:
            bboximg = mapper.plot(tknzd_bbox.detach().cpu(), map_fn, (256, 256))
            bbox_imgs.append(bboximg)

        cond_img = torch.stack(bbox_imgs, dim=0)
        logs['bbox_image'] = cond_img
        return logs
