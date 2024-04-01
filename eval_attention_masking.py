import argparse, os

from typing import Union, Tuple, List, Callable, Dict, Optional
import torch
import torch.nn.functional as nnf
from diffusers import DiffusionPipeline
import numpy as np
from IPython.display import display
from PIL import Image
import abc
from src.p2p import ptp_utils
from src.p2p import seq_aligner

from scripts.txt2img import load_model_from_config
from omegaconf import OmegaConf
import torchvision.transforms.functional as F
from IPython.display import display
import numpy as np
import matplotlib as mpl

from src.p2p.p2p_ldm_utils import *

from notebook.services.config import ConfigManager
cm = ConfigManager().update('notebook', {'limit_output': 10})

def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        '--prompts_string', 
        type=str, 
        nargs='+', 
        default=[])
    parser.add_argument(
        '--emb_path_list', 
        type=str, 
        nargs='+', 
        default=[])
    parser.add_argument(
        '--exp_names', 
        type=str, 
        nargs='+', 
        default=[])
    parser.add_argument(
        '--select_clsses', 
        type=str, 
        nargs='+', 
        default=[])
    parser.add_argument(
        "--ckpt_path",
        type=str,
        const=True,
        default='/home/jovyan/vol-1/root_chen/git_chen/data/models_chen/ldm/text2img-large/model.ckpt',
        nargs="?",
        help="pretrained model path",
    )
    parser.add_argument(
        "--embedding_path",
        type=str,
        const=True,
        default='/home/jovyan/vol-1/root_chen/git_chen/textual_inversion_chen/logs/figure4_lge_sketch2023-06-28T17-21-13_lge_sketch_MC_v2_allPsudo/checkpoints/embeddings_gs-6099.pt',
        nargs="?",
        help="embedding_path",
    )
    parser.add_argument(
        "--out_base",
        type=str,
        const=True,
        default='/home/jovyan/vol-1/root_chen/git_chen/textual_inversion_chen/outputs/iclr2024',
        nargs="?",
        help="out_base",
    )
    parser.add_argument(
        "--log_base",
        type=str,
        const=True,
        default='/home/jovyan/vol-1/root_chen/git_chen/textual_inversion_chen/logs',
        nargs="?",
        help="log_base",
    )
    parser.add_argument(
        "--embedding_base",
        type=str,
        const=True,
        default='checkpoints/embeddings_gs-6099.pt',
        nargs="?",
        help="embedding_base",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        const=True,
        default='lge_cmr_mix_simple_MC_v2_allPsudo',
        nargs="?",
        help="exp_name",
    )
    parser.add_argument(
        "--inf_config",
        type=str,
        const=True,
        default="/home/jovyan/vol-1/root_chen/git_chen/textual_inversion_chen/configs/latent-diffusion/txt2img-1p4B-eval_with_tokens.yaml",
        nargs="?",
        help="inference config",
    )
    parser.add_argument("--num_diff_steps",
        type=int,
        default=50,
        help="num_diff_steps",
    )
    parser.add_argument("--manual_seed",
        type=int,
        default=0,
        help="manual_seed",
    )
    parser.add_argument("--guidance_scale",
        type=float,
        default=5.0,
        help="guidance_scale",
    )
    parser.add_argument("--attn_threshold",
        type=float,
        default=0.5,
        help="attn_threshold",
    )
    parser.add_argument(
        "--save_jpeg", 
        type=str2bool, 
        nargs="?", 
        const=False, 
        default=False, 
        help="Optional save as jpeg to save memory, default save as png")

    return parser




if __name__ == "__main__":
    parser = get_parser()
    opt, unknown = parser.parse_known_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    NUM_DIFFUSION_STEPS = opt.num_diff_steps
    GUIDANCE_SCALE = opt.guidance_scale
    MAX_NUM_WORDS = 77

    ckpt_path = opt.ckpt_path
    embedding_path = os.path.join(opt.log_base, opt.embedding_path, opt.embedding_base)
    config = OmegaConf.load(opt.inf_config)

    ldm = load_model_from_config(config, ckpt_path) 
    ldm.embedding_manager.load(embedding_path)
    ldm = ldm.to(device)
    tokenizer = ldm.cond_stage_model.tknz_fn.tokenizer

    if opt.manual_seed != 0:
        g_gpu = torch.Generator(device='cuda').manual_seed(opt.manual_seed)
    else:
        g_gpu = torch.Generator(device='cuda')
    
    prompts = opt.prompts_string
    controller = AttentionStore()

    if not opt.save_jpeg:
        suffix = '.png'
    else:
        suffix = '.jpg'

    if not os.path.exists(opt.out_base):
        os.mkdir(opt.out_base) 
    out_path_base = os.path.join(opt.out_base, opt.exp_name)
    if not os.path.exists(out_path_base):
        os.mkdir(out_path_base) 
    out_path_prompt = os.path.join(out_path_base, prompts[0])
    if not os.path.exists(out_path_prompt):
        os.mkdir(out_path_prompt) 

    # run bsaeline
    images, x_t = run_and_display(ldm, prompts, controller, \
            config, run_baseline=False, generator=g_gpu, \
            array_latent=True, guidance_scale=GUIDANCE_SCALE, \
            out_path_img=os.path.join(out_path_prompt, 'baseline_plain_MCPL'+suffix), \
            save_img=True)
    show_cross_attention(tokenizer, prompts, controller, \
            res=16, from_where=["up", "down"], \
            out_path_img=os.path.join(out_path_prompt, 'baseline_plain_MCPL_attention'+suffix), \
            save_img=True)
    print('Baseline generation done!')

    # run ours (CL / AttnMask)
    # unconditional generation with ours
    out_dir = out_path_prompt
    out_name = 'ours_CL_mask_attn_unconditioned'+suffix
    emb_path_list = [os.path.join(opt.log_base, emb_path, opt.embedding_base) for emb_path in opt.emb_path_list] 
    exp_names = opt.exp_names
    # print(f'DEBUG: emb_path_list = {emb_path_list}')
    # print(f'DEBUG: exp_names = {exp_names}')
    
    # plot_img_attn_mask(ldm, prompts, emb_path_list, exp_names, \
    #     device, out_dir, out_name, config, latent=None, \
    #     array_latent=False, GUIDANCE_SCALE=GUIDANCE_SCALE, \
    #     attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, mask_concepts=True, g_gpu=g_gpu)
    # print('Unconditional generation done!')
    
    # conditional generation based on baseline
    # out_name = 'ours_CL_mask_attn_conditioned'+suffix
    # plot_img_attn_mask(ldm, prompts, emb_path_list, exp_names, \
    #     device, out_dir, out_name, config, latent=x_t, \
    #     array_latent=True, GUIDANCE_SCALE=GUIDANCE_SCALE, \
    #     attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, g_gpu=g_gpu)
    # print('Conditional generation done!')

    #todo: implement bewlow apply mask for each selected prompt, save each masked image, 
    # and in the plot, add each masked concept and selected key word

    out_name = 'ours_CL_mask_attn_conditioned_masked_concepts'+suffix
    plot_img_attn_mask(ldm, prompts, emb_path_list, exp_names, \
        device, out_dir, out_name, config, latent=x_t, \
        array_latent=True, GUIDANCE_SCALE=GUIDANCE_SCALE, \
        attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, mask_concepts=True, g_gpu=g_gpu)
    print('Conditional generation mask key words done!')

    out_name = 'simple_mask_attn_conditioned'+suffix
    plot_img_mask(ldm, prompts, emb_path_list, exp_names, \
        device, out_dir, out_name, config, latent=x_t, \
        array_latent=True, GUIDANCE_SCALE=GUIDANCE_SCALE, \
        attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, mask_concepts=True, g_gpu=g_gpu)
    print('Conditional simple attention mask done!')

    # conditional generation based on baseline without text
    # out_name = 'ours_CL_mask_attn_conditioned_no_text'+suffix
    # plot_img_attn_mask(ldm, prompts, emb_path_list, exp_names, \
    #     device, out_dir, out_name, config, latent=x_t, \
    #     array_latent=True, GUIDANCE_SCALE=GUIDANCE_SCALE, \
    #     attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, show_text=False, g_gpu=g_gpu)
    # print('Conditional generation (no text) done!')

    out_name = 'ours_CL_mask_attn_conditioned_masked_concepts_no_text'+suffix
    plot_img_attn_mask(ldm, prompts, emb_path_list, exp_names, \
        device, out_dir, out_name, config, latent=x_t, \
        array_latent=True, GUIDANCE_SCALE=GUIDANCE_SCALE, \
        attn_threshold=opt.attn_threshold, select_clsses = opt.select_clsses, \
        show_text=False, mask_concepts=True, g_gpu=g_gpu)
    print('Conditional generation mask key words (no text) done!')

    print('All inference done!')
