import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext

basedir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(basedir)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
#     pl_sd = torch.load(ckpt, map_location={'cuda:1':'cuda:0'})
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.eval()
    return model

class generator:

    def __init__(self):
        self.H = 64
        self.W = 64
        self.config_file = os.path.join(basedir, 'configs/stable-diffusion/v1-inference.yaml')
        self.ckpt =  os.path.join(basedir, 'models/ldm/stable-diffusion-v1-4/sd-v1-4.ckpt')
        self.outdir = "outputs/txt2img-samples"
        self.seed = 42
        self.ddim_steps = 50
        self.f = 8
        self.C = 4
        self.prompt = None
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.model = None
        self.config = None
        self.sampler = None
        self.data = None
        self.scale = 7.5
        self.batch_size = 1
        self.loaded = False
        self.ddim_steps = 50
        self.ddim_eta = 0.0
        self.precision = "autocast"
        self.n_iter = 1

    def load_model(self):
        print('Loading model...')
        self.config = OmegaConf.load(self.config_file)
        self.model = load_model_from_config(self.config, self.ckpt)
        self.model = self.model.to(self.device)
        self.sampler = PLMSSampler(self.model)
        # self.start_code = torch.randn([self.batch_size, self.C, self.H // self.f, self.W // self.f], device=self.device)
        self.start_code = None
        self.precision_scope = autocast if self.precision=="autocast" else nullcontext
        self.loaded = True
        print('Model loaded')

    def generate(self, w, h, prompt):
        self.W = w
        self.H = h
        self.prompt = prompt
        self.data = [self.batch_size * [prompt]]
        # print('width:{}\nheight:{}\ntext:{}'.format(self.W, self.H, self.prompt))

        if not self.loaded:
            return None

        with torch.no_grad():
            with self.model.ema_scope():
                tic = time.time()
                for n in trange(self.n_iter, desc="Sampling"):
                    for prompts in tqdm(self.data, desc="data"):
                        uc = None
                        if self.scale != 1.0:
                            uc = self.model.get_learned_conditioning(self.batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = self.model.get_learned_conditioning(prompts)
                        shape = [self.C, self.H // self.f, self.W // self.f]
                        samples_ddim, _ = self.sampler.sample(S=self.ddim_steps,
                                                         conditioning=c,
                                                         batch_size=self.batch_size,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=self.scale,
                                                         unconditional_conditioning=uc,
                                                         eta=self.ddim_eta,
                                                         x_T=self.start_code)

                        x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                        x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)

                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            # img = Image.fromarray(x_sample.astype(np.uint8))
                            # img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            # print('generated image:', type(img), img)
                        np_img = np.zeros(x_sample.shape)
                        for i in range(np_img.shape[2]):
                            np_img[:, :, i] = x_sample.astype(np.uint8)[:, :, i]
                toc = time.time()
        return np_img
