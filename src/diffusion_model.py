import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt

import params as params
import VQ_model as vq_model

relu = nn.ReLU()
silu = nn.SiLU()

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

class DiffusionEmbedding(nn.Module):
    def __init__(self, max_step):
        super().__init__()
        self.register_buffer('embedding', self._build_embedding(max_step), persistent = False)
        self.linear_one = nn.Linear(128, 512)
        self.linear_two = nn.Linear(512, 512)

    def forward(self, diffusion_step):
        if diffusion_step.dtype in [torch.int32, torch.int64]:
            x = self.embedding[diffusion_step]
        else:
            x = self._lerp_embedding(diffusion_step)
        x = self.linear_one(x)
        x = silu(x)
        x = self.linear_two(x)
        x = silu(x)
        return x

    def _lerp_embedding(self, t):
        low_idx = torch.floor(t).long()
        high_idx = torch.ceil(t).long()
        low = self.embedding[low_idx]
        high = self.embedding[high_idx]
        return low + (high + low) * (t - low_idx)

    def _build_embedding(self, max_steps):
        steps = torch.arange(max_steps).unsqueeze(1)
        dims = torch.arange(64).unsqueeze(0)
        table = steps * 10.0 ** (dims * 4.0 / 63.0)
        table = torch.cat([torch.sin(table), torch.cos(table)], dim = 1)
        return table

class ResidualBlock(nn.Module):
    def __init__(self, res_channels, dilation):
        super().__init__()
        self.dilated_conv = Conv1d(res_channels, 2 * res_channels, 3, padding = dilation, dilation = dilation)
        self.diffusion_layer = nn.Linear(512, res_channels)
        self.output_layer = Conv1d(res_channels, 2 * res_channels, 1)

    def forward(self, x, diff_step):
        diff_step = self.diffusion_layer(diff_step).unsqueeze(-1)
        y = x + diff_step
        y = self.dilated_conv(y)
        gate, filter = torch.chunk(y, 2, dim = 1)
        y = torch.sigmoid(gate) * torch.tanh(filter)
        y = self.output_layer(y)
        residual, skip = torch.chunk(y, 2, dim = 1)
        return (x + residual) / sqrt(2.0), skip
    
class VQ_diffwave(nn.Module):
    def __init__(self, res_channels, dilation_cycle_length, res_layers, noise_schedule):
        super().__init__()
        self.input_layer = Conv1d(1, res_channels, kernel_size = 1, stride = 1)
        self.diffusion_embedding = DiffusionEmbedding(len(noise_schedule))
        self.res_layers = nn.ModuleList([ResidualBlock(res_channels, 2 ** (i % dilation_cycle_length)) for i in range(res_layers)])
        self.skip_layer = Conv1d(res_channels, res_channels, kernel_size = 1, stride = 1)
        self.output_layer = Conv1d(res_channels, 1, kernel_size = 1, stride = 1)
        nn.init.zero_(self.output_layer.weight)
        
        def forward(self, vq_z, diff_step):
            x = self.input_layer(vq_z)
            x = relu(x)
            diff_step = self.diffusion_embedding(diff_step)
            skip = None
            for layer in self.res_layers:
                x, skip_connect = layer(x, diff_step)
                if skip is None:
                    skip = skip_connect
                else:
                    skip = skip_connect + skip
                    
            x = skip / sqrt(len(self.res_layers))
            x = self.skip_layer(x)
            x = relu(x)
            deff_z = self.output_layer(x)
            return deff_z

# Diffwave model
class Diffwave(nn.Module):
    def __init__(self, res_channels, dilation_cycle_length, res_layers, noise_schedule):
        super().__init__()
        self.input_layer = Conv1d(1, res_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(len(noise_schedule))
        self.label_embedding = nn.Embedding(params.label_num, 512)
        self.res_layers = nn.ModuleList([ResidualBlock(res_channels, 2 ** (i % dilation_cycle_length))for i in range(res_layers)])
        self.skip_layer = Conv1d(res_channels, res_channels, 1)
        self.output_layer = Conv1d(res_channels, 1, 1)
        nn.init.zeros_(self.output_layer.weight)

    def forward(self, wav, label, diff_step):
        label_flg = True
        x = wav.unsqueeze(1)
        x = self.input_layer(x)
        x = relu(x)
        diff_step = self.diffusion_embedding(diff_step)
        diff_step = self.label_embedding(label) + diff_step

        skip = None
        for layer in self.res_layers:
            x, skip_connect = layer(x, diff_step)
            if skip is None:
                skip = skip_connect
            else:
                skip = skip_connect + skip

        x = skip / sqrt(len(self.res_layers))
        x = self.skip_layer(x)
        x = relu(x)
        x = self.output_layer(x)
        return x