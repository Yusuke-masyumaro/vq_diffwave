import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import sqrt
from vector_quantize_pytorch import ResidualVQ

import params as params

relu = nn.LeakyReLU(0.2)

class VAE_residual(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_hidden):
        super(VAE_residual, self).__init__()
        self.conv_one = nn.Conv1d(in_channels = in_channels, out_channels = num_residual_hidden, kernel_size = 3, stride = 1, padding = 1)
        self.conv_two = nn.Conv1d(in_channels = num_residual_hidden, out_channels = hidden_dim, kernel_size = 1, stride = 1)

    def forward(self, x):
        h = relu(x)
        h = relu(self.conv_one(h))
        h = self.conv_two(h)
        return x + h

class ResidualStack(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, num_residual_hidden):
        super(ResidualStack, self).__init__()
        self.num_residual_layers = num_residual_layers
        self.layers = nn.ModuleList([VAE_residual(in_channels, hidden_dim, num_residual_hidden) for _ in range(num_residual_layers)])

    def forward(self, x):
        for i in range(self.num_residual_layers):
            x = self.layers[i](x)
        return relu(x)

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_residual_layers = num_residual_layers
        self.residual_hidden_dim = residual_hidden_dim

        self.e_layer_one = nn.Conv1d(in_channels = self.in_channels, out_channels = self.hidden_dim // 2, kernel_size = 3, stride = 1, padding = 1)
        self.e_layer_two = nn.Conv1d(in_channels = self.hidden_dim // 2, out_channels = self.hidden_dim, kernel_size = 3, stride = 1, padding = 1)
        self.e_layer_three = nn.Conv1d(in_channels = self.hidden_dim, out_channels = self.hidden_dim, kernel_size = 3, stride = 1, padding = 1)
        self.residual_stack = ResidualStack(self.hidden_dim, self.hidden_dim, self.num_residual_layers, self.residual_hidden_dim)

        self.encoder_layer = nn.ModuleList([self.e_layer_one, self.e_layer_two, self.e_layer_three])

    def forward(self, x):
        for layer in self.encoder_layer:
            if layer != self.e_layer_three:
                x = layer(x)
                x = relu(x)
            else:
                x = layer(x)
        return self.residual_stack(x)

class Decoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, num_residual_layers, residual_hidden_dim):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim
        self.num_residual_layers = num_residual_layers
        self.residual_hidden_dim = residual_hidden_dim

        self.d_layer_one = nn.Conv1d(in_channels = self.in_channels, out_channels = self.hidden_dim, kernel_size = 3, stride = 1, padding = 1)
        self.residual_stack = ResidualStack(self.hidden_dim, self.hidden_dim, self.num_residual_layers, self.residual_hidden_dim)
        self.d_layer_two = nn.ConvTranspose1d(in_channels = self.hidden_dim, out_channels = self.hidden_dim // 2, kernel_size = 3, stride = 1, padding = 1)
        self.d_layer_three = nn.ConvTranspose1d(in_channels = self.hidden_dim // 2, out_channels = 1, kernel_size = 3, stride = 1, padding = 1)

        self. decoder_layer = nn.ModuleList([self.d_layer_one, self.residual_stack, self.d_layer_two])

    def forward(self, x):
        for layer in self.decoder_layer:
            if layer == self.d_layer_two:
                x = layer(x)
            else:
                x = layer(x)
                x = relu(x)
        x = self.d_layer_three(x)
        return torch.tanh(x)

class VQVAE(nn.Module):
    def __init__(self, encoder, decoder, vq, pre_vq_conv1, data_variance = None):
        super(VQVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vq = vq
        self.pre_vq_conv1 = pre_vq_conv1
        self.data_variance = data_variance

    def forward(self, x):
        z = self.pre_vq_conv1(self.encoder(x))
        z_permute = z.permute(0, 2, 1).contiguous()
        vq_quantize, embedding_idx, vq_loss = self.vq(z_permute)
        vq_quantize_reshape = vq_quantize.permute(0, 2, 1).contiguous()
        output = self.decoder(vq_quantize_reshape)
        if self.data_variance:
            reconstructed_error = torch.mean(torch.square(output - x)) / self.data_variance
            vq_loss = vq_loss.sum() / len(vq_loss)
            loss = reconstructed_error + vq_loss
            return {'z': z, 'x': x, 'loss': loss, 'reconstructed_error': reconstructed_error, 'vq_output': vq_quantize, 'output': output}
        else:
            return {'z': z, 'x': x, 'vq_output': vq_quantize, 'output': output}

def get_model(data_variance = None):
    #encoder = Encoder(in_channels = params.in_channels, hidden_dim = params.hidden_dim, num_residual_layers = params.num_residual_layers, residual_hidden_dim = params.residual_hidden_dim)
    #decoder = Decoder(in_channels = params.embedding_dim, hidden_dim = params.hidden_dim, num_residual_layers = params.num_residual_layers, residual_hidden_dim = params.residual_hidden_dim)
    encoder = Encoder(in_channels = params.vq_params['in_channels'],
                    hidden_dim = params.vq_params['hidden_dim'],
                    num_residual_layers = params.vq_params['num_residual_layers'],
                    residual_hidden_dim = params.vq_params['residual_hidden_dim'])

    decoder = Decoder(in_channels = params.vq_params['embedding_dim'],
                    hidden_dim = params.vq_params['hidden_dim'],
                    num_residual_layers = params.vq_params['num_residual_layers'],
                    residual_hidden_dim = params.vq_params['residual_hidden_dim'])
    pre_vq_conv1 = nn.Conv1d(in_channels = params.vq_params['hidden_dim'], out_channels = params.vq_params['embedding_dim'], kernel_size = 1, stride = 1)
    #vq = VectorQuantize(dim = params.vq_params['embedding_dim'], codebook_size = params.vq_params['codebook_size'])
    vq = ResidualVQ(dim = params.vq_params['embedding_dim'],num_quantizers = 4, codebook_size = params.vq_params['codebook_size'], kmeans_init = True)
    model = VQVAE(encoder, decoder, vq, pre_vq_conv1, data_variance = data_variance)
    optimizer = torch.optim.Adam(model.parameters(), lr = params.vq_params['lr'])
    return model, optimizer

class Encoder_vq(nn.Module):
    def __init__(self, encoder, pre_vq_conv1, vq, data_variance = None):
        super(Encoder_vq, self).__init__()
        self.encoder = encoder
        self.vq = vq
        self.pre_vq_conv1 = pre_vq_conv1
        self.data_variance = data_variance

    def forward(self, x):
        z = self.pre_vq_conv1(self.encoder(x))
        z_permute = z.permute(0, 2, 1).contiguous()
        vq_quantize, embedding_idx, vq_loss = self.vq(z_permute)
        vq_quantize_reshape = vq_quantize.permute(0, 2, 1).contiguous()
        return vq_quantize_reshape

def get_encoder_decoder(data_variance = None):
    encoder = Encoder(in_channels = params.vq_params['in_channels'],
                    hidden_dim = params.vq_params['hidden_dim'],
                    num_residual_layers = params.vq_params['num_residual_layers'],
                    residual_hidden_dim = params.vq_params['residual_hidden_dim'])

    decoder = Decoder(in_channels = params.vq_params['embedding_dim'],
                    hidden_dim = params.vq_params['hidden_dim'],
                    num_residual_layers = params.vq_params['num_residual_layers'],
                    residual_hidden_dim = params.vq_params['residual_hidden_dim'])

    pre_vq_conv1 = nn.Conv1d(in_channels = params.vq_params['hidden_dim'], out_channels = params.vq_params['embedding_dim'], kernel_size = 1, stride = 1)
    #vq = VectorQuantize(dim = params.vq_params['embedding_dim'], codebook_size = params.vq_params['codebook_size'])
    vq = ResidualVQ(dim = params.vq_params['embedding_dim'],num_quantizers = 4, codebook_size = params.vq_params['codebook_size'], kmeans_init = True)
    encoder_vq = Encoder_vq(encoder, decoder, pre_vq_conv1, vq, data_variance = data_variance)
    return encoder_vq, decoder