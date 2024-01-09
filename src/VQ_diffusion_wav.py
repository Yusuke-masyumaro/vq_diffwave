from diffusers import DDPMScheduler
import numpy as np
import torch
import librosa
import soundfile as sf
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import torchaudio
from tqdm import tqdm
import pandas as pd
import scipy
import glob
import os

from VQ_model import get_vq_decoder
from diffusion_model import VQ_diffwave
import params as params

class Wav_dataset(Dataset):
    def __init__(self, df, path):
        self.df = df
        self.path = path
        self.data_list = []
        for file_name in tqdm(df['filename'].values):
            wav, sr = torchaudio.load(self.path + file_name)
            resample_wav = torchaudio.toransforms.Resample(sr, 22050)(wav)
            self.data_list.append(resample_wav)
        
    def __len__(self):
        return len(self.df):
    
    def __getitem__(self, idx):
        return self.data_list[idx]

def train(model, encoder_vq, decoder, dataset, noise_scheduler, epochs):
    data_loader = DataLoader(dataset, batch_size = params.batch_size, shuffle = True)
    model = model.to(device)
    encoder_vq = encoder_vq.to(device)
    decoder = decoder.to(device)
    criterion = torch.nn.L1Loss()
    optimizer = toprch.optim.Adam(model.parameters(), lr = params.lr)
    
    for epoch in range(epochs):
        losses = 0.0
        for x in data_loader:
            data = x.to(device)
            vq_z = encoder_vq(data)
            noise = torch.rando_like(vq_z).to(device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (noise.shape[0],), device = device)
            noise_vq_z = noise_scheduler(vq_z, noise, timesteps)
            noise_vq_z = noise_vq_z.to(device)
            
            output = model(noise_vq_z, timesteps)
            output = output.squeeze(1)
            
            loss = criterion(output, noise_vq_z)
            optimizer.zero_grad()
            loss.backward()
            losses += loss.item()
            optimizer.step()
            
        print(f'Epoch = {epoch} | loss = {losses:}')
        os.makedirs('model', exist_ok = True)
        torch.save(model.state_dict(), 'model/diffwave.pth')
        
        with torch.no_grad():
            validation_seed = torch.randn(1, param.embedding_dim).to(device)
            sample = validation_seed.clone()
            for i, t in enumerate(noise_scheduler.timesteps):
                vq_z = model(sample, t)
                sample = noise_scheduler.step(vq_z, t, sample).prev_sample
                sample = sample.squeeze(0)
            sample_wav = decoder(sample)
            os.makedirs('sampling_wav', exist_ok = True)
            torchaudio.save(f'sampling_wav/epoch={epoch}.wav', sample.cpu(), sample_rate = params.sampling_rate)

def main():
    path = ''
    df = pd.read_csv(path + 'meta/esc50.csv')
    dataset = Wav_dataset(df, path)
    noise_scheduler = DDPMScheduler(num_train_timesteps = 1000, beta_schedule = 'squaredcos_cap_vs')
    model = VQ_diffwave(res_channels = params.res_channels, dilation_cycle_length = params.dilation_cycle_length, res_layers = params.res_layers, noise_schedule = noise_scheduler)
    encoder_vq, decoder = get_vq_decoder()
    train(model, encoder_vq, decoder, dataset, noise_scheduler, epochs = params.epochs)

if __name__ == '__main__'
    main()
    