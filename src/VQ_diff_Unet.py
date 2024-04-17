import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torch.nn.functional as F
import torchaudio
import librosa
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from diffusers import DDPMScheduler, UNet2DModel
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import params as params
from VQ_model import get_model
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ESC_dataset(Dataset):
    def __init__(self, df, path):
        self.df = df
        self.path = path
        self.data_list = []
        for file_name in tqdm(df['filename'].values):
            wav, sr = torchaudio.load(self.path + file_name)
            wav = torchaudio.transforms.Resample(orig_freq = sr, new_freq = 16000)(wav)
            self.data_list.append(wav)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.data_list[idx]
    
class Urban_dataset(Dataset):
    def __init__(self, df, file_path):
        self.wav_data_list = []
        self.label_list = []
        waves = df['slice_file_name'].values
        #labels = df['classID'].values
        folder_num = df['fold'].values
        for i in tqdm(range(len(df))):
            wav_file_path = file_path + 'fold' + str(folder_num[i]) + '/' + waves[i]
            #wav, sr = torchaudio.load(wav_file_path)
            #wav = torchaudio.transforms.Resample(orig_freq = sr, new_freq = 16000)(wav)
            wav, sr = librosa.load(wav_file_path, sr = 16000)
            if len(wav) < params.diff_params['wav_length']:
                max_offset = params.diff_params['wav_length'] - len(wav)
                wav = np.pad(wav, (0, max_offset))
            self.wav_data_list.append(wav)
            #self.labels_list.append(labels[i])
    
    def __len__(self):
        return len(self.wav_data_list)
    
    def __getitem__(self, idx):
        wav = self.wav_data_list[idx]
        #label = self.labels_list[idx]
        return wav, #label


def train(model, vq_model, dataset, noise_scheduler, epochs):
    data_loader = DataLoader(dataset, batch_size = params.diff_params['batch_size'], shuffle = True)
    model = model.to(device)
    vq_model = vq_model.to(device)
    criterion = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr = params.diff_params['lr'])
    
    for epoch in range(epochs):
        losses = 0.0
        for x in tqdm(data_loader):
            data = x.to(device)
            vq_z = vq_model(data)
            print(vq_z.shape)
            noise = torch.randn_like(vq_z).to(device)
            timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (noise.shape[0],), device = device)
            noise_vq_z = noise_scheduler.add_noise(vq_z, noise, timesteps)
            noise_vq_z = noise_vq_z.to(device)
            noise_vq_z = noise_vq_z.unsqueeze(1)
            
            noise_pred = model(noise_vq_z, timesteps, return_dict = False)[0]
            
            loss = criterion(noise_pred, noise)
            optimizer.zero_grad()
            loss.backward()
            losses += loss.item()
            optimizer.step()
        
        print(f'Epoch = {epoch} | loss = {losses:}')
        os.makedirs = ('model', exist_ok == True)
        torch.save(model.state_dict(), 'model/VQ_diff_Unet.pth')
        
        with torch.no_grad():
            validation_seed = torch.randn(1, 64, 80000).to(device)
            sample = validation_seed.clone()
            for i ,t in enumerate(noise_scheduler.timesteps):
                vq_z = model(samplem, t)
                sample = noise_scheduler.step(vq_z, t, sample).prev_sample
            sample_wav = vq_model.decoer(sample)
            sample_wav = sample_wav.squeeze(1)
            os.makedirs('sampling.wav', exist_ok == True)
            torchaudio.save(f'sampling_wav/epoch = {epoch}.wav', sample_wav.cpu(), sample_rate = params.diff_params['sampling_rate'])
            
    
    
if __name__ == '__main__':
    df = pd.read_csv('../dataset/ESC-50-master/meta/test.csv')
    file_path = '../dataset/ESC-50-master/audio/'
    esc_dataset = ESC_dataset(df, file_path)
    '''
    df = pd.read_csv('../dataset/Urban/UrbanSound8K.csv')
    file_path = '../dataset/Urban/'
    urban_dataset = Urban_dataset(df, file_path)
    combined_dataset = ConcatDataset([esc_dataset, urban_dataset])
    '''
    noise_scheduler = DDPMScheduler(num_train_timesteps = 1000, beta_schedule = 'squaredcos_cap_v2')
    model = UNet2DModel(sample_size = (64, 80000),
                    in_channels = 1,
                    out_channels = 1,
                    layers_per_block = 2,
                    block_out_channels = (64, 128, 128, 256),
                    down_block_types = ('DownBlock2D',
                                        'DownBlock2D',
                                        'AttnDownBlock2D',
                                        'AttnDownBlock2D',),
                    up_block_types = ('AttnUpBlock2D',
                                      'AttnUpBlock2D',
                                      'UpBlock2D',
                                      'UpBlock2D',),)
    vq_model = get_model(data_variance = None, inference = True)
    vq_model.load_state_dict(torch.load('VQ_pth/model.pth'))
    train(model, vq_model, esc_dataset, noise_scheduler, epochs = params.diff_params['epochs'])
    