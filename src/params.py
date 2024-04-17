#VQ_model_parameters
vq_params = {'in_channels': 1,
            'hidden_dim': 128,
            'num_residual_layers': 2,
            'residual_hidden_dim': 64,
            'embedding_dim': 64,
            'codebook_size': 512,
            'batch_size': 16,
            'lr': 2e-4,}

diff_params = {'res_channels': 128,
            'dilation_cycle_length': 12,
            'res_layers': 20,
            'batch_size': 8,
            'lr': 2e-4,
            'epochs': 500,
            'wav_length': 80000,
            'sampling_rate': 16000}