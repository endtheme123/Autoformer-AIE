

CONFIGS = {
    "Autoformer": {
        'seq_len': 120,                    # Sequence length
        'label_len': 60,                   # Lookback window size
        'pred_len': 60,                    # Prediction length
        'e_layers': 2,                     # Encoder layers
        'd_layers': 1,                     # Decoder layers
        'n_heads': 8,                      # Number of attention heads
        'd_model': 512,                    # Model embedding dimension
        'factor': 1,                       # Autoformer factor for AutoCorrelation
        'enc_in': 12,                      # Number of input features for encoder
        'dec_in': 12,                      # Number of input features for decoder
        'c_out': 12,                       # Number of output features
        'd_ff': 2048,                      # Feed-forward network dimension
        'moving_avg': 25,                  # Moving average kernel size for decomposition
        'dropout': 0.1,                    # Dropout rate
        'activation': 'gelu',              # Activation function
        'embed': 'timeF',                  # Embedding type (time-frequency-based)
        'freq': 'h',                       # Frequency for embedding
        'output_attention': False,         # Whether to output attention weights
    },
    "Informer": {
        'seq_len': 120,                    # Sequence length
        'label_len': 60,                   # Lookback window size
        'pred_len': 60,                    # Prediction length
        'e_layers': 2,                     # Encoder layers
        'd_layers': 1,                     # Decoder layers
        'n_heads': 8,                      # Number of attention heads
        'd_model': 512,                    # Model embedding dimension
        'factor': 1,                       # Sparsity factor for ProbAttention
        'enc_in': 12,                      # Encoder input features
        'dec_in': 12,                      # Decoder input features
        'c_out': 12,                       # Output features
        'd_ff': 2048,                      # Feed-forward network dimension
        'dropout': 0.1,                    # Dropout rate
        'activation': 'gelu',              # Activation function
        'embed': 'timeF',                  # Embedding type
        'freq': 'h', 
        'output_attention': False,         # Whether to output attention weights
        'distil': True,                    # Distilling option for encoder
       
    },
    "Transformer": {
        'seq_len': 120,                    # Sequence length
        'label_len': 60,                   # Lookback window size
        'pred_len': 60,                    # Prediction length
        'e_layers': 2,                     # Encoder layers
        'd_layers': 1,                     # Decoder layers
        'n_heads': 8,                      # Number of attention heads
        'd_model': 512,                    # Model embedding dimension
        'factor': 1,                       # Sparsity factor for ProbAttention
        'enc_in': 12,                      # Encoder input features
        'dec_in': 12,                      # Decoder input features
        'c_out': 12,         
        'd_ff': 2048,                      # Feed-forward network dimension
        'dropout': 0.1,                    # Dropout rate
        'activation': 'gelu',              # Activation function
        'embed': 'timeF',                  # Embedding type
        'freq': 'h', 
        'output_attention': False,         # Whether to output attention weights
    }
}


configs = CONFIGS
print(configs.get("AutoFormer"))