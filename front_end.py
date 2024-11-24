# import gradio as gr
# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import os
# import zipfile

# # Define paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_PATH = os.path.join(BASE_DIR, "dataset")  # Automatically create this folder
# MODEL_CHECKPOINTS = {
#     "Autoformer": "./checkpoints/truck1_Autoformer_5G_ftM_sl120_ll60_pl60_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth",
#     "Informer": "./checkpoints/truck1_Informer_5G_ftM_sl120_ll60_pl60_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth",
#     "Transformer": "./checkpoints/truck1_Transformer_5G_ftM_sl120_ll60_pl60_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"
# }
# MODELS = ["Autoformer", "Informer", "Transformer"]
# DATASETS = {
#     "electricity": "electricity",
#     "ETT-small": "ETT-small",
#     "exchange_rate": "exchange_rate",
#     "illness": "illness",
#     "traffic": "traffic",
#     "weather": "weather"
# }

# ZIP_FILE_PATH = os.path.join(BASE_DIR, "all_six_datasets.zip")
# MANUAL_DOWNLOAD_URL = "https://drive.google.com/file/d/1alE33S1GmP5wACMXaLu50rDIoVzBM4ik/view?usp=drive_link"

# # Ensure dataset directory and unzipped datasets are present
# def ensure_datasets():
#     # Create dataset folder if not exists
#     if not os.path.exists(DATASET_PATH):
#         os.makedirs(DATASET_PATH)

#     # Check if the dataset has already been extracted
#     dataset_subfolders = os.listdir(DATASET_PATH)
#     if len(dataset_subfolders) == 0:  # No datasets are present
#         print("Dataset not found. Please download the ZIP file manually from:")
#         print(MANUAL_DOWNLOAD_URL)
#         print(f"Place the downloaded ZIP file in: {ZIP_FILE_PATH}")
#         if os.path.exists(ZIP_FILE_PATH):
#             print("Found ZIP file. Extracting...")
#             with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
#                 zip_ref.extractall(DATASET_PATH)
#             print("Extraction complete.")
#         else:
#             raise FileNotFoundError(f"Please download the ZIP file and place it in {ZIP_FILE_PATH}")

# # Load the model dynamically
# def load_model(model_name):
#     checkpoint_file = MODEL_CHECKPOINTS[model_name]
#     checkpoint = torch.load(checkpoint_file)
    
#     if model_name == "Autoformer":
#         from models.Autoformer import Autoformer
#         model = Autoformer(config=checkpoint['config'])
#     elif model_name == "Informer":
#         from models.Informer import Informer
#         model = Informer(config=checkpoint['config'])
#     elif model_name == "Transformer":
#         from models.Transformer import Transformer
#         model = Transformer(config=checkpoint['config'])
#     else:
#         raise ValueError(f"Unknown model name: {model_name}")
    
#     model.load_state_dict(checkpoint['state_dict'])
#     return model

# # Visualize predictions
# def visualize_predictions(model_name, dataset_name):
#     try:
#         model = load_model(model_name)
#     except Exception as e:
#         return f"Error loading model: {e}"

#     # Load dataset dynamically
#     try:
#         dataset_path = os.path.join(DATASET_PATH, DATASETS[dataset_name])
#         if not os.path.exists(dataset_path):
#             return f"Dataset {dataset_name} not found in {DATASET_PATH}. Please ensure it's extracted correctly."

#         # Simulated Predictions (Replace with actual model predictions)
#         true_data = np.sin(np.linspace(0, 2 * np.pi, 100))  # Simulated data
#         pred_data = np.cos(np.linspace(0, 2 * np.pi, 100))  # Simulated predictions
#     except Exception as e:
#         return f"Error loading dataset: {e}"

#     # Create the plot
#     plt.figure(figsize=(10, 5))
#     plt.plot(true_data, label="GroundTruth", color='blue')
#     plt.plot(pred_data, label="Prediction", color='orange')
#     plt.title(f"{model_name} Predictions - {dataset_name}")
#     plt.xlabel("Time Steps")
#     plt.ylabel("Values")
#     plt.legend()
#     plt.grid()
#     plot_path = os.path.join(BASE_DIR, "temp_plot.png")
#     plt.savefig(plot_path)
#     plt.close()

#     return plot_path

# # Ensure datasets are ready
# ensure_datasets()

# # Gradio interface
# with gr.Blocks() as interface:
#     gr.Markdown("# Model Time Series Visualization")
#     gr.Markdown("Visualize predictions from pre-trained models with corresponding checkpoints and datasets.")
    
#     with gr.Row():
#         model_input = gr.Dropdown(choices=MODELS, label="Select Model")
#         dataset_input = gr.Dropdown(choices=list(DATASETS.keys()), label="Select Dataset")
    
#     output_image = gr.Image(type="filepath", label="Time Series Plot")  # Use "filepath" instead of "file"
#     run_button = gr.Button("Run")

#     run_button.click(
#         fn=visualize_predictions,
#         inputs=[model_input, dataset_input],
#         outputs=output_image
#     )

# # Launch the interface
# if __name__ == "__main__":
#     interface.launch()

# import os
# import zipfile
# import gradio as gr
# import numpy as np
# import matplotlib.pyplot as plt
# import logging
# import torch

# from models.Autoformer import Autoformer
# from models.Informer import Informer
# from models.Transformer import Transformer

# # Configure logging
# logging.basicConfig(
#     filename="debug.log",
#     level=logging.DEBUG,
#     format="%(asctime)s - %(levelname)s - %(message)s"
# )

# # Paths
# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATASET_PATH = os.path.join(BASE_DIR, "dataset", "all_six_datasets")
# PREDICTION_OUTPUT_DIR = os.path.join(BASE_DIR, "predict_output_img")
# MODEL_CHECKPOINTS = {
#     "Autoformer": "./checkpoints/truck1_Autoformer_5G_ftM_sl120_ll60_pl60_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth",
#     "Informer": "./checkpoints/truck1_Informer_5G_ftM_sl120_ll60_pl60_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth",
#     "Transformer": "./checkpoints/truck1_Transformer_5G_ftM_sl120_ll60_pl60_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"
# }

# # Consolidated Configurations for All Models
# CONFIGS = {
#     "Autoformer": {
#         'seq_len': 120,                    # Sequence length
#         'label_len': 60,                   # Lookback window size
#         'pred_len': 60,                    # Prediction length
#         'e_layers': 2,                     # Encoder layers
#         'd_layers': 1,                     # Decoder layers
#         'n_heads': 8,                      # Number of attention heads
#         'd_model': 512,                    # Model embedding dimension
#         'factor': 1,                       # Autoformer factor for AutoCorrelation
#         'enc_in': 12,                      # Number of input features for encoder
#         'dec_in': 12,                      # Number of input features for decoder
#         'c_out': 12,                       # Number of output features
#         'd_ff': 2048,                      # Feed-forward network dimension
#         'moving_avg': 25,                  # Moving average kernel size for decomposition
#         'dropout': 0.1,                    # Dropout rate
#         'activation': 'gelu',              # Activation function
#         'embed': 'timeF',                  # Embedding type (time-frequency-based)
#         'freq': 'h',                       # Frequency for embedding
#         'output_attention': False,         # Whether to output attention weights
#         'learning_rate': 0.0001,           # Learning rate for training
#         'batch_size': 32,                  # Batch size for training
#         'train_epochs': 10,                # Number of training epochs
#         'loss': 'mse',                     # Loss function
#         'use_amp': False,                  # Use automatic mixed precision
#     },
#     "Informer": {
#         'seq_len': 120,                    # Sequence length
#         'label_len': 60,                   # Lookback window size
#         'pred_len': 60,                    # Prediction length
#         'e_layers': 2,                     # Encoder layers
#         'd_layers': 1,                     # Decoder layers
#         'n_heads': 8,                      # Number of attention heads
#         'd_model': 512,                    # Model embedding dimension
#         'factor': 1,                       # Sparsity factor for ProbAttention
#         'enc_in': 12,                      # Encoder input features
#         'dec_in': 12,                      # Decoder input features
#         'c_out': 12,                       # Output features
#         'd_ff': 2048,                      # Feed-forward network dimension
#         'dropout': 0.1,                    # Dropout rate
#         'activation': 'gelu',              # Activation function
#         'embed': 'timeF',                  # Embedding type
#         'output_attention': False,         # Whether to output attention weights
#         'distil': True,                    # Distilling option for encoder
#         'patience': 3,                     # Early stopping patience
#         'learning_rate': 0.0001,           # Learning rate
#         'batch_size': 32,                  # Batch size
#         'train_epochs': 10,                # Training epochs
#         'loss': 'mse',                     # Loss function
#         'use_amp': False,                  # Automatic mixed precision
#     },
#     "Transformer": {
#         'seq_len': 120,                    # Sequence length
#         'label_len': 60,                   # Lookback window size
#         'pred_len': 60,                    # Prediction length
#         'e_layers': 2,                     # Encoder layers
#         'n_heads': 8,                      # Number of attention heads
#         'd_model': 512,                    # Model embedding dimension
#         'bucket_size': 64,                 # Bucket size for Reformer attention
#         'n_hashes': 4,                     # Number of hash rounds for LSH attention
#         'enc_in': 12,                      # Encoder input features
#         'c_out': 12,                       # Output features
#         'd_ff': 2048,                      # Feed-forward network dimension
#         'dropout': 0.1,                    # Dropout rate
#         'activation': 'gelu',              # Activation function
#         'embed': 'timeF',                  # Embedding type
#         'output_attention': False,         # Whether to output attention weights
#         'patience': 3,                     # Early stopping patience
#         'learning_rate': 0.0001,           # Learning rate
#         'batch_size': 32,                  # Batch size
#         'train_epochs': 10,                # Training epochs
#         'loss': 'mse',                     # Loss function
#         'use_amp': False,                  # Automatic mixed precision
#     }
# }


# MODELS = list(MODEL_CHECKPOINTS.keys())
# MODEL_CLASS = {"Autoformer", "Informer", "Transformer" ,"Reformer"}

# DATASETS = {
#     "electricity": "electricity/electricity.csv",
#     "ETTh1": "ETT-small/ETTh1.csv",
#     "ETTh2": "ETT-small/ETTh2.csv",
#     "ETTm1": "ETT-small/ETTm1.csv",
#     "ETTm2": "ETT-small/ETTm2.csv",
#     "exchange_rate": "exchange_rate/exchange_rate.csv",
#     "illness": "illness/national_illness.csv",
#     "traffic": "traffic/traffic.csv",
#     "weather": "weather/weather.csv"
# }

# # Ensure datasets are unzipped
# def ensure_datasets_unzipped():
#     dataset_folder = os.path.join(BASE_DIR, "dataset")
#     zip_path = os.path.join(dataset_folder, "all_six_datasets.zip")
#     all_six_datasets_path = os.path.join(dataset_folder, "all_six_datasets")

#     if not os.path.exists(all_six_datasets_path):
#         if os.path.exists(zip_path):
#             print("Extracting datasets...")
#             with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#                 zip_ref.extractall(dataset_folder)
#             print("Datasets extracted.")
#         else:
#             print(f"Dataset zip file not found at {zip_path}. Please ensure it's present.")
#     else:
#         print("Datasets already unzipped.")

# # Load the model dynamically
# # def load_model(model_name):
# #     try:
# #         checkpoint_path = MODEL_CHECKPOINTS.get(model_name)
        
# #         if not checkpoint_path or not checkpoint_path.strip():
# #             raise FileNotFoundError(f"No checkpoint file found for {model_name}.")
        
# #         # Retrieve the appropriate configuration
# #         config = CONFIGS.get(model_name)
# #         if not config:
# #             raise ValueError(f"No configuration found for model {model_name}.")
        
# #         # Load the checkpoint
# #         checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

# #         # Initialize the model dynamically
# #         if model_name == "Autoformer":
# #             model = Autoformer(config=config)
# #         elif model_name == "Informer":
# #             model = Informer(config=config)
# #         elif model_name == "Transformer":
# #             model = Transformer(config=config)
# #         # elif model_name == "Reformer":
# #         # model =  ...
# #         else:
# #             raise ValueError(f"Unknown model name: {model_name}")

# #         # Load weights
# #         model.load_state_dict(checkpoint['state_dict'])
# #         model.eval()

# #         return model
# #     except Exception as e:
# #         logging.error(f"Error loading model {model_name}: {e}")
# #         raise ValueError(f"Error loading model: {e}")

# # Load the model dynamically
# def load_model(model_name):
#     try:
#         # Verify the checkpoint path
#         checkpoint_path = MODEL_CHECKPOINTS.get(model_name)
#         if not checkpoint_path or not os.path.exists(checkpoint_path):
#             raise FileNotFoundError(f"No checkpoint file found for {model_name} at {checkpoint_path}.")

#         # Load checkpoint
#         checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

#         # Ensure 'state_dict' and 'args' keys exist in checkpoint
#         if 'state_dict' not in checkpoint:
#             raise KeyError(f"'state_dict' key missing in the checkpoint for {model_name}.")
#         if 'args' not in checkpoint:
#             raise KeyError(f"'args' key missing in the checkpoint for {model_name}.")

#         # Extract configuration and initialize the model
#         args = checkpoint['args']  # Namespace object containing configuration
#         model_class = MODEL_CLASS.get(model_name)
#         if not model_class:
#             raise ValueError(f"Model class for {model_name} not found.")
#         model = model_class(**vars(args))  # Unpack args as keyword arguments

#         # Load model weights
#         model.load_state_dict(checkpoint['state_dict'])
#         model.eval()

#         return model
#     except Exception as e:
#         logging.error(f"Error loading model {model_name}: {e}")
#         raise ValueError(f"Error loading model: {e}")


# # Load the dataset
# def load_dataset(dataset_name):
#     try:
#         dataset_path = os.path.join(DATASET_PATH, DATASETS[dataset_name])
#         if not os.path.exists(dataset_path):
#             raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}. Please ensure it's correctly extracted.")

#         # Simulated data loader: Replace this with actual data loading logic
#         data = np.loadtxt(dataset_path, delimiter=",", skiprows=1)  # Adjust delimiter/skiprows if necessary
#         features = data[:, :-1]  # Assuming last column is the target
#         targets = data[:, -1]  # Last column as target

#         return features, targets
#     except Exception as e:
#         logging.error(f"Error loading dataset {dataset_name}: {e}")
#         raise ValueError(f"Error loading dataset: {e}")

# # Visualize predictions
# def visualize_predictions(model_name, dataset_name):
#     try:
#         # Load model
#         model = load_model(model_name)

#         # Load dataset
#         features, targets = load_dataset(dataset_name)

#         # Run predictions
#         predictions = model(torch.tensor(features, dtype=torch.float32)).detach().numpy()

#         # Create the plot
#         plt.figure(figsize=(10, 5))
#         plt.plot(targets, label="GroundTruth", color="blue")
#         plt.plot(predictions, label="Prediction", color="orange")
#         plt.title(f"{model_name} Predictions - {dataset_name}")
#         plt.xlabel("Time Steps")
#         plt.ylabel("Values")
#         plt.legend()
#         plt.grid()

#         # Save plot
#         if not os.path.exists(PREDICTION_OUTPUT_DIR):
#             os.makedirs(PREDICTION_OUTPUT_DIR)
#         plot_path = os.path.join(PREDICTION_OUTPUT_DIR, f"{model_name}_{dataset_name}.png")
#         plt.savefig(plot_path)
#         plt.close()

#         logging.info(f"Plot saved at: {plot_path}")
#         return plot_path, None
#     except Exception as e:
#         logging.error(f"Error generating predictions: {e}")
#         return None, str(e)

# # Ensure datasets are ready
# ensure_datasets_unzipped()

# # Gradio Interface
# with gr.Blocks() as interface:
#     gr.Markdown(
#         """
#         # Model Time Series Visualization

#         Welcome to the front-end web interface developed using **Gradio** for Group 2's project in the subject COS40007 **AI for Engineering** in Fall 2024.  
#         This interface has been specifically designed to provide an intuitive and interactive experience for analyzing time series data in prediction 5G Network generated by pre-trained AI models.

#         The platform supports:
#         - Visualization of model predictions alongside actual data to assess performance.
#         - Integration of checkpoints and datasets for a comprehensive evaluation pipeline.
#         - Seamless interaction with AI outputs tailored for engineering-specific applications.
#         """
#     )

#     with gr.Row():
#         model_input = gr.Dropdown(choices=MODELS, label="Model")
#         dataset_input = gr.Dropdown(choices=list(DATASETS.keys()), label="Dataset")
#     plot_output = gr.Image(type="filepath", label="Plot")
#     error_output = gr.Textbox(label="Error", lines=2)
#     gr.Button("Run").click(fn=visualize_predictions, inputs=[model_input, dataset_input], outputs=[plot_output, error_output])

# interface.launch()

import os
import zipfile
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt
import logging
import torch
from utils.tools import dotdict

from models import Autoformer, Informer, Transformer
from model_arguments import CONFIGS
# from exp.exp_basic import Exp_Basic
# class Model_Loader(Exp_Basic):
#     def __init__(self, args):
#         super(Model_Loader, self).__init__(args)

#     def _build_model(self, model_name):
#         model_dict = {
#             'Autoformer': Autoformer,
#             'Transformer': Transformer,
#             'Informer': Informer,

#         }
#         model = model_dict[model_name].Model(self.args).float()

#         return model

# Configure logging
logging.basicConfig(
    filename="debug.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# Assign each model's parameters to the respective variables
autoformer_params = CONFIGS.get("Autoformer")
informer_params = CONFIGS.get("Informer")
transformer_params = CONFIGS.get("Transformer")

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "all_six_datasets")
PREDICTION_OUTPUT_DIR = os.path.join(BASE_DIR, "predict_output_img")
MODEL_CHECKPOINTS = {
    "Autoformer": "./checkpoints/truck1_Autoformer_5G_ftM_sl120_ll60_pl60_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth",
    "Informer": "./checkpoints/truck1_Informer_5G_ftM_sl120_ll60_pl60_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth",
    "Transformer": "./checkpoints/truck1_Transformer_5G_ftM_sl120_ll60_pl60_dm512_nh8_el2_dl1_df2048_fc1_ebtimeF_dtTrue_Exp_0/checkpoint.pth"
}

# Consolidated Model Classes
MODEL_CLASSES = {
    "Autoformer": Autoformer.Model(dotdict(autoformer_params)),
    "Informer": Informer.Model(dotdict(informer_params)),
    "Transformer": Transformer.Model(dotdict(transformer_params)),
}

# Dataset mapping
# DATASETS = {
#     "electricity": "electricity/electricity.csv",
#     "ETTh1": "ETT-small/ETTh1.csv",
#     "ETTh2": "ETT-small/ETTh2.csv",
#     "ETTm1": "ETT-small/ETTm1.csv",
#     "ETTm2": "ETT-small/ETTm2.csv",
#     "exchange_rate": "exchange_rate/exchange_rate.csv",
#     "illness": "illness/national_illness.csv",
#     "traffic": "traffic/traffic.csv",
#     "weather": "weather/weather.csv"

# }
DATASETS = {
    "truck1": "data/truck1-1.csv",
    "truck2": "data/truck2-1.csv",
    "truck3": "data/truck3-1.csv",
}

# Ensure datasets are unzipped
def ensure_datasets_unzipped():
    dataset_folder = os.path.join(BASE_DIR, "dataset")
    zip_path = os.path.join(dataset_folder, "all_six_datasets.zip")
    all_six_datasets_path = os.path.join(dataset_folder, "all_six_datasets")

    if not os.path.exists(all_six_datasets_path):
        if os.path.exists(zip_path):
            print("Extracting datasets...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(dataset_folder)
            print("Datasets extracted.")
        else:
            print(f"Dataset zip file not found at {zip_path}. Please ensure it's present.")
    else:
        print("Datasets already unzipped.")

# Load the model dynamically
def load_model(model_name):
    try:
        # Verify checkpoint existence
        checkpoint_path = MODEL_CHECKPOINTS.get(model_name)
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"No checkpoint file found for {model_name} at {checkpoint_path}.")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
    
        # Ensure 'state_dict' and 'args' keys exist
        # if 'state_dict' not in checkpoint:
        #     raise KeyError(f"'state_dict' missing in checkpoint for {model_name}.")
        # if 'args' not in checkpoint:
        #     raise KeyError(f"'args' missing in checkpoint for {model_name}.")

        # Extract configuration and state_dict
        # args = checkpoint['args']
        # state_dict = checkpoint['state_dict']

        # Dynamically initialize the model
        model_class = MODEL_CLASSES.get(model_name)
        model_class.load_state_dict(checkpoint, strict=False)
        # for key, value in model_class.state_dict().items():
        #     print('key', key, " value: ", value)
        # if not model_class:
        #     raise ValueError(f"Unknown model name: {model_name}.")
        # #model = model_class(**vars(args))  # Initialize model with args
        # model = model_class()
        # # Load the weights
        # #model.load_state_dict(state_dict, strict=False)
        # model.load_state_dict(checkpoint)
        
        model_class.eval()

        return model_class
    except Exception as e:
        logging.error(f"Error loading model {model_name}: {e}")
        raise ValueError(f"Error loading model: {e}")

# Load the dataset
def load_dataset(dataset_name):
    try:
        dataset_path = os.path.join(DATASET_PATH, DATASETS[dataset_name])
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset {dataset_name} not found at {dataset_path}.")

        # Simulated dataset loader
        data = np.loadtxt(dataset_path, delimiter=",", skiprows=1)  # Adjust delimiter/skiprows if necessary
        features = data[:, :-1]  # Assuming last column is the target
        targets = data[:, -1]
        return features, targets
    except Exception as e:
        logging.error(f"Error loading dataset {dataset_name}: {e}")
        raise ValueError(f"Error loading dataset: {e}")

# Visualize predictions
def visualize_predictions(model_name, dataset_name):
    try:
        # Load model
        model = load_model(model_name)

        # Load dataset
        features, targets = load_dataset(dataset_name)

        # Run predictions
        predictions = model(torch.tensor(features, dtype=torch.float32)).detach().numpy()

        # Create the plot
        plt.figure(figsize=(10, 5))
        plt.plot(targets, label="GroundTruth", color="blue")
        plt.plot(predictions, label="Prediction", color="orange")
        plt.title(f"{model_name} Predictions - {dataset_name}")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")
        plt.legend()
        plt.grid()

        # Save the plot
        if not os.path.exists(PREDICTION_OUTPUT_DIR):
            os.makedirs(PREDICTION_OUTPUT_DIR)
        plot_path = os.path.join(PREDICTION_OUTPUT_DIR, f"{model_name}_{dataset_name}.png")
        plt.savefig(plot_path)
        plt.close()

        logging.info(f"Plot saved at: {plot_path}")
        return plot_path, None
    except Exception as e:
        logging.error(f"Error generating predictions: {e}")
        return None, str(e)

# Ensure datasets are ready
ensure_datasets_unzipped()

# Gradio Interface
with gr.Blocks() as interface:
    gr.Markdown(
        """
        # Model Time Series Visualization

        Visualize predictions from pre-trained models with the corresponding checkpoints and datasets.
        """
    )

    with gr.Row():
        model_input = gr.Dropdown(choices=list(MODEL_CHECKPOINTS.keys()), label="Model")
        dataset_input = gr.Dropdown(choices=list(DATASETS.keys()), label="Dataset")
    plot_output = gr.Image(type="filepath", label="Prediction Plot")
    error_output = gr.Textbox(label="Error", lines=2)
    gr.Button("Run").click(fn=visualize_predictions, inputs=[model_input, dataset_input], outputs=[plot_output, error_output])

interface.launch()
