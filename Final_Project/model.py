from neuralnet import SRCNN_model    # Importing the SRCNN_model class from the neuralnet module
import numpy as np                   # Importing NumPy for numerical operations
import os                            # Importing os module for file and directory operations
from utils.common import exists, tensor2numpy  # Importing utility functions from utils.common
import torch                         # Importing PyTorch

# Logger class to manage logging
class logger:
    def __init__(self, path, values) -> None:
        self.path = path            # Path to save the log
        self.values = values        # List of values to log

# SRCNN class to manage the Super-Resolution Convolutional Neural Network
class SRCNN:
    def __init__(self, architecture, device):
        self.device = device        # Device to run the model (CPU or GPU)
        self.model = SRCNN_model(architecture).to(device)  # Initialize the SRCNN model
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=2e-5)  # Initialize the optimizer
        self.loss = torch.nn.MSELoss()  # Loss function
        self.metric = None          # Metric to evaluate model performance
        self.model_path = None      # Path to save the model
        self.ckpt_path = None       # Path to save the checkpoint
        self.ckpt_man = None        # Checkpoint manager
        
    def load_checkpoint(self, ckpt_path):
        # Load the checkpoint from the specified path
        map_location = torch.device("cpu") if self.device == torch.device("cpu") else self.device
        self.ckpt_man = torch.load(ckpt_path, map_location=map_location)
        self.model.load_state_dict(self.ckpt_man['model'])  # Load the model state
        self.optimizer.load_state_dict(self.ckpt_man['optimizer'])  # Load the optimizer state

    def setup(self, optimizer, loss, metric, model_path, ckpt_path):
        self.optimizer = optimizer  # Set the optimizer
        self.loss = loss            # Set the loss function
        self.metric = metric        # Set the evaluation metric
        self.model_path = model_path  # Set the model path
        self.ckpt_path = ckpt_path  # Set the checkpoint path

    def load_weights(self, filepath):
        # Load model weights from the specified file
        self.model.load_state_dict(torch.load(filepath, map_location=self.device))
        
    def evaluate(self, dataset, batch_size=64):
        # Evaluate the model on the given dataset
        losses, metrics = [], []
        isEnd = False
        while not isEnd:
            lr, hr, isEnd = dataset.get_batch(batch_size, shuffle_each_epoch=False)
            lr, hr = lr.to(self.device), hr.to(self.device)
            sr = self.predict(lr)
            loss = self.loss(hr, sr)
            metric = self.metric(hr, sr)
            losses.append(tensor2numpy(loss))
            metrics.append(tensor2numpy(metric))

        metric = np.mean(metrics)
        loss = np.mean(losses)
        return loss, metric

    def predict(self, lr):
        # Predict the high-resolution image from the low-resolution input
        self.model.train(False)
        with torch.no_grad():
            sr = self.model(lr)
        return sr

    def train(self, train_set, valid_set, batch_size, steps, save_every=1,
              save_best_only=False, save_log=False, log_dir=None):
        
        # Ensure log directory is specified if save_log is True
        if save_log and log_dir is None:
            raise ValueError("log_dir must be specified if save_log is True")
        os.makedirs(log_dir, exist_ok=True)  # Create the log directory if it doesn't exist

        # Initialize loggers for training and validation metrics
        dict_logger = {
            "loss": logger(path=os.path.join(log_dir, "losses.npy"), values=[]),
            "metric": logger(path=os.path.join(log_dir, "metrics.npy"), values=[]),
            "val_loss": logger(path=os.path.join(log_dir, "val_losses.npy"), values=[]),
            "val_metric": logger(path=os.path.join(log_dir, "val_metrics.npy"), values=[])
        }

        # Load existing log values if the log files already exist
        for key in dict_logger.keys():
            path = dict_logger[key].path
            if exists(path):
                dict_logger[key].values = np.load(path).tolist()

        # Initialize the current step and maximum steps
        cur_step = 0
        if self.ckpt_man is not None:
            cur_step = self.ckpt_man['step']
        max_steps = cur_step + steps

        prev_loss = np.inf  # Initialize the previous loss to infinity
        if save_best_only and exists(self.model_path):
            self.load_weights(self.model_path)
            prev_loss, _ = self.evaluate(valid_set)
            self.load_checkpoint(self.ckpt_path)

        loss_buffer = []   # Buffer to store losses
        metric_buffer = []  # Buffer to store metrics
        while cur_step < max_steps:
            cur_step += 1
            lr, hr, _ = train_set.get_batch(batch_size)
            loss, metric = self.train_step(lr, hr)
            loss_buffer.append(tensor2numpy(loss))
            metric_buffer.append(tensor2numpy(metric))

            # Save model and log metrics at specified intervals
            if cur_step % save_every == 0 or cur_step >= max_steps:
                loss = np.mean(loss_buffer)
                metric = np.mean(metric_buffer)
                val_loss, val_metric = self.evaluate(valid_set)
                print(f"Step {cur_step}/{max_steps}",
                      f"- loss: {loss:.7f}",
                      f"- {self.metric.__name__}: {metric:.3f}",
                      f"- val_loss: {val_loss:.7f}",
                      f"- val_{self.metric.__name__}: {val_metric:.3f}")
                if save_log:
                    dict_logger["loss"].values.append(loss)
                    dict_logger["metric"].values.append(metric)
                    dict_logger["val_loss"].values.append(val_loss)
                    dict_logger["val_metric"].values.append(val_metric)
                
                loss_buffer = []
                metric_buffer = []
                torch.save({
                    'step': cur_step,
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                }, self.ckpt_path)

                if save_best_only and val_loss > prev_loss:
                    continue
                prev_loss = val_loss
                torch.save(self.model.state_dict(), self.model_path)
                print(f"Save model to {self.model_path}\n")
        
        if save_log:
            for key in dict_logger.keys():
                logger_obj = dict_logger[key]
                path = logger_obj.path
                values = np.array(logger_obj.values, dtype=np.float32)
                np.save(path, values)
  
    def train_step(self, lr, hr):
        # Perform a single training step
        self.model.train(True)
        self.optimizer.zero_grad()  # Clear the gradients

        lr, hr = lr.to(self.device), hr.to(self.device)
        sr = self.model(lr)  # Forward pass

        loss = self.loss(hr, sr)  # Calculate loss
        metric = self.metric(hr, sr)  # Calculate metric
        loss.backward()  # Backward pass
        self.optimizer.step()  # Update model parameters

        return loss, metric