from utils.dataset import dataset  # Importing the dataset module from utils.dataset
from utils.common import PSNR      # Importing the PSNR function from utils.common
from model import SRCNN            # Importing the SRCNN model from the model module
import argparse                    # Importing the argparse module for command-line argument parsing
import torch                       # Importing PyTorch
import os                          # Importing the os module for operating system related functions

# Setting up command-line argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--steps",          type=int, default=100000, help='Number of training steps')
parser.add_argument("--batch-size",     type=int, default=128,    help='Batch size for training')
parser.add_argument("--architecture",   type=str, default="915",  help='Model architecture version')
parser.add_argument("--save-every",     type=int, default=1000,   help='Save model every N steps')
parser.add_argument("--save-log",       type=int, default=0,      help='Save logs if set to 1')
parser.add_argument("--save-best-only", type=int, default=0,      help='Save only the best model if set to 1')
parser.add_argument("--ckpt-dir",       type=str, default="",     help='Checkpoint directory')

# -----------------------------------------------------------
# Global variables
# -----------------------------------------------------------

FLAGS, unparsed = parser.parse_known_args()
steps = FLAGS.steps
batch_size = FLAGS.batch_size
save_every = FLAGS.save_every
save_log = (FLAGS.save_log == 1)
save_best_only = (FLAGS.save_best_only == 1)

# Ensure the architecture is valid
architecture = FLAGS.architecture
if architecture not in ["915", "935", "955"]:
    raise ValueError("architecture must be 915, 935, or 955")

# Set checkpoint directory and paths
ckpt_dir = FLAGS.ckpt_dir
if (ckpt_dir == "") or (ckpt_dir == "default"):
    ckpt_dir = f"checkpoint/SRCNN{architecture}"

model_path = os.path.join(ckpt_dir, f"SRCNN-{architecture}.pt")
ckpt_path = os.path.join(ckpt_dir, "ckpt.pt")

# -----------------------------------------------------------
# Init datasets
# -----------------------------------------------------------

dataset_dir = "dataset"  # Directory containing the datasets
lr_crop_size = 33        # Low-resolution crop size
hr_crop_size = 21        # High-resolution crop size
if architecture == "935":
    hr_crop_size = 19
elif architecture == "955":
    hr_crop_size = 17

# Initialize and load training dataset
train_set = dataset(dataset_dir, "train")
train_set.generate(lr_crop_size, hr_crop_size)
train_set.load_data()

# Initialize and load validation dataset
valid_set = dataset(dataset_dir, "validation")
valid_set.generate(lr_crop_size, hr_crop_size)
valid_set.load_data()

# -----------------------------------------------------------
# Checkpoint Inspection
# -----------------------------------------------------------
def inspect_checkpoint(ckpt_path):
    map_location = torch.device("cpu")  # Load checkpoint on CPU
    checkpoint = torch.load(ckpt_path, map_location=map_location)
    print("Checkpoint keys:", checkpoint.keys())

# -----------------------------------------------------------
# Train
# -----------------------------------------------------------
def main():
    # Determine the device to use (MPS for Apple Silicon, CUDA for GPU, or CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inspect the checkpoint
    inspect_checkpoint(ckpt_path)

    # Initialize the SRCNN model with the specified architecture and device
    srcnn = SRCNN(architecture, device)
    srcnn.setup(optimizer=srcnn.optimizer,
                loss=srcnn.loss,
                model_path=model_path,
                ckpt_path=ckpt_path,
                metric=PSNR)

    # Load checkpoint and start training
    srcnn.load_checkpoint(ckpt_path)
    srcnn.train(train_set, valid_set, steps=steps, batch_size=batch_size,
                save_best_only=save_best_only, save_every=save_every,
                save_log=save_log, log_dir=ckpt_dir)

if __name__ == "__main__":
    main()  # Execute the main function
