# examples/train_mlnise_example.py

"""
Example: Training an MLNISE model in Hogwild mode using a training dataset.
"""

import torch
from torchnise.nise import MLNISEModel  # Or your MLNISE class if you renamed it
from torchnise.mlnise_dataset import MLNiseDrudeDataset
from torchnise.train_mlnise import train_mlnise_hogwild


# 1) Load the training dataset from disk (created by create_datasets.py)
train_ds = MLNiseDrudeDataset(
    dataset_folder="GeneratedHeom",
    dataset_name="mlnise_train_data",
    generate_if_missing=False  # will load existing data
)
print(f"Loaded training dataset: length = {len(train_ds)}")

# 2) Instantiate the MLNISE model
model = MLNISEModel()
model.share_memory()  # needed for Hogwild multi-process
# 3) load previouis checkopoint if available
#model.load_state_dict(torch.load("mlnise_model_final4.pt"))
# 4) Train with Hogwild
device = torch.device("cpu")  # or "cuda" if you have a GPU
trained_model = train_mlnise_hogwild(
    model=model,
    dataset=train_ds,
    num_epochs=3,
    num_processes=16,
    learning_rate=0.0005,
    runname="mlnise_demo_run",
    device=device,
    realizations=100  
)

# 4) Save final model weights
torch.save(trained_model.state_dict(), "mlnise_model_final.pt")
print("Training complete. Model saved to 'mlnise_model_final.pt'.")

