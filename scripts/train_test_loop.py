"""
Contains functions for training and testing a PyTorch model using BoTorch Bayesian Optimisation
"""
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import botorch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
from ax import optimize
from ax.models.torch.botorch import BotorchModel
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.cross_validation import cross_validate
from ax.plot.contour import interact_contour
from ax.plot.diagnostic import interact_cross_validation
from ax.plot.scatter import interact_fitted, plot_objective_vs_constraints, tile_fitted
from ax.plot.slice import plot_slice
from ax.service.ax_client import AxClient, ObjectiveProperties
from ax.utils.notebook.plotting import render, init_notebook_plotting
from ax.utils.testing.mock import fast_botorch_optimize_context_manager
from torchvision import transforms
from model_trainer import train_dataloader, test_dataloader, vest_model
from bayesian_optimisation import evaluate
from model_builder import VestibularNetwork
from data_loader import create_dataloaders
import random

if torch.cuda.is_available():
    print("CUDA is available! You can use the GPU for computation.")
else:
    print("CUDA is not available. You can only use CPU for computation.")

# Utilise device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

botorch_model = BotorchModel(acquisition_function_type='qExpectedImprovement')

init_notebook_plotting()

# Set up BotorchModel for Bayesian optimization
botorch_model = BotorchModel(acquisition_function_type='qExpectedImprovement')

objective_properties_a = ObjectiveProperties(minimize=True)

ax_client = AxClient()
ax_client.create_experiment(
    name="test_visualizations",
    parameters = [
            {
                "name": "batch_size",
                "type": "range", 
                "bounds": [1, 5]
            },
            {
                "name": "max_epochs",
                "type": "range", 
                "bounds": [10, 200]
            },
            {
                "name": "learning_rate", 
                "type": "range", 
                "bounds": [1e-7, 1e-2],
                "log_scale": True
            },
            {
                "name": "weight_decay", 
                "type": "range", 
                "bounds": [1e-8, 1e-2], 
                "log_scale": True
            },
    ],
    objectives={
        "loss": objective_properties_a
    }
)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_random_seed(33)

for i in range(50):
    parameters, trial_index = ax_client.get_next_trial()
    # Evaluate the current set of parameters.
    evaluation_result = evaluate(parameters=parameters,
                                 model=vest_model,
                                 dataloader=train_dataloader,
                                 loss_fn=torch.nn.BCEWithLogitsLoss(),
                                 optimizer=optim.Adam(vest_model.parameters()),
                                 device=device)
    # Ensure evaluation_result is a Python float.
    #if isinstance(evaluation_result, torch.Tensor):
    #    evaluation_result = evaluation_result.item()  # Converts tensor to float

    # Complete the trial with the converted result.
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluation_result)
    #cv = cross_validate(model=ax_client.generation_strategy.model, folds=-1)
   # cv.evaluate(parameters=parameters)
    print(f"finished trial")

print(f"finished optimisation")
model = ax_client.generation_strategy.model
#render(interact_contour(model=model, metric_name="roc_auc_score"))

render(ax_client.get_optimization_trace()) 

  

best_parameters, values = ax_client.get_best_parameters()
best_parameters

print(f"got best parameters")

# Instantiate and train the best model
model = vest_model.to(device)

transform = transforms.Compose([
    transforms.ToPILImage(),    
    transforms.Resize(size=(314, 314)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0,0,0], std=[1,1,1])
])




train_dataloader, test_dataloader, video_dataset = create_dataloaders(
    data_dir="/home/danny/Documents/hpd_clips_test",
    transform=transform,
    test_size=0.2,
    batch_size=1
)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=4e-5)

patience = 10  # Number of epochs to wait before stopping if no improvement is seen
best_val_loss = np.Inf  # Set initial best validation loss to infinity
epochs_without_improvement = 0  # Track epochs without improvement


num_epochs = 75
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    all_true_labels = []
    all_predicted_probabilities = []
    for video_frames_batch, outcomes in train_dataloader:
        X_train_tensor = torch.tensor(video_frames_batch, dtype=torch.float).to(device)
        y_train_tensor = outcomes.float().unsqueeze(1).to(device)
       # X_train_tensor = X_train_tensor.permute(0, 2, 1, 3, 4)
        if torch.any(torch.isnan(X_train_tensor)) | (torch.any(torch.isnan(y_train_tensor))):
            print("nan values found")
        max_norm = 0.25
        #print(f"show the train tensor {X_train_tensor}")

        # Perform forward pass
        print(f"start forward pass")
        outputs = model(X_train_tensor, batch_size = 5)
        #print(f"show the outputs {outputs}")
        #print(f"show the train tensor {X_train_tensor}")
        #print(f"show the true outcomes {y_train_tensor}")
        # Compute loss
        #print("Output shape:", outputs.shape)
        #print("Target shape:", y_train_tensor.shape)
        loss = criterion(outputs, y_train_tensor)
        # Detach the loss from the computation graph, move it to CPU, and convert it to a Python number
        loss_value = loss.detach().cpu().item()
        # Perform backward pass and optimization
        optimizer.zero_grad()

        #print(f"clip grads")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        print(f"start backwards")
        loss.backward()

        #check_gradients(model)
       # for name, param in model.named_parameters():
        #    if param.grad is not None:
         #       print(f"Parameter {name}:")
         #       print(param.grad)
         #   else:
         #       print(f"No gradients for parameter {name}.")
      
        optimizer.step()
        print("optimiser")
        # Store true labels and predicted probabilities for AUC calculation
        all_true_labels.extend(y_train_tensor.cpu().numpy())
        all_predicted_probabilities.extend(torch.sigmoid(outputs).cpu().detach().numpy())
        # Compute ROC AUC score
       # sigmoid = torch.nn.Sigmoid()
       # outputs_probabilities = sigmoid(outputs)
       # print(f"output probabilties{outputs_probabilities}")
       # predictions = (outputs_probabilities > 0.39).float()
       # print(f"predictions{predictions}")
       # print(f"labels{y_train_tensor}")
       # auc_score = roc_auc_score(y_train_tensor.cpu().numpy(), predictions.cpu().numpy())
       # print(f"AUC Score: {auc_score}")
        running_loss += loss.item() * video_frames_batch.size(0)
        print(f"running loss {running_loss}")
    epoch_loss = running_loss / len(train_dataloader.dataset)
   # all_predicted_probabilities.extend(torch.sigmoid(outputs).cpu().detach().numpy())
    # Concatenate all the NumPy arrays into a single NumPy array
    #all_predicted_probabilities_np = np.concatenate(all_predicted_probabilities)
    # Convert the concatenated NumPy array into a PyTorch tensor
   # all_predicted_probabilities_tensor = torch.tensor(all_predicted_probabilities_np)
    #predictions = (all_predicted_probabilities_tensor > 0.39).float()
   # auc_score = roc_auc_score(all_true_labels, predictions)
    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {epoch_loss:.4f}")

print(f"trained model")


# Set model to evaluation mode
model.eval()

# Initialize variables for evaluation
total_loss = 0.0
predictions = []
true_labels = []
correct_predictions = 0

# Iterate through the test data loader
with torch.no_grad():  # No need for gradient calculation during evaluation
    for video_frames_batch, outcomes in test_dataloader:
        X_test_tensor = torch.tensor(video_frames_batch, dtype=torch.float).to(device)
        y_test_tensor = outcomes.float().unsqueeze(1).to(device)

        # Perform forward pass
        outputs = model(X_test_tensor, batch_size=best_parameters['batch_size'])  # Assuming batch_size=1 for testing

        # Compute loss
        loss = criterion(outputs, y_test_tensor)
        total_loss += loss.item() * video_frames_batch.size(0)

        # Convert output probabilities to predictions
        sigmoid = torch.nn.Sigmoid()
        outputs_probabilities = sigmoid(outputs)
        predicted_labels = (outputs_probabilities > 0.39).float()

        # Collect predictions and true labels for further evaluation
        predictions.extend(predicted_labels.cpu().numpy())
        true_labels.extend(outcomes.cpu().numpy())

        correct_predictions += (predicted_labels == y_test_tensor).sum().item()

# Calculate overall loss
eval_loss = total_loss / len(test_dataloader.dataset)

# Calculate evaluation metrics
auc_score = roc_auc_score(true_labels, predictions)

accuracy = correct_predictions / len(test_dataloader.dataset)

# Print or log evaluation results
print(f"Test Loss: {eval_loss:.4f}")
print(f"AUC Score: {auc_score:.4f}")
print(f"Accuracy: {accuracy:.4f}")

print(f"fit model")

torch.save(model.module_.state_dict(), "/home/danny/Documents/Projects/models/cv_dict.pth")
torch.save(model, "/home/danny/Documents/Projects/models/cv.pth")
