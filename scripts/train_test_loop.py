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
from model_trainer import train_dataloader, test_dataloader, vest_model
from bayesian_optimisation import evaluate
from model_builder import VestibularNetwork
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

ax_client = AxClient()
ax_client.create_experiment(
    name="test_visualizations",
    parameters = [
           # {
            #    "name": "batch_size",
            #    "type": "range", 
            #    "bounds": [16, 128]
            #},
            {
                "name": "max_epochs",
                "type": "range", 
                "bounds": [1, 20]
            },
            {
                "name": "learning_rate", 
                "type": "range", 
                "bounds": [1e-4, 1e-3],
                "log_scale": True
            },
            {
                "name": "weight_decay", 
                "type": "range", 
                "bounds": [1e-6, 1e-2], 
                "log_scale": True
            },
    ],
    #objective_name= "accuracy",
    #minimize= False
)

def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

set_random_seed(33)

for i in range(5):
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
    #cv.evaluate(parameters=parameters)
    print(f"finished trial")

print(f"finished optimisation")
model = ax_client.generation_strategy.model
#render(interact_contour(model=model, metric_name="avg_accuracy"))

render(ax_client.get_optimization_trace()) 



best_parameters, values = ax_client.get_best_parameters()
best_parameters

print(f"got best parameters")

# Instantiate and train the best model
best_model = VestibularNetwork(max_epochs=best_parameters['max_epochs'],
                                criterion=torch.nn.BCELoss,
                                device='cuda' if torch.cuda.is_available() else 'cpu',
                                optimizer=optim.Adam,
                                #optimizer__lr=best_parameters['learning_rate'],
                                #optimizer__weight_decay=best_parameters['weight_decay'],
                                iterator_train__batch_size=best_parameters['batch_size']
                                )

print(f"trained model")


# Train the best model on the full training set
best_model.fit(X_train_tensor, y_train_tensor)

print(f"fit model")
