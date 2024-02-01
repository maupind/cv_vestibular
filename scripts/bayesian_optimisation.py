import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

# Utilise device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Define the evaluation function for Bayesian optimization
def evaluate(parameters: str,
             model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module, 
             optimizer: torch.optim.Optimizer,
             device: torch.device) -> Tuple[float, float]:
    auc_values = []
    X_train_list = []
    y_train_list = []

    # Assuming train_loader is your PyTorch DataLoader containing (features, labels) tuples
    for batch in dataloader:
        features, labels = batch
        X_train_list.append(features)
        y_train_list.append(labels)

    # Concatenate the list of tensors along the batch dimension
    X_train_tensor = torch.cat(X_train_list, dim=0)
    y_train_tensor = torch.cat(y_train_list, dim=0)
    
    for train_idx, val_idx in skf.split(X_train_tensor.cpu().numpy(), y_train_tensor.cpu().numpy()):
        X_train_fold, X_val_fold = X_train_tensor[train_idx].to(device), X_train_tensor[val_idx].to(device)
        y_train_fold, y_val_fold = y_train_tensor[train_idx].to(device), y_train_tensor[val_idx].to(device)

        # Determine the optimizer based on the sampled parameters
    # optimizer_choice = parameters['optimizer']
    # if optimizer_choice == 'adam':
    #     optimizer = torch.optim.Adam(lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'])
    # elif optimizer_choice == 'adagrad':
    #     optimizer = torch.optim.Adagrad(lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'])
    # elif optimizer_choice == 'sgd':
    #     optimizer = torch.optim.SGD(lr=parameters['learning_rate'], weight_decay=parameters['weight_decay'])
    # else:
    #     raise ValueError(f"Invalid optimizer choice: {optimizer_choice}")


        # Train your model with the given hyperparameters
        model = model(max_epochs=parameters['max_epochs'],
                                    criterion=torch.nn.BCELoss,
                                    device=device,
                                    optimizer = optimizer,
                                    iterator_train__batch_size=parameters['batch_size']
                                    )
        print(f"Model Defined")
        model.fit(X_train_fold, y_train_fold)

        print(f"Model Fit")

        # Predict probabilities for validation set
        y_val_pred_probs = model.predict_proba(X_val_fold)[:, 1]

        print(f"probabilities predicted")

        y_val_fold_cpu = y_val_fold.cpu().numpy()

        print(f"converted validation fols to cpu")

        # Compute AUC for validation set
        auc_score_fold = roc_auc_score(y_val_fold_cpu, y_val_pred_probs)
        auc_values.append(auc_score_fold)

        print(f"calculated AUC")
        # if auc_score_fold > best_auc:
        best_auc = auc_score_fold
        return{'auc_score_fold': auc_score_fold}

