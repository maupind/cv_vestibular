import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from tqdm.auto import tqdm
from typing import Dict, List, Tuple
import numpy as np
from botorch.cross_validation import batch_cross_validation, gen_loo_cv_folds
from botorch.models import FixedNoiseGP
from gpytorch.mlls import ExactMarginalLogLikelihood
import torch



# Utilise device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=33)


# Define the evaluation function for Bayesian optimization
def evaluate(parameters: str,
             model: torch.nn.Module,
             dataloader: torch.utils.data.DataLoader,
             loss_fn: torch.nn.Module, 
             optimizer: torch.optim.Optimizer,
             device: torch.device) -> Tuple[float, float]:
    auc_values = []
    loss_values = []
    accuracy_values = []
    X_train_list = []
    y_train_list = []  

    max_norm = 1.0

    model = model.to(device)
    model.train()

    all_outputs = torch.tensor([], dtype=torch.float).to(device)
    all_labels = torch.tensor([], dtype=torch.float).to(device)

    # Your existing code for setting up the optimizer

    for batch_idx, (video_frames_batch, outcomes) in enumerate(dataloader):
        X_train_tensor = torch.tensor(video_frames_batch, dtype=torch.float).to(device)
        y_train_tensor = outcomes.float().unsqueeze(1).to(device)
        if torch.any(torch.isnan(X_train_tensor)) | (torch.any(torch.isnan(y_train_tensor))):
            print("nan values found")

        #print(f"show the train tensor {X_train_tensor}")

        # Perform forward pass
        outputs = model(X_train_tensor, batch_size = 1)
        #print(f"show the outputs {outputs}")
        #print(f"show the train tensor {X_train_tensor}")
        #print(f"show the true outcomes {y_train_tensor}")
        # Compute loss
        #print("Output shape:", outputs.shape)
        #print("Target shape:", y_train_tensor.shape)
        loss = loss_fn(outputs, y_train_tensor)
        # Detach the loss from the computation graph, move it to CPU, and convert it to a Python number
        loss_value = loss.detach().cpu().item()

        # Append the detached loss value to the list
        loss_values.append(loss_value)
        print("calculate loss")
        # Perform backward pass and optimization
        optimizer.zero_grad()
        print("zero grad")
        print(f"show the loss {loss}")
        #print(f"clip grads")
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        print(f"start backwards")
        loss.backward()
        #print("backward loss")
        optimizer.step()
        print("optimiser")
        # Compute ROC AUC score
        #all_outputs = torch.cat((all_outputs, outputs.detach()), dim=0)
        #all_labels = torch.cat((all_labels, y_train_tensor), dim=0)
        #print("auc complete")
        # Calculate accuracy
        predictions = (outputs > 0.5).float()  # Assuming binary classification
        correct_predictions = (predictions == y_train_tensor).float().sum().item()
        accuracy = correct_predictions / len(y_train_tensor)
        accuracy_values.append(accuracy)


    # Compute average loss
    avg_loss = sum(loss_values) / len(loss_values)
    avg_accuracy = sum(accuracy_values) / len(accuracy_values)
    #auc_score = roc_auc_score(all_outputs.cpu().numpy(), all_labels.cpu().numpy())
    #print(f"AUC Score over all data: {auc_score}")


    # Compute average ROC AUC score
    return{'accuracy': avg_accuracy}

    #for batch_idx, (video_frames_batch, outcomes) in enumerate(dataloader):
        # Convert video_frames_batch and label_features_batch to tensors
       # print("Label Features non tensor", label_features_batch)
     #   X_train_tensor = torch.tensor(video_frames_batch, dtype=torch.float)  # Assuming your video frames are of type float
      #  print(X_train_tensor.shape)
       # label_features_tensor_list = [torch.tensor(tensor_item) for tensor_item in label_features_batch] 
        #print("Pre Label Features after creating list of tensors", label_features_tensor_list)
        
        #label_features_tensor = torch.Tensor(label_features_tensor_list)
        #print("Pre Label Features after joining list of tensors", label_features_tensor.shape)
        #label_features_tensor_match = label_features_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4)

        #print("Video Frames", video_frames_tensor.shape)
        #print("Label Features", label_features_tensor_match.shape)
        # Append tensors and labels to lists
        #X_train_list.append((video_frames_tensor, label_features_tensor))
        
        # Convert outcomes_batch to a list of flat labels
       # outcomes_flat = [label.item() for label in outcomes]
        #y_train_list.append(outcomes_flat)

    # Concatenate the list of tensors along the batch dimension
       # X_train_tensor_list = [torch.cat((video_frames_tensor, label_features_tensor_match), dim=1).to(device) for video_frames_tensor, label_features_tensor in X_train_list]
        #y_train_tensor = torch.tensor(y_train_list)
        #print(y_train_tensor.shape)
        #y_var_tensor = torch.full_like(y_train_tensor, 0.2)


        # Train your model with the given hyperparameters
        #model = model(max_epochs=parameters['max_epochs'],
         #                       criterion=torch.nn.BCELoss,
          #                      device=device,
           #                     optimizer = optimizer,
            #                    iterator_train__batch_size=parameters['batch_size']
             #                   )
        #print(f"Model Defined")
        #model.fit(X_train_tensor, y_train_tensor)

        #print(f"Model Fit")

        # Predict probabilities for validation set
       ## y_val_pred_probs = model.predict_proba(X_val_fold)[:, 1]

        #print(f"probabilities predicted")

        #y_val_fold_cpu = y_val_fold.cpu().numpy()

        #print(f"converted validation fols to cpu")

        # Compute AUC for validation set
        #auc_score_fold = roc_auc_score(y_val_fold_cpu, y_val_pred_probs)
        #auc_values.append(auc_score_fold)

        #print(f"calculated AUC")
        #if auc_score_fold > best_auc:
         #   best_auc = auc_score_fold
          #  return{'auc_score_fold': auc_score_fold}

        # Generate k-fold cross-validation folds
        #cv_folds = gen_loo_cv_folds(
        #    train_X=X_train_tensor,
        #    train_Y=y_train_tensor
        #)

        # Perform cross-validation
        #model_cls = FixedNoiseGP
        #mll_cls = ExactMarginalLogLikelihood

        #cv_results = batch_cross_validation(
         #   model_cls=model_cls,
          #  mll_cls=mll_cls,
           # cv_folds=2,
            #fit_args=None,
            #observation_noise=False,
        #)

        # Compute the mean and standard deviation of the performance metric
        #mean_auc_score_fold = cv_results.mean().item()
        #std_auc_score_fold = cv_results.std().item()
  
    #for train_idx, val_idx in skf.split(X_train_list, y_train_list):
        #train_idx = np.array(train_idx, dtype=int)  # Convert train_idx to a numpy array
        #val_idx = np.array(val_idx, dtype=int)  # Convert val_idx to a numpy array

        #X_train_fold, X_val_fold = X_train_list[train_idx], X_train_list[val_idx]
        #y_train_fold, y_val_fold = y_train_list[train_idx], y_train_list[val_idx]


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




