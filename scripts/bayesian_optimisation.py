import torch
from data_loader import create_dataloaders

# Utilise device agnostic code
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# Define the evaluation function for Bayesian optimization
def evaluate(parameters: parameters,
             ):
    auc_values = []
    
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
        model = model
        
            NeuralNetClassifier(NeuralNetwork, max_epochs=parameters['max_epochs'],
                                    criterion=torch.nn.BCELoss,
                                    device=device,
                                    optimizer = optim.Adam,
                                    iterator_train__batch_size=parameters['batch_size'],
                                    callbacks=[EarlyStopping(patience=50)]
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

