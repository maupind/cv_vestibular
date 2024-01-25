"""
Contains functions for training and testing a PyTorch model using BoTorch Bayesian Optimisation
"""
import torch
import botorch
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score
from tqdm.auto import tqdm
from typing import Dict, List, Tuple

 # Define the evaluation function for Bayesian optimization
def evaluate(parameters):
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
        model = NeuralNetClassifier(NeuralNetwork, max_epochs=parameters['max_epochs'],
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




botorch_model = BotorchModel(acquisition_function_type='qExpectedImprovement')

init_notebook_plotting()

# Set up BotorchModel for Bayesian optimization
botorch_model = BotorchModel(acquisition_function_type='qExpectedImprovement')

ax_client = AxClient()
ax_client.create_experiment(
    name="test_visualizations",
    parameters = [
            {
                "name": "batch_size",
                "type": "range", 
                "bounds": [16, 128]
            },
            {
                "name": "max_epochs",
                "type": "range", 
                "bounds": [10, 100]
            },
            {
                "name": "learning_rate", 
                "type": "range", 
                "bounds": [1e-4, 1e-1], 
                "log_scale": True
            },
            {
                "name": "weight_decay", 
                "type": "range", 
                "bounds": [1e-6, 1e-2], 
                "log_scale": True
            },
    ],
    objective_name= "auc_score_fold",
    minimize= False,
)


for i in range(50):
    parameters, trial_index = ax_client.get_next_trial()
    ax_client.complete_trial(trial_index=trial_index, raw_data=evaluate(parameters))

model = ax_client.generation_strategy.model
render(interact_contour(model=model, metric_name="auc_score_fold"))

render(ax_client.get_optimization_trace()) 



best_parameters, values = ax_client.get_best_parameters()
best_parameters



# Instantiate and train the best model
best_model = NeuralNetClassifier(NeuralNetwork, 
                                max_epochs=best_parameters['max_epochs'],
                                criterion=torch.nn.BCELoss,
                                device='cuda' if torch.cuda.is_available() else 'cpu',
                                optimizer=optim.Adam,
                                #optimizer__lr=best_parameters['learning_rate'],
                                #optimizer__weight_decay=best_parameters['weight_decay'],
                                iterator_train__batch_size=best_parameters['batch_size'],
                                callbacks=[EarlyStopping(patience=50)]
                                )




# Train the best model on the full training set
best_model.fit(X_train_tensor, y_train_tensor)


