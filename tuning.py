# hyperparameter_tuning.py
import torch
import torch.nn as nn
from torch.optim import Adam
from model import get_model
from data_preparation import get_loader, create_data_splits
from train import train_one_epoch, validate  # Assuming these functions are modular and imported

def tune_hyperparameters():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    param_grid = {
        'learning_rate': [1e-3, 1e-4, 1e-5],
        'batch_size': [2, 4, 8]
    }
    best_loss = float('inf')
    best_params = {}

    for lr in param_grid['learning_rate']:
        for batch_size in param_grid['batch_size']:
            model = get_model(1).to(device)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = Adam(model.parameters(), lr=lr)
            # Load the data
            train_indices, val_indices, test_indices = create_data_splits(1000)
            train_loader = get_loader('data', batch_size, True, subset_indices=train_indices)
            val_loader = get_loader('data', batch_size, False, subset_indices=val_indices)
            #Start training
            print(f"Training with learning rate: {lr} and batch size: {batch_size}")
            for epoch in range(5):  # Shorter number of epochs for quick tuning
                train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
                val_loss = validate(model, val_loader, criterion, device)
                print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

                if val_loss < best_loss:
                    best_loss = val_loss
                    best_params = {'lr': lr, 'batch_size': batch_size}
                    torch.save(model.state_dict(), f'model_best.pth')

    print(f"Best Params: {best_params}, Best Validation Loss: {best_loss}")

if __name__ == "__main__":
    tune_hyperparameters()
