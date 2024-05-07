import datetime
import torch
import torch.nn as nn
from torch.optim import Adam
from data_preparation import get_loader, create_data_splits
from model import get_model
from torch.utils.tensorboard import SummaryWriter


# TensorBoard logging 
#writer = SummaryWriter() 
#writer = SummaryWriter('runs/your_experiment_name')
# Define parameters


def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    for images, true_masks in train_loader:
        images = images.to(device)
        true_masks = true_masks.to(device).float()# For binary classification
        #true_masks = true_masks.to(device).long()  # For multi-class
        optimizer.zero_grad()
        pred_masks = model(images)
        loss = criterion(pred_masks.squeeze(1), true_masks)
        # loss = criterion(pred_masks, true_masks) # For multi-class classification tasks
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader)

def validate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for images, true_masks in val_loader:
            images= images.to(device)
            true_masks = true_masks.to(device).float().unsqueeze(1)
            pred_masks = model(images)
            loss = criterion(pred_masks, true_masks)
            val_loss += loss.item()
    return val_loss / len(val_loader)

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for images, true_masks in test_loader:
            images = images.to(device)
            true_masks = true_masks.to(device).float()  # Ensure correct format
            pred_masks = model(images)
            loss = criterion(pred_masks.squeeze(1), true_masks)
            test_loss += loss.item()
    return test_loss / len(test_loader)
def log_results(filepath, epoch, train_loss, val_loss, test_loss=None):

    with open(filepath, 'a') as f:
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"{now} - Epoch {epoch}: Train Loss = {train_loss}, Val Loss = {val_loss}\n")
        if test_loss:
            f.write(f"{now} - Test Loss: {test_loss}\n")

if __name__ == '__main__':

    # Define hyperparameters
    learning_rate = 0.001 #fine tuned
    num_epochs = 25
    num_classes = 1
    batch_size = 2 #fine tuned
    # Define device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print(device.type)

    # Initialize model
    model = get_model(num_classes).to(device)
    criterion = nn.BCEWithLogitsLoss()  # Appropriate for binary classification tasks
    #criterion = nn.CrossEntropyLoss() # For multi-calss classification tasks
    optimizer = Adam(model.parameters(), lr=learning_rate)
    train_indices, val_indices, test_indices = create_data_splits(1000) # Split the data

    # Prepare data loaders
    train_loader = get_loader('data', batch_size, True, subset_indices=train_indices)
    val_loader = get_loader('data', batch_size, False, subset_indices=val_indices)
    test_loader = get_loader('data', batch_size, False, subset_indices=test_indices)

    # File for logging
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    log_filepath = f'outputs/training_log_{now}.txt'
    # Training loop
    for epoch in range(num_epochs):

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss = validate(model, val_loader, criterion, device)
        log_results(log_filepath, epoch, train_loss, val_loss)
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')   

    # Save the model after training
    torch.save(model.state_dict(), 'models/binary_model_ft1.pth')
    print('Model saved to model.pth')
    # Evaluate the model after training and validation
    test_loss = evaluate_model(model, test_loader, criterion, device)
    log_results(log_filepath, epoch, train_loss, val_loss, test_loss)
    print(f'Test Loss: {test_loss}')