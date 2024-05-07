import pandas as pd
import matplotlib.pyplot as plt

# Path to the log file
log_file_path = 'outputs/training_log_2024-04-28_14:55:17.txt'

# Reading the file
data = []
with open(log_file_path, 'r') as file:
    for line in file:
        parts = line.strip().split(' - ')
        date_time = parts[0]
        epoch_info = parts[1]

        # Correctly extracting the epoch number and loss values
        print(epoch_info.split(':')[0].split()[1])
        print(epoch_info.split('=')[1].split(',')[0].strip())
        print(epoch_info.split('=')[2].strip())
        epoch_number = int(epoch_info.split(':')[0].split()[1])
        train_loss = float(epoch_info.split('=')[1].split(',')[0].strip())
        val_loss = float(epoch_info.split('=')[2].strip())

        data.append({
            'DateTime': date_time,
            'Epoch': epoch_number,
            'Train Loss': train_loss,
            'Validation Loss': val_loss
        })

# Convert to DataFrame
df = pd.DataFrame(data)

# Plotting
plt.figure(figsize=(10, 5))
plt.plot(df['Epoch'].values, df['Train Loss'].values, label='Train Loss')
plt.plot(df['Epoch'].values, df['Validation Loss'].values, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()
