import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt

def swish(x):
    return x * torch.sigmoid(x)

# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, fc_data, labels):
        self.fc_data = fc_data
        self.labels = labels

    # 样本数量
    def __len__(self):
        return len(self.fc_data)

    # 根据给定索引idx返回一个样本
    def __getitem__(self, idx):
        fc = self.fc_data[idx]
        label = self.labels[idx]
        return fc, label


class CrossNet(nn.Module):
    def __init__(self):
        super(CrossNet, self).__init__()
        self.conv_row = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, 120), stride=1, padding=0)
        self.conv_col = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(120, 1), stride=1, padding=0)
        self.conv_row2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1, 120), stride=1, padding=0)
        self.conv_col2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(120, 1), stride=1, padding=0)
        self.conv_E2N = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1, 120), stride=1, padding=0)
        #self.relu = nn.LeakyReLU(negative_slope=0.33)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120*64, 30)
        #self.fc2 = nn.Linear(256, 128)
        #self.fc3 = nn.Linear(128, 30)
        self.fc4 = nn.Linear(30, 1)
        self.dropout = nn.Dropout(p=0.5)
        self.dropout_conv = nn.Dropout(p=0.3)
    def forward(self, x):
        # 在行方向上进行卷积
        conv_row_out = self.conv_row(x)
        #print("row", conv_row_out.size())
        # 在列方向上进行卷积
        conv_col_out = self.conv_col(x)
        #print("col", conv_col_out.size())
        # 将行方向和列方向的特征图相加
        x_cat = conv_row_out + conv_col_out
        x = swish(x_cat)
        #x = self.dropout_conv(x)
        #print("x_cat", x.size())
       # xr = self.conv_row2(x)
       # xc = self.conv_col2(x)
        #x = xr + xc
        #x = swish(x)
       # x = self.dropout_conv(x)
        x_E2N = self.conv_E2N(x)
        #print("x_E2N", x_E2N.size())
        x = swish(x_E2N)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        #print("x_E2N", x.size())
        x1 = self.fc1(x)
        x = self.relu(x1)
        x = self.dropout(x)
        #x2 = self.fc2(x)
       #x = self.relu(x2)
        #x = self.dropout(x)
        #x3 = self.fc3(x)
       # x = self.relu(x3)
        #x = self.dropout(x)
        x = self.fc4(x)

        return x

# Load .npy file and .csv file
fc_data = np.load('E:/NIT-zrl/zanyong/hcp1200/data_all_27Jan2018_transpose.npy')
labels_df = pd.read_csv('E:/NIT-zrl/zanyong/hcp1200/lb_new.csv')

# 处理NumPy数组中的NaN值
fc_data = np.nan_to_num(fc_data, nan=0.0)

# Extract feature matrix and target labels
X = fc_data
y = labels_df['ReadEng_AgeAdj'].values
# CogCrystalComp_AgeAdj CogTotalComp_AgeAdj Flanker_AgeAdj PicSeq_AgeAdj
# Find indices of samples without NaN values in y
valid_indices = ~np.isnan(y)

# Filter X and y based on valid indices
X = X[valid_indices]
y = y[valid_indices]
# Split the dataset into train, validation, and test sets
X_train, X_val_test, y_train, y_val_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=42)

# Expand dimensions for grayscale channel
X_train = np.expand_dims(X_train, axis=1)
X_val = np.expand_dims(X_val, axis=1)
X_test = np.expand_dims(X_test, axis=1)

# Convert to tensors and create data loaders
train_dataset = CustomDataset(torch.Tensor(X_train), torch.Tensor(y_train))
val_dataset = CustomDataset(torch.Tensor(X_val), torch.Tensor(y_val))
test_dataset = CustomDataset(torch.Tensor(X_test), torch.Tensor(y_test))
batch_size = 16
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define hyperparameters
input_dim = X.shape[1]  # Number of brain areas (M)
output_dim = 1
learning_rate = 0.0005
num_epochs = 512

# Create model instance
cnn_model = CrossNet()

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(cnn_model.parameters(), lr=learning_rate)
# 创建空列表来存储训练过程中的损失值
train_losses = []
val_losses = []
# Train the model
for epoch in range(num_epochs):
    cnn_model.train()
    train_loss_sum = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = cnn_model(data)
        loss = criterion(output.squeeze(), target.view(-1))
        loss.backward()
        optimizer.step()
        train_loss_sum += loss.item()
        # 计算平均训练损失并保存
    train_loss_avg = train_loss_sum / len(train_loader)
    train_losses.append(train_loss_avg)

    # Evaluate the model
    cnn_model.eval()
    with torch.no_grad():
        train_preds = []
        train_targets = []
        train_mae = []
        for data, target in train_loader:
            output = cnn_model(data)
            train_preds.extend(output.squeeze().tolist())
            train_targets.extend(target.tolist())
            train_mae.extend(torch.abs(output.squeeze() - target).tolist())

        val_preds = []
        val_targets = []
        val_mae = []
        val_loss_sum = 0.0
        for data, target in val_loader:
            output = cnn_model(data)
            loss = criterion(output.squeeze(), target.view(-1))
            val_loss_sum += loss.item()
            val_preds.extend(output.squeeze().tolist())
            val_targets.extend(target.tolist())
            val_mae.extend(torch.abs(output.squeeze() - target).tolist())
            # 计算平均验证损失并保存
        val_loss_avg = val_loss_sum / len(val_loader)
        val_losses.append(val_loss_avg)


        test_preds = []
        test_targets = []
        test_mae = []
        for data, target in test_loader:
            output = cnn_model(data)
            test_preds.extend(output.squeeze().tolist())
            test_targets.extend(target.tolist())
            test_mae.extend(torch.abs(output.squeeze() - target).tolist())

    train_rmse = mean_squared_error(train_targets, train_preds, squared=False)
    val_rmse = mean_squared_error(val_targets, val_preds, squared=False)
    test_rmse = mean_squared_error(test_targets, test_preds, squared=False)
    train_r2 = r2_score(train_targets, train_preds)
    val_r2 = r2_score(val_targets, val_preds)
    test_r2 = r2_score(test_targets, test_preds)
    train_mae = np.mean(train_mae)
    val_mae = np.mean(val_mae)
    test_mae = np.mean(test_mae)

    print(f"Epoch {epoch + 1}/{num_epochs}:")
    print(f"Train RMSE: {train_rmse:.4f} | Train R^2: {train_r2:.4f} | Train MAE: {train_mae:.4f}")
    print(f"Val RMSE: {val_rmse:.4f} | Val R^2: {val_r2:.4f} | Val MAE: {val_mae:.4f}")
    print(f"Test RMSE: {test_rmse:.4f} | Test R^2: {test_r2:.4f} | Test MAE: {test_mae:.4f}")
    print("")

# 绘制loss曲线
plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
# Save the trained model
torch.save(cnn_model.state_dict(), 'language(reading).pth')