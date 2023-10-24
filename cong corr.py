import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LinearRegression
# Custom dataset class
class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, fc_data, labels):
        self.fc_data = fc_data
        self.labels = labels

    def __len__(self):
        return len(self.fc_data)

    def __getitem__(self, idx):
        fc = self.fc_data[idx]
        label = self.labels[idx]
        return fc, label

def swish(x):
    return x * torch.sigmoid(x)

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

# 读取数据
# Load .npy file and .csv file
fc_data = np.load('E:/NIT-zrl/zanyong/hcp1200/data_all_27Jan2018_transpose.npy')
labels_df = pd.read_csv('E:/NIT-zrl/zanyong/hcp1200/lb_new.csv')
# 处理NumPy数组中的NaN值
fc_data = np.nan_to_num(fc_data, nan=0.0)
# 处理Pandas DataFrame中的NaN值
# labels.fillna(0.0, inplace=True)
X = fc_data
y = labels_df["Flanker_AgeAdj"].values
# PicSeq_AgeAdj CogCrystalComp_AgeAdj ReadEng_AgeAdj
# Find indices of samples without NaN values in y
valid_indices = ~np.isnan(y)
# Filter X and y based on valid indices
X = X[valid_indices]
y = y[valid_indices]
# Convert to tensors and create dataset
dataset = CustomDataset(torch.unsqueeze(torch.Tensor(X), dim=1), torch.Tensor(y))
batch_size = 16
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

# Create model instance
model_path = 'predict_Flanker attention.pth'
model = CrossNet()
model.load_state_dict(torch.load(model_path))
model.eval()

predictions = []
labels = []
gaps = []

with torch.no_grad():
    for data, target in data_loader:
        output = model(data)
        predictions.extend(output.squeeze().tolist())
        labels.extend(target.tolist())
        gaps.extend((output.squeeze() - target).tolist())

# Convert predictions, labels, and gaps to NumPy arrays
predictions = np.array(predictions)
labels = np.array(labels)
# Calculate GAP values
gaps = predictions - labels
gaps = np.array(gaps)

# Create a DataFrame to store the results
results_df = pd.DataFrame({'prediction': predictions, 'label': labels, 'diff': gaps})

# Save results to a CSV file
results_df.to_csv('attention.csv', index=False)

# Perform linear regression
regression_model = LinearRegression()
regression_model.fit(labels.reshape(-1, 1), predictions.reshape(-1, 1))
regression_line = regression_model.predict(labels.reshape(-1, 1))

# Plot scatter plot with regression line and diagonal line
plt.scatter(labels, predictions, s=10, alpha=0.6)
plt.plot(labels, regression_line, color='m', linestyle=':', label='Regression Line')
plt.plot(labels, labels, color='r', linestyle='--', label='Diagonal Line')
plt.xlabel('label')
plt.ylabel('prediction')
plt.title('Attention')
plt.legend()
plt.grid(True)
plt.show()
