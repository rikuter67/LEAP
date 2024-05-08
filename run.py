import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from model import MLP  # model.pyからMLPクラスをインポート
import pdb

# デバイスの設定 (CUDAが利用可能かどうかを確認)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_data(file_path, nrows=None):
    print(f"Loading data from {file_path}...")
    return pd.read_csv(file_path, nrows=nrows)

def prepare_dataset(features, targets=None, test=False):
    scaler = StandardScaler()  # 標準化
    features = scaler.fit_transform(features)  # 特徴量のスケーリング
    if test:
        return DataLoader(TensorDataset(torch.tensor(features, dtype=torch.float32)), batch_size=32, shuffle=False)
    else:
        targets = torch.tensor(targets, dtype=torch.float32).view(-1, 368)  # ターゲットを適切に形状変換
        return DataLoader(TensorDataset(torch.tensor(features, dtype=torch.float32), targets), batch_size=32, shuffle=True)

def r_squared(outputs, targets):
    ss_res = ((targets - outputs) ** 2).sum()
    ss_tot = ((targets - targets.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot
    return r2.item()


def train_and_validate(model, train_loader, val_loader, num_epochs=1000, early_stopping_rounds=20, device='cuda'):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    no_improve_epochs = 0

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)  # データとターゲットをデバイスに移動
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        val_loss, val_r2 = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Validation Loss = {val_loss:.4f}, R^2 = {val_r2:.4f}")

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= early_stopping_rounds:
            print("Early stopping triggered.")
            break

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)  # データとターゲットをデバイスに移動
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            predictions.append(output.view(-1))
            actuals.append(target.view(-1))

    avg_loss = total_loss / len(loader)
    r2 = r_squared(torch.cat(predictions), torch.cat(actuals))
    return avg_loss, r2



# Load and preprocess data
train_df = prepare_data('train.csv', nrows=1000000)
features = train_df.iloc[:, 1:-368].values  # Adjust to correct slicing if needed
targets = train_df.iloc[:, -368:].values  # Adjust to correct slicing if needed

# Prepare data loaders
X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=0.2, random_state=42)
train_loader = prepare_dataset(X_train, y_train[:, :369])  # ターゲットの列を選択して渡す
val_loader = prepare_dataset(X_val, y_val[:, :369])  # ターゲットの列を選択して渡す

# モデルの初期化とデバイスへの移動
model = MLP(input_dim=X_train.shape[1], output_dim=368).to(device)

# Train model with early stopping
train_and_validate(model, train_loader, val_loader)

# Load best model for predictions
model.load_state_dict(torch.load('best_model.pth'))

# Predict using the model on CUDA
predictions = []
chunksize = 500  # 適切なチャンクサイズを設定
test_df = prepare_data('test.csv')
for i in range(0, len(test_df), chunksize):
    test_features = test_df.iloc[i:i+chunksize, 1:].values
    test_loader = prepare_dataset(test_features, test=True)
    with torch.no_grad():
        for data in test_loader:
            data = data[0].to(device)  # データをGPUに移動
            output = model(data)
            predictions.append(output.cpu().numpy())  # 結果をCPUに戻してnumpy配列に変換

# Flatten the list of arrays into a single array
predictions = np.vstack(predictions)

# Load only the ID column
ids = pd.read_csv('sample_submission.csv', usecols=[0])

# Create the final DataFrame
submission_df = pd.DataFrame(data=predictions, columns=[f'target_{i}' for i in range(368)])
submission_df.insert(0, 'Id', ids)

# Save to CSV
submission_df.to_csv('submission.csv', index=False)
print("Saved predictions to submission.csv")