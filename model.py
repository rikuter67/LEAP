import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_dim, 100)  # 1層目のユニット数を100に設定
        self.relu = nn.ReLU()  # 活性化関数
        self.layer2 = nn.Linear(100, output_dim)  # 出力層のユニット数を368に設定（ターゲットの特徴量数に合わせる）

    def forward(self, x):
        x = self.relu(self.layer1(x))  # 1層目の処理
        x = self.layer2(x)  # 出力層の処理
        return x
