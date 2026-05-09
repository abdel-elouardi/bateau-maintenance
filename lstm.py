import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# ← CHANGER CES LIGNES
FICHIER        = 'data/raw/mon_fichier.csv'
CIBLE          = 'target'
COLONNE_GROUPE = 'groupe'  # colonne pour grouper les séquences
SEQUENCE       = 30        # cycles à regarder en arrière

# 1. CHARGER
df = pd.read_csv(FICHIER)
df.columns = df.columns.str.strip()
df = df.drop_duplicates()
cols_constantes = [col for col in df.columns if df[col].nunique() == 1]
df = df.drop(columns=cols_constantes)
print(f"✅ Données : {df.shape}")

# 2. NORMALISER
cols_features = [col for col in df.columns if col not in [CIBLE, COLONNE_GROUPE]]
scaler = StandardScaler()
df[cols_features] = scaler.fit_transform(df[cols_features])

# 3. SÉQUENCES
X_seq, y_seq = [], []
for groupe in df[COLONNE_GROUPE].unique():
    df_g = df[df[COLONNE_GROUPE] == groupe].reset_index(drop=True)
    for i in range(SEQUENCE, len(df_g)):
        X_seq.append(df_g[cols_features].iloc[i-SEQUENCE:i].values)
        y_seq.append(df_g[CIBLE].iloc[i])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
print(f"✅ Séquences : {X_seq.shape}")

# 4. SPLIT
split   = int(0.8 * len(X_seq))
X_train = torch.FloatTensor(X_seq[:split])
X_test  = torch.FloatTensor(X_seq[split:])
y_train = torch.FloatTensor(y_seq[:split])
y_test  = torch.FloatTensor(y_seq[split:])
loader  = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
print(f"✅ Train:{len(X_train)} Test:{len(X_test)}")

# 5. MODÈLE LSTM
class LSTM(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, 64, 2, batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(64, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze()

# 6. ENTRAÎNEMENT
device    = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model     = LSTM(X_train.shape[2]).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
print(f"🚀 Entraînement sur {device}...")
for epoch in range(1, 51):
    model.train()
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        optimizer.step()
    if epoch % 10 == 0:
        print(f"  Epoch {epoch}/50 | Loss: {loss.item():.4f}")

# 7. ÉVALUATION
model.eval()
with torch.no_grad():
    y_pred = model(X_test.to(device)).cpu().numpy()
rmse = np.sqrt(mean_squared_error(y_test.numpy(), y_pred))
r2   = r2_score(y_test.numpy(), y_pred)
print(f"\n📊 LSTM — RMSE:{rmse:.2f} | R2:{r2:.4f}")

# 8. SAUVEGARDER
torch.save(model.state_dict(), 'models/lstm.pt')
print("✅ Modèle LSTM sauvegardé")