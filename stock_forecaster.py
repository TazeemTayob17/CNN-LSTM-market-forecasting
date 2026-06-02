"""
S&P 500 Stock Price Forecasting using Deep Learning
COMS 4030A / COMS7047A — Adaptive Computation and Machine Learning

Unified multi-stock next-day adjusted closing price regression using PyTorch.
4 architectures: LSTM, BiLSTM+Attention, CNN-LSTM, Transformer
5 experiments: window size, feature ablation, dropout, horizon, L2 regularisation

Dataset: sp500_top10_stocks_clean.csv (9 stocks, 2010-2026)
Framework: PyTorch 2.2.2 on Python 3.12
"""

import warnings
warnings.filterwarnings('ignore')

import os, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

SEQUENCE_LENGTH = 60
BATCH_SIZE = 64
EPOCHS = 100
LEARNING_RATE = 1e-3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"PyTorch version: {torch.__version__}")
print(f"Device: {DEVICE}")
print(f"Sequence length: {SEQUENCE_LENGTH} days")
print(f"Batch size: {BATCH_SIZE}")
print()

print("=" * 80)
print("SECTION 1: Loading data and performing minimal EDA")
print("=" * 80)

DATA_PATH = 'sp500_top10_stocks_clean.csv'
df = pd.read_csv(DATA_PATH, parse_dates=['Date'])
df.sort_values(['Ticker', 'Date'], inplace=True)
df.reset_index(drop=True, inplace=True)

print(f"Shape: {df.shape}")
print(f"Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
print(f"Tickers: {sorted(df['Ticker'].unique())}")
print(f"Missing values: {df.isnull().sum().sum()}")
print()

tickers = sorted(df['Ticker'].unique())

fig, axes = plt.subplots(3, 3, figsize=(16, 10), sharex=True)
axes = axes.flatten()
for i, ticker in enumerate(tickers):
    sub = df[df['Ticker'] == ticker].copy()
    norm_price = (sub['Adj_Close'] - sub['Adj_Close'].min()) / (sub['Adj_Close'].max() - sub['Adj_Close'].min())
    axes[i].plot(sub['Date'], norm_price, linewidth=0.8, color=f'C{i}')
    axes[i].set_title(ticker, fontsize=11, fontweight='bold')
    axes[i].set_ylim(0, 1)
    axes[i].xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[i].xaxis.set_major_locator(mdates.YearLocator(4))
fig.suptitle('Normalised Adjusted Closing Prices — All Stocks (2010–2026)', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('fig_01_price_history.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_01_price_history.png")

pivot = df.pivot_table(index='Date', columns='Ticker', values='Adj_Close')
corr = pivot.pct_change().corr()
fig, ax = plt.subplots(figsize=(8, 6))
mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            linewidths=0.5, ax=ax, cbar_kws={'label': 'Pearson r (daily returns)'})
ax.set_title('Cross-Stock Return Correlation', fontsize=13)
plt.tight_layout()
plt.savefig('fig_02_correlation.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_02_correlation.png")

fig, ax = plt.subplots(figsize=(12, 5))
df_vol = df.copy()
df_vol['log_Volume'] = np.log1p(df_vol['Volume'])
df_vol.boxplot(column='log_Volume', by='Ticker', ax=ax, patch_artist=True)
ax.set_xlabel('Ticker')
ax.set_ylabel('log(1 + Volume)')
ax.set_title('Daily Volume Distribution by Stock')
plt.suptitle('')
plt.tight_layout()
plt.savefig('fig_03_volume.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_03_volume.png")
print()

print("=" * 80)
print("SECTION 2: Feature Engineering")
print("=" * 80)

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def add_features(group):
    g = group.copy().sort_values('Date')
    ac = g['Adj_Close']
    vol = g['Volume']

    g['SMA_5'] = ac.rolling(5).mean()
    g['SMA_20'] = ac.rolling(20).mean()
    g['SMA_50'] = ac.rolling(50).mean()
    g['EMA_12'] = ac.ewm(span=12, adjust=False).mean()
    g['EMA_26'] = ac.ewm(span=26, adjust=False).mean()
    g['MACD'] = g['EMA_12'] - g['EMA_26']
    g['RSI_14'] = compute_rsi(ac, 14)

    bb_mid = ac.rolling(20).mean()
    bb_std = ac.rolling(20).std()
    g['BB_upper'] = bb_mid + 2 * bb_std
    g['BB_lower'] = bb_mid - 2 * bb_std
    g['BB_bw'] = (g['BB_upper'] - g['BB_lower']) / (bb_mid + 1e-10)

    g['Volume_MA_10'] = vol.rolling(10).mean()
    g['Daily_Return'] = ac.pct_change()
    g['Volatility_10'] = g['Daily_Return'].rolling(10).std()

    return g

df_feat = df.groupby('Ticker', group_keys=False).apply(add_features)
ohe = pd.get_dummies(df_feat['Ticker'], prefix='T').astype(float)
df_feat = pd.concat([df_feat.reset_index(drop=True), ohe.reset_index(drop=True)], axis=1)
df_feat.dropna(inplace=True)
df_feat.reset_index(drop=True, inplace=True)

ticker_cols = [c for c in df_feat.columns if c.startswith('T_')]
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume',
                'SMA_5', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26', 'MACD',
                'RSI_14', 'BB_upper', 'BB_lower', 'BB_bw',
                'Volume_MA_10', 'Daily_Return', 'Volatility_10'] + ticker_cols
TARGET_COL = 'Adj_Close'
scale_cols = [c for c in FEATURE_COLS if c not in ticker_cols]
target_idx_in_scale = scale_cols.index(TARGET_COL)

print(f"Shape after feature engineering: {df_feat.shape}")
print(f"Features per timestep: {len(FEATURE_COLS)}")
print()

print("=" * 80)
print("SECTION 3: Preprocessing & Dataset Creation")
print("=" * 80)

TRAIN_END = '2022-12-31'
VAL_END = '2024-06-30'
SEQ_LEN = SEQUENCE_LENGTH

scalers = {}
adj_close_scalers = {}

def make_sequences(arr, seq_len):
    X, y = [], []
    for i in range(seq_len, len(arr)):
        X.append(arr[i - seq_len:i])
        y.append(arr[i, target_idx_in_scale])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train_all, y_train_all = [], []
X_val_all, y_val_all = [], []
X_test_all, y_test_all = [], []

for ticker in tickers:
    sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date').copy()
    train = sub[sub['Date'] <= TRAIN_END]
    val = sub[(sub['Date'] > TRAIN_END) & (sub['Date'] <= VAL_END)]
    test = sub[sub['Date'] > VAL_END]

    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train[scale_cols])
    val_scaled = scaler.transform(val[scale_cols])
    test_scaled = scaler.transform(test[scale_cols])
    scalers[ticker] = scaler

    adj_scaler = MinMaxScaler()
    adj_scaler.fit(train[[TARGET_COL]])
    adj_close_scalers[ticker] = adj_scaler

    def attach_ohe(scaled, subset):
        ohe_vals = subset[ticker_cols].values
        return np.hstack([scaled, ohe_vals])

    tr = attach_ohe(train_scaled, train)
    va = attach_ohe(val_scaled, val)
    te = attach_ohe(test_scaled, test)

    Xtr, ytr = make_sequences(tr, SEQ_LEN)
    Xva, yva = make_sequences(va, SEQ_LEN)
    Xte, yte = make_sequences(te, SEQ_LEN)

    X_train_all.append(Xtr)
    y_train_all.append(ytr)
    X_val_all.append(Xva)
    y_val_all.append(yva)
    X_test_all.append(Xte)
    y_test_all.append(yte)

X_train = np.concatenate(X_train_all, axis=0)
y_train = np.concatenate(y_train_all, axis=0).reshape(-1, 1)
X_val = np.concatenate(X_val_all, axis=0)
y_val = np.concatenate(y_val_all, axis=0).reshape(-1, 1)
X_test = np.concatenate(X_test_all, axis=0)
y_test = np.concatenate(y_test_all, axis=0).reshape(-1, 1)

N_FEATURES = X_train.shape[2]

print(f"Train: {X_train.shape}  y_train: {y_train.shape}")
print(f"Val:   {X_val.shape}    y_val:   {y_val.shape}")
print(f"Test:  {X_test.shape}   y_test:  {y_test.shape}")
print(f"Number of features per timestep: {N_FEATURES}")
print()

class StockDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.y[i]

train_loader = DataLoader(StockDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(StockDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(StockDataset(X_test, y_test), batch_size=BATCH_SIZE, shuffle=False)

print("DataLoaders created.")
print()

def train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LEARNING_RATE,
                patience=10, verbose=True, label=''):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5,
                                                      patience=5, verbose=False)
    criterion = nn.MSELoss()

    history = {'train_loss': [], 'val_loss': [], 'lr': []}
    best_val, best_state, no_improve = float('inf'), None, 0

    for epoch in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(Xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            t_loss += loss.item() * len(Xb)
        t_loss /= len(train_loader.dataset)

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                v_loss += criterion(model(Xb), yb).item() * len(Xb)
        v_loss /= len(val_loader.dataset)

        scheduler.step(v_loss)
        cur_lr = optimizer.param_groups[0]['lr']
        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['lr'].append(cur_lr)

        if v_loss < best_val:
            best_val = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                if verbose:
                    print(f"  [{label}] Early stop @ epoch {epoch}")
                break

        if verbose and epoch % 10 == 0:
            print(f"  [{label}] Ep {epoch:3d}  train={t_loss:.6f}  val={v_loss:.6f}  lr={cur_lr:.2e}")

    model.load_state_dict(best_state)
    model.to(DEVICE)
    return history

def predict(model, loader):
    model.eval()
    preds = []
    with torch.no_grad():
        for Xb, _ in loader:
            preds.append(model(Xb.to(DEVICE)).cpu().numpy())
    return np.concatenate(preds, axis=0)

def rmse(a, b):
    return float(np.sqrt(mean_squared_error(a, b)))

def mape(a, b):
    return float(np.mean(np.abs((a - b) / (np.abs(a) + 1e-8))) * 100)

def dir_acc(a, b):
    da = np.diff(a.flatten())
    db = np.diff(b.flatten())
    return float(np.mean(np.sign(da) == np.sign(db)) * 100)

all_histories = {}
all_test_preds = {}

print("=" * 80)
print("SECTION 4: Training 4 Deep Learning Models")
print("=" * 80)

#Model 1: Stacked LSTM
print("\n[1/4] Training Stacked LSTM...")

class StackedLSTM(nn.Module):
    def __init__(self, n_features, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.drop2 = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(hidden2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.drop1(out)
        out, _ = self.lstm2(out)
        out = self.drop2(out[:, -1, :])
        return self.fc(out)

lstm_model = StackedLSTM(N_FEATURES)
total_params = sum(p.numel() for p in lstm_model.parameters())
print(f"  Parameters: {total_params:,}")

hist_lstm = train_model(lstm_model, train_loader, val_loader, label='LSTM')
all_histories['LSTM'] = hist_lstm
print(f"  Best val loss: {min(hist_lstm['val_loss']):.6f}")

#Model 2: BiLSTM + Attention
print("\n[2/4] Training BiLSTM with Attention...")

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        e = torch.tanh(self.score(h))
        alpha = torch.softmax(e, dim=1)
        self.last_weights = alpha.detach()
        context = (alpha * h).sum(dim=1)
        return context

class BiLSTMAttention(nn.Module):
    def __init__(self, n_features, hidden1=128, hidden2=64, dropout=0.2):
        super().__init__()
        self.bilstm1 = nn.LSTM(n_features, hidden1, batch_first=True, bidirectional=True)
        self.drop1 = nn.Dropout(dropout)
        self.bilstm2 = nn.LSTM(hidden1 * 2, hidden2, batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(dropout)
        self.attention = BahdanauAttention(hidden2 * 2)
        self.fc = nn.Sequential(nn.Linear(hidden2 * 2, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.bilstm1(x)
        out = self.drop1(out)
        out, _ = self.bilstm2(out)
        out = self.drop2(out)
        return self.fc(self.attention(out))

bilstm_model = BiLSTMAttention(N_FEATURES)
total_params = sum(p.numel() for p in bilstm_model.parameters())
print(f"  Parameters: {total_params:,}")

hist_bilstm = train_model(bilstm_model, train_loader, val_loader, label='BiLSTM+Attn')
all_histories['BiLSTM+Attn'] = hist_bilstm
print(f"  Best val loss: {min(hist_bilstm['val_loss']):.6f}")

#Model 3: CNN-LSTM Hybrid
print("\n[3/4] Training CNN-LSTM Hybrid...")

class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size):
        super().__init__()
        self.pad = kernel_size - 1
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size)

    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))

class CNNLSTM(nn.Module):
    def __init__(self, n_features, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(n_features, 64, 3)
        self.conv2 = CausalConv1d(64, 64, 3)
        self.pool = nn.MaxPool1d(2)
        self.conv3 = CausalConv1d(64, 32, 3)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(32, 64, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = self.relu(self.conv3(x))
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.drop(out[:, -1, :])
        return self.fc(out)

cnnlstm_model = CNNLSTM(N_FEATURES)
total_params = sum(p.numel() for p in cnnlstm_model.parameters())
print(f"  Parameters: {total_params:,}")

hist_cnnlstm = train_model(cnnlstm_model, train_loader, val_loader, label='CNN-LSTM')
all_histories['CNN-LSTM'] = hist_cnnlstm
print(f"  Best val loss: {min(hist_cnnlstm['val_loss']):.6f}")

#Model 4: Transformer Encoder
print("\n[4/4] Training Transformer Encoder...")

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + self.drop1(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.drop2(ffn_out))
        return x

class TransformerForecaster(nn.Module):
    def __init__(self, n_features, seq_len, embed_dim=64, num_heads=4, ff_dim=128, n_blocks=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(n_features, embed_dim)
        self.pos_emb = nn.Embedding(seq_len, embed_dim)
        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, ff_dim, dropout) for _ in range(n_blocks)]
        )
        self.fc = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(),
                                nn.Dropout(dropout), nn.Linear(64, 1))

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device)
        x = self.input_proj(x) + self.pos_emb(positions)
        x = self.blocks(x)
        x = x.mean(dim=1)
        return self.fc(x)

transformer_model = TransformerForecaster(N_FEATURES, SEQ_LEN)
total_params = sum(p.numel() for p in transformer_model.parameters())
print(f"  Parameters: {total_params:,}")

hist_transformer = train_model(transformer_model, train_loader, val_loader, label='Transformer')
all_histories['Transformer'] = hist_transformer
print(f"  Best val loss: {min(hist_transformer['val_loss']):.6f}")

print("\n All models trained successfully!")
print()

print("=" * 80)
print("SECTION 5: Hyperparameter Experiments")
print("=" * 80)

#Experiment 1: Window size
print("\nExperiment 1: Window size ablation...")
exp1_results = {}
for wsize in [20, 30, 60]:
    print(f"  Testing window size {wsize}...", end=' ', flush=True)
    Xtr_w, ytr_w, Xva_w, yva_w = [], [], [], []
    for ticker in tickers:
        sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date').copy()
        train_s = sub[sub['Date'] <= TRAIN_END]
        val_s = sub[(sub['Date'] > TRAIN_END) & (sub['Date'] <= VAL_END)]
        sc = scalers[ticker]
        tr_sc = np.hstack([sc.transform(train_s[scale_cols]), train_s[ticker_cols].values])
        va_sc = np.hstack([sc.transform(val_s[scale_cols]), val_s[ticker_cols].values])
        def seqs(arr, w):
            X, y = [], []
            for i in range(w, len(arr)):
                X.append(arr[i-w:i])
                y.append(arr[i, target_idx_in_scale])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        X_, y_ = seqs(tr_sc, wsize)
        Xtr_w.append(X_)
        ytr_w.append(y_)
        X_, y_ = seqs(va_sc, wsize)
        Xva_w.append(X_)
        yva_w.append(y_)
    Xtr_w = np.concatenate(Xtr_w)
    ytr_w = np.concatenate(ytr_w).reshape(-1, 1)
    Xva_w = np.concatenate(Xva_w)
    yva_w = np.concatenate(yva_w).reshape(-1, 1)
    tl_w = DataLoader(StockDataset(Xtr_w, ytr_w), BATCH_SIZE, shuffle=True)
    vl_w = DataLoader(StockDataset(Xva_w, yva_w), BATCH_SIZE, shuffle=False)
    m = StackedLSTM(N_FEATURES)
    h = train_model(m, tl_w, vl_w, label=f'LSTM-w{wsize}', verbose=False)
    val_rmse = float(np.sqrt(min(h['val_loss'])))
    exp1_results[wsize] = val_rmse
    print(f"RMSE={val_rmse:.6f}")

#Experiment 2: Feature ablation
print("\nExperiment 2: Feature ablation...")
exp2_results = {}
ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume']
target_ohlcv_idx = ohlcv_cols.index('Adj_Close')
for feat_label, feat_cols, t_idx in [
    ('OHLCV-only', ohlcv_cols, target_ohlcv_idx),
    ('Full features', scale_cols, target_idx_in_scale)
]:
    print(f"  Testing {feat_label}...", end=' ', flush=True)
    Xtr_f, ytr_f, Xva_f, yva_f = [], [], [], []
    for ticker in tickers:
        sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date').copy()
        train_s = sub[sub['Date'] <= TRAIN_END]
        val_s = sub[(sub['Date'] > TRAIN_END) & (sub['Date'] <= VAL_END)]
        sc_f = MinMaxScaler().fit(train_s[feat_cols])
        tr_sc = sc_f.transform(train_s[feat_cols])
        va_sc = sc_f.transform(val_s[feat_cols])
        def seqs_f(arr, t_i):
            X, y = [], []
            for i in range(SEQ_LEN, len(arr)):
                X.append(arr[i-SEQ_LEN:i])
                y.append(arr[i, t_i])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        X_, y_ = seqs_f(tr_sc, t_idx)
        Xtr_f.append(X_)
        ytr_f.append(y_)
        X_, y_ = seqs_f(va_sc, t_idx)
        Xva_f.append(X_)
        yva_f.append(y_)
    Xtr_f = np.concatenate(Xtr_f)
    ytr_f = np.concatenate(ytr_f).reshape(-1, 1)
    Xva_f = np.concatenate(Xva_f)
    yva_f = np.concatenate(yva_f).reshape(-1, 1)
    nf = Xtr_f.shape[2]
    tl = DataLoader(StockDataset(Xtr_f, ytr_f), BATCH_SIZE, shuffle=True)
    vl = DataLoader(StockDataset(Xva_f, yva_f), BATCH_SIZE, shuffle=False)
    m = StackedLSTM(nf)
    h = train_model(m, tl, vl, label=f'LSTM-{feat_label}', verbose=False)
    val_rmse = float(np.sqrt(min(h['val_loss'])))
    exp2_results[feat_label] = val_rmse
    print(f"RMSE={val_rmse:.6f} (n_feat={nf})")

#Experiment 3: Dropout rate
print("\nExperiment 3: Dropout rate...")
exp3_results = {}
for dr in [0.1, 0.2, 0.3]:
    print(f"  Testing dropout={dr}...", end=' ', flush=True)
    m = StackedLSTM(N_FEATURES, dropout=dr)
    h = train_model(m, train_loader, val_loader, label=f'LSTM-drop{dr}', verbose=False)
    val_rmse = float(np.sqrt(min(h['val_loss'])))
    exp3_results[dr] = val_rmse
    print(f"RMSE={val_rmse:.6f}")

#Experiment 4: Prediction horizon
print("\nExperiment 4: Prediction horizon...")
exp4_results = {}
for horizon in [1, 5]:
    print(f"  Testing {horizon}-day horizon...", end=' ', flush=True)
    Xtr_h, ytr_h, Xva_h, yva_h = [], [], [], []
    for ticker in tickers:
        sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date').copy()
        train_s = sub[sub['Date'] <= TRAIN_END]
        val_s = sub[(sub['Date'] > TRAIN_END) & (sub['Date'] <= VAL_END)]
        sc = scalers[ticker]
        tr_sc = np.hstack([sc.transform(train_s[scale_cols]), train_s[ticker_cols].values])
        va_sc = np.hstack([sc.transform(val_s[scale_cols]), val_s[ticker_cols].values])
        def seqs_h(arr, h):
            X, y = [], []
            for i in range(SEQ_LEN, len(arr) - h + 1):
                X.append(arr[i-SEQ_LEN:i])
                y.append(arr[i + h - 1, target_idx_in_scale])
            return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
        X_, y_ = seqs_h(tr_sc, horizon)
        Xtr_h.append(X_)
        ytr_h.append(y_)
        X_, y_ = seqs_h(va_sc, horizon)
        Xva_h.append(X_)
        yva_h.append(y_)
    Xtr_h = np.concatenate(Xtr_h)
    ytr_h = np.concatenate(ytr_h).reshape(-1, 1)
    Xva_h = np.concatenate(Xva_h)
    yva_h = np.concatenate(yva_h).reshape(-1, 1)
    tl = DataLoader(StockDataset(Xtr_h, ytr_h), BATCH_SIZE, shuffle=True)
    vl = DataLoader(StockDataset(Xva_h, yva_h), BATCH_SIZE, shuffle=False)
    m = StackedLSTM(N_FEATURES)
    h_ = train_model(m, tl, vl, label=f'LSTM-h{horizon}', verbose=False)
    val_rmse = float(np.sqrt(min(h_['val_loss'])))
    exp4_results[horizon] = val_rmse
    print(f"RMSE={val_rmse:.6f}")

#Experiment 5: L2 regularisation
print("\nExperiment 5: L2 weight decay (Transformer)...")
exp5_results = {}
for wd in [0.0, 1e-4, 1e-3]:
    print(f"  Testing weight_decay={wd:.0e}...", end=' ', flush=True)
    m = TransformerForecaster(N_FEATURES, SEQ_LEN)
    m.to(DEVICE)
    optimizer = optim.Adam(m.parameters(), lr=LEARNING_RATE, weight_decay=wd)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    criterion = nn.MSELoss()
    best_val = float('inf')
    no_imp = 0
    best_st = None
    for epoch in range(1, EPOCHS + 1):
        m.train()
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(m(Xb), yb).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            optimizer.step()
        m.eval()
        v_loss = 0.0
        with torch.no_grad():
            for Xb, yb in val_loader:
                v_loss += criterion(m(Xb.to(DEVICE)), yb.to(DEVICE)).item() * len(Xb)
        v_loss /= len(val_loader.dataset)
        scheduler.step(v_loss)
        if v_loss < best_val:
            best_val = v_loss
            best_st = {k: v.cpu().clone() for k, v in m.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= 10:
                break
    val_rmse = float(np.sqrt(best_val))
    exp5_results[wd] = val_rmse
    print(f"RMSE={val_rmse:.6f}")

print("\n All experiments completed!")
print()

#Visualizations
print("=" * 80)
print("SECTION 6: Generating Training Visualisations")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))
axes = axes.flatten()
model_names = ['LSTM', 'BiLSTM+Attn', 'CNN-LSTM', 'Transformer']
colors = ['steelblue', 'darkorange', 'green', 'crimson']

for ax, name, col in zip(axes, model_names, colors):
    h = all_histories[name]
    epochs_range = range(1, len(h['train_loss']) + 1)
    ax.plot(epochs_range, h['train_loss'], label='Train loss', color=col, linewidth=1.5)
    ax.plot(epochs_range, h['val_loss'], label='Val loss', color=col, linewidth=1.5, linestyle='--')
    lrs = h['lr']
    for i in range(1, len(lrs)):
        if lrs[i] < lrs[i-1]:
            ax.axvline(i+1, color='grey', linewidth=0.8, linestyle=':')
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend(fontsize=9)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

fig.suptitle('Training & Validation Loss Curves (grey dashes = LR reduction)', fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('fig_04_loss_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_04_loss_curves.png")

fig, axes = plt.subplots(1, 3, figsize=(16, 4))

ax = axes[0]
ax.bar([str(k) for k in exp1_results], list(exp1_results.values()), color='steelblue', edgecolor='k')
ax.set_title('Exp 1: Window Size vs. Val RMSE')
ax.set_xlabel('Lookback (days)')
ax.set_ylabel('Val RMSE')

ax = axes[1]
ax.bar(list(exp2_results.keys()), list(exp2_results.values()), color=['salmon', 'steelblue'], edgecolor='k')
ax.set_title('Exp 2: Feature Set vs. Val RMSE')
ax.set_xlabel('Feature set')
ax.set_ylabel('Val RMSE')

ax = axes[2]
ax.plot(list(exp3_results.keys()), list(exp3_results.values()), 'o-', color='green', linewidth=2, markersize=8)
ax.set_title('Exp 3: Dropout Rate vs. Val RMSE')
ax.set_xlabel('Dropout rate')
ax.set_ylabel('Val RMSE')
ax.set_xticks(list(exp3_results.keys()))

plt.tight_layout()
plt.savefig('fig_05_experiments.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_05_experiments.png")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))

ax = axes[0]
ax.bar([f'{k}-day' for k in exp4_results], list(exp4_results.values()), color='darkorange', edgecolor='k')
ax.set_title('Exp 4: Prediction Horizon vs. Val RMSE')
ax.set_xlabel('Forecast horizon')
ax.set_ylabel('Val RMSE')

ax = axes[1]
ax.plot([str(k) for k in exp5_results], list(exp5_results.values()), 's-', color='crimson', linewidth=2, markersize=8)
ax.set_title('Exp 5: L2 Weight Decay vs. Val RMSE (Transformer)')
ax.set_xlabel('Weight decay')
ax.set_ylabel('Val RMSE')

plt.tight_layout()
plt.savefig('fig_06_experiments2.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_06_experiments2.png")

bilstm_model.eval()
sample_X = torch.tensor(X_test[:1], dtype=torch.float32).to(DEVICE)
with torch.no_grad():
    _ = bilstm_model(sample_X)
attn_w = bilstm_model.attention.last_weights[0, :, 0].detach().cpu().numpy()

fig, ax = plt.subplots(figsize=(14, 2))
im = ax.imshow(attn_w.reshape(1, -1), aspect='auto', cmap='YlOrRd')
ax.set_yticks([])
ax.set_xlabel('Timestep (0 = oldest, 59 = most recent)')
ax.set_title('BiLSTM Attention Weights over 60-day Lookback Window (first test sample)')
plt.colorbar(im, ax=ax, orientation='horizontal', pad=0.5, fraction=0.05)
plt.tight_layout()
plt.savefig('fig_07_attention.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_07_attention.png")

print()

#Results
print("=" * 80)
print("SECTION 7: Test Results & Analysis")
print("=" * 80)

models_dict = {
    'LSTM': lstm_model,
    'BiLSTM+Attn': bilstm_model,
    'CNN-LSTM': cnnlstm_model,
    'Transformer': transformer_model,
}

print("\nComputing test predictions...")
for name, model in models_dict.items():
    all_test_preds[name] = predict(model, test_loader)
    print(f"  {name}: {all_test_preds[name].shape}")

print("\nComputing inverse-transformed metrics...")
results_rows = []
for name in models_dict:
    preds_norm = all_test_preds[name].flatten()
    true_norm = y_test.flatten()

    pred_prices, true_prices = [], []
    offset = 0
    for ticker in tickers:
        sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date')
        te = sub[sub['Date'] > VAL_END]
        n = max(0, len(te) - SEQ_LEN)
        if n == 0:
            continue
        adj_sc = adj_close_scalers[ticker]
        p_norm = preds_norm[offset:offset + n].reshape(-1, 1)
        t_norm = true_norm[offset:offset + n].reshape(-1, 1)
        pred_prices.append(adj_sc.inverse_transform(p_norm).flatten())
        true_prices.append(adj_sc.inverse_transform(t_norm).flatten())
        offset += n

    pp = np.concatenate(pred_prices)
    tp = np.concatenate(true_prices)

    r = rmse(tp, pp)
    m_mae = float(mean_absolute_error(tp, pp))
    m_mape = mape(tp, pp)
    da = dir_acc(tp, pp)

    results_rows.append({
        'Model': name,
        'RMSE ($)': round(r, 3),
        'MAE ($)': round(m_mae, 3),
        'MAPE (%)': round(m_mape, 3),
        'Dir. Accuracy (%)': round(da, 2)
    })

results_df = pd.DataFrame(results_rows).set_index('Model')
print("\n" + results_df.to_string())
print()

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
metrics = ['RMSE ($)', 'MAE ($)', 'MAPE (%)']
colors_bars = ['steelblue', 'darkorange', 'green']

for ax, metric, col in zip(axes, metrics, colors_bars):
    vals = results_df[metric]
    bars = ax.bar(vals.index, vals.values, color=col, edgecolor='k', alpha=0.85)
    ax.set_title(metric, fontsize=12)
    ax.set_ylabel(metric)
    ax.tick_params(axis='x', rotation=20)
    for bar, val in zip(bars, vals.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002 * vals.max(),
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)

fig.suptitle('Test Set Performance — All Models', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('fig_08_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_08_metrics_comparison.png")

focus_tickers = ['AAPL', 'NVDA', 'TSLA']
fig, axes = plt.subplots(len(focus_tickers), 1, figsize=(15, 10))

for ax, ticker in zip(axes, focus_tickers):
    sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date')
    te = sub[sub['Date'] > VAL_END]
    n = max(0, len(te) - SEQ_LEN)
    if n == 0:
        continue
    dates = te['Date'].values[SEQ_LEN:]
    adj_sc = adj_close_scalers[ticker]

    cumulative = sum(
        max(0, len(df_feat[(df_feat['Ticker'] == t) & (df_feat['Date'] > VAL_END)]) - SEQ_LEN)
        for t in tickers[:tickers.index(ticker)]
    )
    true_norm_t = y_test[cumulative:cumulative + n]
    true_p = adj_sc.inverse_transform(true_norm_t).flatten()
    ax.plot(dates, true_p, label='Actual', color='black', linewidth=1.5)

    for name, col in zip(models_dict.keys(), ['steelblue', 'darkorange', 'green', 'crimson']):
        pred_norm = all_test_preds[name][cumulative:cumulative + n]
        pred_p = adj_sc.inverse_transform(pred_norm.reshape(-1, 1)).flatten()
        ax.plot(dates, pred_p, label=name, linewidth=1.0, alpha=0.8, color=col, linestyle='--')

    ax.set_title(f'{ticker} — Test Set Predictions', fontsize=11, fontweight='bold')
    ax.set_ylabel('Adj Close ($)')
    ax.legend(fontsize=8, loc='upper left')
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fig_09_predictions.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_09_predictions.png")

best_model_name = results_df['RMSE ($)'].idxmin()
best_model = models_dict[best_model_name]
best_preds = all_test_preds[best_model_name].flatten()
true_flat = y_test.flatten()

fig, axes = plt.subplots(3, 3, figsize=(14, 9))
axes = axes.flatten()

for i, ticker in enumerate(tickers):
    adj_sc = adj_close_scalers[ticker]
    sub = df_feat[df_feat['Ticker'] == ticker].sort_values('Date')
    te = sub[sub['Date'] > VAL_END]
    n = max(0, len(te) - SEQ_LEN)
    if n == 0:
        continue
    offset = sum(max(0, len(df_feat[(df_feat['Ticker'] == t) & (df_feat['Date'] > VAL_END)]) - SEQ_LEN)
                 for t in tickers[:i])
    pred_p = adj_sc.inverse_transform(best_preds[offset:offset+n].reshape(-1, 1)).flatten()
    true_p = adj_sc.inverse_transform(true_flat[offset:offset+n].reshape(-1, 1)).flatten()
    residuals = pred_p - true_p
    axes[i].hist(residuals, bins=30, color='steelblue', edgecolor='k', alpha=0.7)
    axes[i].axvline(0, color='red', linewidth=1.5)
    axes[i].set_title(ticker)
    axes[i].set_xlabel('Residual ($)')
    axes[i].set_ylabel('Count')

fig.suptitle(f'Residual Error Distribution — {best_model_name} (Test Set)', fontsize=13)
plt.tight_layout()
plt.savefig('fig_10_residuals.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved: fig_10_residuals.png")

print()

print("=" * 80)
print("PROJECT SUMMARY")
print("=" * 80)

print(f"""
 Data Pipeline:
  - Loaded {df.shape[0]:,} rows, {len(tickers)} stocks, 2010-2026
  - Engineered {len(FEATURE_COLS)} features (OHLCV + technical indicators + ticker OHE)
  - Created {X_train.shape[0]:,} train / {X_val.shape[0]:,} val / {X_test.shape[0]:,} test sequences

 Models Trained (on full dataset):
  - LSTM: 132K params, val loss = {min(all_histories['LSTM']['val_loss']):.6f}
  - BiLSTM+Attention: 330K params, val loss = {min(all_histories['BiLSTM+Attn']['val_loss']):.6f}
  - CNN-LSTM: 51K params, val loss = {min(all_histories['CNN-LSTM']['val_loss']):.6f}
  - Transformer: 76K params, val loss = {min(all_histories['Transformer']['val_loss']):.6f}

 Experiments:
  - Window size: {list(exp1_results.keys())} → {[f'{v:.4f}' for v in exp1_results.values()]}
  - Feature ablation: {list(exp2_results.keys())}
  - Dropout: {[f'{v:.4f}' for v in exp3_results.values()]}
  - Horizon: {list(exp4_results.keys())}
  - L2 regularisation: {[f'{v:.4f}' for v in exp5_results.values()]}

 Test Results ({best_model_name} — best performer):
  - RMSE: ${results_df.loc[best_model_name, 'RMSE ($)']:.2f}
  - MAE: ${results_df.loc[best_model_name, 'MAE ($)']:.2f}
  - MAPE: {results_df.loc[best_model_name, 'MAPE (%)']:.2f}%
  - Directional Accuracy: {results_df.loc[best_model_name, 'Dir. Accuracy (%)']:.2f}%

 Visualisations Generated:
  - fig_01_price_history.png
  - fig_02_correlation.png
  - fig_03_volume.png
  - fig_04_loss_curves.png
  - fig_05_experiments.png
  - fig_06_experiments2.png
  - fig_07_attention.png
  - fig_08_metrics_comparison.png
  - fig_09_predictions.png
  - fig_10_residuals.png

  Total runtime: {time.time()} seconds
""")

print("=" * 80)
print("ALL TASKS COMPLETED SUCCESSFULLY!")
print("=" * 80)
