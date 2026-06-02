# S&P 500 Deep Learning Forecaster 📈

A comparative deep learning analysis built to forecast next-day adjusted closing prices for nine major S&P 500 technology stocks (AAPL, MSFT, NVDA, etc.). 

This project processes over 15 years of historical market data, engineered with custom technical indicators (MACD, RSI, Bollinger Bands), to evaluate the predictive power of various neural network architectures in a high-stakes, highly volatile financial environment.

### 🧠 Architectures Evaluated
* **CNN-LSTM Hybrid (Best Performer):** Utilizes 1D causal convolutions for local momentum extraction prior to sequential modeling, achieving the lowest test RMSE ($41.83).
* **Bidirectional LSTM with Attention:** Employs a custom Bahdanau-style attention mechanism to capture forward and backward temporal context over a 60-day lookback window.
* **Custom Transformer Encoder:** Built entirely from scratch using PyTorch primitives, featuring multi-head self-attention and learned positional embeddings.
* **Stacked LSTM:** A robust, two-layer baseline sequential model utilizing dropout regularization.

### 🚀 Key Findings
* **Feature Engineering:** Integrating engineered technical indicators significantly reduced forecast error compared to relying solely on raw OHLCV price and volume data.
* **Sequence Length:** A 60-day (~3 month) sliding lookback window captured quarterly market trends effectively, drastically outperforming shorter 20- and 30-day windows.
* **Optimal Extraction:** The CNN-LSTM hybrid outperformed pure attention-based and recurrent models, proving that causal convolutional feature extraction is highly effective at identifying short-term momentum shifts before sequence processing.

### 🛠️ Tech Stack
* **Language:** Python
* **Framework:** PyTorch
* **Data Processing:** Pandas, NumPy, Scikit-learn
