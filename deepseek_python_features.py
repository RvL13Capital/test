import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
from project.config import Config
import pandas_ta as ta
import pywt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit

def make_stationary(df):
    """Konvertiert Preisdaten in stationäre Renditen, behält aber Original für bestimmte Features."""
    original_cols = df.copy()
    df_stat = df[['open', 'high', 'low', 'close']].pct_change()
    df_stat['volume'] = np.log(df['volume'] + 1).diff()
    df_stat.dropna(inplace=True)
    df_stat = df_stat.join(original_cols[['open', 'high', 'low', 'close']], rsuffix='_orig')
    return df_stat

def select_features(train_df):
    """Wählt die besten Features aus Trainingsdaten mit TimeSeriesSplit"""
    df_features = _create_features(train_df)
    df_features.dropna(inplace=True)
    
    initial_cols = get_feature_columns()
    X = df_features[initial_cols]

    # Identifiziere hochkorrelierte Features für VIF
    corr_matrix = X.corr().abs()
    high_corr_features = set()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if corr_matrix.iloc[i, j] > 0.8:
                colname = corr_matrix.columns[i]
                high_corr_features.add(colname)
    
    # Berechne VIF nur für hochkorrelierte Features
    vif_data = pd.DataFrame()
    if high_corr_features:
        vif_data["feature"] = list(high_corr_features)
        vif_data["VIF"] = [variance_inflation_factor(X[list(high_corr_features)].values, i) 
                          for i in range(len(high_corr_features))]
        high_vif_features = vif_data[vif_data['VIF'] > 10]['feature']
        X_filtered = X.drop(columns=high_vif_features)
    else:
        X_filtered = X
    
    # Zeitreihenvalidierung für Feature-Importanz
    tscv = TimeSeriesSplit(n_splits=3)
    importances = []
    
    for train_idx, val_idx in tscv.split(X_filtered):
        X_train, X_val = X_filtered.iloc[train_idx], X_filtered.iloc[val_idx]
        y_train = df_features['close'].iloc[train_idx]
        
        model = xgb.XGBRegressor(**Config.XGBOOST_HPARAMS)
        model.fit(X_train, y_train)
        importances.append(model.feature_importances_)
    
    avg_importances = np.mean(importances, axis=0)
    feature_importance_df = pd.DataFrame({
        'feature': X_filtered.columns,
        'importance': avg_importances
    }).sort_values('importance', ascending=False)
    
    selected_features = feature_importance_df.head(20)['feature'].tolist()
    print(f"Ausgewählte Features: {selected_features}")
    return selected_features

def prepare_sequences(df, window_size, prediction_length, selected_features=None, scaler=None):
    df_stat = make_stationary(df)
    df_with_features = _create_features(df_stat)
    if selected_features:
        df_final = df_with_features[selected_features]
    else:
        df_final = df_with_features[get_feature_columns()]
    
    if scaler is None:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(df_final)
    else:
        data = scaler.transform(df_final)
    
    src, trg = [], []
    for i in range(0, len(data) - window_size - prediction_length + 1, 1):
        src.append(data[i:i + window_size])
        trg.append(data[i + window_size:i + window_size + prediction_length])
    
    return np.array(src), np.array(trg), scaler

def prepare_inference_from_df(df, scaler, window_size, selected_features=None):
    df_stat = make_stationary(df)
    df_with_features = _create_features(df_stat)
    if selected_features:
        df_final = df_with_features[selected_features]
    else:
        df_final = df_with_features[get_feature_columns()]
    if len(df_final) < window_size:
        return np.array([])
    data = scaler.transform(df_final)
    return np.array([data[-window_size:]])

class FeatureForecaster:
    def __init__(self):
        self.models = {}
        
    def train(self, df, feature_columns):
        for col in feature_columns:
            if col == 'close':
                continue
            model = xgb.XGBRegressor(**Config.XGBOOST_HPARAMS)
            X, y = [], []
            for i in range(Config.DATA_WINDOW_SIZE, len(df)):
                X.append(df.iloc[i-Config.DATA_WINDOW_SIZE:i][feature_columns].values.flatten())
                y.append(df.iloc[i][col])
            model.fit(np.array(X), np.array(y))
            self.models[col] = model
            
    def forecast_features(self, last_window, steps):
        forecasts = {col: [] for col in self.models}
        current_data = last_window.copy()
        
        for _ in range(steps):
            features = _create_features(current_data)
            new_row = {}
            flattened = features[get_feature_columns()].values.flatten()
            for col, model in self.models.items():
                new_row[col] = model.predict([flattened])[0]
            current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
            for col in forecasts:
                forecasts[col].append(new_row[col])
        return forecasts

def iterative_xgboost_forecast(model, initial_input, steps, feature_cols, last_known_data, feature_forecaster=None):
    predictions = []
    current_data = last_known_data.copy()
    if feature_forecaster is None:
        feature_forecaster = FeatureForecaster()
        feature_forecaster.train(last_known_data, feature_cols)
    for _ in range(steps):
        input_df = current_data.tail(Config.DATA_WINDOW_SIZE)
        input_features = _create_features(input_df)
        flattened = input_features[feature_cols].values.flatten()
        predicted_close = model.predict(flattened.reshape(1, -1))[0]
        predictions.append(predicted_close)
        feature_forecasts = feature_forecaster.forecast_features(
            current_data.tail(Config.DATA_WINDOW_SIZE),
            1
        )
        new_row = {
            'open': feature_forecasts.get('open', [predicted_close * 0.998])[0],
            'high': feature_forecasts.get('high', [predicted_close * 1.005])[0],
            'low': feature_forecasts.get('low', [predicted_close * 0.995])[0],
            'close': predicted_close,
            'volume': feature_forecasts.get('volume', [current_data['volume'].iloc[-1] * 0.97])[0]
        }
        current_data = pd.concat([current_data, pd.DataFrame([new_row])], ignore_index=True)
        current_data = _create_features(current_data)
    return predictions

def get_feature_columns():
    return [
        'open', 'high', 'low', 'close', 'volume', 'range', 'avg_price', 'RSI_14',
        'MACD_12_26_9', 'ATRr_14', 'ADX_14', 'OBV', 'AD_line', 'volume_near_price',
        'avg_volume', 'volume_profile_strength', 'wavelet_coherence'
    ]

def _create_features(df):
    df_copy = df.copy()
    df_copy.ta.rsi(length=14, append=True)
    df_copy.ta.macd(fast=12, slow=26, signal=9, append=True)
    df_copy.ta.atr(length=14, append=True)
    df_copy.ta.adx(length=14, append=True)
    df_copy.ta.obv(append=True)
    df_copy.ta.ad(append=True)
    df_copy['range'] = df_copy['high'] - df_copy['low']
    df_copy['avg_price'] = (df_copy['high'] + df_copy['low']) / 2
    df_copy['avg_volume'] = df_copy['volume'].rolling(window=20).mean()
    df_copy['volume_near_price'] = compute_volume_near_price(df_copy)
    df_copy['volume_profile_strength'] = compute_volume_profile(df_copy)
    df_copy['volatility_contraction'] = compute_volatility_contraction(df_copy)
    df_copy['price_action_pattern'] = compute_price_action_patterns(df_copy)
    df_copy['momentum_divergence'] = compute_momentum_divergence(df_copy)
    df_copy['ignition_impulse'] = compute_ignition_impulse(df_copy)
    df_copy['wavelet_coherence'] = compute_wavelet_coherence(df_copy['close'].values, df_copy['volume'].values)
    df_copy.fillna(method='ffill', inplace=True)
    df_copy.fillna(method='bfill', inplace=True)
    return df_copy

def compute_wavelet_coherence(close, volume, wavelet='morl'):
    if len(close) < 2 or len(volume) < 2:
        return 0.0
    
    # Optimierte Skalen für Performance
    scales = [1, 16, 32, 64, 128]
    
    coef_close, _ = pywt.cwt(close, scales, wavelet)
    coef_volume, _ = pywt.cwt(volume, scales, wavelet)
    
    # Stabilitätsverbesserung
    epsilon = 1e-10
    coherence = np.abs(np.mean(coef_close * np.conj(coef_volume), axis=0) ** 2
    denominator = (np.mean(np.abs(coef_close)**2, axis=0) * (np.mean(np.abs(coef_volume)**2, axis=0) + epsilon)
    
    return np.mean(coherence / denominator)

def compute_volume_near_price(df, price_col='close', window=20, threshold=0.01):
    recent = df[price_col].tail(window)
    mean_price = recent.mean()
    volume_near = df['volume'][(df[price_col] >= mean_price * (1 - threshold)) &
                              (df[price_col] <= mean_price * (1 + threshold))].sum()
    return volume_near / df['volume'].tail(window).sum() if df['volume'].tail(window).sum() > 0 else 0.0

def compute_volume_profile(df, price_col='close', bins=50):
    prices = df[price_col]
    volumes = df['volume']
    hist, bin_edges = np.histogram(prices, bins=bins, weights=volumes)
    max_volume_bin = np.argmax(hist)
    return hist[max_volume_bin] / volumes.sum() if volumes.sum() > 0 else 0.0

def compute_volatility_contraction(df, price_col='close', window=20):
    middle = df[price_col].rolling(window).mean()
    std = df[price_col].rolling(window).std()
    bandwidth = (middle + 2 * std - (middle - 2 * std)) / middle
    return bandwidth.iloc[-1] if not bandwidth.isnull().all() else 0.0

def compute_price_action_patterns(df, window=5):
    candles = df.tail(window)
    if len(candles) < 2:
        return 0.0
    body = abs(candles['close'] - candles['open'])
    is_doji = (body.iloc[-1] / candles['range'].iloc[-1] < 0.1) if candles['range'].iloc[-1] > 0 else False
    is_engulfing = (body.iloc[-1] > body.iloc[-2] * 1.5 and
                    candles['close'].iloc[-1] > candles['open'].iloc[-2] and
                    candles['open'].iloc[-1] < candles['close'].iloc[-2]) if len(candles) > 1 else False
    return 1.0 if is_doji or is_engulfing else 0.0

def compute_momentum_divergence(df, price_col='close', rsi_col='RSI_14', window=14):
    price_diff = df[price_col].diff().tail(window)
    rsi_diff = df[rsi_col].diff().tail(window)
    if price_diff.iloc[-1] > 0 and rsi_diff.iloc[-1] < 0:
        return 1.0
    elif price_diff.iloc[-1] < 0 and rsi_diff.iloc[-1] > 0:
        return -1.0
    return 0.0

def compute_ignition_impulse(df, price_col='close', window=10):
    returns = df[price_col].pct_change().tail(window)
    return returns.abs().mean() if not returns.isnull().all() else 0.0