
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt

# Daten laden und reinigen
def load_data(sources):
    """Lädt OHLCV-Daten aus mehreren CSV-Dateien und kombiniert sie."""
    data_list = []
    for source in sources:
        data = pd.read_csv(source)
        required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"Daten in {source} müssen OHLCV-Spalten enthalten.")
        data_list.append(data)
    combined_data = pd.concat(data_list, ignore_index=True)
    combined_data = combined_data.dropna()
    combined_data = combined_data.reset_index(drop=True)
    return combined_data

# Hilfsfunktion: ATR berechnen
def calculate_atr(data, period):
    """Berechnet den Average True Range (ATR) über ein gegebenes Zeitfenster."""
    high_low = data['High'] - data['Low']
    high_close = np.abs(data['High'] - data['Close'].shift())
    low_close = np.abs(data['Low'] - data['Close'].shift())
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()

# Hilfsfunktion: OBV berechnen
def calculate_obv(data):
    """Berechnet das On-Balance-Volume (OBV)."""
    obv = [0]
    for i in range(1, len(data)):
        if data['Close'].iloc[i] > data['Close'].iloc[i-1]:
            obv.append(obv[-1] + data['Volume'].iloc[i])
        elif data['Close'].iloc[i] < data['Close'].iloc[i-1]:
            obv.append(obv[-1] - data['Volume'].iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=data.index)

# Hilfsfunktion: MACD berechnen
def calculate_macd(data, fast=12, slow=26, signal=9):
    """Berechnet das MACD-Histogramm mit Standardparametern."""
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - signal_line
    return macd_hist

# Konsolidierung erkennen
def detect_consolidation(data, vol_threshold=0.5, std_percentile=20, vol_percentile=30, min_period=10):
    """Identifiziert Konsolidierungsphasen basierend auf Volatilität, Preisspanne und Volumen."""
    # Adaptive Volatilität
    data['ATR_10'] = calculate_atr(data, 10)
    data['ATR_100'] = calculate_atr(data, 100)
    data['Vol_Ratio'] = data['ATR_10'] / data['ATR_100']
    
    # Preisspanne
    data['Price_STD'] = data['Close'].rolling(window=20).std()
    threshold_std = np.percentile(data['Price_STD'].dropna(), std_percentile)
    
    # Volumenvalidierung
    data['Avg_Volume'] = data['Volume'].rolling(window=20).mean()
    volume_threshold = np.percentile(data['Volume'], vol_percentile)
    
    # Konsolidierungsbedingung
    data['Consolidation'] = ((data['Vol_Ratio'] < vol_threshold) & 
                             (data['Price_STD'] < threshold_std) & 
                             (data['Avg_Volume'] > volume_threshold)).astype(int)
    
    # Mindestens 10 Tage Konsolidierung
    data['Consolidation'] = data['Consolidation'].rolling(window=min_period).sum() >= min_period
    return data

# Features berechnen
def compute_features(data):
    """Berechnet Features für das Modell basierend auf vergangenen Daten."""
    # Volatilität der Volatilität (VoV)
    data['Daily_Vol'] = np.log(data['High'] / data['Low']) / np.sqrt(4 * np.log(2))
    data['VoV'] = data['Daily_Vol'].rolling(window=30).std()
    
    # Normalisierte OBV-Änderung
    data['OBV'] = calculate_obv(data)
    data['OBV_Change'] = (data['OBV'] - data['OBV'].shift(10)) / data['OBV'].rolling(window=10).mean()
    
    # MACD-Histogramm
    data['MACD_Hist'] = calculate_macd(data)
    
    return data

# Beschriftung für Zündungsdetektor
def label_breakouts(data):
    """Labelt Tage mit Ausbrüchen innerhalb der nächsten 5 Tage."""
    labels = []
    for i in range(len(data) - 5):
        s = max(0, i - 20)  # Start der Konsolidierung (max. 20 Tage zurück)
        range_high = data['High'].iloc[s:i+1].max() * 1.02
        range_low = data['Low'].iloc[s:i+1].min() * 0.98
        future_prices = data['Close'].iloc[i+1:i+6]
        if any(future_prices > range_high) or any(future_prices < range_low):
            labels.append(1)
        else:
            labels.append(0)
    labels.extend([0] * 5)  # Letzte 5 Tage ohne Zukunft
    return pd.Series(labels, index=data.index)

# Beschriftung für Richtungs-Klassifikator
def label_direction(data, breakout_indices):
    """Labelt die Richtung von Ausbrüchen (bullisch=1, bärisch=0)."""
    labels = []
    for i in breakout_indices:
        s = max(0, i - 20)
        range_high = data['High'].iloc[s:i+1].max() * 1.02
        range_low = data['Low'].iloc[s:i+1].min() * 0.98
        future_prices = data['Close'].iloc[i+1:i+6]
        if any(future_prices > range_high):
            labels.append(1)
        elif any(future_prices < range_low):
            labels.append(0)
    return pd.Series(labels, index=breakout_indices)

# Modelle trainieren mit Hyperparameter-Tuning
def train_models(data):
    """Trainiert die zwei XGBoost-Modelle mit Hyperparameter-Tuning."""
    data = compute_features(data)
    consolidation_data = data[data['Consolidation'] == 1].dropna()
    if len(consolidation_data) < 10:
        raise ValueError("Nicht genügend Konsolidierungsdaten für das Training.")
    
    # Features für Zündungsdetektor
    X_ignition = consolidation_data[['VoV', 'OBV_Change', 'MACD_Hist']].abs()
    y_ignition = label_breakouts(consolidation_data)
    
    # Datenaufteilung
    X_train, X_test, y_train, y_test = train_test_split(X_ignition, y_ignition, test_size=0.2, shuffle=False)
    
    # Hyperparameter-Tuning für Zündungsdetektor
    param_grid_ignition = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    model_ignition = xgb.XGBClassifier(scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train))
    grid_search_ignition = GridSearchCV(model_ignition, param_grid_ignition, cv=5, scoring='roc_auc')
    grid_search_ignition.fit(X_train, y_train)
    best_model_ignition = grid_search_ignition.best_estimator_
    ignition_pred = best_model_ignition.predict_proba(X_test)[:, 1]
    print(f"Beste Parameter Zündungsdetektor: {grid_search_ignition.best_params_}")
    print(f"AUC-ROC Zündungsdetektor: {roc_auc_score(y_test, ignition_pred):.4f}")
    
    # Daten für Richtungs-Klassifikator
    breakout_indices = consolidation_data.index[y_ignition == 1]
    breakout_data = consolidation_data.loc[breakout_indices].dropna()
    X_direction = breakout_data[['VoV', 'OBV_Change', 'MACD_Hist']]
    y_direction = label_direction(breakout_data, breakout_indices)
    
    if len(y_direction) < 2:
        raise ValueError("Nicht genügend Ausbruchdaten für den Richtungs-Klassifikator.")
    
    # Hyperparameter-Tuning für Richtungs-Klassifikator
    param_grid_direction = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'n_estimators': [50, 100, 200]
    }
    model_direction = xgb.XGBClassifier()
    grid_search_direction = GridSearchCV(model_direction, param_grid_direction, cv=5, scoring='accuracy')
    grid_search_direction.fit(X_direction, y_direction)
    best_model_direction = grid_search_direction.best_estimator_
    direction_pred = best_model_direction.predict(X_direction)
    print(f"Beste Parameter Richtungs-Klassifikator: {grid_search_direction.best_params_}")
    print(f"Genauigkeit Richtungs-Klassifikator: {accuracy_score(y_direction, direction_pred):.4f}")
    
    return best_model_ignition, best_model_direction

# Vorhersage machen
def predict(data, model_ignition, model_direction, threshold=0.65):
    """Macht Vorhersagen für Tage in Konsolidierungsphasen."""
    data = compute_features(data)
    if data['Consolidation'].iloc[-1]:
        X = data[['VoV', 'OBV_Change', 'MACD_Hist']].iloc[-1:]
        ignition_wert = model_ignition.predict_proba(X.abs())[:, 1][0]
        if ignition_wert > threshold:
            direction = model_direction.predict(X)[0]
            return ignition_wert, 'bullisch' if direction else 'bärisch'
    return None, None

# Visualisierung
def plot_features(data, ignition_wert, direction):
    """Visualisiert die Features und Vorhersagen."""
    plt.figure(figsize=(12, 6))
    plt.plot(data['Close'], label='Close')
    plt.plot(data['Consolidation'] * data['Close'].max(), label='Konsolidierung', alpha=0.5)
    if ignition_wert is not None:
        plt.axvline(x=data.index[-1], color='r', linestyle='--', label=f'Ignition-Wert: {ignition_wert:.4f}, Richtung: {direction}')
    plt.legend()
    plt.show()

# Hauptprogramm
if __name__ == "__main__":
    # Daten laden
    sources = ['path_to_data1.csv', 'path_to_data2.csv']  # Liste der Dateipfade
    data = load_data(sources)
    
    # Konsolidierung erkennen
    data = detect_consolidation(data)
    
    # Modelle trainieren
    model_ignition, model_direction = train_models(data)
    
    # Vorhersage für den letzten Tag
    ignition_wert, direction = predict(data, model_ignition, model_direction)
    if ignition_wert is not None:
        print(f"Ignition-Wert: {ignition_wert:.4f}, Richtung: {direction}")
    else:
        print("Keine Vorhersage möglich (keine Konsolidierung oder Ignition-Wert unter Schwellenwert).")
    
    # Visualisierung
    plot_features(data, ignition_wert, direction)
