

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from flask import Flask, request, render_template_string
import talib

app = Flask(__name__)

# HTML template for the interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Universeller Konsolidierungsindikator v2</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head>
<body>
    <h1>Universeller Konsolidierungsindikator v2 (UKI)</h1>
    <p>Upload a CSV file with columns: Date, Open, High, Low, Close, Volume</p>
    <form action="/" method="post" enctype="multipart/form-data">
        <input type="file" name="file" accept=".csv"><br><br>
        <h3>Perioden für Berechnungen</h3>
        <label>ATR Short Period: <input type="number" name="atr_short_len" value="10" min="1"></label><br>
        <label>ATR Long Period: <input type="number" name="atr_long_len" value="100" min="1"></label><br>
        <label>ADX Period: <input type="number" name="adx_len" value="14" min="1"></label><br>
        <label>Price Channel Period: <input type="number" name="channel_len" value="20" min="1"></label><br>
        <label>PKV Percent-Rank Period: <input type="number" name="pkv_rank_len" value="100" min="1"></label><br>
        <label>OBV Trend Period: <input type="number" name="obv_len" value="20" min="1"></label><br>
        <h3>Gewichtung der Komponenten</h3>
        <label>Volatility (VR): <input type="number" name="w1" value="35" min="0" step="0.1"></label><br>
        <label>Trend Strength (TSF): <input type="number" name="w2" value="35" min="0" step="0.1"></label><br>
        <label>Price Channel (PKV): <input type="number" name="w3" value="15" min="0" step="0.1"></label><br>
        <label>Volume Flow (VF): <input type="number" name="w4" value="15" min="0" step="0.1"></label><br>
        <h3>Schwellenwerte</h3>
        <label>High Threshold: <input type="number" name="threshold_high" value="75" min="1" max="100"></label><br>
        <label>Low Threshold: <input type="number" name="threshold_low" value="25" min="1" max="100"></label><br>
        <input type="submit" value="Generate Chart">
    </form>
    {% if chart_json %}
    <div id="chart" data-chart='{{ chart_json }}'></div>
    <script>
        Plotly.newPlot('chart', JSON.parse(document.getElementById('chart').getAttribute('data-chart')));
    </script>
    {% endif %}
</body>
</html>
"""

def calculate_uki(df, atr_short_len, atr_long_len, adx_len, channel_len, pkv_rank_len, obv_len, w1, w2, w3, w4, threshold_high, threshold_low):
    # Volatility Ratio (VR)
    atr_short = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_short_len)
    atr_long = talib.ATR(df['High'], df['Low'], df['Close'], timeperiod=atr_long_len)
    volatility_ratio = np.where(atr_long > 0, atr_short / atr_long, 0)
    vr_score = 100 * (1 - np.minimum(1, volatility_ratio))

    # Trend Strength Filter (TSF)
    _, _, adx_val = talib.DX(df['High'], df['Low'], df['Close'], timeperiod=adx_len)
    tsf_score = 100 * np.maximum(0, 1 - (adx_val - 20) / 20)

    # Price Channel Ratio (PKV)
    channel_high = df['High'].rolling(window=channel_len).max()
    channel_low = df['Low'].rolling(window=channel_len).min()
    channel_width = channel_high - channel_low
    pkv = np.where(atr_long > 0, channel_width / atr_long, 0)
    pkv_rank = pkv.rolling(window=pkv_rank_len).apply(lambda x: (pd.Series(x).rank(pct=True) * 100).iloc[-1], raw=True)
    pkv_score = 100 - pkv_rank

    # Volume Flow (VF)
    obv = talib.OBV(df['Close'], df['Volume'])
    obv_sma = obv.rolling(window=obv_len).mean()
    vf_score = np.where(obv > obv_sma, (obv - obv_sma) / obv_sma * 100 + 50, 50 - (obv_sma - obv) / obv_sma * 100)
    vf_score = np.clip(vf_score, 0, 100)

    # Consolidation Score
    consolidation_strength = (w1 * vr_score + w2 * tsf_score + w3 * pkv_score) / (w1 + w2 + w3)
    konsolidierungs_score = consolidation_strength

    # Colors for plotting
    colors = []
    shapes = []
    for i, (score, vf) in enumerate(zip(konsolidierungs_score, vf_score)):
        if score >= threshold_high:
            if w4 > 0 and vf > 55:
                colors.append('green')
                if w4 > 0:
                    shapes.append(dict(type='rect', x0=df.index[i], x1=df.index[i+1] if i < len(df)-1 else df.index[i] + (df.index[i] - df.index[i-1]),
                                       y0=threshold_high, y1=100, fillcolor='rgba(0,255,0,0.2)', layer='below', line_width=0))
            elif w4 > 0 and vf < 45:
                colors.append('red')
                if w4 > 0:
                    shapes.append(dict(type='rect', x0=df.index[i], x1=df.index[i+1]_to_datetime(df.index[-1]) if i < len(df)-1 else df.index[i] + (df.index[i] - df.index[i-1]),
                                       y0=threshold_high, y1=100, fillcolor='rgba(255,0,0,0.2)', layer='below', line_width=0))
            else:
                colors.append('orange')
        elif score <= threshold_low:
            colors.append('blue')
        else:
            colors.append('gray')

    return konsolidierungs_score, colors, shapes

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            df = pd.read_csv(file)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)

            # Get parameters
            params = {key: float(request.form[key]) for key in ['atr_short_len', 'atr_long_len', 'adx_len', 'channel_len', 'pkv_rank_len', 'obv_len', 'w1', 'w2', 'w3', 'w4', 'threshold_high', 'threshold_low']}
            
            # Calculate indicator
            score, colors, shapes = calculate_uki(df, **params)

            # Create Plotly figure
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                                subplot_titles=('Price Chart', 'UKI v2'))
            
            # Price chart
            fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price'), row=1, col=1)
            
            # Indicator chart
            fig.add_trace(go.Scatter(x=df.index, y=score, mode='lines', name='Consolidation Score', line=dict(color=colors)), row=2, col=1)
            fig.add_hline(y=params['threshold_high'], line_dash='dash', line_color='orange', row=2, col=1)
            fig.add_hline(y=params['threshold_low'], line_dash='dash', line_color='blue', row=2, col=1)
            fig.add_hline(y=50, line_dash='dot', line_color='gray', row=2, col=1)
            fig.update_layout(shapes=shapes, height=800, title_text="Universeller Konsolidierungsindikator v2")

            chart_json = fig.to_json()
            return render_template_string(HTML_TEMPLATE, chart_json=chart_json)
    return render_template_string(HTML_TEMPLATE)

if __name__ == '__main__':
    app.run(debug=True)

