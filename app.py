from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from yahooquery import Ticker
import os

app = Flask(__name__)

def plot_ticker_analysis(ticker):
    try:
        stock = Ticker(ticker)
        data = stock.history(period='6mo', interval='1d')
        if isinstance(data, dict) or data.empty:
            return False

        data = data.reset_index()
        data = data.pivot(index='date', columns='symbol')
        data.columns = ['_'.join(col).strip() for col in data.columns.values]
        data = data.filter(like=f'_{ticker}')
        data.columns = [col.replace(f'_{ticker}', '') for col in data.columns]
        data.index = pd.to_datetime(data.index, utc=True).tz_localize(None)
    except:
        return False

    if 'volume' not in data.columns:
        return False

    data['20_MA'] = data['close'].rolling(window=20).mean()
    data['20_STD'] = data['close'].rolling(window=20).std()
    data['Upper_BB'] = data['20_MA'] + (data['20_STD'] * 2)
    data['Lower_BB'] = data['20_MA'] - (data['20_STD'] * 2)

    data['TP'] = (data['high'] + data['low'] + data['close']) / 3
    data['volume'] = data['volume'].fillna(0)
    data['Raw_MF'] = data['TP'] * data['volume']
    data['Positive_MF'] = np.where(data['TP'].diff() > 0, data['Raw_MF'], 0)
    data['Negative_MF'] = np.where(data['TP'].diff() < 0, data['Raw_MF'], 0)
    data['MFR'] = data['Positive_MF'].rolling(window=14).sum() / data['Negative_MF'].rolling(window=14).sum()
    data['MFI'] = 100 - (100 / (1 + data['MFR']))

    data['EMA_12'] = data['close'].ewm(span=12, adjust=False).mean()
    data['EMA_26'] = data['close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()
    data['MACD_Gap'] = data['MACD'] - data['Signal_Line']

    macd_min = data['MACD_Gap'].min()
    macd_max = data['MACD_Gap'].max()
    data['MACD_Normalized'] = (data['MACD_Gap'] - macd_min) / (macd_max - macd_min) * 200 - 100

    top_5 = data['MACD_Normalized'].nlargest(5)
    bottom_5 = data['MACD_Normalized'].nsmallest(5)

    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(3, 2, width_ratios=[3, 1])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(data['close'], label='Close Price', color='blue')
    ax1.plot(data['20_MA'], label='20-day MA', color='orange')
    ax1.plot(data['Upper_BB'], label='Upper BB', color='green')
    ax1.plot(data['Lower_BB'], label='Lower BB', color='red')
    ax1.set_title(f'{ticker} - Bollinger Bands')
    ax1.legend()

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(data['MFI'], label='MFI', color='purple')
    ax2.axhline(80, color='red', linestyle='--')
    ax2.axhline(20, color='green', linestyle='--')
    ax2.set_title(f'{ticker} - Money Flow Index (MFI)')
    ax2.legend()

    ax3 = fig.add_subplot(gs[2, 0])
    ax3.bar(data.index, data['MACD_Gap'], label='MACD Gap',
            color=['green' if x > 0 else 'red' for x in data['MACD_Gap']], alpha=0.5)
    ax3.plot(data['MACD'], label='MACD', color='blue')
    ax3.plot(data['Signal_Line'], label='Signal Line', color='red')
    ax3.set_title(f'{ticker} - MACD with Gap')
    ax3.legend()

    ax_values = fig.add_subplot(gs[:, 1])
    ax_values.axis("off")
    sorted_values = data['MACD_Normalized'].dropna()
    num_entries = len(sorted_values)
    mid_idx = num_entries // 2
    row_spacing = 0.025

    for idx, (date, value) in enumerate(sorted_values.items()):
        color = 'green' if date in top_5.index else 'red' if date in bottom_5.index else 'black'
        col_x = 0.05 if idx < mid_idx else 0.55
        row_y = 1.0 - (idx % mid_idx) * row_spacing
        ax_values.text(col_x, row_y, f"{date.strftime('%Y-%m-%d')}: {value:.1f}%", fontsize=8, color=color,
                       transform=ax_values.transAxes)

    plt.tight_layout()
    output_path = os.path.join("static", "output.png")
    plt.savefig(output_path)
    plt.close()
    return True

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        ticker = request.form.get('ticker')
        if plot_ticker_analysis(ticker):
            return redirect(url_for('result'))
        else:
            return render_template('index.html', error="데이터를 불러오지 못했습니다.")
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port, debug=False)
