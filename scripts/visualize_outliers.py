import matplotlib.pyplot as plt
import seaborn as sns


def visualize_outliers(df, ticker):
    """
    Generates plots to identify outliers in stock data.
    """
    plt.figure(figsize=(12, 6))

    # Plot Bollinger Bands
    df['ma_20'] = df['close'].rolling(20).mean()
    df['upper_band'] = df['ma_20'] + 2 * df['close'].rolling(20).std()
    df['lower_band'] = df['ma_20'] - 2 * df['close'].rolling(20).std()

    plt.plot(df['date'], df['close'], label='Close Price')
    plt.plot(df['date'], df['upper_band'], label='Upper Band', linestyle='--')
    plt.plot(df['date'], df['lower_band'], label='Lower Band', linestyle='--')
    plt.title(f'{ticker} Price with Bollinger Bands (20-day)')
    plt.legend()
    plt.show()

    # Boxplot of daily returns
    df['daily_return'] = df['close'].pct_change()
    sns.boxplot(x=df['daily_return'].dropna())
    plt.title(f'{ticker} Daily Returns Distribution')


# Usage
visualize_outliers(df_aapl, 'AAPL')
visualize_outliers(df_msft, 'MSFT')