import yfinance as yf


class StockData:
    def __init__(self, symbol: str = "AAPL", start_date: str = "2009-01-01", end_date: str = "2023-12-31"):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date

    def get_historical_symbol_data(self):
        data_df = yf.download(self.symbol, start=self.start_date, end=self.end_date)
        data_df = data_df[['Adj Close', 'Open', 'High', 'Low', 'Close', 'Volume']].round(2)
        return data_df


