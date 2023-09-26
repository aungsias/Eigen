import pandas as pd
import yfinance as yf

class SNP500:

    def __init__(self):
        self.url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        self.tickers = None
        self.sectors = None
        self.stocks_by_sector = None
        self.data = None

    def get(self, start, end, progress=True):
        table = self._get_table()
        self.tickers = table["Symbol"].str.replace(".", "-", regex=False).to_list()
        self.sectors = table["GICS Sector"].to_list()
        self.stocks_by_sector = table[["Symbol", "GICS Sector"]].set_index("GICS Sector").sort_index()
        self.data = yf.download(self.tickers, start, end, progress=progress)

    def _get_table(self):
        return pd.read_html(self.url)[0]