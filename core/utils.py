import yfinance as yf
from django.core.cache import cache

class CachedTicker:
    def __init__(self, ticker_symbol):
        self.ticker = ticker_symbol
        self.yf = yf.Ticker(ticker_symbol)

    def _get_cached_attr(self, attr_name):
        # Create a unique key for this stock + attribute (e.g., "RELIANCE.NS_info")
        cache_key = f"{self.ticker}_{attr_name}"

        # Try to get from cache; if missing, fetch from Yahoo and save it
        return cache.get_or_set(cache_key, lambda: getattr(self.yf, attr_name), timeout=1800)

    @property
    def info(self):
        return self._get_cached_attr('info')

    @property
    def balance_sheet(self):
        return self._get_cached_attr('balance_sheet')

    @property
    def financials(self):
        return self._get_cached_attr('financials')

    @property
    def quarterly_financials(self):
        return self._get_cached_attr('quarterly_financials')

    @property
    def cashflow(self):
        return self._get_cached_attr('cashflow')

    @property
    def major_holders(self):
        return self._get_cached_attr('major_holders')

    @property
    def institutional_holders(self):
        return self._get_cached_attr('institutional_holders')

    @property
    def mutualfund_holders(self):
        return self._get_cached_attr('mutualfund_holders')

    @property
    def news(self):
        return self._get_cached_attr('news')

    # History is a method, not a property, so we handle it differently
    def history(self, period="max"):
        cache_key = f"{self.ticker}_history_{period}"
        return cache.get_or_set(cache_key, lambda: self.yf.history(period=period), timeout=1800)