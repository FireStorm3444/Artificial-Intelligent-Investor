import yfinance as yf
from django.core.cache import cache
import time
import random


class CachedTicker:
    def __init__(self, ticker_symbol):
        self.ticker = ticker_symbol
        self.yf = yf.Ticker(ticker_symbol)

    def _fetch_with_retry(self, attr_name, retries=3):
        """
        Tries to fetch data. If Rate Limited, waits and retries.
        """
        for i in range(retries):
            try:
                # Try to fetch the attribute (e.g., .info, .financials)
                data = getattr(self.yf, attr_name)
                return data
            except Exception as e:
                error_msg = str(e).lower()
                # If it's a rate limit error, wait and try again
                if "too many requests" in error_msg or "429" in error_msg:
                    print(f"Rate limit hit for {self.ticker}. Retrying {i + 1}/{retries}...")
                    time.sleep(random.uniform(1.5, 3.5))  # Sleep 1.5-3.5 seconds
                else:
                    # If it's another error (like Symbol Not Found), stop trying
                    print(f"Error fetching {attr_name} for {self.ticker}: {e}")
                    return None
        return None

    def _get_cached_attr(self, attr_name):
        cache_key = f"{self.ticker}_{attr_name}"

        # 1. Check Cache First
        cached_data = cache.get(cache_key)
        if cached_data is not None:
            return cached_data

        # 2. If missing, Fetch with Retry Logic
        fresh_data = self._fetch_with_retry(attr_name)

        # 3. Save to Cache (valid for 30 mins)
        if fresh_data is not None:
            cache.set(cache_key, fresh_data, timeout=1800)

        return fresh_data

    @property
    def info(self):
        # Return empty dict if failed, so views don't crash with 500 error
        data = self._get_cached_attr('info')
        return data if data is not None else {}

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

    def history(self, period="max"):
        cache_key = f"{self.ticker}_history_{period}"
        cached_data = cache.get(cache_key)

        if cached_data is not None:
            return cached_data

        # Fetch with simple retry for history
        try:
            fresh_data = self.yf.history(period=period)
            if not fresh_data.empty:
                cache.set(cache_key, fresh_data, timeout=1800)
            return fresh_data
        except Exception:
            return None