import yfinance as yf
from django.http import HttpResponse
from functools import wraps
import requests

def yf_ticker_required(view_func):
    @wraps(view_func)
    def wrapper(request, ticker, *args, **kwargs):
        try:
            # This is the repeated logic
            session = requests.Session()
            session.headers.update({
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            })
            yf_ticker = yf.Ticker(ticker + ".NS", session=session)
            # We can also check here if the ticker is valid
            if not yf_ticker.info.get('regularMarketPrice'):
                return HttpResponse(f"Invalid ticker or no data found for {ticker}.", status=404)
        except Exception as e:
            return HttpResponse(f"Error fetching data from yfinance: {str(e)}", status=500)

        # Pass the created yf_ticker object to the actual view
        return view_func(request, ticker, yf_ticker=yf_ticker, *args, **kwargs)

    return wrapper