import yfinance as yf
from django.http import HttpResponse
from functools import wraps

def yf_ticker_required(view_func):
    @wraps(view_func)
    def wrapper(request, ticker, *args, **kwargs):
        try:
            # This is the repeated logic
            yf_ticker = yf.Ticker(ticker + ".NS")
            # We can also check here if the ticker is valid
            if not yf_ticker.info.get('regularMarketPrice'):
                return HttpResponse(f"Invalid ticker or no data found for {ticker}.", status=404)
        except Exception as e:
            return HttpResponse(f"Error fetching data from yfinance: {str(e)}", status=500)

        # Pass the created yf_ticker object to the actual view
        return view_func(request, ticker, yf_ticker=yf_ticker, *args, **kwargs)

    return wrapper