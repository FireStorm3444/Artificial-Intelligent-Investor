import yfinance as yf
from django.core.management.base import BaseCommand
from core.models import Stock

class Command(BaseCommand):
    help = "Updates the list of industries"
    def handle(self, *args, **options):
        counter = 1
        total = Stock.objects.count()
        for stock in Stock.objects.all():
            ticker = stock.ticker
            try:
                ticker_data = yf.Ticker(ticker + ".NS")
                # industry = ticker_data.info["industry"]
                # stock.industry = industry
                # sector = ticker_data.info["sector"]
                # stock.sector = sector
                # stock.save()
                market_cap = ticker_data.info.get("marketCap", 0)
                stock.market_cap = market_cap
                stock.save()
                print(f"{(counter/total)*100:.2f}% Done!! {total-counter} remaining.")
            except Exception as e:
                print(f"Error fetching industry for {ticker}: {e}")
            counter += 1