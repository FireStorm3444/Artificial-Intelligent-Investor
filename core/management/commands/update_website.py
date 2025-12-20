import yfinance as yf
from django.core.management.base import BaseCommand
from core.models import Stock

class Command(BaseCommand):
    help = "Updates the website URLs for all stocks"

    def handle(self, *args, **options):
        counter = 1
        total = Stock.objects.count()
        for stock in Stock.objects.all():
            ticker = stock.ticker
            try:
                ticker_data = yf.Ticker(ticker + ".NS")
                website = ticker_data.info.get("website", "")
                stock.website = website
                stock.save()
                print(f"{(counter/total)*100:.2f}% Done!! {total-counter} remaining.")
            except Exception as e:
                print(f"Error fetching website for {ticker}: {e}")
            counter += 1