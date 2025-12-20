import csv
from django.core.management.base import BaseCommand
from core.models import Stock

class Command(BaseCommand):
    help = 'Loads stocks from a CSV file into the database'

    def handle(self, *args, **kwargs):
        Stock.objects.all().delete()
        with open('nse_stocks.csv', 'r') as file:
            reader = csv.DictReader(file)

            for row in reader:
                ticker = row['SYMBOL']
                name = row['NAME OF COMPANY']
                Stock.objects.create(ticker=ticker, name=name)