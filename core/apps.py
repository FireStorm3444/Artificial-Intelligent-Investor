from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        # pass
        # populate the stock trie with stock data from the database
        from core.trie_instance import stock_trie
        from core.models import Stock

        stocks = Stock.objects.all()

        for stock in stocks:
            stock_trie.insert(stock.name.lower(), stock)
            stock_trie.insert(stock.ticker.lower(), stock)