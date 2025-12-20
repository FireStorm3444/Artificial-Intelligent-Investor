import sys

from django.apps import AppConfig


class CoreConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'core'

    def ready(self):
        # pass
        # populate the stock trie with stock data from the database
        from core.trie_instance import stock_trie
        from core.models import Stock
        from django.db.utils import OperationalError, ProgrammingError

        # 1. Check if we are running a management command (like collectstatic or migrate)
        # If so, we should skip loading the data to prevent errors during build.
        if 'collectstatic' in sys.argv or 'migrate' in sys.argv or 'makemigrations' in sys.argv:
            return

        # 2. Wrap the DB query in a try-except block
        # This allows the app to start even if the table doesn't exist yet.
        try:
            stocks = Stock.objects.all()
            for stock in stocks:
                stock_trie.insert(stock.name.lower(), stock)
                stock_trie.insert(stock.ticker.lower(), stock)
        except (OperationalError, ProgrammingError):
            pass