from django.contrib.auth.decorators import login_required
from core.models import Stock
from users.models import Watchlist
from django.shortcuts import render, get_object_or_404

# Create your views here.
@login_required
def toggle_watchlist(request, ticker):
    # Get the Stock object using the ticker
    stock = get_object_or_404(Stock, ticker__iexact=ticker)

    # Try to find a Watchlist object for the current user and stock
    try:
        watchlist_item = Watchlist.objects.get(user=request.user, stock=stock)
        # If found, user wants to remove it
        watchlist_item.delete()
        is_in_watchlist = False
    except Watchlist.DoesNotExist:
        # If not found, user wants to add it
        Watchlist.objects.create(user=request.user, stock=stock)
        is_in_watchlist = True

    # Return the updated button by rendering a partial template
    return render(request, 'users/partials/watchlist_button.html', {
        'stock': stock,
        'is_in_watchlist': is_in_watchlist
    })

@login_required
def watchlist_view(request):
    watchlist = Watchlist.objects.filter(user=request.user)
    stock_ids = watchlist.values_list('stock_id', flat=True)
    added_stocks = Stock.objects.filter(id__in=stock_ids)
    print(added_stocks)
    return render(request, 'users/watchlist.html', {'stocks': added_stocks})
