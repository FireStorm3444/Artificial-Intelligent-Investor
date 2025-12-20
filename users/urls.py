from django.urls import path
from . import views

app_name = 'users'
urlpatterns = [
    path('toggle-watchlist/<str:ticker>/', views.toggle_watchlist, name='toggle_watchlist'),
    path('watchlist/', views.watchlist_view, name='watchlist'),
]

