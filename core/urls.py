from django.urls import path, include
from .views import emergency_password_reset
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('search/', views.search_stocks, name='search'),
    path('stock/<str:ticker>', views.stock_details, name='stock_detail'),
    path('stock/<str:ticker>/qualitative_analysis/', views.get_qualitative_partial, name='get_qualitative_analysis'),
    path('stock/<str:ticker>/info/', views.get_stats_partial, name='get_stats'),
    path('stock/<str:ticker>/chart/', views.get_chart_partial, name='get_chart'),
    path('stock/<str:ticker>/aii_analysis', views.get_analysis_partial, name='get_analysis'),
    path('stock/<str:ticker>/news/', views.get_news_partial, name='get_news'),
    path('stock/<str:ticker>/peer_comparison/', views.get_peer_comparison_partial, name='get_peers'),
    path('stock/<str:ticker>/quarters/', views.get_profit_loss_quarterly_partial, name='get_profit_loss_quarterly'),
    path('stock/<str:ticker>/profit_loss/', views.get_profit_loss_partial, name='get_profit_loss'),
    path('stock/<str:ticker>/balance_sheet/', views.get_balance_sheet_partial, name='get_balance_sheet'),
    path('stock/<str:ticker>/cash_flow/', views.get_cash_flow_partial, name='get_cash_flow'),
    path('stock/<str:ticker>/ratios/', views.get_ratios_partial, name='get_ratios'),
    path('stock/<str:ticker>/shareholding/', views.get_shareholding_partial, name='get_shareholding'),
    path('stock/<str:ticker>/valuation/', views.get_valuation_partial, name='get_valuation'),
    path('emergency-reset-999/', emergency_password_reset), # Secret hard-to-guess URL
]