from django.db import models
from core.models import Stock
from django.contrib.auth.models import User

# Create your models here.
class Watchlist(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    stock = models.ForeignKey(Stock, on_delete=models.CASCADE)

    class Meta:
        unique_together = ('user', 'stock')