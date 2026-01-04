from django.db import models

class Stock(models.Model):
    ticker = models.CharField(max_length=10, unique=True)
    name = models.CharField(max_length=255)
    industry = models.CharField(max_length=255, null=True, blank=True)
    sector = models.CharField(max_length=255, null=True, blank=True)
    website = models.URLField(null=True, blank=True)
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.name} ({self.ticker})"

class StockAnalysis(models.Model):
    stock = models.OneToOneField(Stock, on_delete=models.CASCADE, primary_key=True)
    stock_ticker = models.CharField(max_length=255)
    analysis_text = models.TextField()
    last_updated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Analysis for {self.stock.ticker}"

class QualitativeAnalysis(models.Model):
    stock = models.OneToOneField(Stock, on_delete=models.CASCADE, primary_key=True)
    stock_ticker = models.CharField(max_length=255)
    qualitative_analysis = models.TextField(null=True, blank=True)
    last_generated = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"Qualitative analysis for {self.stock.ticker}"