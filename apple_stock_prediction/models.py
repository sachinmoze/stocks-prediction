from django.db import models

# Create your models here.

class AppleStockData(models.Model):
    datetime = models.DateTimeField()
    open = models.DecimalField(max_digits=10, decimal_places=5)
    high = models.DecimalField(max_digits=10, decimal_places=5)
    low = models.DecimalField(max_digits=10, decimal_places=5)
    close = models.DecimalField(max_digits=10, decimal_places=5)
    volume = models.IntegerField()
    New_RSI_7 = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_RSI_14 = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_MACD = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_SMA_50 = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_SMA_100 = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_EMA_50 = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_EMA_100 = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_Upper_Band = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_Lower_Band = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_TrueRange = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_ATR_7 = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_ATR_14 = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_CCI_7 = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
    New_CCI_14 = models.DecimalField(max_digits=10, decimal_places=5, default=0.0)
