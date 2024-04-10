# Generated by Django 5.0.3 on 2024-04-09 20:57

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='AppleStockData',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('datetime', models.DateTimeField()),
                ('open', models.DecimalField(decimal_places=5, max_digits=10)),
                ('high', models.DecimalField(decimal_places=5, max_digits=10)),
                ('low', models.DecimalField(decimal_places=5, max_digits=10)),
                ('close', models.DecimalField(decimal_places=5, max_digits=10)),
                ('volume', models.IntegerField()),
                ('New_RSI_7', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_RSI_14', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_MACD', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_SMA_50', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_SMA_100', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_EMA_50', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_EMA_100', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_Upper_Band', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_Lower_Band', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_TrueRange', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_ATR_7', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_ATR_14', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_CCI_7', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
                ('New_CCI_14', models.DecimalField(decimal_places=5, default=0.0, max_digits=10)),
            ],
        ),
    ]