# Generated manually to fix timestamp fields

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('core', '0007_qualitativeanalysis'),
    ]

    operations = [
        # Change StockAnalysis.last_updated from auto_now to auto_now_add
        migrations.AlterField(
            model_name='stockanalysis',
            name='last_updated',
            field=models.DateTimeField(auto_now_add=True),
        ),
        # Change QualitativeAnalysis.last_generated from auto_now to auto_now_add
        migrations.AlterField(
            model_name='qualitativeanalysis',
            name='last_generated',
            field=models.DateTimeField(auto_now_add=True),
        ),
    ]

