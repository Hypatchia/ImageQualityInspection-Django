# Generated by Django 4.2.4 on 2023-08-07 17:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('ClassifierModel', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='uploadedimage',
            name='image',
            field=models.ImageField(upload_to='uploaded_images'),
        ),
    ]