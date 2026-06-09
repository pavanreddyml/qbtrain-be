from django.apps import AppConfig


class PoisonedDatasetConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.aisecurity.poisoneddataset"
    label = "aisecurity_poisoneddataset"
