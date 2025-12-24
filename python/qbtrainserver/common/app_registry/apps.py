# python/qbtrainserver/common/app_registry/apps.py
from django.apps import AppConfig
from django.core.exceptions import ImproperlyConfigured

class AppRegistryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "common.app_registry"

    def ready(self):
        """
        Validate that all apps in the registry have unique appIds.
        """
        from .data import APPS
        
        seen_ids = set()
        duplicates = []

        # Iterate through Categories -> Subcategories -> Apps
        for category, subcats in APPS.items():
            for subcat in subcats:
                for app in subcat.get("apps", []):
                    # Only validate if appId exists (allows gradual migration)
                    app_id = app.get("appId")
                    
                    if app_id:
                        if app_id in seen_ids:
                            duplicates.append(f"{app_id} (in {subcat['subcategory']})")
                        seen_ids.add(app_id)

        if duplicates:
            raise ImproperlyConfigured(
                f"Duplicate appIds detected in registry: {', '.join(duplicates)}. "
                "appIds must be unique for routing."
            )