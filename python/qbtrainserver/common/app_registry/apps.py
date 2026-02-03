# python/qbtrainserver/common/app_registry/apps.py
from django.apps import AppConfig
from django.core.exceptions import ImproperlyConfigured

class AppRegistryConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "common.app_registry"

    def ready(self):
        """
        Validate that all apps in the registry have globally-unique **ids**
        (slugs), which are used for routing (e.g. /app/<id>).
        Falls back to appId only for legacy entries that have no id.
        Compatible with both the new normalized structure:
            APPS: Dict[category_id -> Dict[subcategory_id -> List[app_dict]]]
        and the legacy structure:
            APPS: Dict[category_name, List[section_dict{ apps: [...] }]]
        """
        from .data import APPS
        
        seen_keys = set()
        duplicates = []

        if not isinstance(APPS, dict):
            return

        for cat_key, sub in APPS.items():
            # New structure: Dict[subcat_id -> List[apps]]
            if isinstance(sub, dict):
                for sub_key, apps in (sub.items() if sub else []):
                    for app in apps or []:
                        # Prefer id (new routing), fallback to legacy appId if no id present
                        key = app.get("id") or app.get("appId")
                        if not key:
                            continue
                        if key in seen_keys:
                            duplicates.append(f"{key} (category={cat_key}, subcategory={sub_key})")
                        else:
                            seen_keys.add(key)
                continue

            # Legacy structure: List[sections], each with {"apps": [...]}
            if isinstance(sub, list):
                for section in sub:
                    apps = (section or {}).get("apps") or []
                    for app in apps:
                        key = app.get("id") or app.get("appId")
                        if not key:
                            continue
                        if key in seen_keys:
                            sec_name = (section or {}).get("subcategory") or (section or {}).get("name") or "unknown"
                            duplicates.append(f"{key} (legacy; category={cat_key}, section={sec_name})")
                        else:
                            seen_keys.add(key)

        if duplicates:
            raise ImproperlyConfigured(
                f"Duplicate app ids detected in registry: {', '.join(duplicates)}. "
                "App ids (or legacy appIds when id is missing) must be globally unique for routing."
            )