from django.urls import path

from . import views

urlpatterns = [
    path("registry/categories", views.get_app_categories, name="registry_categories"),
    path("registry/apps", views.get_apps, name="registry_apps"),
    # Updated: query params ?type=(category|subcategory|app)&tag=<tag>
    path("registry/image", views.get_registry_image, name="registry_image"),
]
