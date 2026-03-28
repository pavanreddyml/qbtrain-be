from django.urls import path
from . import views

urlpatterns = [
    path("models/", views.list_models, name="modeltheftimages_models"),
    path("generate/", views.generate, name="modeltheftimages_generate"),
    path("download/", views.download_all, name="modeltheftimages_download"),
    path("download/status/", views.download_status, name="modeltheftimages_download_status"),
    path("unload/", views.unload_model, name="modeltheftimages_unload"),
]
