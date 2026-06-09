# apps/aisecurity/imscaler/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path("query/", views.query, name="imscaler_query"),
    path("generate-image/", views.generate_image, name="imscaler_generate_image"),
    path("preprocess-image/", views.preprocess_image, name="imscaler_preprocess_image"),
    path("read-image-metadata/", views.read_image_metadata, name="imscaler_read_image_metadata"),
    path("defense-preview/", views.defense_preview, name="imscaler_defense_preview"),
]
