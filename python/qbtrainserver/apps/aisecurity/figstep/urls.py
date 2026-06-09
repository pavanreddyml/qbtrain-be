# apps/aisecurity/figstep/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path("query/", views.query, name="figstep_query"),
    path("generate-image/", views.generate_image, name="figstep_generate_image"),
    path("analyze-image/", views.analyze_image, name="figstep_analyze_image"),
]
