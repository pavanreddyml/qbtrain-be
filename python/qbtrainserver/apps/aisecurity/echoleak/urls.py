# apps/aisecurity/echoleak/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path("query/", views.query, name="query"),
    path("pdf/exfil/", views.pdf_exfil, name="pdf_exfil"),
]
