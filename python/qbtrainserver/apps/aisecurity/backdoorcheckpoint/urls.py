# apps/aisecurity/backdoorcheckpoint/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path("meta/",                       views.meta,              name="bc_meta"),
    path("models/",                     views.list_models,       name="bc_list_models"),
    path("models/download/",            views.start_download,    name="bc_start_download"),
    path("models/download-status/",     views.download_status,   name="bc_download_status"),
    path("models/download-stream/<str:model_id>/", views.download_stream, name="bc_download_stream"),
    path("query/",                      views.query,             name="bc_query"),
    path("samples/",                    views.samples,           name="bc_samples"),
    path("samples/image/",              views.sample_image,      name="bc_sample_image"),
]
