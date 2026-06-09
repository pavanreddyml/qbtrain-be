# apps/aisecurity/imageadvattacks/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path("meta/", views.meta, name="imageadvattacks_meta"),
    path("models/", views.list_models, name="imageadvattacks_models"),
    path("start/", views.start, name="imageadvattacks_start"),
    path("stop/", views.stop, name="imageadvattacks_stop"),
    path("status/<str:job_id>/", views.session_status, name="imageadvattacks_status"),
    path("stream/<str:job_id>/", views.stream, name="imageadvattacks_stream"),
]
