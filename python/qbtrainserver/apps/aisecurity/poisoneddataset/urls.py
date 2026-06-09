# apps/aisecurity/poisoneddataset/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path("meta/", views.meta, name="poisoneddataset_meta"),
    path("start/", views.start, name="poisoneddataset_start"),
    path("stop/", views.stop, name="poisoneddataset_stop"),
    path("status/<str:job_id>/", views.session_status, name="poisoneddataset_status"),
    path("stream/<str:job_id>/", views.stream, name="poisoneddataset_stream"),
    path("analyze/", views.analyze_samples, name="poisoneddataset_analyze"),
    path("test/", views.test_model, name="poisoneddataset_test"),
]
