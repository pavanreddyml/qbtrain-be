# apps/aisecurity/cursedpixels/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path("meta/", views.meta, name="cursedpixels_meta"),
    path("start/", views.start, name="cursedpixels_start"),
    path("stop/", views.stop, name="cursedpixels_stop"),
    path("status/<str:job_id>/", views.session_status, name="cursedpixels_status"),
    path("stream/<str:job_id>/", views.stream, name="cursedpixels_stream"),
    path("test/", views.test, name="cursedpixels_test"),
]
