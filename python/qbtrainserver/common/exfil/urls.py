from django.urls import path
from . import views

urlpatterns = [
    path("health/", views.health, name="el-health"),
    path("add-log/", views.add_log, name="el-add-log"),
    path("fetch-logs/", views.fetch_logs, name="el-fetch-logs"),
    path("delete-logs/", views.delete_logs, name="el-delete-logs"),
    path("get-image/", views.get_image, name="el-get-image"),
    path("get-image", views.get_image, name="el-get-image-noslash"),
]
