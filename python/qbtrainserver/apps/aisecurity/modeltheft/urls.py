from django.urls import path
from . import views

urlpatterns = [
    path("session/", views.get_session, name="modeltheft_session"),
    path("methods/", views.get_methods, name="modeltheft_methods"),
    path("configure/", views.configure, name="modeltheft_configure"),
    path("reset/", views.reset, name="modeltheft_reset"),
    path("train/", views.train, name="modeltheft_train"),
    path("test/", views.test, name="modeltheft_test"),

    # Student model management
    path("students/", views.student_models, name="modeltheft_student_models"),
    path("students/download/", views.download_student, name="modeltheft_download_student"),
    path("students/status/", views.download_status, name="modeltheft_download_status"),
    path("students/delete/", views.delete_student, name="modeltheft_delete_student"),
]
