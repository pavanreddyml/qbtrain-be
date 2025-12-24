# qbtrainserver/common/clients/urls.py
from django.urls import path

from . import views

app_name = "qbtrain_llm"

urlpatterns = [
    # Health
    path("health/", views.health, name="health"),

    # Hugging Face
    path("hf/download/", views.hf_download, name="hf_download"),
    path("hf/status/", views.hf_status, name="hf_status"),
    path("hf/models/", views.hf_list_models, name="hf_list_models"),
    path("hf/models/delete/", views.hf_delete_model, name="hf_delete_model"),

    # Ollama
    path("ollama/download/", views.ollama_download, name="ollama_download"),
    path("ollama/status/", views.ollama_status, name="ollama_status"),
    path("ollama/models/", views.ollama_list_models, name="ollama_list_models"),
    path("ollama/models/delete/", views.ollama_delete_model, name="ollama_delete_model"),

    # LLM calls
    path("llm/response/", views.llm_response, name="llm_response"),
    path("llm/json/", views.llm_json_response, name="llm_json_response"),
    path("llm/stream/", views.llm_stream_response, name="llm_stream_response"),

    # Client specs
    path("specs/", views.clients_specs, name="clients_specs"),
]
