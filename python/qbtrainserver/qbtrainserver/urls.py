from django.urls import path, include
from drf_spectacular.views import SpectacularAPIView, SpectacularRedocView, SpectacularSwaggerView
from . import views

urlpatterns = [
    path('api/schema/', SpectacularAPIView.as_view(), name='schema'),
    path('swagger/', SpectacularSwaggerView.as_view(url_name='schema'), name='swagger-ui'),
    path('redoc/', SpectacularRedocView.as_view(url_name='schema'), name='redoc'),

    path('api/health/', views.health, name='health'),

    path('api/apps/', include('common.app_registry.urls')),
    path('api/clients/', include('common.clients.urls')),

    path('api/apps/aisecurity/crdlr/', include('apps.aisecurity.crdlr.urls')),
    path('api/apps/aisecurity/echoleak/', include('apps.aisecurity.echoleak.urls')),
    path('api/apps/aisecurity/codeexec/', include('apps.aisecurity.codeexec.urls')),
    path('api/apps/aisecurity/modeltheft/', include('apps.aisecurity.modeltheft.urls')),

    path('ex/', include('common.exfil.urls')),
]
