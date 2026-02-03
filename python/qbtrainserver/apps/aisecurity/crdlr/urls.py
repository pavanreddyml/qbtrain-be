# apps/aisecurity/crdlr/urls.py
from django.urls import path

from . import views

urlpatterns = [
    path("health/", views.health, name="health"),
    path("db/reset/", views.db_reset, name="db_reset"),
    # Model images (query param: ?id=<model_id>)
    path("models/image/", views.model_image, name="model_image"),
    # Dealerships: intentionally no endpoints (read-only helpers exist in functions.py)
    # Brands
    path("brands/", views.brands_list, name="brands_list"),
    path("brands/add/", views.brands_create, name="brands_create"),
    path("brands/<int:brand_id>/", views.brand_get, name="brand_get"),
    path("brands/<int:brand_id>/update/", views.brand_update, name="brand_update"),
    path("brands/<int:brand_id>/delete/", views.brand_delete, name="brand_delete"),
    # Models
    path("models/", views.models_list, name="models_list"),
    path("models/add/", views.models_create, name="models_create"),
    path("models/<int:model_id>/", views.model_get, name="model_get"),
    path("models/<int:model_id>/update/", views.model_update, name="model_update"),
    path("models/<int:model_id>/delete/", views.model_delete, name="model_delete"),
    # Vehicles
    path("vehicles/", views.vehicles_list, name="vehicles_list"),
    path("vehicles/add/", views.vehicles_create, name="vehicles_create"),
    path("vehicles/<int:vehicle_id>/", views.vehicle_get, name="vehicle_get"),
    path("vehicles/<int:vehicle_id>/update/", views.vehicle_update, name="vehicle_update"),
    path("vehicles/<int:vehicle_id>/delete/", views.vehicle_delete, name="vehicle_delete"),
    # Customers
    path("customers/", views.customers_list, name="customers_list"),
    path("customers/add/", views.customers_create, name="customers_create"),
    path("customers/<int:customer_id>/", views.customer_get, name="customer_get"),
    path("customers/<int:customer_id>/update/", views.customer_update, name="customer_update"),
    path("customers/<int:customer_id>/delete/", views.customer_delete, name="customer_delete"),
    # Employees
    path("employees/", views.employees_list, name="employees_list"),
    path("employees/add/", views.employees_create, name="employees_create"),
    path("employees/<int:employee_id>/", views.employee_get, name="employee_get"),
    path("employees/<int:employee_id>/update/", views.employee_update, name="employee_update"),
    path("employees/<int:employee_id>/delete/", views.employee_delete, name="employee_delete"),
    # Orders
    path("orders/", views.orders_list, name="orders_list"),
    path("orders/add/", views.orders_create, name="orders_create"),
    path("orders/<int:order_id>/", views.order_get, name="order_get"),
    path("orders/<int:order_id>/update/", views.order_update, name="order_update"),
    path("orders/<int:order_id>/delete/", views.order_delete, name="order_delete"),
    path("orders/<int:order_id>/items/", views.order_items_list, name="order_items_list"),
    path("orders/<int:order_id>/items/add/", views.order_items_add, name="order_items_add"),
    path("orders/<int:order_id>/items/<int:order_item_id>/update/", views.order_item_update, name="order_item_update"),
    path("orders/<int:order_id>/items/<int:order_item_id>/delete/", views.order_item_delete, name="order_item_delete"),
    path("orders/<int:order_id>/payments/", views.order_payments_list, name="order_payments_list"),
    path("orders/<int:order_id>/payments/add/", views.order_payments_add, name="order_payments_add"),
    path("orders/<int:order_id>/status/", views.order_set_status, name="order_set_status"),
    path("orders/<int:order_id>/history/", views.order_history, name="order_history"),
    # Reports
    path("reports/inventory/", views.report_inventory, name="report_inventory"),
    path("reports/sales/", views.report_sales, name="report_sales"),
    # Assistant
    path("assistant/query/", views.assistant_query, name="assistant_query"),
    path("assistant/stream/", views.assistant_stream, name="assistant_stream"),
    # perms and users
    path("permissions/", views.get_available_permissions, name="perms_list"),
    path("permissions/bypass/", views.get_bypass_permission, name="perms_bypass"),
    path("users/", views.get_available_users, name="users_list"),
    path("stored_procedures/", views.get_available_stored_procedures, name="stored_procedures_list"),
]
