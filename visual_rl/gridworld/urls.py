from django.urls import path

from . import views

urlpatterns = [
    path('', views.tabular_grid),
    path('add', views.add),
]