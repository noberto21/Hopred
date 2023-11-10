from django.urls import path
from . import views

urlpatterns = [
    path('', views.home ,name ='home'),
    path('predict/', views.predict),
    path('predict/predictions/', views.predictions),
    path('about/', views.about, name='about')
]