from django.urls import path
from . import views

urlpatterns = [
    path('', views.home ,name ='home'),
    path('predict/', views.predict),
    path('predict/predictions/', views.predictions),
    path('about/', views.about, name='about'),
    path('signup', views.signup, name='signup'),
    path('login', views.loginpage, name='login'),
    path('logout', views.logoutUser, name='logout'),
]