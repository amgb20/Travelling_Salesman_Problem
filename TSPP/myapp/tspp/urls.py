from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('download_csv/<str:algorithm>/<int:Length>x<int:Width>/', views.download_csv, name='download_csv'),
    # path('result/', views.result, name='result'),
]

