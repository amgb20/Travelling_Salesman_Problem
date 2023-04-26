from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('index/', views.index, name='index'),
    path('tspp_results/', views.tspp_results, name='tspp_results'),

    # path('', views.index, name='index'),
    path('download_path_csv/<str:algorithm>/<int:Length>x<int:Width>/', views.download_path_csv, name='download_path_csv'),
    # path('download_elapsed_time_csv/<str:algorithm>/<int:Length>x<int:Width>/', views.download_elapsed_time_csv, name='download_elapsed_time_csv'),
    path('download_csv/<str:algorithm>/<int:Length>/<int:Width>/', views.download_elapsed_time_csv, name='download_csv'),
    path('download_csv/<str:algorithm>/<int:Length>/<int:Width>/', views.download_cpu_usages_csv, name='download_csv'),
    # path('result/', views.result, name='result'),
]

