from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('index/', views.index, name='index'),
    path('tspp_results/', views.tspp_results, name='tspp_results'),
    path('download_path_csv/<str:algorithm>/<int:Length>/<int:Width>/', views.download_path_csv, name='download_path_csv'),
    path('download_csv/<str:algorithm>/<int:Length>/<int:Width>/', views.download_elapsed_time_csv, name='download_csv'),
    path('download_cpu_csv/<str:algorithm>/<int:Length>/<int:Width>/', views.download_cpu_usages_csv, name='download_cpu_csv'),
]

