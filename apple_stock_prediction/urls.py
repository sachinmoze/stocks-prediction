from django.urls import path
from . import views


urlpatterns=[
    path("",views.apple_stock_home,name="apple_stock_home"),
    path('predict/', views.predict_view, name='predict'),
    path('predict-manual/', views.predict_manual_view, name='predict_manual'),
    path('stock-price/', views.get_current_stock_price, name='current_stock_price'),
    path('start_ws_client/', views.start_ws_client, name='start_ws_client'),
    path('fetch_store_data/', views.fetch_and_store_stock_data, name='fetch_and_store_stock_data'),
    path('fetch_data/', views.custom_data_store, name='fetch_stock_data'),
    path('latest_data/', views.latest_stock_data_view, name='latest_stock_data_view'),
    path('fetch-chart-data/', views.fetch_chart_data, name='fetch_chart_data'),

    
]