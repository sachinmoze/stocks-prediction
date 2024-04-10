from datetime import datetime, timedelta
from django.shortcuts import render
from django.http import HttpResponse,JsonResponse     
import yfinance as yf
# Import required libraries
import asyncio
import json
import websockets
import requests
from django.shortcuts import render
import random
from django.http import JsonResponse
from .models import AppleStockData

from django.http import JsonResponse
from .models import AppleStockData
from django.db.models import Max
import pandas as pd
import numpy as np

import pickle
import os
# Create your views here.

model_file = os.path.join(os.path.dirname(__file__), 'apple_random_forest_model.pkl')

# Load the pre-trained model from the pickle file
with open(model_file, 'rb') as f:
    apple_model = pickle.load(f)

    
   



# Function to parse real-time data and extract stock price
def parse_stock_price(data):
    try:
        # Parse the received JSON data
        parsed_data = json.loads(data)
        # Extract the stock price from the parsed data
        stock_price = parsed_data['price']  # Adjust this based on the actual structure of your data
        return stock_price
    except json.JSONDecodeError:
        # Handle JSON decoding errors
        print("Error decoding JSON data")
        return None
    except KeyError:
        # Handle missing or incorrect keys in the data
        print("Missing or incorrect key in data")
        return None
    
# Function to update real-time stock price in Django view or template
def update_stock_price(stock_price):
    # Implement logic to update the real-time stock price in the view or template
    # For example, you can update a variable or HTML element in your template
    # Here's a hypothetical example of updating a variable in the view
    # You need to adapt this based on your actual implementation
    context = {}
    if stock_price is not None:
        # Update the stock price in the view or template
        # For example, update a variable named 'real_time_stock_price' in the context
        context['real_time_stock_price'] = stock_price
    else:
        # Handle cases where the stock price is not available or invalid
        pass

# WebSocket client to subscribe to real-time stock data
async def stock_price_ws():
    async with websockets.connect('ws://provider-websocket-url') as websocket:
        # Subscribe to real-time data for the desired stock(s)
        await websocket.send(json.dumps({'action': 'subscribe', 'symbols': ['AAPL']}))
        
        # Continuously receive and process real-time data updates
        while True:
            data = await websocket.recv()
            # Parse and extract the stock price from the received data
            stock_price = parse_stock_price(data)
            # Update the real-time stock price in the Django view or template
            update_stock_price(stock_price)



# Django view function to start the WebSocket client
def start_ws_client(request):
    # Start the WebSocket client in a separate thread or process
    asyncio.ensure_future(stock_price_ws())
    return HttpResponse("WebSocket client started successfully.")

# Django view function to fetch real-time stock data from Alpha Vantage API
def fetch_realtime_stock_data(request):
    # API endpoint for Alpha Vantage
    api_key = 'YOUR_ALPHA_VANTAGE_API_KEY'
    symbol = 'AAPL'  # Example stock symbol
    
    # Make a request to Alpha Vantage API to fetch real-time data
    url = f'https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    
    if response.status_code == 200:
        # Extract real-time stock price from the response
        data = response.json()
        stock_price = data['Global Quote']['05. price']  # Example key for real-time price
        
        # Update the real-time stock price in the Django view or template
        update_stock_price(stock_price)
        return JsonResponse({'success': True})
    else:
        return JsonResponse({'success': False, 'message': 'Failed to fetch real-time stock data.'})



# Define RSI calculation function
def calculate_rsi(data, period):
    delta = data.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Define MACD calculation function
def calculate_macd(data):
    ema_12 = data.ewm(span=12, min_periods=0, adjust=False).mean()
    ema_26 = data.ewm(span=26, min_periods=0, adjust=False).mean()
    macd_line = ema_12 - ema_26
    return macd_line

# Define Bollinger Bands calculation function
def calculate_bollinger_bands(data, window=20, num_std=2):
    rolling_mean = data.rolling(window=window).mean()
    rolling_std = data.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Define CCI calculation function
def calculate_cci(high, low, close, open, period):
    typical_price = (high + low + close + open) / 4
    sma_typical_price = typical_price.rolling(window=period).mean()
    mean_deviation = (typical_price - sma_typical_price).abs().rolling(window=period).mean()
    cci = (typical_price - sma_typical_price) / (0.015 * mean_deviation)
    return cci

def fetch_stock_data(stock_name, selected_date=None, end_date=None, Interval="1min"):
    stock_name = "AAPL"
    #response = requests.get("https://api.twelvedata.com/time_series?apikey=d2128066f27e41aaa450634291878f8f&interval=1min&symbol=AAPL&format=JSON&timezone=Europe/Dublin")
    if (selected_date is None) and (end_date is None):
        response = requests.get(f"https://api.twelvedata.com/time_series?apikey=d2128066f27e41aaa450634291878f8f&interval={Interval}&symbol={stock_name}&format=JSON&timezone=Europe/Dublin")

        if response.status_code == 200:
            data = response.json().get("values")

            return data
        else:
            return {}
    else:
        response = requests.get(f"https://api.twelvedata.com/time_series?apikey=d2128066f27e41aaa450634291878f8f&interval={Interval}&symbol=AAPL&timezone=Europe/Dublin&start_date={selected_date} 00:00:00&end_date={end_date} 23:59:00")
        if response.status_code == 200:
            data = response.json().get("values")

            return data
        else:
            return {}    


def get_data_from_db():
    # Query all objects from the AppleStockData model
    stock_data_objects = AppleStockData.objects.all()
    
    # Convert the query result into a list of dictionaries
    stock_data_list = [
        {
            'datetime': obj.datetime,
            'open': obj.open,
            'high': obj.high,
            'low': obj.low,
            'close': obj.close,
            'volume': obj.volume,
            'New_RSI_7': obj.New_RSI_7,
            'New_RSI_14': obj.New_RSI_14,
            'New_MACD': obj.New_MACD,
            'New_SMA_50': obj.New_SMA_50,
            'New_SMA_100': obj.New_SMA_100,
            'New_EMA_50': obj.New_EMA_50,
            'New_EMA_100': obj.New_EMA_100,
            'New_Upper_Band': obj.New_Upper_Band,
            'New_Lower_Band': obj.New_Lower_Band,
            'New_TrueRange': obj.New_TrueRange,
            'New_ATR_7': obj.New_ATR_7,
            'New_ATR_14': obj.New_ATR_14,
            'New_CCI_7': obj.New_CCI_7,
            'New_CCI_14': obj.New_CCI_14,
        }
        for obj in stock_data_objects
    ]
    
    # Create a pandas DataFrame from the list of dictionaries
    df = pd.DataFrame(stock_data_list)
    
    return df

def calculate_technical_indicators_and_return_df(data):
    # Convert JSON data to DataFrame
    df = pd.DataFrame(data)
    
    # Convert numerical columns to numeric data type
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    # Convert 'datetime' column to datetime data type
    df['datetime'] = pd.to_datetime(df['datetime'])
    

    # Filter DataFrame to get only new data after the latest datetime
    # if latest_datetime:
    #     new_data = df[df['datetime'] > latest_datetime]
    # else:
    #     new_data = df  # If no existing data, store all data
    new_data = df.sort_values(by='datetime')
    # Apply calculations to the new data
    new_data['New_RSI_7'] = calculate_rsi(new_data['close'], 7)
    new_data['New_RSI_14'] = calculate_rsi(new_data['close'], 14)
    new_data['New_MACD'] = calculate_macd(new_data['close'])
    new_data['New_SMA_50'] = new_data['close'].rolling(window=50).mean()
    new_data['New_SMA_100'] = new_data['close'].rolling(window=100).mean()
    new_data['New_EMA_50'] = new_data['close'].ewm(span=50, min_periods=0, adjust=False).mean()
    new_data['New_EMA_100'] = new_data['close'].ewm(span=100, min_periods=0, adjust=False).mean()
    new_data['New_Upper_Band'], new_data['New_Lower_Band'] = calculate_bollinger_bands(new_data['close'])
    new_data['New_TrueRange'] = np.maximum.reduce([new_data['high'] - new_data['low'], abs(new_data['high'] - new_data['close'].shift()), abs(new_data['low'] - new_data['close'].shift())])
    new_data['New_ATR_7'] = new_data['New_TrueRange'].rolling(window=7).mean()
    new_data['New_ATR_14'] = new_data['New_TrueRange'].rolling(window=14).mean()
    new_data['New_CCI_7'] = calculate_cci(new_data['high'], new_data['low'], new_data['close'], new_data['open'], 7)
    new_data['New_CCI_14'] = calculate_cci(new_data['high'], new_data['low'], new_data['close'], new_data['open'], 14)
    
    new_data.fillna(0, inplace=True) # Fill NaN values with 0

    return new_data

def store_stock_data(stock_data):
    # Assume you receive JSON data in the request
    #json_data = request.body.decode('utf-8')
    data = stock_data

    # Fetch the latest datetime from the database
    latest_datetime = AppleStockData.objects.aggregate(Max('datetime'))['datetime__max']
    
    data_from_db=get_data_from_db()

    # Calculate technical indicators for the new data
    new_data = calculate_technical_indicators_and_return_df(data)

    # Convert DataFrame back to dictionary
    new_data_dict = new_data.to_dict(orient='records')
    
    

    # Create AppleStockData objects and save to the database
    for item in new_data_dict:
        stock_data = AppleStockData(**item)
        stock_data.save()

    return JsonResponse({'message': 'Stock data stored successfully'})

def fetch_and_store_stock_data(request):
    stock_data = fetch_stock_data("AAPL")

    if stock_data:
        return store_stock_data(stock_data)
    else:
        return JsonResponse({'message': 'Failed to fetch stock data'})
    
def only_store_stock_data_database(data):
    
        # Convert JSON data to DataFrame
    df = pd.DataFrame(data)
    
    # Convert numerical columns to numeric data type
    numeric_cols = ['open', 'high', 'low', 'close', 'volume']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)

    # Convert 'datetime' column to datetime data type
    df['datetime'] = pd.to_datetime(df['datetime'])

    for index, row in df.iterrows():
        stock_data = AppleStockData(
            datetime=row['datetime'],
            open=row['open'],
            high=row['high'],
            low=row['low'],
            close=row['close'],
            volume=row['volume'],
        )
        stock_data.save()        

def custom_data_store(request):
    stock_data = fetch_stock_data("AAPL", "2023-11-01 01:41:00", "2024-04-10 09:42:00")
    only_store_stock_data_database(stock_data)
    return JsonResponse({'message': 'Stock data stored successfully'})

def fetch_latest_stock_data():
    # Retrieve the latest datetime value from the database
    latest_datetime = AppleStockData.objects.aggregate(Max('datetime'))['datetime__max']
    
    if latest_datetime:
        # Fetch the first row with the latest datetime value
        latest_stock_data = AppleStockData.objects.filter(datetime=latest_datetime).first()
        data_dict = {
        'datetime': latest_stock_data.datetime,
        'open': latest_stock_data.open,
        'high': latest_stock_data.high,
        'low': latest_stock_data.low,
        'close': latest_stock_data.close,
        'volume': latest_stock_data.volume,
        'New_RSI_7': latest_stock_data.New_RSI_7,
        'New_RSI_14': latest_stock_data.New_RSI_14,
        'New_MACD': latest_stock_data.New_MACD,
        'New_SMA_50': latest_stock_data.New_SMA_50,
        'New_SMA_100': latest_stock_data.New_SMA_100,
        'New_EMA_50': latest_stock_data.New_EMA_50,
        'New_EMA_100': latest_stock_data.New_EMA_100,
        'New_Upper_Band': latest_stock_data.New_Upper_Band,
        'New_Lower_Band': latest_stock_data.New_Lower_Band,
        'New_TrueRange': latest_stock_data.New_TrueRange,
        'New_ATR_7': latest_stock_data.New_ATR_7,
        'New_ATR_14': latest_stock_data.New_ATR_14,
        'New_CCI_7': latest_stock_data.New_CCI_7,
        'New_CCI_14': latest_stock_data.New_CCI_14,
        }
        return data_dict
    else:
        # Handle case when there is no data in the database
        return {}



def latest_stock_data_view(request):
    latest_stock_data=fetch_latest_stock_data()
    
    # Display the latest stock data
    #pd.DataFrame(data_dict)
    #return render(request, 'latest_stock_data.html', {'latest_stock_data': data_dict})
    return JsonResponse({'latest_stock_data': latest_stock_data})
# Display the latest stock data

def get_current_stock_price(request):
    # API endpoint for Alpha Vantage
    latest_stock_data=fetch_latest_stock_data()
    #print(latest_stock_data)
    if request.method == 'GET' and request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
        selected_date = request.GET.get('date')
        stock_name = "AAPL"

        symbol = 'AAPL'  # Example stock symbol

        #if response.status_code == 200:
        if True:
            # Extract real-time stock price from the response
            #data = response.json()
            #stock_price = data['Global Quote']['05. price']  # Example key for real-time price
            stock_price = random.randint(100, 200)
            return JsonResponse(latest_stock_data)
        else:
            JsonResponse({'error': 'Invalid request','price':latest_stock_data}, status=400) 
    return JsonResponse({'error': 'Invalid request'}, status=400)   

def apple_stock_home(request):
    stock_name = "AAPL"
    latest_stock_data=fetch_latest_stock_data()

    test_data=pd.DataFrame([latest_stock_data])
    test_data=test_data.drop(columns=['datetime'])
    # Perform prediction using the fetched data
    prediction_result = apple_model.predict(test_data)

    end_time = datetime.now().strftime("%Y-%m-%d")
    end_time_ = datetime.now()
    start_time_ = end_time_ - timedelta(days=1)
    start_time = start_time_.strftime("%Y-%m-%d")
    get_data_date=fetch_stock_data(stock_name, start_time, end_time)

    datetime_list = [entry['datetime'][-8:] for entry in get_data_date]  # Extract only the time part
    close_list = [float(entry['close']) for entry in get_data_date]

    return render(request, 'apple_stock_home.html', {"market_prediction":prediction_result[0],"labels":datetime_list,"data":close_list})

def predict_view(request):
    if request.method == 'GET' and request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
        selected_date = request.GET.get('date')
        stock_name = "AAPL"
        get_data_date=fetch_stock_data(stock_name, selected_date, selected_date)
        
        formatted_data=calculate_technical_indicators_and_return_df(get_data_date)
        formatted_data=formatted_data.drop(columns=['datetime'])
        test_data=formatted_data.iloc[[-1]]

        # Perform prediction using the fetched data
        result = apple_model.predict(test_data)[0]
        prediction_result={"stock_name":stock_name,"date":selected_date,"prediction":f"{result}"}

        # Return the prediction result as JSON response
        return JsonResponse({'result': prediction_result})

    # Handle invalid requests
    return JsonResponse({'error': 'Invalid request'}, status=400) 

def predict_manual_view(request):
    if request.method == 'POST' and request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
        data = json.loads(request.body)
        stock_name = data.get('stock_name')
        open_price = data.get('open')
        high_price = data.get('high')
        low_price = data.get('low')
        close_price = data.get('close')
        volume = data.get('volume')
        print(open_price,high_price,low_price,close_price,volume)
        selected_date = "2023-11-01"
        print(selected_date)
        # get_data_date=fetch_stock_data(stock_name, selected_date, selected_date)
        
        # formatted_data=calculate_technical_indicators_and_return_df(get_data_date)
        # formatted_data=formatted_data.drop(columns=['datetime'])
        # test_data=formatted_data.iloc[[-1]]
        # print(test_data)
        # Fetch data for the specified stock name and date
        # Replace this with your actual data fetching logic

        # Perform prediction using the fetched data
        #result = apple_model.predict(test_data)[0]
        prediction_result={"stock_name":stock_name,"date":selected_date,"prediction":f"Manula prediction"}

        # Return the prediction result as JSON response
        return JsonResponse({'result': prediction_result})

    # Handle invalid requests
    return JsonResponse({'error': 'Invalid request'}, status=400)

def fetch_chart_data(request):
    if request.method == 'GET' and request.META.get('HTTP_X_REQUESTED_WITH') == 'XMLHttpRequest':
        selected_date = request.GET.get('dateRange')
        stock_name = "AAPL"
        if selected_date == "1Day":
            end_time = datetime.now().strftime("%Y-%m-%d")
            
            end_time_ = datetime.now()
            start_time_ = end_time_ - timedelta(days=1)
            start_time = start_time_.strftime("%Y-%m-%d")
            
            get_data_date=fetch_stock_data(stock_name, start_time, end_time)
        
            datetime_list = [entry['datetime'][-8:] for entry in get_data_date]  # Extract only the time part
            close_list = [float(entry['close']) for entry in get_data_date]
        elif selected_date == "1Week":
            end_time = datetime.now().strftime("%Y-%m-%d")
            end_time_ = datetime.now()
            start_time_ = end_time_ - timedelta(days=8)
            start_time = start_time_.strftime("%Y-%m-%d")
            get_data_date=fetch_stock_data(stock_name,start_time,end_time,Interval="1day")
            datetime_list = [entry['datetime'] for entry in get_data_date]
            close_list = [float(entry['close']) for entry in get_data_date]

        elif selected_date == "1Month":
            end_time = datetime.now().strftime("%Y-%m-%d")
            end_time_ = datetime.now()
            start_time_ = end_time_ - timedelta(days=30)
            start_time = start_time_.strftime("%Y-%m-%d")
            get_data_date=fetch_stock_data(stock_name, start_time, end_time,Interval="1day")
            datetime_list = [entry['datetime'] for entry in get_data_date]
        
            close_list = [float(entry['close']) for entry in get_data_date]

        # Return the prediction result as JSON response
        return JsonResponse({'labels': datetime_list,'data': close_list})

    # Handle invalid requests
    return JsonResponse({'error': 'Invalid request'}, status=400)