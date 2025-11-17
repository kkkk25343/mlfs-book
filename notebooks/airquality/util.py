import os
import datetime
import time
import requests
import pandas as pd
import json
from geopy.geocoders import Nominatim
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import openmeteo_requests
import requests_cache
from retry_requests import retry
import hopsworks
import hsfs
from pathlib import Path

def get_historical_weather(city, start_date,  end_date, latitude, longitude):
    # latitude, longitude = get_city_coordinates(city)

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = -1)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ["temperature_2m_mean", "precipitation_sum", "wind_speed_10m_max", "wind_direction_10m_dominant"]
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process daily data. The order of variables needs to be the same as requested.
    daily = response.Daily()
    daily_temperature_2m_mean = daily.Variables(0).ValuesAsNumpy()
    daily_precipitation_sum = daily.Variables(1).ValuesAsNumpy()
    daily_wind_speed_10m_max = daily.Variables(2).ValuesAsNumpy()
    daily_wind_direction_10m_dominant = daily.Variables(3).ValuesAsNumpy()

    daily_data = {"date": pd.date_range(
        start = pd.to_datetime(daily.Time(), unit = "s"),
        end = pd.to_datetime(daily.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = daily.Interval()),
        inclusive = "left"
    )}
    daily_data["temperature_2m_mean"] = daily_temperature_2m_mean
    daily_data["precipitation_sum"] = daily_precipitation_sum
    daily_data["wind_speed_10m_max"] = daily_wind_speed_10m_max
    daily_data["wind_direction_10m_dominant"] = daily_wind_direction_10m_dominant

    daily_dataframe = pd.DataFrame(data = daily_data)
    daily_dataframe = daily_dataframe.dropna()
    daily_dataframe['city'] = city
    return daily_dataframe

def get_hourly_weather_forecast(city, latitude, longitude):

    # latitude, longitude = get_city_coordinates(city)

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after = 3600)
    retry_session = retry(cache_session, retries = 5, backoff_factor = 0.2)
    openmeteo = openmeteo_requests.Client(session = retry_session)

    # Make sure all required weather variables are listed here
    # The order of variables in hourly or daily is important to assign them correctly below
    url = "https://api.open-meteo.com/v1/ecmwf"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": ["temperature_2m", "precipitation", "wind_speed_10m", "wind_direction_10m"]
    }
    responses = openmeteo.weather_api(url, params=params)

    # Process first location. Add a for-loop for multiple locations or weather models
    response = responses[0]
    print(f"Coordinates {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation {response.Elevation()} m asl")
    print(f"Timezone {response.Timezone()} {response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0 {response.UtcOffsetSeconds()} s")

    # Process hourly data. The order of variables needs to be the same as requested.

    hourly = response.Hourly()
    hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
    hourly_precipitation = hourly.Variables(1).ValuesAsNumpy()
    hourly_wind_speed_10m = hourly.Variables(2).ValuesAsNumpy()
    hourly_wind_direction_10m = hourly.Variables(3).ValuesAsNumpy()

    hourly_data = {"date": pd.date_range(
        start = pd.to_datetime(hourly.Time(), unit = "s"),
        end = pd.to_datetime(hourly.TimeEnd(), unit = "s"),
        freq = pd.Timedelta(seconds = hourly.Interval()),
        inclusive = "left"
    )}
    hourly_data["temperature_2m_mean"] = hourly_temperature_2m
    hourly_data["precipitation_sum"] = hourly_precipitation
    hourly_data["wind_speed_10m_max"] = hourly_wind_speed_10m
    hourly_data["wind_direction_10m_dominant"] = hourly_wind_direction_10m

    hourly_dataframe = pd.DataFrame(data = hourly_data)
    hourly_dataframe = hourly_dataframe.dropna()
    return hourly_dataframe



def get_city_coordinates(city_name: str):
    """
    Takes city name and returns its latitude and longitude (rounded to 2 digits after dot).
    """
    # Initialize Nominatim API (for getting lat and long of the city)
    geolocator = Nominatim(user_agent="MyApp")
    city = geolocator.geocode(city_name)

    latitude = round(city.latitude, 2)
    longitude = round(city.longitude, 2)

    return latitude, longitude

def trigger_request(url:str):
    response = requests.get(url)
    if response.status_code == 200:
        # Extract the JSON content from the response
        data = response.json()
    else:
        print("Failed to retrieve data. Status Code:", response.status_code)
        raise requests.exceptions.RequestException(response.status_code)

    return data


def get_pm25(aqicn_url: str, country: str, city: str, street: str, day: datetime.date, AQI_API_KEY: str):
    """
    Returns DataFrame with air quality (pm25) as dataframe
    """
    # The API endpoint URL
    url = f"{aqicn_url}/?token={AQI_API_KEY}"

    # Make a GET request to fetch the data from the API
    data = trigger_request(url)

    # if we get 'Unknown station' response then retry with city in url
    if data['data'] == "Unknown station":
        url1 = f"https://api.waqi.info/feed/{country}/{street}/?token={AQI_API_KEY}"
        data = trigger_request(url1)

    if data['data'] == "Unknown station":
        url2 = f"https://api.waqi.info/feed/{country}/{city}/{street}/?token={AQI_API_KEY}"
        data = trigger_request(url2)


    # Check if the API response contains the data
    if data['status'] == 'ok':
        # Extract the air quality data
        aqi_data = data['data']
        aq_today_df = pd.DataFrame()
        aq_today_df['pm25'] = [aqi_data['iaqi'].get('pm25', {}).get('v', None)]
        aq_today_df['pm25'] = aq_today_df['pm25'].astype('float32')

        aq_today_df['country'] = country
        aq_today_df['city'] = city
        aq_today_df['street'] = street
        aq_today_df['date'] = day
        aq_today_df['date'] = pd.to_datetime(aq_today_df['date'])
        aq_today_df['url'] = aqicn_url
    else:
        print("Error: There may be an incorrect  URL for your Sensor or it is not contactable right now. The API response does not contain data.  Error message:", data['data'])
        raise requests.exceptions.RequestException(data['data'])

    return aq_today_df


def plot_air_quality_forecast(city: str, street: str, df: pd.DataFrame, file_path: str, hindcast=False):
    fig, ax = plt.subplots(figsize=(10, 6))

    day = pd.to_datetime(df['date']).dt.date
    # Plot each column separately in matplotlib
    ax.plot(day, df['predicted_pm25'], label='Predicted PM2.5', color='red', linewidth=2, marker='o', markersize=5, markerfacecolor='blue')

    # Set the y-axis to a logarithmic scale
    ax.set_yscale('log')
    ax.set_yticks([0, 10, 25, 50, 100, 250, 500])
    ax.get_yaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_ylim(bottom=1)

    # Set the labels and title
    ax.set_xlabel('Date')
    ax.set_title(f"PM2.5 Predicted (Logarithmic Scale) for {city}, {street}")
    ax.set_ylabel('PM2.5')

    colors = ['green', 'yellow', 'orange', 'red', 'purple', 'darkred']
    labels = ['Good', 'Moderate', 'Unhealthy for Some', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
    ranges = [(0, 49), (50, 99), (100, 149), (150, 199), (200, 299), (300, 500)]
    for color, (start, end) in zip(colors, ranges):
        ax.axhspan(start, end, color=color, alpha=0.3)

    # Add a legend for the different Air Quality Categories
    patches = [Patch(color=colors[i], label=f"{labels[i]}: {ranges[i][0]}-{ranges[i][1]}") for i in range(len(colors))]
    legend1 = ax.legend(handles=patches, loc='upper right', title="Air Quality Categories", fontsize='x-small')

    # Aim for ~10 annotated values on x-axis, will work for both forecasts ans hindcasts
    if len(df.index) > 11:
        every_x_tick = len(df.index) / 10
        ax.xaxis.set_major_locator(MultipleLocator(every_x_tick))

    plt.xticks(rotation=45)

    if hindcast == True:
        ax.plot(day, df['pm25'], label='Actual PM2.5', color='black', linewidth=2, marker='^', markersize=5, markerfacecolor='grey')
        legend2 = ax.legend(loc='upper left', fontsize='x-small')
        ax.add_artist(legend1)

    # Ensure everything is laid out neatly
    plt.tight_layout()

    # # Save the figure, overwriting any existing file with the same name
    plt.savefig(file_path)
    return plt


def delete_feature_groups(fs, name):
    try:
        for fg in fs.get_feature_groups(name):
            fg.delete()
            print(f"Deleted {fg.name}/{fg.version}")
    except hsfs.client.exceptions.RestAPIError:
        print(f"No {name} feature group found")

def delete_feature_views(fs, name):
    try:
        for fv in fs.get_feature_views(name):
            fv.delete()
            print(f"Deleted {fv.name}/{fv.version}")
    except hsfs.client.exceptions.RestAPIError:
        print(f"No {name} feature view found")

def delete_models(mr, name):
    models = mr.get_models(name)
    if not models:
        print(f"No {name} model found")
    for model in models:
        model.delete()
        print(f"Deleted model {model.name}/{model.version}")

def delete_secrets(proj, name):
    secrets = secrets_api(proj.name)
    try:
        secret = secrets.get_secret(name)
        secret.delete()
        print(f"Deleted secret {name}")
    except hopsworks.client.exceptions.RestAPIError:
        print(f"No {name} secret found")

# WARNING - this will wipe out all your feature data and models
def purge_project(proj):
    fs = proj.get_feature_store()
    mr = proj.get_model_registry()

    # Delete Feature Views before deleting the feature groups
    delete_feature_views(fs, "air_quality_fv")

    # Delete ALL Feature Groups
    delete_feature_groups(fs, "air_quality")
    delete_feature_groups(fs, "weather")
    delete_feature_groups(fs, "aq_predictions")

    # Delete all Models
    delete_models(mr, "air_quality_xgboost_model")
    delete_secrets(proj, "SENSOR_LOCATION_JSON")

def check_file_path(file_path):
    my_file = Path(file_path)
    if my_file.is_file() == False:
        print(f"Error. File not found at the path: {file_path} ")
    else:
        print(f"File successfully found at the path: {file_path}")

def backfill_predictions_for_monitoring(weather_fg, air_quality_df, monitor_fg, model):
    """
    从历史 weather_fg + air_quality_df 里造一批“伪预测”，
    用于第一次跑监控时填充 hindcast 数据。

    注意：模型现在用的是 5 个特征：
      pm25_rolling_mean_3d + 4 个天气特征
    但监控的 Feature Group 只需要存预测结果和天气特征、位置信息，
    不一定要把 rolling 特征也存进去。
    """
    # 1. 先读出天气特征，按时间排序，取最近 10 天
    features_df = weather_fg.read()
    features_df = features_df.sort_values(by=['date'], ascending=True)
    features_df = features_df.tail(10)

    # 2. 从 air_quality_df 里取 rolling 特征，并按 date merge 上来
    #    air_quality_df 里应该已经有 pm25_rolling_mean_3d 这一列（在训练前你已经加过）
    rolling_df = air_quality_df[['date', 'pm25_rolling_mean_3d']].copy()
    features_df = features_df.merge(rolling_df, on='date', how='left')

    # 没有 rolling 特征的行先丢掉，避免模型抱怨
    features_df = features_df.dropna(subset=['pm25_rolling_mean_3d'])
    features_df['pm25_rolling_mean_3d'] = features_df['pm25_rolling_mean_3d'].astype('float32')

    # 3. 用和训练时完全一致的 5 个特征做预测
    feature_cols = [
        'pm25_rolling_mean_3d',
        'temperature_2m_mean',
        'precipitation_sum',
        'wind_speed_10m_max',
        'wind_direction_10m_dominant',
    ]
    features_df['predicted_pm25'] = model.predict(features_df[feature_cols])

    # 4. 把真实 pm25 和位置信息（street、country、city）拼回来
    #    weather_fg 里一般有 city，air_quality_df 里也有 city，所以 merge 后可能会出现 city_x / city_y
    outcome_cols = ['date', 'pm25', 'street', 'country']
    if 'city' in air_quality_df.columns:
        outcome_cols.append('city')

    df = features_df.merge(
        air_quality_df[outcome_cols],
        on='date',
        how='inner'
    )

    # ---- 关键修复点：整理 city 列 ----
    # 如果 merge 后出现了 city_x / city_y，把它们合并成一个 city
    if 'city_x' in df.columns or 'city_y' in df.columns:
        if 'city_x' in df.columns and 'city_y' in df.columns:
            # 一般两个是一样的，这里用 city_x
            df['city'] = df['city_x']
        elif 'city_x' in df.columns:
            df['city'] = df['city_x']
        elif 'city_y' in df.columns:
            df['city'] = df['city_y']
        # 把多余的 city_x / city_y 列删掉
        df = df.drop(columns=[c for c in ['city_x', 'city_y'] if c in df.columns])

    df['days_before_forecast_day'] = 1

    # hindcast_df：用来画对比图，需要真实值和预测值
    hindcast_df = df[['date', 'pm25', 'predicted_pm25']].copy()

    # 5. 写入监控 Feature Group：
    #    只保留和 notebook 里 batch_data.insert() 一致的列，
    #    注意这里 **不要** 把 pm25（真实值）和 city_x/city_y 写进去。
     # ---- 根据 monitor_fg 的 schema 动态确定要写的列 ----
    fg_cols = {feat.name.lower() for feat in monitor_fg.features}

    insert_cols = [
        'date',
        'temperature_2m_mean',
        'precipitation_sum',
        'wind_speed_10m_max',
        'wind_direction_10m_dominant',
        'predicted_pm25',
        'street',
        'city',
        'country',
        'days_before_forecast_day',
    ]
    # 如果 FG schema 里有 rolling，就一起写入
    if 'pm25_rolling_mean_3d' in fg_cols:
        insert_cols.insert(5, 'pm25_rolling_mean_3d')  # 放在天气特征之后

    # 只保留现有列（防 KeyError）
    insert_cols = [c for c in insert_cols if c in df.columns]
    insert_df = df[insert_cols].copy()

    if 'pm25_rolling_mean_3d' in insert_df.columns:
        insert_df['pm25_rolling_mean_3d'] = insert_df['pm25_rolling_mean_3d'].astype('float32')

    monitor_fg.insert(insert_df, write_options={"wait_for_job": True})

    return hindcast_df
