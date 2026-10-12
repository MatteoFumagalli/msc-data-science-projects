# Import useful libraries
import numpy as np
import os
import pandas as pd
import time
from tqdm import tqdm

from datetime import timedelta
from opencage.geocoder import OpenCageGeocode
from shapely.geometry import Point
from shapely.wkt import loads
from timezonefinder import TimezoneFinder

import openmeteo_requests
import pytz
import requests_cache
from retry_requests import retry


# Definition of functions
def check_for_nan_values(df):
    '''Return the list of columns with a NaN value for each row interested'''
    
    nan_rows = df[df.isna().any(axis=1)].apply(
        lambda row: row.index[row.isna()].tolist(), axis=1)
    
    for idx, cols in nan_rows.items():
        print(f'Row {idx} has a NaN value in the following columns: {cols}')

    return nan_rows


def handle_missing_save_stats(df, nan_rows):
    '''Manage NaN values for save columns'''

    save_cols = ['home_saves_made', 'away_saves_made', 
                 'home_saves_faced', 'away_saves_faced']
    
    for idx in nan_rows.index:
        if df.loc[idx, save_cols].isna().all():
            if (df.loc[idx, 'home_shots_on_target'] == 0 and 
                df.loc[idx, 'away_shots_on_target'] == 0):
                df.loc[idx, save_cols] = 0

    return df


def handle_missing_attendance(df, nan_rows):
    '''Fill NaN values for attendance of specific matches (diretta.it)'''
    
    fill_values = {
        1441: 42212,
        3447: 12774,
        3472: 31171,
        4495: 14358,
        4989: 34000
        }
    
    for idx, cols in nan_rows.items():
        if 'attendance' in cols:
            if idx in fill_values:
                df.loc[idx, 'attendance'] = fill_values[idx]

    return df


def restructure_schema(df):
    '''Prepare the result of the scraping for the database'''

    competitions_df = (
        df[['comp_name', 'season_year']].copy().drop_duplicates(
            subset=['comp_name', 'season_year']).reset_index(drop=True))
    competitions_df['id'] = competitions_df.index + 1
    competitions_df = competitions_df[['id', 'comp_name', 'season_year']]

    teams_home = df[['home_team']].rename(columns={'home_team': 'team_name'})
    teams_away = df[['away_team']].rename(columns={'away_team': 'team_name'})
    teams_df = (pd.concat([teams_home, teams_away])
                .drop_duplicates(subset=['team_name']).sort_values(
                    'team_name').reset_index(drop=True))
    teams_df['id'] = teams_df.index + 1
    teams_df = teams_df[['id', 'team_name']]

    stadiums_df = (
        df[['stadium_name', 'city']].copy().drop_duplicates(
            subset=['stadium_name', 'city']).sort_values(
                'stadium_name').reset_index(drop=True))
    stadiums_df['id'] = stadiums_df.index + 1
    stadiums_df = stadiums_df[['id', 'stadium_name', 'city']]

    matches_df = df.merge(
        competitions_df, on=['comp_name', 'season_year']).rename(
            columns={'id': 'competition_id'})
    matches_df = matches_df.merge(stadiums_df, on='stadium_name').rename(
        columns={'id': 'stadium_id'})
    matches_df = matches_df.merge(
        teams_df, left_on='home_team', right_on='team_name').rename(
            columns={'id': 'home_team_id'})
    matches_df = matches_df.merge(
        teams_df, left_on='away_team', right_on='team_name').rename(
            columns={'id': 'away_team_id'})
    matches_df['date'] = pd.to_datetime(
        matches_df['date'] + ' ' + matches_df['time'].fillna('00:00'))
    matches_df['id'] = matches_df.index + 1
    matches_df = matches_df[[
        'id', 'competition_id', 'stadium_id', 'home_team_id', 'away_team_id',
        'date', 'home_goals', 'away_goals', 'attendance']]
    
    stats_cols = [c for c in df.columns if c not in [
        'comp_name', 'season_year', 'home_team', 'away_team', 'date', 'time', 
        'stadium_name', 'city', 'home_goals', 'away_goals', 'attendance']]
    match_stats_df = df[stats_cols].copy()
    match_stats_df['match_id'] = matches_df['id'].values
    match_stats_df['id'] = match_stats_df.index + 1
    cols = ['id', 'match_id'] + [
        col for col in match_stats_df if col not in ['id', 'match_id']]
    match_stats_df = match_stats_df[cols]  

    return competitions_df, teams_df, stadiums_df, matches_df, match_stats_df


def preprocess_and_restructure(df):
    '''Restructure the DataFrame by aligning with the database schema'''

    df = df.sort_values(
        by=['season_year', 'comp_name', 'date']
    ).reset_index(drop=True)

    print('NaN values before preprocessing:')
    nan_before = check_for_nan_values(df)

    df = handle_missing_save_stats(df, nan_before)
    df = handle_missing_attendance(df, nan_before)

    print('\nNaN values after correction:')
    nan_after = check_for_nan_values(df)

    competitions_df, teams_df, stadiums_df, matches_df, match_stats_df = (
        restructure_schema(df)
    )

    return {
        'full_df': df,
        'nan_before': nan_before,
        'nan_after': nan_after,
        'competitions': competitions_df,
        'teams': teams_df,
        'stadiums': stadiums_df,
        'matches': matches_df,
        'match_stats': match_stats_df
    }


def enrich_matches_data(matches_df):
    '''Enrich matches data with meteorological season and time_slot columns'''

    def get_meteorological_season(date):
        
        month = date.month
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Autumn'
        
    matches_df['season'] = matches_df['date'].apply(get_meteorological_season)

    def get_time_of_day(date):
        '''Return the time of day based on the hour'''
        
        hour = date.hour
        if hour < 14:
            return 'Midday'
        elif hour < 17:
            return 'Afternoon'
        elif hour < 20:
            return 'Evening'
        else:
            return 'Night'
        
    matches_df['time_slot'] = matches_df['date'].apply(get_time_of_day)

    # Reorder columns
    matches_df = matches_df[[
    'id', 'competition_id', 'stadium_id', 'home_team_id', 'away_team_id',
    'date', 'season', 'time_slot', 'home_goals', 'away_goals', 'attendance']]
    
    return matches_df


def get_coordinates(stadiums_df):
    '''Use OpenCage API to get country and coordinates of stadiums'''

    key = os.getenv('OPENCAGE_API_KEY')
    if not key:
        raise ValueError("Set OPENCAGE_API_KEY in your .env file")
    
    geocoder = OpenCageGeocode(key)
    geometries = []
    countries = []

    for _, row in tqdm(stadiums_df.iterrows(), total=len(stadiums_df), 
                       desc='Geocoding stadiums'):
        stadium = row['stadium_name']
        city = row['city']
        query = f'{stadium}, {city}' if pd.notna(city) else stadium

        try:
            results = geocoder.geocode(
                query, language='en', no_annotations=1, limit=1)

            if results:
                location = results[0]
                formatted = location.get('formatted', '')
                countries.append(formatted.split(',')[-1].strip() 
                                 if formatted else None)
                lat = float(location['geometry']['lat'])
                lon = float(location['geometry']['lng'])
                geometries.append(Point(lon, lat))
            else:
                print(f'Geocoding failed for: {query}')
                countries.append(None)
                geometries.append(None)
        except Exception as e:
            print(f'Error geocoding {query}: {e}')
            countries.append(None)
            geometries.append(None)

        # Insert a pause between requests to avoid rate limit
        time.sleep(1.1)

    stadiums_df['country'] = countries
    stadiums_df['geometry'] = geometries
    
    return stadiums_df


def add_timezone(stadiums_df):
    '''Get the timezone using each stadium coordinates'''

    tf = TimezoneFinder(in_memory=True)

    def get_timezone(point):
        if point is None:
            return None
        
        if isinstance(point, str):
            try:
                point = loads(point)
            except:
                return None
        
        if point is None:
            return None
        else:
            return tf.timezone_at(lng=point.x, lat=point.y)
        
    stadiums_df['timezone'] = stadiums_df['geometry'].apply(get_timezone)
    return stadiums_df


def get_meteo_conditions(matches_df, stadiums_df):
    '''Use Open-Meteo historical API to retrieve match weather conditions'''

    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=-1)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)
    url = 'https://archive-api.open-meteo.com/v1/archive'

    weather_vars = ['temperature_2m', 'precipitation', 'wind_speed_10m']
    weather_rows = []
    
    matches_with_stadium = matches_df.merge(
        stadiums_df[['id', 'geometry', 'timezone']], 
        left_on='stadium_id', right_on='id'
    ).rename(columns={'id_x': 'match_id', 'id_y': 'stadium_api_id'}) 
    matches_with_stadium.drop(columns=['stadium_api_id'], inplace=True)
    
    # Define the request
    for _, match in tqdm(matches_with_stadium.iterrows(), 
                         total=len(matches_with_stadium),
                         desc='Fetching weather data'):
        match_id = match['match_id']
        point = match['geometry']
        timezone = match['timezone']
        match_time = match['date']

        if point is None or timezone is None:
            print(f"Skipping match {match_id}")
            continue
        
        # Define the parameters for the request
        lat, lng = point.y, point.x
        yesterday = match_time - timedelta(days=1)
        tomorrow = match_time + timedelta(days=1)
        start_date_api = yesterday.strftime('%Y-%m-%d')
        end_date_api = tomorrow.strftime('%Y-%m-%d')
        
        params = {
            'latitude': lat,
            'longitude': lng,
            'start_date': start_date_api,
            'end_date': end_date_api,
            'hourly': weather_vars,
            'timezone': timezone
        }

        try:
            response = openmeteo.weather_api(url, params=params)[0]

            # Process hourly data
            hourly = response.Hourly()
            times_start_raw = hourly.Time()
            temp_values = hourly.Variables(0).ValuesAsNumpy()

            # Define the hour list
            if isinstance(times_start_raw, np.ndarray) and (
                times_start_raw.size == temp_values.size):
                times_utc = pd.to_datetime(times_start_raw, unit='s', utc=True)
            else:
                if isinstance(times_start_raw, np.ndarray) and (
                    times_start_raw.size == 1):
                    start_time = pd.to_datetime(
                        times_start_raw[0], unit='s', utc=True)
                else:
                    start_time = pd.to_datetime(start_date_api, utc=True)
            
                interval_seconds = hourly.Interval() if hasattr(
                    hourly, 'Interval') else 3600
                times_utc = pd.date_range(
                    start=start_time, periods=temp_values.size, 
                    freq=pd.Timedelta(seconds=interval_seconds),
                    inclusive='left', tz='UTC')

            # Process hourly data
            hourly_data = {
                'time': times_utc,
                'temperature_2m': temp_values,
                'precipitation': hourly.Variables(1).ValuesAsNumpy(),
                'wind_speed_10m': hourly.Variables(2).ValuesAsNumpy()}
            hourly_df = pd.DataFrame(hourly_data)

            # Define the window of ± 2 hours
            tz = pytz.timezone(timezone)
            match_time_localized = tz.localize(match_time) 
            
            start_window_utc = (
                match_time_localized - timedelta(hours=2)).astimezone(pytz.UTC)
            end_window_utc = (
                match_time_localized + timedelta(hours=2)).astimezone(pytz.UTC)
            window = (hourly_df['time'] >= start_window_utc) & (
                hourly_df['time'] <= end_window_utc)
                        
            if not window.any():
                print('No weather data found in window for match '
                      f"{match_id} ({match_time})")
                continue
            
            # Calculate the statistics within the window
            summary = {'match_id': match_id}
            for var in weather_vars:
                summary[var] = hourly_df[var][window].mean()

            weather_rows.append(summary)

        except Exception as e:
            print(f"Error in fetching weather for match {match_id}: {e}")
            summary_none = {
                'match_id': match_id,
                'temperature_2m': None,
                'precipitation': None,
                'wind_speed_10m': None
            }
            weather_rows.append(summary_none)

        # Introduce a pause to respect the limit of 5000 requests per hour
        time.sleep(0.75)

    weather_conditions_df = pd.DataFrame(weather_rows)
    if not weather_conditions_df.empty:
        weather_conditions_df['id'] = weather_conditions_df.index + 1
        cols = ['id', 'match_id'] + [col 
                                     for col in weather_conditions_df.columns 
                                     if col not in ['id', 'match_id']]
        weather_conditions_df = weather_conditions_df[cols]
        weather_conditions_df.rename(columns={'temperature_2m': 'temperature', 
                                          'wind_speed_10m': 'wind_speed'},
                                          inplace=True)
    else:
        weather_conditions_df = pd.DataFrame(columns=[
            'id', 'match_id', 'temperature', 'precipitation', 'wind_speed'])

    return weather_conditions_df


def categorize_weather_conditions(weather_conditions_df):
    '''Transform raw data into categories'''

    weather_conditions_df['temperature_category'] = pd.cut(
        weather_conditions_df['temperature'],
        bins=[-float('inf'), 12, 25, float('inf')],
        labels=['Cold', 'Mild', 'Hot']
    )

    weather_conditions_df['is_precipitation'] = (
        weather_conditions_df['precipitation'] > 0.0)

    # Simpler Beaufort classification
    weather_conditions_df['wind_speed_category'] = pd.cut(
        weather_conditions_df['wind_speed'],
        bins=[-0.1, 7, 28, float('inf')],
        labels=['Low', 'Medium', 'High']
    )

    return weather_conditions_df