# Import useful libraries
import numpy as np
import os
import pandas as pd

from dotenv import load_dotenv
from geoalchemy2.shape import from_shape
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from sqlalchemy import create_engine, inspect, text
from sqlalchemy import Integer, String, Float, DateTime, Boolean
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm.attributes import InstrumentedAttribute
from shapely.wkt import loads

from src.models import Base, Competition, Team, Stadium
from src.models import Match, MatchStatistic, Weather


# Definition of functions
def load_db_env():
    '''Read and return the credentials for the database from file .env'''

    if not load_dotenv():
        raise FileNotFoundError(
            'File .env not found. Copy .env.example in .env and fill the values'
        )

    PGDATABASE = os.getenv('PGDATABASE')
    PGUSER = os.getenv('PGUSER')
    PGPASSWORD = os.getenv('PGPASSWORD')
    PGHOST = os.getenv('PGHOST')
    PGPORT = os.getenv('PGPORT')

    if not all([PGDATABASE, PGUSER, PGPASSWORD, PGHOST, PGPORT]):
        raise ValueError('File .env does not contain a value for all variables')
    
    return PGDATABASE, PGUSER, PGPASSWORD, PGHOST, PGPORT


def setup_database(db_name, user, password, host='localhost', port=5432):
    '''Create the database db_name if not exists and enable PostGIS'''

    conn = psycopg2.connect(
        dbname='postgres',
        user=user,
        password=password,
        host=host,
        port=port
    )

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()
    cur.execute(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'")
    exists = cur.fetchone()
    if not exists:
        cur.execute(f'CREATE DATABASE {db_name}')
        print(f'Database {db_name} created')
    else:
        print(f'Database {db_name} already existent')

    cur.close()
    conn.close()

    engine_setup = create_engine(
        f'postgresql://{user}:{password}@{host}:{port}/{db_name}',
        isolation_level='AUTOCOMMIT'
    )

    with engine_setup.connect() as conn:
        conn.execute(text('CREATE EXTENSION IF NOT EXISTS postgis;'))
        print('PostGIS extension enabled')


def create_engine_inspector_session(
        db_name, user, password, host='localhost', port=5432):
    '''Create the engine, inspector, and Session'''

    try:
        engine = create_engine(
            f'postgresql://{user}:{password}@{host}:{port}/{db_name}')
    except Exception as e:
        print('Unable to access PostgreSQL database: ', e)

    inspector = inspect(engine)
    Session = sessionmaker(bind=engine)
    return engine, inspector, Session


def create_tables(engine):
    '''Create the tables for the database'''

    Base.metadata.create_all(engine)


def initialize_database():
    '''Retrieve the credentials and initialize the database'''

    db_name, user, password, host, port = load_db_env()
    setup_database(db_name, user, password, host, port)
    engine, inspector, Session = create_engine_inspector_session(
        db_name, user, password, host, port)
    create_tables(engine)
    return engine, inspector, Session


def check_structure(df, Model):
    '''Check for the compatibility between database schema and DataFrame'''

    report = {}
    model_columns = {
        name: col 
        for name, col in Model.__dict__.items()
        if isinstance(
            col, InstrumentedAttribute) and hasattr(col.property, 'columns')
    }

    for col_name, col_attr in model_columns.items():
        col_type = type(col_attr.property.columns[0].type)
        nullable = col_attr.property.columns[0].nullable

        if col_name not in df.columns:
            report[col_name] = {'status': 'missing in df'}
            continue

        values = df[col_name]
        if not nullable and values.isna().any():
            report[col_name] = {'status': 'null values in non nullable column'}
            continue

        if col_type == Integer:
            if not pd.api.types.is_numeric_dtype(values):
                report[col_name] = {
                    'status': 'not numeric values in integer column'}
                continue
                
            if (values.dropna() % 1 != 0).any():
                report[col_name] = {'status': 'float values in integer column'}
                continue
            
        if col_type == Float:
            if not pd.api.types.is_numeric_dtype(values):
                report[col_name] = {
                    'status': 'not numeric values in float column'}

        if col_type == String:
            if not pd.api.types.is_string_dtype(values):
                report[col_name] = {
                    'status': 'not string values'}
                
        if col_type == DateTime:
            if not pd.api.types.is_datetime64_any_dtype(values):
                report[col_name] = {
                    'status': 'not datetime values'}
                continue

        if col_type == Boolean:
            if not pd.api.types.is_bool_dtype(values):
                report[col_name] = {
                    'status': 'not boolean values'}
                continue

    additional_columns = [col for col in df.columns if col not in model_columns]
    if additional_columns:
        report['additional_columns'] = {
            'status': 'more columns in the DataFrame'}
        
    return report


def check_db_compatibility(dfs):
    '''Evaluate the compatibility of each DataFrame with the database schema'''

    reports = {
        'Competitions': check_structure(dfs['competitions'], Competition),
        'Teams': check_structure(dfs['teams'], Team),
        'Stadiums': check_structure(dfs['stadiums'], Stadium),
        'Matches': check_structure(dfs['matches'], Match),
        'Match Stats': check_structure(dfs['match_stats'], MatchStatistic),
        'Weather Conditions': check_structure(dfs['weather'], Weather)
    }
    return reports


def check_types(matches_df):
    '''Correct the columns type before the population of the database'''

    matches_df = matches_df.replace({pd.NA: None, np.nan: None})
    for col in ['home_goals', 'away_goals', 'attendance']:
        if matches_df[col].isnull().any():
            matches_df[col] = matches_df[col].astype('object')
        else:
            matches_df[col] = matches_df[col].astype('int64')

    return matches_df


def preprocess_for_insert(matches_df, match_stats_df):
    '''Correct the type of specific columns'''

    matches_df = check_types(matches_df)
    match_stats_df = (
        match_stats_df.astype(object)
        .where(pd.notna(match_stats_df), None)
        .to_dict(orient='records')
    )
    return matches_df, match_stats_df


def populate_db(Session, Competition, Team, Stadium, Match, MatchStatistic, 
                Weather, competitions_df, teams_df, stadiums_df, matches_df, 
                match_stats_df, weather_conditions_df):
    '''Insert table data into the database'''

    with Session() as session:
        for _, row in competitions_df.iterrows():
            session.add(Competition(
                id=row['id'], 
                comp_name=row['comp_name'],
                season_year=row['season_year']
            ))

        for _, row in teams_df.iterrows():
            session.add(Team(
                id=row['id'],
                team_name=row['team_name']
            ))

        for _, row in stadiums_df.iterrows():
            session.add(Stadium(
                id=row['id'],
                stadium_name=row['stadium_name'],
                city=row['city'],
                country=row['country'],
                geometry=from_shape(
                    loads(row['geometry']), srid=4326) 
                    if row['geometry'] else None,
                timezone=row['timezone']
            ))

        session.commit()
        print('\nCompetitions inserted')
        print('Teams inserted')
        print('Stadiums inserted')

        for _, row in matches_df.iterrows():
            session.add(Match(
                id=row['id'],
                competition_id=row['competition_id'],
                stadium_id=row['stadium_id'],
                home_team_id=row['home_team_id'],
                away_team_id=row['away_team_id'],
                date=row['date'],
                season=row['season'],
                time_slot=row['time_slot'],
                home_goals=row['home_goals'],
                away_goals=row['away_goals'],
                attendance=row['attendance']
            ))
            
        session.commit()
        print('Matches inserted')

        columns = ['home_xg', 'away_xg', 'home_possession', 'away_possession',
                   'home_passes_completed', 'away_passes_completed',
                   'home_passes_attempted', 'away_passes_attempted',
                   'home_shots_on_target', 'away_shots_on_target',
                   'home_shots_total', 'away_shots_total', 'home_saves_made', 
                   'away_saves_made', 'home_saves_faced', 'away_saves_faced', 
                   'home_yellow_cards', 'away_yellow_cards', 'home_red_cards', 
                   'away_red_cards', 'home_fouls', 'away_fouls', 'home_corners', 
                   'away_corners', 'home_crosses', 'away_crosses', 
                   'home_touches', 'away_touches', 'home_tackles', 
                   'away_tackles', 'home_interceptions', 'away_interceptions', 
                   'home_aerials_won', 'away_aerials_won', 'home_clearances', 
                   'away_clearances', 'home_offsides', 'away_offsides', 
                   'home_goal_kicks', 'away_goal_kicks', 'home_throw_ins', 
                   'away_throw_ins', 'home_long_balls', 'away_long_balls']

        for record in match_stats_df:
            session.add(MatchStatistic(**record))

        for _, row in weather_conditions_df.iterrows():
            session.add(Weather(
                id=row['id'],
                match_id=row['match_id'],
                temperature=row['temperature'],
                precipitation=row['precipitation'],
                wind_speed=row['wind_speed'],
                temperature_category=row['temperature_category'],
                is_precipitation=row['is_precipitation'],
                wind_speed_category=row['wind_speed_category']
            ))

        session.commit()
        print('Match statistics inserted')
        print('Weather conditions inserted')


def connect_db():
    '''Retrieve the credentials and create a new session'''

    db_name, user, password, host, port = load_db_env()
    engine, inspector, Session = create_engine_inspector_session(
        db_name, user, password, host, port)
    return engine, inspector, Session