# Library imports
import pandas as pd
import psycopg2
from psycopg2.extensions import connection
from psycopg2.extras import execute_values


def get_connection(db_config: dict[str, str]) -> connection | None:
    '''
    Open and return a connection to the PostgreSQL database
    '''
    print('== POSTGRESQL DATABASE CONNECTION ==')
    try:
        conn = psycopg2.connect(
            dbname=db_config['dbname'],
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
        print(' -> Successfully connected to PostgreSQL database')
        return conn
    except Exception as e:
        print(f' -> Error when connecting to PostgreSQL database: {e}')
        return None
    

def create_database(db_config: dict[str, str]) -> None:
    '''
    Create the database if it does not already exist
    '''
    # Connect to the default postgres database
    conn = psycopg2.connect(
            dbname='postgres',
            user=db_config['user'],
            password=db_config['password'],
            host=db_config['host'],
            port=db_config['port']
        )
    conn.autocommit = True

    # Create the new database
    print('== POSTGRESQL DATABASE CREATION ==')
    db_name = db_config['dbname']
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT 1 FROM pg_database WHERE datname='{db_name}'")
            if cur.fetchone():
                print(f' -> Database {db_name} already exists\n')
            else:
                cur.execute(F'CREATE DATABASE {db_name}')
                print(f' -> Database {db_name} created successfully\n')
    finally:
        conn.close()


def execute_sql_file(conn: connection, file_path: str) -> None:
    '''
    Read a .sql file and execute it
    '''
    with open(file_path, 'r', encoding='utf-8') as file:
        sql_script = file.read()

    try:
        with conn.cursor() as cur:
            cur.execute(sql_script)
        conn.commit()
        print(f' -> SQL file executed successfully\n')
    except Exception as e:
        conn.rollback()
        print(f' -> Error when executing SQL file: {e}\n')
        raise


def insert_df(conn: connection, table_name: str, df: pd.DataFrame):
    '''
    Insert a DataFrame into a table within the database
    '''
    columns = ', '.join(df.columns)
    if table_name.lower() == 'order':
        formatted_table_name = '"order"'
    else:
        formatted_table_name = table_name
    query = f'INSERT INTO {formatted_table_name} ({columns}) VALUES %s'
    values = df.values.tolist()

    try:
        with conn.cursor() as cur:
            execute_values(cur, query, values)
        conn.commit()
        print(f' -> Table {table_name}: {len(df)} rows inserted')
    except Exception as e:
        conn.rollback()
        print(f' -> Error when inserting data into table {table_name}: {e}')
        raise


def insert_all_dataframes(
        conn: connection, dataframes: dict[str, pd.DataFrame]) -> None:
    '''
    Insert all DataFrames into the corresponding tables within the database
    '''
    print(f'== DATA INGESTION ==')
    for table_name, df in dataframes.items():
        insert_df(conn, table_name, df)

    print('\nData ingestion completed')