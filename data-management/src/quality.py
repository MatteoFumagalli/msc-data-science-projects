# Import useful libraries
import pandas as pd
import folium
import geopandas as gpd
from geopandas.tools import sjoin

from sqlalchemy import text


# Definition of functions
def evaluate_completeness(engine, inspector, table_name):
    '''Evaluate the completeness of a table'''
    
    if not inspector.has_table(table_name):
        raise ValueError(f'Table {table_name} does not exist in the database')
    
    columns = [col['name'] for col in inspector.get_columns(table_name)]
    df = pd.read_sql_table(table_name, engine)

    if df.empty:
        raise ValueError(f'Table {table_name} exists but is empty')

    # Calculate the completeness metrics
    tuple_completeness = (
        df.notnull().sum(axis=1) / len(columns) * 100).mean().round(2)
    attribute_completeness = (df.notnull().sum() / len(df) * 100).round(2)
    table_completeness = (
        df.notnull().sum().sum() / (df.shape[0] * df.shape[1]) * 100).round(2)
    
    return {
        'table': table_name,
        'tuple_completeness (%)': tuple_completeness,
        'attribute_completeness (%)': attribute_completeness.to_dict(),
        'table_completeness (%)': table_completeness
    }


def evaluate_all_tables_completeness(engine, inspector):
    '''Evaluate the completeness of all tables'''

    tuple_completeness_all = []
    table_completeness_all = []
    attribute_completeness_all = []

    for table in ['competitions', 'teams', 'stadiums', 
                'matches', 'match_statistics', 'weather_conditions']:
        report = evaluate_completeness(engine, inspector, table)

        tuple_completeness_all.append(report['tuple_completeness (%)'])
        table_completeness_all.append(report['table_completeness (%)'])
        attribute_completeness_all.extend(
            report['attribute_completeness (%)'].values()
        )

        print(f"\nTable: {report['table']}")
        print(f"Tuple completeness: {report['tuple_completeness (%)']}%")
        print(f"Table completeness: {report['table_completeness (%)']}%")
        print("\nAttribute completeness:")
        for attr, val in report['attribute_completeness (%)'].items():
            print(f'  - {attr}: {val}%')

    overall_stats = {
        'avg_tuple_completeness': 
        sum(tuple_completeness_all)/len(tuple_completeness_all),
        'avg_table_completeness': 
        sum(table_completeness_all)/len(table_completeness_all),
        'attribute_completeness_min': min(attribute_completeness_all),
        'attribute_completeness_mean': 
        sum(attribute_completeness_all)/len(attribute_completeness_all),
        'attribute_completeness_max': max(attribute_completeness_all)
    }

    print("\nOverall statistics:")
    print('Average tuple completeness: '
          f"{overall_stats['avg_tuple_completeness']:.2f}%")
    print('Average table completeness: '
          f"{overall_stats['avg_table_completeness']:.2f}%")
    print("Overall attribute completeness:")
    print(f"  - Min:  {overall_stats['attribute_completeness_min']:.2f}%")
    print(f"  - Mean: {overall_stats['attribute_completeness_mean']:.2f}%")
    print(f"  - Max:  {overall_stats['attribute_completeness_max']:.2f}%")


def evaluate_consistency_stadiums(engine, inspector):
    '''Plot stadium locations on a map, using 1-km buffered country polygons'''

    for table in ['stadiums', 'countries_admin0']:
        if not inspector.has_table(table):
            raise ValueError(f'Table {table} does not exist in the database')
    
    countries_gdf = gpd.read_postgis(
        'SELECT name, geom FROM countries_admin0;', engine, geom_col='geom')

    stadiums_gdf = gpd.read_postgis(
        'SELECT * FROM stadiums;', engine, geom_col='geometry')

    # Create the 1-km buffer to avoid false outliers
    countries_gdf_utm = countries_gdf.to_crs(epsg=3857)
    countries_gdf_utm['geom_buffered'] = countries_gdf_utm['geom'].buffer(1000)
    countries_gdf['geom_buffered'] = countries_gdf_utm[
        'geom_buffered'].to_crs(epsg=4326)

    # Check wheter the stadium is outside its country buffer
    stadiums_gdf['outside_country'] = ~stadiums_gdf.apply(
        lambda row: countries_gdf.loc[
            countries_gdf['name']==row['country'], 'geom_buffered']
        .apply(lambda poly: row['geometry'].within(poly)).any(),
        axis=1
    )

    # Create the map
    mean_lat = stadiums_gdf.geometry.y.mean()
    mean_lon = stadiums_gdf.geometry.x.mean()
    m = folium.Map(
        location=[mean_lat, mean_lon], zoom_start=4, tiles='CartoDB positron')

    for _, row in stadiums_gdf.iterrows():
        color = 'red' if row['outside_country'] else 'green'
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5, color=color,
            fill=True, fill_color=color, fill_opacity=0.7,
            popup=f"{row['stadium_name']} ({row['country']})"
        ).add_to(m)

    folium.GeoJson(
        countries_gdf['geom_buffered'],
        style_function=lambda x: {
            'color': 'black', 'weight': 1, 'fillOpacity': 0}
    ).add_to(m)

    return m, stadiums_gdf[stadiums_gdf['outside_country']]


def evaluate_stadiums_overlapping(engine, inspector):
    '''Evaluate whether different stadiums are inside a buffer of 500-meter'''

    for table in ['stadiums', 'countries_admin0']:
        if not inspector.has_table(table):
            raise ValueError(f'Table {table} does not exist in the database')

    stadiums = gpd.read_postgis(
        'SELECT * FROM stadiums;', engine, geom_col='geometry')

    stadiums_utm = stadiums.to_crs(epsg=3857)
    stadiums_utm['buffer'] = stadiums_utm.geometry.buffer(500)

    buffers = stadiums_utm[['id', 'stadium_name', 'buffer']].rename(
        columns={
            'id': 'id_left',
            'stadium_name': 'stadium_name_left'
        }).set_geometry('buffer')

    points = stadiums_utm[['id', 'stadium_name', 'geometry']].rename(
        columns={
            'id': 'id_right',
            'stadium_name': 'stadium_name_right'
        })

    # Spatial self-join between buffers and coordinates
    overlaps = sjoin(buffers, points, predicate='contains', how='inner')

    # Remove self matches and mirrored duplicates
    overlaps = overlaps[overlaps['id_left'] != overlaps['id_right']]
    overlaps = overlaps[overlaps['id_left'] < overlaps['id_right']]

    # Add original geometries
    geom_map = stadiums_utm.set_index('id').geometry
    overlaps['geometry_left'] = overlaps['id_left'].map(geom_map)
    overlaps['geometry_right'] = overlaps['id_right'].map(geom_map)

    # Compute distance
    overlaps['distance_meter'] = overlaps.geometry_left.distance(
        overlaps.geometry_right
    )

    overlaps['geometry_left'] = gpd.GeoSeries(
        overlaps['geometry_left'], crs=3857
    ).to_crs(4326)

    overlaps['geometry_right'] = gpd.GeoSeries(
        overlaps['geometry_right'], crs=3857
    ).to_crs(4326)

    result = overlaps[[
        'id_left', 'stadium_name_left',
        'id_right', 'stadium_name_right',
        'geometry_left', 'geometry_right',
        'distance_meter'
    ]]

    return result


def evaluate_matches_stats_consistency(engine, inspector):
    '''Check the consistency between different statistics'''
    
    if not inspector.has_table('match_statistics'):
        raise ValueError(
            f'Table match_statistics does not exist in the database')
    
    match_stats_df = pd.read_sql_table('match_statistics', engine)

    rules = {
        'passes': '''home_passes_completed > home_passes_attempted
        OR away_passes_completed > away_passes_attempted''',

        'shots': '''home_shots_on_target > home_shots_total
        OR away_shots_on_target > away_shots_total''',

        'possession': 'home_possession + away_possession <> 1',

        'cards': '''home_yellow_cards < 0 OR home_red_cards < 0 
        OR away_yellow_cards < 0 OR away_red_cards < 0''',

        'saves vs goals': 
        '''home_goals = 0 AND away_saves_made <> home_shots_on_target
        OR away_goals = 0 AND home_saves_made <> away_shots_on_target'''
    }

    results = {}
    for rule, condition in rules.items():
        query = f'''
        SELECT COUNT(*) FROM match_statistics 
        JOIN matches ON match_statistics.match_id=matches.id 
        WHERE {condition};'''
        violations = pd.read_sql(query, engine).iloc[0,0]
        results[rule] = {
            'violations': int(violations),
            'consistency': float(
                round(1 - violations/match_stats_df.shape[0], 4))
        }

    return results


def check_possession_violations(engine):
    '''Identify matches that violate possession constraint'''

    query = '''
    SELECT match_id, home_possession, away_possession,
        (home_possession + away_possession) AS total_possession
    FROM match_statistics
    WHERE home_possession + away_possession <> 1'''

    df = pd.read_sql(query, engine)
    return df


def check_saves_goals_violations(engine):
    '''Identify matches that violate saves vs goals constraint'''

    query = '''
    SELECT m.id AS match_id, m.date,
        ht.team_name AS home_team, at.team_name AS away_team, 
        m.home_goals, m.away_goals, 
        ms.home_saves_made, ms.away_saves_made, 
        ms.home_shots_on_target, ms.away_shots_on_target
    FROM match_statistics ms JOIN matches m ON ms.match_id = m.id
    JOIN teams ht ON m.home_team_id = ht.id 
    JOIN teams at ON m.away_team_id = at.id
    WHERE (m.home_goals = 0 AND ms.away_saves_made <> ms.home_shots_on_target)
        OR (m.away_goals = 0 AND ms.home_saves_made <> ms.away_shots_on_target);'''

    df = pd.read_sql(query, engine)
    return df


def evaluate_temporal_consistency(engine, inspector):
    '''Check the consistency between date and competition year'''
    
    for table in ['matches', 'competitions']:
        if not inspector.has_table(table):
            raise ValueError(
                f'Table {table} does not exist in the database')

    query = '''
    SELECT m.id as match_id, m.date as match_date, c.season_year as season
    FROM matches m JOIN competitions c ON m.competition_id = c.id
    WHERE EXTRACT(YEAR FROM m.date) NOT IN (
        CAST(SUBSTRING(c.season_year FROM 1 FOR 4) AS int),
        CAST(SUBSTRING(c.season_year FROM 6 FOR 4) AS int)
    );
    '''

    inconsistencies = pd.read_sql(query, engine)
    return inconsistencies


def create_view_matches_quality(engine):
    '''Create a view for evaluating to quality of matches'''

    query = '''
    CREATE OR REPLACE VIEW view_matches_quality AS
    SELECT m.id as match_id,

        CASE
            WHEN ms.match_id IS NOT NULL AND (
                ms.home_xg IS NULL OR
                ms.away_xg IS NULL
            ) THEN TRUE
            ELSE FALSE
        END AS is_suspended,

        CASE
            WHEN (
                (m.home_goals = 0 AND 
                ms.away_saves_made <> ms.home_shots_on_target)
                OR
                (m.away_goals = 0 AND 
                ms.home_saves_made <> ms.away_shots_on_target)
            ) THEN TRUE
            ELSE FALSE
        END AS is_awarded,

        CASE
            WHEN m.attendance IS NULL THEN TRUE
            ELSE FALSE
        END AS closed_doors

    FROM matches m JOIN match_statistics ms ON m.id=ms.match_id;
    '''

    with engine.begin() as conn:
        conn.execute(text(query))


def create_view_match_statistics_quality(engine):
    '''Create a view for evaluating to quality of statistics'''

    query = '''
    CREATE OR REPLACE VIEW view_match_statistics_quality AS
    SELECT 
        ms.match_id, ms.home_possession, ms.away_possession,
        (ms.home_possession + ms.away_possession) AS total_possession,

        ABS(ms.home_possession + ms.away_possession - 1) <= 0.015
            AS possession_consistent 
    FROM match_statistics ms;
    '''

    with engine.begin() as conn:
        conn.execute(text(query))


def create_view_overlapping_stadiums(engine):
    '''Create a view for overlapping stadiums'''

    query = '''
    CREATE OR REPLACE VIEW view_overlapping_stadiums AS
    SELECT
        s1.id AS stadium_id_1, s1.stadium_name AS stadium_name_1,
        s2.id AS stadium_id_2, s2.stadium_name AS stadium_name_2,
        ST_Distance(
            s1.geometry::geography,
            s2.geometry::geography
        ) AS distance_meter     
    FROM stadiums s1
    JOIN stadiums s2
      ON s1.id < s2.id AND
        ST_DWithin(
            s1.geometry::geography,
            s2.geometry::geography,
            500
        );
    '''

    with engine.begin() as conn:
        conn.execute(text(query))


def create_view_matches_clean(engine):
    '''Create a view excluding suspended and awarded matches'''

    query = '''
    CREATE OR REPLACE VIEW view_matches_clean AS
    SELECT m.*
    FROM matches m
    JOIN view_matches_quality mq ON m.id = mq.match_id
    WHERE mq.is_suspended = false AND mq.is_awarded = false;
    '''

    with engine.begin() as conn:
        conn.execute(text(query))


def create_view_match_statistics_clean(engine):
    '''Create a view excluding inconsistent statistics'''

    query = '''
    CREATE OR REPLACE VIEW view_match_statistics_clean AS
    SELECT ms.*
    FROM match_statistics ms
    JOIN view_match_statistics_quality msq ON ms.match_id = msq.match_id
    WHERE msq.possession_consistent = true;   
    '''

    with engine.begin() as conn:
        conn.execute(text(query))


def create_view_stadiums_clean(engine):
    '''Create a view with the indication of canonical stadiums'''

    query = '''
    CREATE OR REPLACE VIEW view_stadiums_clean AS
    WITH mapping AS (
        SELECT stadium_id_1 AS original_id,
            LEAST(stadium_id_1, stadium_id_2) AS canonical_id
        FROM view_overlapping_stadiums
        UNION
        SELECT stadium_id_2 AS original_id,
            LEAST(stadium_id_1, stadium_id_2) AS canonical_id
        FROM view_overlapping_stadiums
    )
    SELECT s.id AS original_id,
        COALESCE(m.canonical_id, s.id) AS canonical_id,
        s.stadium_name, s.country, s.geometry
    FROM stadiums s
    LEFT JOIN mapping m
        ON s.id = m.original_id;
    '''

    with engine.begin() as conn:
        conn.execute(text(query))