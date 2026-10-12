# Import useful libraries
import numpy as np
import pandas as pd

from sqlalchemy import inspect
from src.models import Match, MatchStatistic, Weather

from IPython.display import display
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
import seaborn as sns
sns.set_theme(style='whitegrid')

from scipy.stats import chi2_contingency
from scipy.stats import kruskal
from scikit_posthocs import posthoc_dunn


# Definition of functions (RQ1)
def get_result(row):
    '''Obtain the result from the number of goals'''
    if row['home_goals'] > row['away_goals']:
        return 'Home Win'
    elif row['home_goals'] < row['away_goals']:
        return 'Away Win'
    else:
        return 'Draw'
        
colors = ['#3B6FB6', '#FF8C1A', '#4CAF50']


def distribution_is_precipitation(Session):
    '''Display the distribution of matches with or without precipitation'''

    # Data query
    with Session() as session:
        weather_data = session.query(Weather.is_precipitation).all()

    df_weather = pd.DataFrame(weather_data, columns=['is_precipitation'])

    df_weather['precipitation'] = df_weather['is_precipitation'].map(
        {False: 'No Precipitation', True: 'Precipitation'}
    )
    category_order = ['No Precipitation', 'Precipitation']
    counts = df_weather['precipitation'].value_counts().reindex(category_order)

    # Plot
    colors = ['tomato', 'steelblue']
    plt.figure(figsize=(8,5))
    bars = plt.bar(
        counts.index,
        counts.values,
        color=colors,
        edgecolor='black',
        linewidth=1.2
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + max(counts.values)*0.02,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=12
        )

    plt.xlabel('')
    plt.ylabel('Number of Matches')
    plt.title(
        'Distribution of Matches with or without Precipitation',
        fontsize=19, pad=20
    )

    plt.ylim(0, max(counts.values)*1.1)
    plt.tight_layout()
    plt.show()


def distribution_precipitations(Session): 
    '''Display the distribution of numerical precipitations''' 
    # Data query 
    with Session() as session: 
        data = session.query( Weather.precipitation ).all() 
        
    df = pd.DataFrame(data, columns=['precipitation']) 
    
    # Plot histogram with log scale on Y axis 
    plt.figure(figsize=(8, 5)) 
    plt.hist(df['precipitation'], bins=30, edgecolor='black', color='steelblue') 
    plt.yscale('log') 
    plt.title('Distribution of Precipitation (mm) in Matches', 
              fontsize=20, fontweight='bold', pad=20) 
    plt.xlabel('Precipitation (mm)', fontsize=14) 
    plt.ylabel('Number of Matches (log scale)', fontsize=14) 
    plt.grid(axis='y', alpha=0.75) 
    plt.show()


def distribution_matches_precipitations(Session):
    '''Display the distribution of results for is_precipitation'''

    # Data Query
    with Session() as session:
        data = session.query(
            Match.home_goals,
            Match.away_goals,
            Weather.is_precipitation
        ).join(Weather, Match.id == Weather.match_id).all()

    df = pd.DataFrame(data, 
                      columns=['home_goals', 'away_goals', 'is_precipitation'])

    df['result'] = df.apply(get_result, axis=1)
    
    df_rain = df[df['is_precipitation'] == True]
    df_no_rain = df[df['is_precipitation'] == False]
    rain_counts = df_rain['result'].value_counts(normalize=True) * 100
    no_rain_counts = df_no_rain['result'].value_counts(normalize=True) * 100

    # Plot parameters
    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 14,
        'axes.titlesize': 19,
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Rain pie
    axes[0].pie(
        rain_counts.values,
        labels=rain_counts.index,
        autopct="%1.1f%%",
        pctdistance=0.75,
        labeldistance=1.08,
        colors=colors,
        startangle=90,
        wedgeprops={'linewidth': 1.2, 'edgecolor': 'black'}
    )
    axes[0].set_title('Match Outcomes - Precipitation'
                      f'\n(Total matches: {len(df_rain)})', 
                      fontsize=20, fontweight='bold')

    # No rain pie
    axes[1].pie(
        no_rain_counts.values,
        labels=no_rain_counts.index,
        autopct="%1.1f%%",
        pctdistance=0.75,
        labeldistance=1.08,
        colors=colors,
        startangle=90,
        wedgeprops={'linewidth': 1.2, 'edgecolor': 'black'}
    )
    axes[1].set_title(
        'Match Outcomes - No Precipitation'
        f'\n(Total matches: {len(df_no_rain)})', 
        fontsize=20, fontweight='bold')

    plt.tight_layout()
    plt.show()


def chi_square_test_rain(Session):
    '''Apply a chi-square test on is_precipitation'''

    # Data Query
    with Session() as session:
        data = session.query(
            Match.home_goals,
            Match.away_goals,
            Weather.is_precipitation
        ).join(Weather, Match.id == Weather.match_id).all()

    df = pd.DataFrame(data, 
                      columns=['home_goals', 'away_goals', 'is_precipitation'])

    df['result'] = df.apply(get_result, axis=1)

    # Contingency table
    contingency_table = pd.crosstab(df['is_precipitation'], df['result'])
    print('Contingency Table:')
    print(contingency_table)

    # Chi-Square Test
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print('\nChi-Square Statistic:', chi2)
    print('Degrees of Freedom:', dof)
    print('P-value:', p)
    print('\nExpected Frequencies if independent:')
    print(pd.DataFrame(expected, index=contingency_table.index, 
                       columns=contingency_table.columns))

    if p < 0.05:
        print('\nConclusion: Rain affects match outcomes '
              '(reject null hypothesis)')
    else:
        print('\nConclusion: Rain does NOT significantly affect match outcomes '
              '(fail to reject null hypothesis)')
        

def match_outcomes_is_precipitation(Session):
    '''Evaluate the impact of precipitation on match outcomes'''

    distribution_is_precipitation(Session)
    distribution_precipitations(Session)
    distribution_matches_precipitations(Session)
    chi_square_test_rain(Session)


def distribution_wind_category(Session):
    '''Display the distribution of wind speed categories'''

    # Data query
    with Session() as session:
        data = session.query(Weather.wind_speed_category).all()

    df = pd.DataFrame(data, columns=['wind_category'])

    category_order = ['Low', 'Medium', 'High']
    df['wind_category'] = pd.Categorical(
        df['wind_category'], categories=category_order, ordered=True
    )

    counts = df['wind_category'].value_counts().reindex(category_order)

    # Plot
    colors = ['#A8E6A3', '#4CAF50', '#006400']
    plt.figure(figsize=(8,5))
    bars = plt.bar(
        counts.index,
        counts.values,
        color=colors,
        edgecolor='black',
        linewidth=1.2
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + max(counts.values)*0.02,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=12
        )

    plt.xticks(
        range(len(counts)),
        ['Low Wind', 'Medium Wind', 'High Wind']
    )
    plt.xlabel('')
    plt.ylabel('Number of Matches')
    plt.title('Distribution of Matches by Wind Speed Category', pad=20)
    plt.ylim(0, max(counts.values)*1.1)

    plt.tight_layout()
    plt.show()


def distribution_wind_speed(Session):
    '''Display the distribution of wind speed (km/h)'''

    with Session() as session:
        data = session.query(
            Weather.wind_speed).all()

    df = pd.DataFrame(data, columns=['wind_speed'])

    plt.figure(figsize=(8, 5))
    plt.hist(df['wind_speed'], bins=30, edgecolor='black', color='#4CAF50')

    plt.title(
        'Distribution of Wind Speed (km/h) in Matches',
        fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Wind Speed (km/h)', fontsize=14)
    plt.ylabel('Number of Matches', fontsize=14)
    plt.grid(axis='y', alpha=0.75)

    plt.show()


def distribution_matches_wind(Session):
    '''Display the distribution of results for wind_category'''

    # Data Query
    with Session() as session:
        data = session.query(
            Match.home_goals,
            Match.away_goals,
            Weather.wind_speed_category
        ).join(Weather, Match.id == Weather.match_id).all()

    df = pd.DataFrame(data, columns=[
        'home_goals', 'away_goals', 'wind_speed_category'])

    df['result'] = df.apply(get_result, axis=1)

    ordered_categories = ['Low', 'Medium', 'High']
    result_order = ['Home Win', 'Away Win', 'Draw']

    wind_groups = (
        df.groupby('wind_speed_category')['result']
        .value_counts(normalize=True)
        .mul(100)
        .unstack()
        .reindex(ordered_categories)
    )[result_order]

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 14,
        'axes.titlesize': 19,
    })

    num_categories = len(wind_groups.index)
    fig, axes = plt.subplots(1, num_categories, figsize=(6 * num_categories, 7))
    if num_categories == 1:
        axes = [axes]

    for i, category in enumerate(wind_groups.index):
        values = wind_groups.loc[category].values
        labels = wind_groups.columns
        total_matches = df[df['wind_speed_category'] == category].shape[0]
        
        # Draw pie
        axes[i].pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            pctdistance=0.80,
            labeldistance=1.10,
            colors=colors,
            startangle=90,
            wedgeprops={'linewidth': 1.2, 'edgecolor': 'black'}
        )
        axes[i].set_title(
            f'Match Outcomes - {category} '
            f'Wind\n(Total matches: {total_matches})', 
            fontsize=20, fontweight='bold', pad=-10)

    plt.tight_layout()
    plt.show()


def chi_square_test_wind(Session):
    '''Apply the chi-square test on wind_category'''

    # Data Query
    with Session() as session:
        data = session.query(
            Match.home_goals,
            Match.away_goals,
            Weather.wind_speed_category
        ).join(Weather, Match.id == Weather.match_id).all()

    df = pd.DataFrame(data, 
                      columns=['home_goals', 'away_goals', 'wind_category'])

    df['result'] = df.apply(get_result, axis=1)

    # Contingency table
    contingency_table = pd.crosstab(
        df['wind_category'], df['result']).reindex(
            index=['Low', 'Medium', 'High'], 
            columns=['Away Win', 'Draw', 'Home Win'])
    
    print('Contingency Table:')
    print(contingency_table)

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print('\nChi-Square Statistic:', chi2)
    print('Degrees of Freedom:', dof)
    print('P-value:', p)
    print('\nExpected Frequencies if independent:')
    print(pd.DataFrame(expected, 
                       index=contingency_table.index, 
                       columns=contingency_table.columns))

    # P-value test
    if p < 0.05:
        print('\nConclusion: Wind category significantly affects match '
              'outcomes (reject null hypothesis)')
    else:
        print('\nConclusion: Wind category does NOT significantly affect match '
              'outcomes (fail to reject null hypothesis)')
        

def match_outcomes_wind_category(Session):
    '''Evaluate the impact of wind on match outcomes'''
    
    distribution_wind_category(Session)
    distribution_wind_speed(Session)
    distribution_matches_wind(Session)
    chi_square_test_wind(Session)


def distribution_temperature_category(Session):
    '''Display the distribution of temperature categories'''

    # Data query
    with Session() as session:
        data = session.query(Weather.temperature_category).all()

    df = pd.DataFrame(data, columns=['temperature_category'])

    category_order = ['Cold', 'Mild', 'Hot']
    df['temperature_category'] = pd.Categorical(
        df['temperature_category'],
        categories=category_order,
        ordered=True
    )
    counts = df['temperature_category'].value_counts().reindex(category_order)

    # Plot
    colors = ['#FFEDA0', '#FEB24C', '#F03B20']
    plt.figure(figsize=(8,5))
    bars = plt.bar(
        counts.index,
        counts.values,
        color=colors,
        edgecolor='black',
        linewidth=1.2
    )

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width()/2,
            height + max(counts.values)*0.02,
            f'{int(height)}',
            ha='center',
            va='bottom',
            fontweight='bold',
            fontsize=12
        )

    plt.xticks(range(len(counts)), ['Cold', 'Mild', 'Hot'])
    plt.xlabel('')
    plt.ylabel('Number of Matches')
    plt.title('Distribution of Matches by Temperature Category', pad=20)
    plt.ylim(0, max(counts.values)*1.1)

    plt.tight_layout()
    plt.show()


def distribution_temperature(Session):
    '''Display the distribution of temperature (°C)'''

    with Session() as session:
        data = session.query(Weather.temperature).all()

    df = pd.DataFrame(data, columns=['temperature'])

    plt.figure(figsize=(8, 5))
    plt.hist(df['temperature'], bins=30, edgecolor='black', color='#FEB24C')

    plt.title(
        'Distribution of Temperature (°C) in Matches',
        fontsize=20, fontweight='bold', pad=20)
    plt.xlabel('Temperature (°C)', fontsize=14)
    plt.ylabel('Number of Matches', fontsize=14)
    plt.grid(axis='y', alpha=0.75)
    plt.show()


def distribution_matches_temperature(Session):
    '''Display the distribution of results for temperature_category'''

    # Data query
    with Session() as session:
        data = session.query(
            Match.home_goals,
            Match.away_goals,
            Weather.temperature_category
        ).join(Weather, Match.id == Weather.match_id).all()

    df = pd.DataFrame(
        data, columns=['home_goals', 'away_goals', 'temperature_category'])
    df['result'] = df.apply(get_result, axis=1)

    result_order = ['Home Win', 'Away Win', 'Draw']
    temp_groups = df.groupby('temperature_category')['result'].value_counts(
        normalize=True).mul(100).unstack()[result_order]
    temp_groups = temp_groups.reindex(['Cold', 'Mild', 'Hot'])

    plt.rcParams.update({
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 14,
        'axes.titlesize': 19,
    })

    num_categories = len(temp_groups.index)
    fig, axes = plt.subplots(1, num_categories, figsize=(6 * num_categories, 7))
    if num_categories == 1:
        axes = [axes]

    for i, category in enumerate(temp_groups.index):
        values = temp_groups.loc[category].values
        labels = temp_groups.columns
        total_matches = df[df['temperature_category'] == category].shape[0]
    
        # Draw pie
        axes[i].pie(
            values,
            labels=labels,
            autopct="%1.1f%%",
            pctdistance=0.80,
            labeldistance=1.10,
            colors=colors,
            startangle=90,
            wedgeprops={'linewidth': 1.2, 'edgecolor': 'black'}
        )
        axes[i].set_title(
            f'Match Outcomes - {category} Temperature\n'
            f'(Total matches: {total_matches})', 
            fontsize=20, fontweight='bold', pad=20)

    plt.tight_layout()
    plt.show()


def chi_square_test_temperature(Session):
    '''Apply the chi-square test on temperature_category'''

    # Data Query
    with Session() as session:
        data = session.query(
            Match.home_goals,
            Match.away_goals,
            Weather.temperature_category
        ).join(Weather, Match.id == Weather.match_id).all()

    df = pd.DataFrame(
        data,
        columns=['home_goals', 'away_goals', 'temperature_category']
    )

    df['result'] = df.apply(get_result, axis=1)

    # Contingency table
    contingency_table = pd.crosstab(
        df['temperature_category'],
        df['result']
    ).reindex(
        index=['Cold', 'Mild', 'Hot'],
        columns=['Away Win', 'Draw', 'Home Win']
    )

    print('Contingency Table:')
    print(contingency_table)

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print('\nChi-Square Statistic:', chi2)
    print('Degrees of Freedom:', dof)
    print('P-value:', p)

    print('\nExpected Frequencies if independent:')
    print(pd.DataFrame(expected,
                       index=contingency_table.index, 
                       columns=contingency_table.columns))

    if p < 0.05:
        print('\nConclusion: Temperature category significantly affects match '
              'outcomes (reject null hypothesis)')
    else:
        print('\nConclusion: Temperature category does NOT significantly '
              'affect match outcomes (fail to reject null hypothesis)')
        

def match_outcomes_temperature_category(Session):
    '''Evaluate the impact of temperature on match outcomes'''
    
    distribution_temperature_category(Session)
    distribution_temperature(Session)
    distribution_matches_temperature(Session)
    chi_square_test_temperature(Session)


def distribution_attendance_is_precipitation(Session):
    '''Display the distribution of attendance for is_precipitation'''

    # Attendance vs Precipitation
    with Session() as session:
        data = session.query(
                Match.attendance, Weather.is_precipitation).join(
                    Weather, Match.id == Weather.match_id).all()
    df = pd.DataFrame(data, columns=['attendance', 'is_precipitation'])

    median_no_rain = df[df['is_precipitation'] == False]['attendance'].median()
    median_rain = df[df['is_precipitation'] == True]['attendance'].median()

    print(f"Median attendance without rain: {median_no_rain:.0f}")
    print(f"Median attendance with rain: {median_rain:.0f}")

    # Boxplot
    plt.figure(figsize=(8,5))
    sns.boxplot(
        x='is_precipitation',
        y='attendance',
        hue='is_precipitation',
        data=df,
        palette={False:'tomato', True:'steelblue'},
        dodge=False
    )
    plt.xticks([0,1], ['No Precipitation','Precipitation'])
    plt.ylabel('Attendance')
    plt.title('Attendance vs Precipitation')
    sns.despine()
    plt.legend([],[], frameon=False)
    plt.show()


def distribution_attendance_wind_category(Session):
    '''Display the distribution of attendance for wind_category'''

    # Attendance vs Wind Speed Category
    with Session() as session:
        data = session.query(
              Match.attendance, Weather.wind_speed_category).join(
                  Weather, Match.id == Weather.match_id).all()
    df = pd.DataFrame(data, columns=['attendance','wind_speed_category'])

    # Median
    medians = df.groupby(
        'wind_speed_category')['attendance'].median().reindex(
            ['Low','Medium','High'])
    for cat, med in medians.items():
        print(f"Median attendance for {cat} wind: {med:.0f}")

    # Boxplot 
    plt.figure(figsize=(8,5))
    sns.boxplot(
        x='wind_speed_category',
        y='attendance',
        hue='wind_speed_category',
        data=df,
        palette={'Low':'#A8E6A3', 'Medium':'#4CAF50', 'High':'#006400'},
        order=['Low','Medium','High'],
        dodge=False
    )
    plt.ylabel('Attendance')
    plt.title('Attendance vs Wind Speed Category')
    sns.despine()
    plt.legend([],[], frameon=False)
    plt.show()


def distribution_attendance_temperature_category(Session):
    '''Display the distribution of attendance for temperature_category'''

    # Attendance vs Temperature Category
    with Session() as session:
        data = session.query(
               Match.attendance, Weather.temperature_category).join(
                   Weather, Match.id == Weather.match_id).all()
    df = pd.DataFrame(data, columns=['attendance','temperature_category'])

    # Median
    medians = df.groupby(
        'temperature_category')['attendance'].median().reindex(
            ['Cold', 'Mild', 'Hot'])
    for cat, med in medians.items():
        print(f"Median attendance for {cat} temperature: {med:.0f}")

    # Boxplot
    plt.figure(figsize=(8,5))
    sns.boxplot(
        x='temperature_category',
        y='attendance',
        hue='temperature_category',
        data=df,
        palette={'Cold':'#FFEDA0', 'Mild':'#FEB24C', 'Hot':'#F03B20'},
        order=['Cold', 'Mild', 'Hot'],
        dodge=False
    )
    plt.ylabel('Attendance')
    plt.title('Attendance vs Temperature Category')
    sns.despine()
    plt.legend([],[], frameon=False)
    plt.show()


def distribution_attendance(Session):
    '''Evaluate the distribution of attendance for different conditions'''

    distribution_attendance_is_precipitation(Session)
    distribution_attendance_wind_category(Session)
    distribution_attendance_temperature_category(Session)


def cohens_d(x, y):
    '''Compute Cohen's d for all numeric columns'''

    nx, ny = len(x), len(y)
    pooled_std = np.sqrt(
        ((nx-1)*np.std(x, ddof=1)**2 + (ny-1)*np.std(y, ddof=1)**2) / (nx+ny-2))
    if pooled_std == 0:
        return 0
    return (np.mean(x) - np.mean(y)) / pooled_std


def match_statistics_is_precipitation(Session):
    '''Display the distribution of match statistics for is_precipitation'''

    # Numeric columns from MatchStatistic
    stat_numeric_cols = [
        c.name for c in inspect(MatchStatistic).columns
        if c.type.python_type in [int, float] 
        and c.name not in ['id', 'match_id']
    ]

    goal_cols = ['home_goals', 'away_goals']
    numeric_cols = stat_numeric_cols + goal_cols

    exclude_cols = [
        'home_passes_completed', 'away_passes_completed',
        'home_passes_attempted', 'away_passes_attempted',
        'home_touches', 'away_touches'
    ]

    group1_cols = [c for c in numeric_cols if c not in exclude_cols]
    group2_cols = [c for c in numeric_cols if c in exclude_cols]

    # Data Query
    with Session() as session:
        query = session.query(
            *[getattr(MatchStatistic, c) for c in stat_numeric_cols],
            Match.home_goals,
            Match.away_goals,
            Weather.is_precipitation
        ).join(Match, MatchStatistic.match_id == Match.id).join(
            Weather, Match.id == Weather.match_id)

        data = query.all()

    df = pd.DataFrame(
        data, columns=stat_numeric_cols + goal_cols + ['is_precipitation'])
    df['is_precipitation'] = df['is_precipitation'].map(
        {False: 'No Precipitation', True: 'Precipitation'})
    
    df['is_precipitation'] = pd.Categorical(
        df['is_precipitation'],
        categories=['No Precipitation', 'Precipitation'], ordered=True)

    # Group 1: Main stats + goals
    df_melt1 = df.melt(
        id_vars='is_precipitation',
        value_vars=group1_cols,
        var_name='Statistic',
        value_name='Value'
    )

    plt.figure(figsize=(18,6))
    sns.boxplot(
        x='Statistic', y='Value',
        hue='is_precipitation',
        data=df_melt1,
        palette={'No Precipitation': 'tomato', 'Precipitation': 'steelblue'}
    )
    plt.xticks(rotation=90)
    plt.title(
        'Match Statistics by Precipitation', fontsize=25)
    plt.xlabel('')
    plt.ylabel('Value')
    sns.despine()
    plt.legend(title='Precipitation')
    plt.tight_layout()
    plt.show()

    # Group 1: Main stats + goals (filtrate)
    filtered_cols = group1_cols[16:-10]
    df_melt1 = df.melt(
        id_vars='is_precipitation',
        value_vars=filtered_cols,
        var_name='Statistic', 
        value_name='Value'
    )

    plt.figure(figsize=(18,6))
    sns.boxplot(
        x='Statistic', y='Value',
        hue='is_precipitation',
        data=df_melt1,
        palette={'No Precipitation': 'tomato', 'Precipitation': 'steelblue'}
    )
    plt.xticks(rotation=90)
    plt.title(
        'Filtered Match Statistics by Precipitation', fontsize=25)
    plt.xlabel('')
    plt.ylabel('Value')
    sns.despine()
    plt.legend(title='Precipitation')
    plt.tight_layout()
    plt.show()

    effect_sizes = {}
    for col in numeric_cols:
        rain_vals = df[df['is_precipitation'] == 'Precipitation'][col].dropna()
        no_rain_vals = df[
            df['is_precipitation'] == 'No Precipitation'][col].dropna()
        effect_sizes[col] = cohens_d(rain_vals, no_rain_vals)

    effect_df = pd.DataFrame.from_dict(
        effect_sizes, orient='index', columns=['Cohen_d'])
    effect_df['Abs_Cohen_d'] = effect_df['Cohen_d'].abs()
    effect_df = effect_df.sort_values('Abs_Cohen_d', ascending=False)

    # Top 15 statistics by effect size
    top_stats = effect_df.head(15)
    print('Top 15 statistics with highest effect size '
          '(Precipitation vs No Precipitation):')
    print(top_stats)

    top15 = top_stats.sort_values('Cohen_d', key=abs, ascending=False).head(15)

    colors = ['tomato' if x < 0 else 'steelblue' for x in top15['Cohen_d']]

    plt.figure(figsize=(10,7))
    sns.barplot(x='Cohen_d', y=top15.index, data=top15, 
                hue=top15.index, palette=colors, 
                legend=False, edgecolor='black', linewidth=0.5)

    plt.axvline(0, color='black', linewidth=1)
    plt.xlabel("Cohen's d (Effect Size, Precipitation vs No Precipitation)")
    plt.ylabel('Statistic')
    plt.title('Top 15 Match Statistics - Diverging Effect of Precipitation', 
              fontsize=25, pad=20)

    # Legend
    red_patch = mpatches.Patch(
        color='tomato', label='Higher with No Precipitation')
    green_patch = mpatches.Patch(
        color='steelblue', label='Higher with Precipitation')
    plt.legend(handles=[green_patch, red_patch], loc='lower right')

    plt.tight_layout()
    plt.show()


def match_statistics_wind_category(Session):
    '''Display the distribution of match statistics for wind_category'''

    stat_numeric_cols = [c.name for c in inspect(MatchStatistic).columns 
                        if c.type.python_type in [int, float] 
                        and c.name not in ['id','match_id']]
    goal_cols = ['home_goals', 'away_goals']

    numeric_cols = stat_numeric_cols + goal_cols 

    exclude_cols = ['home_passes_completed', 'away_passes_completed', 
                    'home_passes_attempted', 'away_passes_attempted', 
                    'home_touches', 'away_touches']

    group1_cols = [c for c in numeric_cols if c not in exclude_cols] 
    group2_cols = [c for c in numeric_cols if c in exclude_cols]     

    # Data Query
    with Session() as session:
        query = session.query(
            *[getattr(MatchStatistic, c) for c in stat_numeric_cols],
            Match.home_goals,
            Match.away_goals,
            Weather.wind_speed_category
        ).join(Match, MatchStatistic.match_id == Match.id).join(
            Weather, Match.id == Weather.match_id)

        data = query.all()

    df = pd.DataFrame(
        data, columns=stat_numeric_cols + goal_cols + ['wind_speed_category'])
    df['wind_speed_category'] = pd.Categorical(
        df['wind_speed_category'], categories=['Low', 'Medium', 'High'], 
        ordered=True)

    # Group 1: Main stats + goals
    df_melt1 = df.melt(id_vars='wind_speed_category', value_vars=group1_cols, 
                       var_name='Statistic', value_name='Value')
    plt.figure(figsize=(16,6))
    sns.boxplot(
        x='Statistic', y='Value', 
        hue='wind_speed_category', 
        data=df_melt1, 
        palette={'Low': '#A8E6A3', 'Medium':'#4CAF50', 'High':'#006400'}
    )
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('Value')
    plt.title('Match Statistics by Wind Speed Category',fontsize = 25)
    plt.legend(title='Wind Speed')
    sns.despine()
    plt.tight_layout()
    plt.show()

    # Group 1: Main stats + goals (filtrate)
    filtered_group1_cols = group1_cols[16:-10]
    df_melt1_filtered = df.melt(
        id_vars='wind_speed_category',
        value_vars=filtered_group1_cols,
        var_name='Statistic',
        value_name='Value'
    )

    plt.figure(figsize=(16,6))
    sns.boxplot(
        x='Statistic', y='Value',
        hue='wind_speed_category',
        data=df_melt1_filtered,
        palette={'Low': '#A8E6A3', 'Medium':'#4CAF50', 'High':'#006400'}
    )
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('Value')
    plt.title('Filtered Match Statistics by Wind Speed Category',fontsize = 25)
    plt.legend(title='Wind Speed')
    sns.despine()
    plt.tight_layout()
    plt.show()

    # Compute Cohen's d for each statistic (High vs Low wind)
    effect_sizes = {}
    for stat in numeric_cols:
        high_vals = df[df['wind_speed_category']=='High'][stat].dropna().values
        low_vals = df[df['wind_speed_category']=='Low'][stat].dropna().values
        effect_sizes[stat] = cohens_d(high_vals, low_vals)

    effect_df = pd.DataFrame.from_dict(
        effect_sizes, orient='index', columns=['Cohen_d'])
    effect_df['Abs_Cohen_d'] = effect_df['Cohen_d'].abs()
    top15_wind_effect = effect_df.sort_values(
        'Abs_Cohen_d', ascending=False).head(15)

    print('Top 15 statistics most affected by wind speed (High vs Low):')
    print(top15_wind_effect)

    # Palette colors
    colors = ['steelblue' if x > 0 else 'tomato' 
              for x in top15_wind_effect['Cohen_d']]

    plt.figure(figsize=(10,7))
    sns.barplot(
        x='Cohen_d',
        y=top15_wind_effect.index,
        hue=top15_wind_effect.index,
        data=top15_wind_effect,
        palette=colors,
        legend=False,
        edgecolor='black',
        linewidth=0.5
    )

    plt.axvline(0, color='black', linewidth=1)
    legend_elements = [
        Patch(facecolor='steelblue', label='Higher with High Wind Speed'),
        Patch(facecolor='tomato', label='Higher with Low Wind Speed')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.xlabel("Cohen's d (Effect Size, High vs Low Wind Speed)")
    plt.ylabel('Statistic')
    plt.title('Top 15 Match Statistics - Diverging Effect of Wind Speed', 
              fontsize=25, pad=20)
    plt.tight_layout()
    plt.show()


def match_statistics_temperature_category(Session):
    '''Display the distribution of match statistics for temperature_category'''

    stat_numeric_cols = [c.name for c in inspect(MatchStatistic).columns 
                        if c.type.python_type in [int, float] 
                        and c.name not in ['id','match_id']]
    goal_cols = ['home_goals', 'away_goals']

    numeric_cols = stat_numeric_cols + goal_cols

    exclude_cols = ['home_passes_completed', 'away_passes_completed', 
                    'home_passes_attempted', 'away_passes_attempted', 
                    'home_touches', 'away_touches']

    group1_cols = [c for c in numeric_cols if c not in exclude_cols]
    group2_cols = [c for c in numeric_cols if c in exclude_cols]      

    # Data Query
    with Session() as session:
        query = session.query(
            *[getattr(MatchStatistic, c) for c in stat_numeric_cols],
            Match.home_goals,
            Match.away_goals,
            Weather.temperature_category
        ).join(Match, MatchStatistic.match_id == Match.id).join(
            Weather, Match.id == Weather.match_id)

        data = query.all()

    df = pd.DataFrame(
        data, columns=stat_numeric_cols + goal_cols + ['temperature_category'])
    df['temperature_category'] = pd.Categorical(
        df['temperature_category'], 
        categories=['Cold','Mild','Hot'], ordered=True)

    # Group 1: Main stats + goals
    df_melt1 = df.melt(id_vars='temperature_category', 
                       value_vars=group1_cols, var_name='Statistic', 
                       value_name='Value')
    plt.figure(figsize=(16,6))
    sns.boxplot(
        x='Statistic', y='Value', 
        hue='temperature_category', 
        data=df_melt1, 
        palette={'Cold':'#FFEDA0', 'Mild':'#FEB24C', 'Hot':'#F03B20'}
    )
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('Value')
    plt.title('Match Statistics by Temperature Category', fontsize = 25)
    plt.legend(title='Temperature')
    sns.despine()
    plt.tight_layout()
    plt.show()

    # Group 1: Main stats + goals (filtrate)
    filtered_group1_cols = group1_cols[16:-10]
    df_melt1_filtered = df.melt(
        id_vars='temperature_category',
        value_vars=filtered_group1_cols,
        var_name='Statistic',
        value_name='Value'
    )

    plt.figure(figsize=(16,6))
    sns.boxplot(
        x='Statistic', y='Value',
        hue='temperature_category',
        data=df_melt1_filtered,
        palette={'Cold':'#FFEDA0', 'Mild':'#FEB24C', 'Hot':'#F03B20'}
    )
    plt.xticks(rotation=90)
    plt.xlabel('')
    plt.ylabel('Value')
    plt.title('Filtered Match Statistics by Temperature Category', fontsize=25)
    plt.legend(title='Temperature')
    sns.despine()
    plt.tight_layout()
    plt.show()

    # Compute Cohen's d for each statistic (Hot vs Cold)
    effect_sizes_temp = {}
    for stat in numeric_cols:
        hot_vals = df[df['temperature_category']=='Hot'][stat].dropna().values
        cold_vals = df[df['temperature_category']=='Cold'][stat].dropna().values
        effect_sizes_temp[stat] = cohens_d(hot_vals, cold_vals)
        
    effect_df_temp = pd.DataFrame.from_dict(
        effect_sizes_temp, orient='index', columns=['Cohen_d'])
    effect_df_temp['Abs_Cohen_d'] = effect_df_temp['Cohen_d'].abs()
    top15_temp_effect = effect_df_temp.sort_values(
        'Abs_Cohen_d', ascending=False).head(15)

    print('Top 15 statistics most affected by temperature (Hot vs Cold):')
    print(top15_temp_effect)

    # Palette colors
    colors = ['steelblue' if x > 0 else 'tomato' 
              for x in top15_temp_effect['Cohen_d']]

    plt.figure(figsize=(10,7))
    sns.barplot(
        x='Cohen_d',
        y=top15_temp_effect.index,
        hue=top15_temp_effect.index,
        data=top15_temp_effect,
        palette=colors,
        legend=False,
        edgecolor='black',
        linewidth=0.5
    )

    plt.axvline(0, color='black', linewidth=1)
    legend_elements = [
        Patch(facecolor='steelblue', label='Higher with Hot'),
        Patch(facecolor='tomato', label='Higher with Cold')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.xlabel("Cohen's d (Effect Size, Hot vs Cold)")
    plt.ylabel('Statistic')
    plt.title('Top 15 Match Statistics - Diverging Effect of Temperature', 
              fontsize=25, pad=20)
    plt.tight_layout()
    plt.show()


# Definition of functions (RQ2)
def distribution_season(Session):
    '''Display the distribution of matches per season'''

    # Data query
    with Session() as session:
        matches_data = session.query(Match.id, Match.season).all()

    df_matches = pd.DataFrame(matches_data, columns=['match_id', 'season'])

    # Count number of matches per season
    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
    counts = df_matches['season'].value_counts().reindex(season_order)

    # Create barplot with mixed colors
    plt.figure(figsize=(10,6))
    colors = sns.color_palette('pastel', len(season_order))
    bars = plt.bar(counts.index, counts.values, color=colors, 
                   edgecolor='black', linewidth=1.2)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, height + 2, f'{int(height)}', 
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # Titles and labels
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Number of Matches', fontsize=12)
    plt.title('Number of Matches by Season',
              fontsize=19, fontweight='bold', pad=20)
    plt.xticks(fontsize=11)
    plt.yticks(fontsize=11)
    plt.ylim(0, max(counts.values)*1.1) 

    plt.show()


def distribution_matches_season(Session):
    '''Distribution of match outcomes across seasons using engine'''

    with Session() as session:
        matches_data = session.query(
            Match.home_goals,
            Match.away_goals,
            Match.season
        ).all()

    df = pd.DataFrame(
        matches_data, columns=['home_goals', 'away_goals', 'season'])
    df['result'] = df.apply(get_result, axis=1)

    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
    result_order = ['Home Win', 'Draw', 'Away Win']

    counts = (
        df.groupby(['season', 'result'])
          .size()
          .unstack(fill_value=0)
          .reindex(season_order)
    )
    percentages = counts.div(counts.sum(axis=1), axis=0) * 100

    # Plot stacked bar chart
    percentages[result_order].plot(
        kind='bar',
        stacked=True,
        figsize=(10,6),
        edgecolor='black',
        color=['#3B6FB6', '#4CAF50', '#FF8C1A']
    )

    for i, season in enumerate(percentages.index):
        cumulative = 0
        for result in result_order:
            height = percentages.loc[season, result]
            if height > 0:
                plt.text(
                    i,
                    cumulative + height / 2,
                    f'{height:.1f}%',
                    ha='center',
                    va='center',
                    fontsize=10,
                    color='black',
                    fontweight='bold'
                )
                cumulative += height

    plt.ylabel('Percentage of Matches', fontsize=12)
    plt.xlabel('Season', fontsize=12)
    plt.title('Distribution of Match Outcomes by Season',
              fontsize=19, fontweight='bold', pad=20)
    plt.legend(title='Result', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(fontsize=12, rotation=0)
    plt.tight_layout()
    plt.show()


def chi_square_test_season(Session):
    '''Chi-square test between season and match outcome using engine'''

    with Session() as session:
        matches_data = session.query(
            Match.home_goals,
            Match.away_goals,
            Match.season
        ).all()

    df = pd.DataFrame(
        matches_data, columns=['home_goals', 'away_goals', 'season'])
    df['result'] = df.apply(get_result, axis=1)

    # Contingency table
    contingency_table = pd.crosstab(df['season'], df['result'])
    print('Contingency Table:')
    print(contingency_table)

    # Chi-square test
    chi2, p, dof, expected = chi2_contingency(contingency_table)
    print('\nChi-Square Statistic:', chi2)
    print('Degrees of Freedom:', dof)
    print('P-value:', p)

    print('\nExpected Frequencies if independent:')
    print(pd.DataFrame(expected,
                       index=contingency_table.index,
                       columns=contingency_table.columns))

    if p < 0.05:
        print('\nConclusion: Match outcomes are significantly influenced '
              'by seasonality')
    else:
        print('\nConclusion: No significant seasonal effect on match outcomes')


def match_outcomes_season(Session):
    '''Evaluate the impact of seasonality on match outcomes'''

    distribution_season(Session)
    distribution_matches_season(Session)
    chi_square_test_season(Session)


def epsilon_squared_kw(H, n, k):
    '''Compute Epsilon Squared for Kruskal-Wallis test'''
    return (H - k + 1) / (n - k)


def seasonal_effect_ranking(engine, alpha=0.05):
    '''Rank match and match_statistics metrics by seasonal effect
    using Kruskal-Wallis and Epsilon Squared'''

    inspector = inspect(engine)
    results = []

    goal_cols = ['home_goals', 'away_goals']
    
    stats_cols = [c['name'] for c in inspector.get_columns('match_statistics')
                  if c['name'] not in ['id', 'match_id']]

    for metric in goal_cols:
        query = f"""
        SELECT {metric}, season
        FROM matches
        WHERE {metric} IS NOT NULL
        """
        df = pd.read_sql(query, engine)

        if df['season'].nunique() < 2:
            continue

        groups = [df[df['season'] == s][metric].values
                  for s in df['season'].unique()]
        
        H, p = kruskal(*groups)
        eps = epsilon_squared_kw(H, len(df), len(groups))

        results.append({
            'metric': metric,
            'H': H,
            'p_value': p,
            'epsilon_sq': eps
        })

    for metric in stats_cols:
        query = f"""
        SELECT m.season, ms.{metric}
        FROM matches m
        JOIN match_statistics ms ON m.id = ms.match_id
        WHERE ms.{metric} IS NOT NULL
        """
        df = pd.read_sql(query, engine)

        if df['season'].nunique() < 2:
            continue

        groups = [
            df[df['season'] == s][metric].values
            for s in df['season'].unique()
        ]

        H, p = kruskal(*groups)
        eps = epsilon_squared_kw(H, len(df), len(groups))

        results.append({
            'metric': metric,
            'H': H,
            'p_value': p,
            'epsilon_sq': eps
        })

    result_df = pd.DataFrame(results)
    result_df = result_df[result_df['p_value'] < alpha]
    result_df = result_df.sort_values(
        'epsilon_sq', ascending=False
    ).reset_index(drop=True)

    return result_df


def seasonal_summary_stats(engine, ranking_df, top_n=10):
    '''Display summary statistics for top N seasonal effect metrics'''

    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
    top_metrics = ranking_df['metric'].head(top_n).tolist()

    summary_rows = []

    for metric in top_metrics:
        if metric in ['home_goals', 'away_goals']:
            query = f"""
            SELECT {metric}, season
            FROM matches
            WHERE {metric} IS NOT NULL
            """
        else:
            query = f"""
            SELECT m.season, ms.{metric}
            FROM matches m
            JOIN match_statistics ms ON m.id = ms.match_id
            WHERE ms.{metric} IS NOT NULL
            """

        df = pd.read_sql(query, engine)
        df['season'] = pd.Categorical(
            df['season'], categories=season_order, ordered=True)

        stats = df.groupby('season', observed=True)[metric].agg(['mean','std'])

        row = {
            season: 
            f"{stats.loc[season, 'mean']:.2f} ± {stats.loc[season, 'std']:.2f}"
            for season in season_order if season in stats.index}
        
        row['metric'] = metric
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows).set_index('metric')
    return summary


def seasonal_heatmap(engine, ranking_df, top_n=9):
    '''Plot heatmap of seasonal effects for top N metrics'''

    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
    top_metrics = ranking_df['metric'].head(top_n).tolist()

    heatmap_data = []

    for metric in top_metrics:
        if metric in ['home_goals', 'away_goals']:
            query = f"""
            SELECT {metric}, season
            FROM matches
            WHERE {metric} IS NOT NULL
            """
        else:
            query = f"""
            SELECT m.season, ms.{metric}
            FROM matches m
            JOIN match_statistics ms ON m.id = ms.match_id
            WHERE ms.{metric} IS NOT NULL
            """

        df = pd.read_sql(query, engine)
        df['season'] = pd.Categorical(
            df['season'], categories=season_order, ordered=True)

        seasonal_means = df.groupby(
            'season', observed=True)[metric].mean().reindex(season_order)
        norm_values = (
            seasonal_means - seasonal_means.mean()) / seasonal_means.std()
        heatmap_data.append(norm_values)

    heatmap_df = pd.DataFrame(
        heatmap_data, index=top_metrics, columns=season_order)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        heatmap_df,
        annot=True,
        fmt='.2f',
        cmap='coolwarm',
        center=0,
        linewidths=0.5,
        cbar_kws={'label': 'Normalized Mean (z-score)'}
    )
    plt.title(f'Seasonal Effects on Top {top_n} Match Statistics', 
              fontsize=19, pad=20)
    plt.xlabel('Season', fontsize=12)
    plt.ylabel('Metric', fontsize=12)
    plt.tight_layout()
    plt.show()


def match_statistics_season_kruskal_wallis(engine):
    '''Rank match statistics and display summary statistics'''

    seasonal_ranking = seasonal_effect_ranking(engine)
    seasonal_ranking = seasonal_ranking.head(15)
    print('Top 15 statistics most affected by seasonality:')
    display(seasonal_ranking)
    print()
    summary_stats = seasonal_summary_stats(engine, seasonal_ranking)
    display(summary_stats)
    print()
    seasonal_heatmap(engine, seasonal_ranking)


def posthoc_seasonal(Session, metric):
    '''Perform Dunn's post-hoc test for a given metric across seasons'''

    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']

    with Session() as session:
        if metric in ['home_goals', 'away_goals']:
            rows = (
                session.query(
                    getattr(Match, metric),
                    Match.season
                )
                .filter(getattr(Match, metric).isnot(None))
                .all()
            )
        else:
            rows = (
                session.query(
                    getattr(MatchStatistic, metric),
                    Match.season
                )
                .join(MatchStatistic, MatchStatistic.match_id == Match.id)
                .filter(getattr(MatchStatistic, metric).isnot(None))
                .all()
            )

    df = pd.DataFrame(rows, columns=[metric, 'season'])
    df = df[df['season'].isin(season_order)]

    dunn = posthoc_dunn(
        df,
        val_col=metric,
        group_col='season',
        p_adjust='bonferroni'
    )

    dunn = dunn.loc[season_order, season_order]
    return dunn


def combined_heatmaps(Session, ranking_df, top_n=9):
    '''Plot combined heatmaps for top N seasonal effect metrics'''
    
    top_metrics = ranking_df['metric'].head(top_n).tolist()
    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']

    n = len(top_metrics)
    rows = int(np.ceil(n / 3))
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))

    for ax, metric in zip(axes.flatten(), top_metrics):
        dunn = posthoc_seasonal(Session, metric)

        sns.heatmap(
            dunn,
            annot=True,
            cmap='coolwarm_r',
            fmt='.3f',
            linewidths=0.5,
            ax=ax,
            cbar=False
        )
        ax.set_title(metric, fontsize=18)
        ax.set_xlabel('Season', fontsize=12)
        ax.set_ylabel('Season', fontsize=12)

    for ax in axes.flatten()[n:]:
        ax.axis('off')

    plt.suptitle(f'Seasonal Trends for Top {top_n} Match Statistics', 
              fontsize=28, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()


def facet_seasonal_trends(Session, ranking_df, top_n=9):
    '''Plot seasonal trends for top N seasonal effect metrics'''

    season_order = ['Summer', 'Autumn', 'Winter', 'Spring']
    top_metrics = ranking_df['metric'].head(top_n).tolist()

    n = len(top_metrics)
    rows = int(np.ceil(n / 3))
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(18, 5*rows))

    with Session() as session:
        for ax, metric in zip(axes.flatten(), top_metrics):
            if metric in ['home_goals', 'away_goals']:
                rows_data = (
                    session.query(
                        getattr(Match, metric),
                        Match.season
                    )
                    .filter(getattr(Match, metric).isnot(None))
                    .all()
                )
            else:
                rows_data = (
                    session.query(
                        getattr(MatchStatistic, metric),
                        Match.season
                    )
                    .join(MatchStatistic, MatchStatistic.match_id == Match.id)
                    .filter(getattr(MatchStatistic, metric).isnot(None))
                    .all()
                )

            df = pd.DataFrame(rows_data, columns=[metric, 'season'])
            df = df[df['season'].isin(season_order)]
            df[metric] = pd.to_numeric(df[metric], errors='coerce')

            summary = (df.groupby(
                'season')[metric].agg(['mean', 'std']).reindex(season_order))

            # Line plot
            ax.plot(
                season_order,
                summary['mean'],
                marker='o',
                linewidth=2,
                markersize=6,
                color='#1f77b4'
            )

            ax.set_title(metric, fontsize=18)
            ax.grid(alpha=0.3)
            ax.set_xlabel('Season', fontsize=12)
            ax.set_ylabel(
                f"{metric.replace('_', ' ').title()} Mean", fontsize=12)
            
        for ax in axes.flatten()[n:]:
            ax.axis('off')

    plt.suptitle(f'Seasonal Trends for Top {top_n} Match Statistics', 
              fontsize=28, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.3)
    plt.show()


def match_statistics_season_dunn(Session, engine):
    '''Plot seasonal trends for top 9 metrics'''

    seasonal_ranking = seasonal_effect_ranking(engine)
    seasonal_ranking = seasonal_ranking.head(9)
    combined_heatmaps(Session, seasonal_ranking)
    facet_seasonal_trends(Session, seasonal_ranking)