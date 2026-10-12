# Import useful libraries
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from sqlalchemy import ForeignKey, CheckConstraint, UniqueConstraint
from sqlalchemy.orm import declarative_base, relationship
from geoalchemy2 import Geometry


# Define the structure of the tables for the database
Base = declarative_base()

class Competition(Base):
        __tablename__ = 'competitions'
        id = Column(Integer, primary_key=True)
        comp_name = Column(String, nullable=False)
        season_year = Column(String)

        matches = relationship('Match', back_populates='competition')


class Team(Base):
    __tablename__ = 'teams'
    id = Column(Integer, primary_key=True)
    team_name = Column(String, nullable=False)

    home_matches = relationship(
        'Match', back_populates='home_team_rel', 
        foreign_keys='Match.home_team_id')
    away_matches = relationship(
        'Match', back_populates='away_team_rel', 
        foreign_keys='Match.away_team_id')


class Stadium(Base):
    __tablename__ = 'stadiums'
    id = Column(Integer, primary_key=True)
    stadium_name = Column(String, nullable=False)
    city = Column(String)
    country = Column(String)
    geometry = Column(Geometry('POINT', srid=4326))
    timezone = Column(String)

    matches = relationship('Match', back_populates='stadium')


class Match(Base):
    __tablename__ = 'matches'
    id = Column(Integer, primary_key=True)
    competition_id = Column(Integer, ForeignKey('competitions.id'))
    stadium_id = Column(Integer, ForeignKey('stadiums.id'))
    home_team_id = Column(Integer, 
                          ForeignKey('teams.id', ondelete='RESTRICT'))
    away_team_id = Column(Integer, 
                          ForeignKey('teams.id', ondelete='RESTRICT'))
    date = Column(DateTime)
    season = Column(String)
    time_slot = Column(String)
    home_goals = Column(Integer)
    away_goals = Column(Integer)
    attendance = Column(Integer)

    competition = relationship('Competition', back_populates='matches')
    stadium = relationship('Stadium', back_populates='matches')
    home_team_rel = relationship('Team', back_populates='home_matches', 
                                 foreign_keys=[home_team_id])
    away_team_rel = relationship('Team', back_populates='away_matches', 
                                 foreign_keys=[away_team_id])
    weather = relationship('Weather', back_populates='match', uselist=False)
    statistics = relationship('MatchStatistic', 
                              back_populates='match', uselist=False)

    __table_args__ = (
        CheckConstraint(
            'home_goals >= 0 AND away_goals >= 0', 
            name='check_goals_positive'),
        UniqueConstraint('stadium_id', 'date', name='unique_stadium_date'),
        CheckConstraint("season IN "
            "('Spring', 'Summer', 'Autumn', 'Winter')", 
            name='check_season'),
        CheckConstraint("time_slot IN "
            "('Midday','Afternoon','Evening','Night')", 
            name='check_time_slot')
        )


class MatchStatistic(Base):
    __tablename__ = 'match_statistics'
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, 
                      ForeignKey('matches.id', ondelete='CASCADE'),
                      unique=True)
    home_xg = Column(Float)
    away_xg = Column(Float)
    home_possession = Column(Float)
    away_possession = Column(Float)
    home_passes_completed = Column(Integer)
    away_passes_completed = Column(Integer)
    home_passes_attempted = Column(Integer)
    away_passes_attempted = Column(Integer)
    home_shots_on_target = Column(Integer)
    away_shots_on_target = Column(Integer)
    home_shots_total = Column(Integer)
    away_shots_total = Column(Integer)
    home_saves_made = Column(Integer)
    away_saves_made = Column(Integer)
    home_saves_faced = Column(Integer)
    away_saves_faced = Column(Integer)
    home_yellow_cards = Column(Integer)
    away_yellow_cards = Column(Integer)
    home_red_cards = Column(Integer)
    away_red_cards = Column(Integer)
    home_fouls = Column(Integer)
    away_fouls = Column(Integer)
    home_corners = Column(Integer)
    away_corners = Column(Integer)
    home_crosses = Column(Integer)
    away_crosses = Column(Integer)
    home_touches = Column(Integer)
    away_touches = Column(Integer)
    home_tackles = Column(Integer)
    away_tackles = Column(Integer)
    home_interceptions = Column(Integer)
    away_interceptions = Column(Integer)
    home_aerials_won = Column(Integer)
    away_aerials_won = Column(Integer)
    home_clearances = Column(Integer)
    away_clearances = Column(Integer)
    home_offsides = Column(Integer)
    away_offsides = Column(Integer)
    home_goal_kicks = Column(Integer)
    away_goal_kicks = Column(Integer)
    home_throw_ins = Column(Integer)
    away_throw_ins = Column(Integer)
    home_long_balls = Column(Integer)
    away_long_balls = Column(Integer)       

    match = relationship('Match', back_populates='statistics')


class Weather(Base):
    __tablename__ = 'weather_conditions'
    id = Column(Integer, primary_key=True)
    match_id = Column(Integer, 
                      ForeignKey('matches.id', ondelete='CASCADE'),
                      unique=True)
    temperature = Column(Float)
    precipitation = Column(Float)
    wind_speed = Column(Float)
    temperature_category = Column(String)
    is_precipitation = Column(Boolean)
    wind_speed_category = Column(String)

    match = relationship('Match', back_populates='weather')

    __table_args__ = (
    CheckConstraint("temperature_category IN ('Cold', 'Mild', 'Hot')", 
                    name='check_temperature_category'),
    CheckConstraint(
        "is_precipitation IN (TRUE, FALSE)", name='check_is_precipitation'),
    CheckConstraint("wind_speed_category IN ('Low', 'Medium', 'High')", 
                    name='check_wind_speed_category'),
    )