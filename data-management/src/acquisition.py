# Import useful libraries
import pandas as pd
import re
from tqdm import tqdm

from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager


# Definition of functions
def create_driver(headless=False):
    '''Create a driver for Chrome'''

    # Configure options for Chrome
    options = Options()

    if headless:
        options.add_argument('--headless=new')
        options.add_argument('--window-size=1920,1080')
    else:
        options.add_argument('--start-maximized')
    
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--no_sandbox')
    options.add_argument(
        'user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
        'AppleWebKit/537.86 (KHTML, like Gecko) Chrome/120 Safari/537.36'
    )

    # Install driver and define Chrome instance
    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)

    return driver


def close_cookie_banner(driver, timeout=5):
    '''Try to close the cookie banner if present'''

    wait = WebDriverWait(driver, timeout)
    try: 
        # Obtain the cookie banner and click the button
        btn = wait.until(EC.element_to_be_clickable((
            By.CSS_SELECTOR, '.osano-cm-button')))
        btn.click()

    except Exception as e:
        pass


def fetch_competitions(driver):
    '''Return the list of the big five competitions'''

    url = 'https://fbref.com/en/comps'

    try:
        driver.get(url)
        wait = WebDriverWait(driver, 5)

        close_cookie_banner(driver)

        # Search for the CSS selector targeting the table with all competitions
        try:
            wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, '#comps_club')))

        except Exception as e:
            print('CSS selector not found: ', e)
            wait.until(EC.presence_of_element_located((By.TAG_NAME, 'body')))

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Retrieve the table with all competitions
        table = soup.select_one('#comps_club')
        if not table:
            print('No table found with the specific CSS selector')
            return []
        
        # Select the big five competitions
        rows = table.select('tbody tr')
        rows = rows[:-1]

        competitions = []

        for row in rows:
            a = row.select_one('th a')
            if not a:
                continue
            
            name = a.text.strip()
            name = name.replace('Fußball-', '').strip()
            
            href = a.get('href')
            if not href:
                continue

            if href.startswith('/'):
                href = 'https://fbref.com' + href
            
            competitions.append({'name': name, 'url': href})
          
        return competitions

    except Exception as e:
        print('Error in fetch competitions: ', e)
        return []


def fetch_competition_seasons(driver, competition_url):
    '''Return the list of the last three seasons for a competition'''

    try:
        driver.get(competition_url)
        wait = WebDriverWait(driver, 5)

        # Define the CSS selector targeting the table with all seasons
        try:
            wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, '#seasons')))

        except Exception as e:
            print('Seasons table not found: ', e)
            return []

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Retrieve the table with all seasons
        table = soup.select_one('#seasons')
        if not table:
            print('No table found with the specific CSS selector')

        # Select the last three complete seasons
        rows = table.select('tbody tr')
        rows = rows[1:4]
        
        seasons = []
        
        for row in rows:
            a = row.select_one('th a')
            if not a:
                continue

            name = a.text.strip()
            
            href = a.get('href')
            if not href:
                continue

            if href.startswith('/'):
                href = 'https://fbref.com' + href

            seasons.append({'name': name, 'url': href})

        return seasons        

    except Exception as e:
        print('Error in fetch seasons: ', e)
        return []
    

def get_scores_fixtures_url(driver, season_url):
    '''Return the list of correct url for matches table of a season'''

    try:
        driver.get(season_url)
        wait = WebDriverWait(driver, 5)

        # Define the CSS selector targeting the navigation menu
        try:
            wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, '#inner_nav')))
        
        except Exception as e:
            print('Navigation menu not found: ', e)
            return None
        
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        nav = soup.select_one('#inner_nav')
        if not nav:
            return None
        
        for a in nav.select('a'):
            if a.text.strip() == 'Scores & Fixtures':
                href = a.get('href')
                
                if href.startswith('/'):
                    href = 'https://fbref.com' + href
                
                return href
            
    except Exception as e:
        print('Scores & Fixtures not found: ', e)
        return None
    

def fetch_season_matches(driver, scores_url):
    '''Return the list of matches for a season of a competition'''

    try:
        driver.get(scores_url)
        wait = WebDriverWait(driver, 5)

        # Define the CSS selector targeting the table with all matches
        try:
            wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, '#all_sched')))
        
        except Exception as e:
            print('Matches table non found: ', e)
            return []
        
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Retrieve the table with all matches
        table = soup.select_one('#all_sched')
        if not table:
            print('No table found with the specific CSS selector')

        matches = []
        for row in table.select('tbody tr'):
            gameweek = row.select_one(
                "td[data-stat='gameweek'], th[data-stat='gameweek']")
            if not gameweek or not gameweek.text.strip().isdigit():
                continue

            a = row.select_one("td[data-stat='match_report'] a")
            if not a:
                continue

            href = a.get('href')
            if href.startswith('/'):
                href = 'https://fbref.com' + href

            home = row.select_one("td[data-stat='home_team']").text.strip()
            away = row.select_one("td[data-stat='away_team']").text.strip()

            matches.append({'home_team': home, 'away_team': away, 'url': href})

        # Check for duplicated matches
        seen = set()
        unique_matches = []
        for m in matches:
            key = (m['home_team'], m['away_team'], m['url'])
            if key not in seen:
                seen.add(key)
                unique_matches.append(m)

        return unique_matches        
    
    except Exception as e:
        print('Error in fetch matches: ', e)
        return []
    

def parse_main_statistics(soup, match):
    '''Parse the main statistics of a match'''

    stats_list = ['goals', 'xg', 'possession', 'passes_completed', 
        'passes_attempted', 'shots_on_target', 'shots_total', 
        'saves_made', 'saves_faced']
    fixed_keys = ['date', 'time', 'stadium_name', 'city', 'attendance']

    for key in fixed_keys:
        match[key] = None

    for stat in stats_list:
        match[f'home_{stat}'] = None
        match[f'away_{stat}'] = None

    # Extract score, xG, date, time and stadium
    scorebox = soup.select_one('.scorebox')
    if scorebox:
        scores = scorebox.select('.score')
        if len(scores) > 0:
            home_raw = scores[0].text.strip()
            home_clean = ''.join(c for c in home_raw if c.isdigit())
            match['home_goals'] = int(home_clean) if home_clean else None
        else:
            match['home_goals'] = None 

        if len(scores) > 1:
            away_raw = scores[1].text.strip()
            away_clean = ''.join(c for c in away_raw if c.isdigit())
            match['away_goals'] = int(away_clean) if away_clean else None
        else:
            match['away_goals'] = None 

        xgs = scorebox.select('.score_xg')
        match['home_xg'] = float(xgs[0].text.strip()) if len(xgs) > 0 else None
        match['away_xg'] = float(xgs[1].text.strip()) if len(xgs) > 1 else None
        
        meta = soup.select_one('.scorebox_meta')
        if meta:
            date_span = meta.select_one('span.venuetime')
            if date_span:
                match['date'] = date_span.get('data-venue-date', None)
                match['time'] = date_span.get('data-venue-time', None)
            else:
                match['date'] = match['time'] = None

            venue_div = meta.find('strong', string='Venue')
            if venue_div:
                venue_small = venue_div.find_next_sibling('small')
                if venue_small:
                    venue_text = venue_small.get_text(strip=True)
                    if ',' in venue_text:
                        match['stadium_name'], match['city'] = [
                        x.strip() for x in venue_text.split(',', 1)]
                    else:
                        match['stadium_name'] = venue_text
                        match['city'] = None
                else:
                    match['stadium_name'] = match['city'] = None
            else:
                match['stadium_name'] = match['city'] = None

            attendance_div = meta.find('strong', string='Attendance')
            if attendance_div:
                att_small = attendance_div.find_next_sibling('small')
                if att_small:
                    att_text = att_small.get_text(strip=True).replace(',', '')
                    match['attendance'] = (int(att_text) 
                                           if att_text.isdigit() else None)
                else:
                    match['attendance'] = None
            else:
                match['attendance'] = None
        else:
            match['date'] = match['time'] = None
            match['stadium_name'] = match['city'] = None
            match['attendance'] = None

    # Extract possession, passes, shots and saves
    team_table = soup.select_one('#team_stats table')
    if team_table:
        rows = team_table.select('tr')
        current_stat = None

        for row in rows:
            th = row.select_one('th')
            tds = row.select('td')
                
            if th and not tds:
                current_stat = th.get_text(strip=True).lower().replace(' ', '_')
                continue
                       
            if current_stat == 'possession' and len(tds) >= 2:
                home_div = tds[0].select_one('strong')
                away_div = tds[1].select_one('strong')
                if home_div and away_div:
                    try:
                        match['home_possession'] = (
                            float(home_div.get_text(strip=True)
                                  .replace('%', '')) / 100)
                        match['away_possession'] = (
                            float(away_div.get_text(strip=True)
                                  .replace('%', '')) / 100)
                    except:
                        match['home_possession'] = None
                        match['away_possession'] = None
                else:
                    match['home_possession'] = None
                    match['away_possession'] = None
                continue

            if current_stat in ['passing_accuracy', 'shots_on_target', 
                                      'saves'] and len(tds) >= 2:
                for stat_name, key_home, key_away in [
                    ('passing_accuracy', (
                        'home_passes_completed', 'home_passes_attempted'), 
                        ('away_passes_completed', 'away_passes_attempted')),
                    ('shots_on_target', (
                        'home_shots_on_target', 'home_shots_total'), 
                        ('away_shots_on_target', 'away_shots_total')),
                    ('saves', ('home_saves_made', 'home_saves_faced'), 
                        ('away_saves_made', 'away_saves_faced'))
                ]:
                    if current_stat == stat_name:
                        home_match = (
                            re.search(r'(\d+)\s*of\s*(\d+)', tds[0].text))
                        away_match = (
                            re.search(r'(\d+)\s*of\s*(\d+)', tds[1].text))

                        match[key_home[0]] = (
                            int(home_match.group(1)) if home_match else None)
                        match[key_home[1]] = (
                            int(home_match.group(2)) if home_match else None)
                        
                        match[key_away[0]] = (
                            int(away_match.group(1)) if away_match else None)
                        match[key_away[1]] = (
                            int(away_match.group(2)) if away_match else None)

    return match


def parse_cards(soup, match):
    '''Count the number of yellow and red cards of a match'''
    
    # Track cards per team
    match['home_yellow_cards'] = match['away_yellow_cards'] = None
    match['home_red_cards'] = match['away_red_cards'] = None

    events = soup.select('#events_wrap .event')
    if events:
        match['home_yellow_cards'] = match['away_yellow_cards'] = 0
        match['home_red_cards'] = match['away_red_cards'] = 0

        for ev in events:
            icon_div = ev.select_one('.event_icon')
            if not icon_div:
                continue

            classes = icon_div.get('class', [])
            if 'a' in ev.get('class', []):
                team = 'home'
            elif 'b' in ev.get('class', []):
                team = 'away'
            else:
                continue

            # Count the number of cards per each team
            key_y = f'{team}_yellow_cards'
            key_r = f'{team}_red_cards'
            if 'yellow_card' in classes and 'yellow_red_card' not in classes:
                match[key_y] += 1
            elif 'red_card' in classes and 'yellow_red_card' not in classes:
                match[key_r] += 1
            elif 'yellow_red_card' in classes:
                match[key_y] += 1
                match[key_r] += 1

    return match


def parse_secondary_statistics(soup, match):
    '''Parse the secondary statistics of a match'''

    secondary_stats_list = [ 'fouls', 'corners', 'crosses', 'touches', 
            'tackles', 'interceptions', 'aerials_won', 'clearances', 
            'offsides', 'goal_kicks', 'throw_ins', 'long_balls']

    for s in secondary_stats_list:
        match[f'home_{s}'] = None
        match[f'away_{s}'] = None

    # Extract other match statistics    
    extra_div = soup.select_one('#team_stats_extra')
    if extra_div:
        groups = extra_div.select('div > div')
        if groups:
            for group in groups:
                divs = group.find_all('div')
                for i in range(0, len(divs), 3):
                    try:
                        stat_name = divs[i+1].text.strip().lower().replace(
                            ' ', '_')
                    except:
                        stat_name = None
                        
                    try:
                        home_val = int(divs[i].text.strip())
                    except:
                        home_val = None

                    try:
                        away_val = int(divs[i+2].text.strip())
                    except:
                        away_val = None
                        
                    if stat_name:
                        match[f'home_{stat_name}'] = home_val
                        match[f'away_{stat_name}'] = away_val

    return match


def fetch_match_stats(driver, match):
    '''Return match result, xG, date, time, stadium, ball possession,
    shots (on target and total), passes (completed and attempted), 
    saves (made and faced), cards, additional numeric stats'''

    try:
        driver.get(match['url'])
        wait = WebDriverWait(driver, 5)

        # Define the CSS selector targeting the container of cards and stats
        try:
            wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, '.scorebox')))
        except Exception as e:
            print('Scorebox section not found: ', e)

        try:
            wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, '#events_wrap')))
        except Exception as e:
            print('Events container not found: ', e)

        try:
            wait.until(EC.presence_of_element_located((
                By.CSS_SELECTOR, '#team_stats')))
        except Exception as e:
            print('Team stats table not found: ', e)

        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Extract score, xG, date, time, stadium, 
        # possession, passes, shots and saves
        match = parse_main_statistics(soup, match)

        # Extract the number of cards
        match = parse_cards(soup, match)

        # Extract other statistics
        match = parse_secondary_statistics(soup, match)

    except Exception as e:
        print(f"Error in {match['home_team']} vs {match['away_team']}: {e}")

    return match


def run_scraper():
    '''Execute the overall workflow about the scraper'''
    
    # Create the driver
    driver = create_driver()

    try:
        # Extract the list of competitions
        competitions = fetch_competitions(driver)

        # Extract the list of seasons for each competition
        all_matches = []
        for competition in competitions:
            seasons = fetch_competition_seasons(driver, competition['url'])
            if not seasons:
                print(f"No season found ({competition['name']})")
                continue

            # Extract the list of matches for each season
            for season in seasons:
                scores_url = get_scores_fixtures_url(driver, season['url'])
                matches = fetch_season_matches(driver, scores_url)

                if not matches:
                    print(f'No matches found '
                        f"({competition['name']} {season['name']})")
                    continue

                # Extract available statistics for each match
                for m in tqdm(matches, desc=
                            f"Processing {competition['name']} "
                            f"{season['name']}"):
                    match_data = fetch_match_stats(driver, m)
                    match_data['comp_name'] = competition['name']
                    match_data['season_year'] = season['name']
                    match_data.pop('url', None)
                    all_matches.append(match_data)
    finally:
        # Close the driver
        driver.quit()

    # Create the overall DataFrame
    all_matches_df = pd.DataFrame(all_matches)
    cols = ['comp_name', 'season_year'] + [
            c for c in all_matches_df.columns 
            if c not in ['comp_name', 'season_year']]
    all_matches_df = all_matches_df[cols]
    
    return all_matches_df