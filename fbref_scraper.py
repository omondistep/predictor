"""
FBref squad stats scraper.

Scrapes xG/xGA/possession/passing/defensive stats for all teams in a league.
Uses BeautifulSoup first, falls back to Selenium for dynamic content.
Stores data in DB table `fbref_squad_stats` for the current season.
"""

import re
import time
import sqlite3
from datetime import datetime
from typing import Optional

import cloudscraper
from bs4 import BeautifulSoup

# ── Selenium imports (lazy) ──────────────────────────────────────────────
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.chrome.service import Service as ChromeService
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    HAS_SELENIUM = True
except ImportError:
    HAS_SELENIUM = False

# ── FBref league config ──────────────────────────────────────────────────
FBREF_LEAGUES = {
    "9":  {"name": "Premier League", "country": "ENG", "slug": "9/Premier-League-Stats"},
    "12": {"name": "La Liga", "country": "ESP", "slug": "12/La-Liga-Stats"},
    "13": {"name": "Bundesliga", "country": "GER", "slug": "13/Bundesliga-Stats"},
    "11": {"name": "Serie A", "country": "ITA", "slug": "11/Serie-A-Stats"},
    "16": {"name": "Ligue 1", "country": "FRA", "slug": "16/Ligue-1-Stats"},
    "23": {"name": "Eredivisie", "country": "NED", "slug": "23/Eredivisie-Stats"},
    "32": {"name": "Primeira Liga", "country": "POR", "slug": "32/Primeira-Liga-Stats"},
    "37": {"name": "Belgian Pro League", "country": "BEL", "slug": "37/Belgian-Pro-League-Stats"},
    "8":  {"name": "Champions League", "country": "EUR", "slug": "8/Champions-League-Stats"},
    "24": {"name": "Eredivisie Vrouwen", "country": "NED", "slug": "24/Eredivisie-Vrouwen-Stats"},
}

# ── Team name normalization (Forebet → FBref) ────────────────────────────
# Comprehensive mapping: Forebet name → FBref slug fragment
TEAM_NAME_MAP = {
    # England
    "Arsenal": "Arsenal",
    "Aston Villa": "Aston-Villa",
    "Bournemouth": "Bournemouth",
    "Brentford": "Brentford",
    "Brighton": "Brighton-and-Hove-Albion",
    "Chelsea": "Chelsea",
    "Crystal Palace": "Crystal-Palace",
    "Everton": "Everton",
    "Fulham": "Fulham",
    "Ipswich Town": "Ipswich-Town",
    "Leicester City": "Leicester-City",
    "Liverpool": "Liverpool",
    "Manchester City": "Manchester-City",
    "Manchester Utd": "Manchester-United",
    "Newcastle Utd": "Newcastle-United",
    "Nott'ham Forest": "Nottingham-Forest",
    "Southampton": "Southampton",
    "Tottenham": "Tottenham-Hotspur",
    "West Ham": "West-Ham-United",
    "Wolves": "Wolverhampton-Wanderers",
    # Spain
    "Barcelona": "Barcelona",
    "Real Madrid": "Real-Madrid",
    "Atletico Madrid": "Atletico-Madrid",
    "Athletic Club": "Athletic-Club",
    "Real Sociedad": "Real-Sociedad",
    "Real Betis": "Real-Betis",
    "Villarreal": "Villarreal",
    "Sevilla": "Sevilla",
    "Girona": "Girona",
    "Valencia": "Valencia",
    "Celta Vigo": "Celta-Vigo",
    "Rayo Vallecano": "Rayo-Vallecano",
    "Mallorca": "Mallorca",
    "Las Palmas": "Las-Palmas",
    "Getafe": "Getafe",
    "Osasuna": "Osasuna",
    "Alaves": "Alaves",
    "Espanyol": "Espanyol",
    "Leganes": "Leganes",
    "Real Valladolid": "Real-Valladolid",
    # Germany
    "Bayern Munich": "Bayern-Munich",
    "Bayer Leverkusen": "Bayer-Leverkusen",
    "Borussia Dortmund": "Borussia-Dortmund",
    "RB Leipzig": "RB-Leipzig",
    "Eintracht Frankfurt": "Eintracht-Frankfurt",
    "VfB Stuttgart": "VfB-Stuttgart",
    "VfL Wolfsburg": "VfL-Wolfsburg",
    "SC Freiburg": "SC-Freiburg",
    "Borussia Monchengladbach": "Borussia-Monchengladbach",
    "Werder Bremen": "Werder-Bremen",
    "1. FC Heidenheim": "1-FC-Heidenheim",
    "1. FC Union Berlin": "1-FC-Union-Berlin",
    "FC Augsburg": "FC-Augsburg",
    "TSG Hoffenheim": "TSG-Hoffenheim",
    "1. FSV Mainz 05": "1-FSV-Mainz-05",
    "VfL Bochum": "VfL-Bochum",
    "1. FC Koln": "1-FC-Koln",
    "SV Darmstadt 98": "SV-Darmstadt-98",
    # Italy
    "Inter Milan": "Internazionale",
    "AC Milan": "Milan",
    "Napoli": "Napoli",
    "Juventus": "Juventus",
    "AS Roma": "AS-Roma",
    "Lazio": "Lazio",
    "Atalanta": "Atalanta",
    "Fiorentina": "Fiorentina",
    "Bologna": "Bologna",
    "Torino": "Torino",
    "Monza": "Monza",
    "Genoa": "Genoa",
    "Lecce": "Lecce",
    "Cagliari": "Cagliari",
    "Udinese": "Udinese",
    "Sassuolo": "Sassuolo",
    "Empoli": "Empoli",
    "Verona": "Verona",
    "Frosinone": "Frosinone",
    "Salernitana": "Salernitana",
    # France
    "Paris Saint-Germain": "Paris-Saint-Germain",
    "Marseille": "Marseille",
    "Monaco": "Monaco",
    "Lille": "Lille",
    "Lyon": "Lyon",
    "Nice": "Nice",
    "Rennes": "Rennes",
    "Lens": "Lens",
    "Strasbourg": "Strasbourg",
    "Nantes": "Nantes",
    "Brest": "Brest",
    "Toulouse": "Toulouse",
    "Montpellier": "Montpellier",
    "Reims": "Reims",
    "Le Havre": "Le-Havre",
    "Metz": "Metz",
    "Lorient": "Lorient",
    "Clermont": "Clermont",
    "Auxerre": "Auxerre",
    "Angers": "Angers",
    # South America (partial)
    "River Plate": "River-Plate",
    "Boca Juniors": "Boca-Juniors",
    "Flamengo": "Flamengo",
    "Palmeiras": "Palmeiras",
    "Corinthians": "Corinthians",
    "Sao Paulo": "Sao-Paulo",
    "Santos": "Santos",
    "Gremio": "Gremio",
    "Internacional": "Internacional",
    "Cruzeiro": "Cruzeiro",
    "Atletico Mineiro": "Atletico-Mineiro",
    # Turkey
    "Galatasaray": "Galatasaray",
    "Fenerbahce": "Fenerbahce",
    "Besiktas": "Besiktas",
    "Trabzonspor": "Trabzonspor",
    "Istanbul Basaksehir": "Istanbul-Basaksehir",
    # Netherlands
    "Ajax": "Ajax",
    "PSV Eindhoven": "PSV-Eindhoven",
    "Feyenoord": "Feyenoord",
    "AZ Alkmaar": "AZ-Alkmaar",
    "FC Twente": "FC-Twente",
    # Portugal
    "Benfica": "Benfica",
    "FC Porto": "Porto",
    "Sporting CP": "Sporting-CP",
    "Braga": "Braga",
    "Vitoria Guimaraes": "Vitoria-Guimaraes",
    # Belgium
    "Club Brugge": "Club-Brugge",
    "Anderlecht": "Anderlecht",
    "Union SG": "Union-SG",
    "Genk": "Genk",
    "Royal Antwerp": "Royal-Antwerp",
    "Gent": "Gent",
    # Colombia
    "Atletico Nacional": "Atletico-Nacional",
    "Millonarios": "Millonarios",
    "America de Cali": "America-de-Cali",
    "Junior": "Junior",
    "Independiente Medellin": "Independiente-Medellin",
    "Deportivo Cali": "Deportivo-Cali",
    "Once Caldas": "Once-Caldas",
    "Deportes Tolima": "Deportes-Tolima",
    "Santa Fe": "Santa-Fe",
    "Alianza Petrolera": "Alianza-Petrolera",
    # Argentina
    "Boca Juniors": "Boca-Juniors",
    "River Plate": "River-Plate",
    "Racing Club": "Racing-Club",
    "Independiente": "Independiente",
    "San Lorenzo": "San-Lorenzo",
    "Estudiantes": "Estudiantes",
    "Velez Sarsfield": "Velez-Sarsfield",
    "Defensa y Justicia": "Defensa-y-Justicia",
    "Talleres": "Talleres",
    "Godoy Cruz": "Godoy-Cruz",
    "Colon": "Colon",
    "Argentinos Juniors": "Argentinos-Juniors",
    "Lanus": "Lanus",
    "Newells Old Boys": "Newells-Old-Boys",
    "Rosario Central": "Rosario-Central",
    "Banfield": "Banfield",
    "Gimnasia LP": "Gimnasia-LP",
    "Tigre": "Tigre",
    "Belgrano": "Belgrano",
    "Barracas Central": "Barracas-Central",
    "Platense": "Platense",
    "Sarmiento": "Sarmiento",
    "Union Santa Fe": "Union-Santa-Fe",
    "Instituto": "Instituto",
    "Central Cordoba": "Central-Cordoba",
    "Atletico Tucuman": "Atletico-Tucuman",
    "Estudiantes BA": "Estudiantes-BA",
    "San Martin SJ": "San-Martin-SJ",
    "Chacarita Juniors": "Chacarita-Juniors",
    # Uruguay
    "Penarol": "Penarol",
    "Nacional": "Nacional",
    "Defensor Sporting": "Defensor-Sporting",
    "Liverpool Montevideo": "Liverpool",
    "Boston River": "Boston-River",
    "Danubio": "Danubio",
    "Progreso": "Progreso",
    "Torque": "Torque",
    "Cerro": "Cerro",
    "Montevideo Wanderers": "Montevideo-Wanderers",
    "Plaza Colonia": "Plaza-Colonia",
    "Fenix": "Fenix",
    "Miramar Misiones": "Miramar-Misiones",
    "Racing Montevideo": "Racing-Montevideo",
    "CA Rentistas": "Rentistas",
    "Colón": "Colon",
    "Cerro Largo": "Cerro-Largo",
    "Central Español": "Central-Espanol",
    "Juventud Las Piedras": "Juventud",
    "Deportivo Maldonado": "Deportivo-Maldonado",
    "Huracan FC": "Huracan-FC",
    "Atenas": "Atenas",
    "La Luz": "La-Luz",
    "Miramar Misiones": "Miramar-Misiones",
    "Oriental": "Oriental",
    "Paysandu": "Paysandu",
    "Sportivo Cerrito": "Sportivo-Cerrito",
    "Tacuarembo": "Tacuarembo",
    "Uruguay Montevideo": "Uruguay-Montevideo",
    # Colombia
    "Atletico Nacional": "Atletico-Nacional",
    "Tigres FC": "Tigres-FC",
    "Millonarios": "Millonarios",
    "America de Cali": "America-de-Cali",
    "Junior": "Junior",
    "Independiente Medellin": "Independiente-Medellin",
    "Deportivo Cali": "Deportivo-Cali",
    "Once Caldas": "Once-Caldas",
    "Deportes Tolima": "Deportes-Tolima",
    "Santa Fe": "Santa-Fe",
    "Alianza Petrolera": "Alianza-Petrolera",
    "Envigado": "Envigado",
    "Deportivo Pereira": "Deportivo-Pereira",
    "Aguilas Doradas": "Aguilas-Doradas",
    "Jaguares": "Jaguares",
    "Alianza FC": "Alianza-FC",
    "Atletico Bucaramanga": "Atletico-Bucaramanga",
    "Deportivo Pasto": "Deportivo-Pasto",
    "Fortaleza CEIF": "Fortaleza-CEIF",
    "Patriotas": "Patriotas",
    "Union Magdalena": "Union-Magdalena",
    "La Equidad": "La-Equidad",
    "Chico FC": "Chico-FC",
    "Cortulua": "Cortulua",
    "Deportes Quindio": "Deportes-Quindio",
    "Expreso Rojo": "Expreso-Rojo",
    "Bogota FC": "Bogota-FC",
    "Barranquilla FC": "Barranquilla-FC",
    "Real Santander": "Real-Santander",
    "Tulua": "Tulua",
    "Valledupar": "Valledupar",
    "Buenaventura": "Buenaventura",
    "Leones": "Leones",
    "Orsomarso": "Orsomarso",
    "Boca Juniors de Cali": "Boca-Juniors-de-Cali",
    "Cortuluá": "Cortulua",
    "Deportes Quindío": "Deportes-Quindio",
    "Fortaleza CEIF": "Fortaleza-CEIF",
    "Universitario Popayan": "Universitario-Popayan",
    # Norway
    "Bodo/Glimt": "Bodo-Glimt",
    "Molde": "Molde",
    "Rosenborg": "Rosenborg",
    "Brann": "Brann",
    "Viking FK": "Viking-FK",
    "Lillestrom SK": "Lillestrom-SK",
    "Tromso": "Tromso",
    "Sarpsborg 08": "Sarpsborg-08",
    "Valerenga": "Valerenga",
    "HamKam": "HamKam",
    "Aalesund FK": "Aalesund-FK",
    "Sandefjord": "Sandefjord",
    "Kristiansund BK": "Kristiansund-BK",
    "KFUM Oslo": "KFUM-Oslo",
    "Fredrikstad": "Fredrikstad",
    # Sweden
    "Malmo FF": "Malmo-FF",
    "AIK Fotboll": "AIK",
    "Hammarby IF": "Hammarby-IF",
    "Djurgardens": "Djurgardens",
    "Elfsborg": "Elfsborg",
    "Hacken": "Hacken",
    "IFK Goteborg": "IFK-Goteborg",
    "Kalmar FF": "Kalmar-FF",
    "Sirius IK": "Sirius-IK",
    "Mjallby AIF": "Mjallby-AIF",
    "Vasteras SK FK": "Vasteras-SK-FK",
    "Osters IF": "Osters-IF",
    "Brommapojkarna": "Brommapojkarna",
    "Halmstads BK": "Halmstads-BK",
    "Orgryte IS": "Orgryte-IS",
    "Norrkopings": "Norrkopings",
    # Iceland
    "Breidablik": "Breidablik",
    "Valur Reykjavik": "Valur",
    "FH Hafnarfjordur": "FH-Hafnarfjordur",
    "KA Akureyri": "KA-Akureyri",
    "KR Reykjavik": "KR-Reykjavik",
    "Stjarnan FC": "Stjarnan",
    "Vikingur Reykjavik": "Vikingur-Reykjavik",
    "IA Akranes": "IA-Akranes",
    "Keflavik IF": "Keflavik-IF",
    "IBV Vestmannaeyjar": "IBV-Vestmannaeyjar",
    "Fram Reykjavik": "Fram-Reykjavik",
    "Thor Akureyri": "Thor-Akureyri",
    "Leiknir Reykjavik": "Leiknir-Reykjavik",
    "Grotta": "Grotta",
    "Fylkir FC": "Fylkir",
    "Afturelding": "Afturelding",
    "HK Kopavogur": "HK-Kopavogur",
    "IR Reykjavik": "IR-Reykjavik",
    "Trottur Reykjavik": "Trottur-Reykjavik",
    "UMF Grindavik": "UMF-Grindavik",
    "UMF Njardvik": "UMF-Njardvik",
    "Vestri": "Vestri",
    "Volsungur": "Volsungur",
    "Dalvik/Reynir": "Dalvik-Reynir",
    "Fjardabyggd/Leiknir": "Fjardabyggd-Leiknir",
    "Fjölnir": "Fjolnir",
    "Haukar": "Haukar",
    "KFG Gardabaer": "KFG-Gardabaer",
    "Kormakur": "Kormakur",
    "Kári Akranes": "Kari-Akranes",
    "IF Magni": "IF-Magni",
    "Hviti Riddarinn": "Hviti-Riddarinn",
    "UMF Selfoss": "UMF-Selfoss",
    "Trottur Vogum": "Trottur-Vogum",
    "Vikingur Olafsvik": "Vikingur-Olafsvik",
    "UMF Alftanes": "UMF-Alftanes",
    # Denmark
    "FC Copenhagen": "FC-Copenhagen",
    "Midtjylland": "Midtjylland",
    "Brondby": "Brondby",
    "Nordsjaelland": "Nordsjaelland",
    "AGF Aarhus": "AGF",
    "AaB Aalborg": "AaB",
    "Silkeborg IF": "Silkeborg-IF",
    "OB Odense": "OB",
    "Viborg FF": "Viborg-FF",
    "Randers FC": "Randers-FC",
    "SonderjyskE": "Sonderjyske",
    "Lyngby BK": "Lyngby-BK",
    "FC Helsingor": "FC-Helsingor",
    "HB Koge": "HB-Koge",
    "Vendsyssel FF": "Vendsyssel-FF",
    # Scotland
    "Celtic": "Celtic",
    "Rangers": "Rangers",
    "Hearts": "Hearts",
    "Hibernian": "Hibernian",
    "Aberdeen": "Aberdeen",
    "Dundee United": "Dundee-United",
    "Kilmarnock": "Kilmarnock",
    "St Mirren": "St-Mirren",
    "Motherwell": "Motherwell",
    "St Johnstone": "St-Johnstone",
    "Ross County": "Ross-County",
    "Livingston": "Livingston",
    "Dundee FC": "Dundee-FC",
    "Inverness CT": "Inverness-CT",
    "Partick Thistle": "Partick-Thistle",
    # Brazil
    "Flamengo": "Flamengo",
    "Palmeiras": "Palmeiras",
    "Botafogo": "Botafogo",
    "Fortaleza": "Fortaleza",
    "Sao Paulo": "Sao-Paulo",
    "Internacional": "Internacional",
    "Bahia": "Bahia",
    "Cruzeiro": "Cruzeiro",
    "Vasco da Gama": "Vasco-da-Gama",
    "Corinthians": "Corinthians",
    "Atletico Mineiro": "Atletico-Mineiro",
    "Fluminense": "Fluminense",
    "Juventude": "Juventude",
    "Santos": "Santos",
    "Gremio": "Gremio",
    "Cuiaba": "Cuiaba",
    "Goias": "Goias",
    "Bragantino": "Bragantino",
    "Athletico Paranaense": "Athletico-Paranaense",
    "Criciuma": "Criciuma",
    "Vitoria": "Vitoria",
    "Atletico Goianiense": "Atletico-Goianiense",
    "Cuiaba": "Cuiaba",
    "Internacional": "Internacional",
    "Sao Paulo": "Sao-Paulo",
    "Vasco da Gama": "Vasco-da-Gama",
    # Mexico
    "Club America": "Club-America",
    "Guadalajara Chivas": "Guadalajara",
    "Cruz Azul": "Cruz-Azul",
    "Tigres": "Tigres-UNAL",
    "Monterrey CF Monterrey": "Monterrey",
    "Pumas UNAM": "Pumas-UNAM",
    "Santos Laguna": "Santos-Laguna",
    "Toluca": "Toluca",
    "Leon Club Leon": "Club-Leon",
    "Atlas": "Atlas",
    "Puebla FC Puebla": "Puebla",
    "Necaxa Club Necaxa": "Necaxa",
    "Mazatlan FC Mazatlan": "Mazatlan",
    "FC Juarez": "FC-Juarez",
    "San Luis Atletico": "Atletico-San-Luis",
    "Pachuca FC Pachuca": "Pachuca",
    "Queretaro FC": "Queretaro",
    # Chile
    "Colo-Colo": "Colo-Colo",
    "Universidad de Chile": "Universidad-de-Chile",
    "Union La Calera": "Union-La-Calera",
    "Deportes Temuco": "Deportes-Temuco",
    "Audax Italiano": "Audax-Italiano",
    "Nublense": "Nublense",
    "Cobresal": "Cobresal",
    "Copiapo": "Copiapo",
    "O'Higgins": "OHiggins",
    "U. de Concepcion": "U-de-Concepcion",
    "Cobreloa": "Cobreloa",
    "Curico Unido": "Curico-Unido",
    "Deportes La Serena": "Deportes-La-Serena",
    "Espanol": "Espanol",
    "La Calera": "La-Calera",
    "Palestino": "Palestino",
    "San Luis": "San-Luis",
    "Santiago Wanderers": "Santiago-Wanderers",
    "Union Espanola": "Union-Espanola",
    # Peru
    "Sporting Cristal": "Sporting-Cristal",
    "Universitario": "Universitario",
    "Alianza Lima": "Alianza-Lima",
    "FBC Melgar": "FBC-Melgar",
    "Cienciano": "Cienciano",
    "Sport Boys": "Sport-Boys",
    "Cusco FC": "Cusco-FC",
    "Juan Aurich": "Juan-Aurich",
    "Universidad San Martin": "Universidad-San-Martin",
    "Alianza Atletico": "Alianza-Atletico",
    "ADT": "ADT",
    "Comerciantes Unidos": "Comerciantes-Unidos",
    "Grau": "Grau",
    "Carlos A. Mannucci": "Carlos-A-Mannucci",
    "Sport Huancayo": "Sport-Huancayo",
    "UTC": "UTC",
    "Unión Comercio": "Union-Comercio",
    "Los Chankas": "Los-Chankas",
    "Deportivo Garcilaso": "Deportivo-Garcilaso",
    "FC Cajamarca": "FC-Cajamarca",
    "UCV Moquegua": "UCV-Moquegua",
    # Ecuador
    "LDU Quito": "LDU-Quito",
    "Barcelona SC": "Barcelona-SC",
    "Emelec": "Emelec",
    "Independiente del Valle": "Independiente-del-Valle",
    "Aucas": "Aucas",
    "Universidad Catolica": "Universidad-Catolica",
    "Delfin SC": "Delfin-SC",
    "Deportivo Cuenca": "Deportivo-Cuenca",
    "Macara": "Macara",
    "Orense SC": "Orense-SC",
    "Gualaceo SC": "Gualaceo-SC",
    "Cumbaya FC": "Cumbaya-FC",
    "Tecnico Universitario": "Tecnico-Universitario",
    "Libertad FC": "Libertad-FC",
    "9 de Octubre": "9-de-Octubre",
    "Deportivo Santo": "Deportivo-Santo",
    "El Nacional": "El-Nacional",
    "America de Quito": "America-de-Quito",
    "Mushuc Runa": "Mushuc-Runa",
    "Guayaquil City": "Guayaquil-City",
    "Vinotinto Ecuador": "Vinotinto-Ecuador",
    "Universidad Catolica del Ecuador": "Universidad-Catolica",
}

# ── DB Schema ────────────────────────────────────────────────────────────
CREATE_FBREF_TABLE = """
CREATE TABLE IF NOT EXISTS fbref_squad_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    league TEXT NOT NULL,
    season_year INTEGER NOT NULL,
    team_name TEXT NOT NULL,
    fbref_slug TEXT,
    matches_played INTEGER DEFAULT 0,
    possession_pct REAL,
    squad_xg REAL,
    squad_xga REAL,
    squad_xgd REAL,
    squad_xgd_per90 REAL,
    squad_shots INTEGER,
    squad_shots_on_target INTEGER,
    squad_shots_on_target_pct REAL,
    squad_passes_completed INTEGER,
    squad_passes_attempted INTEGER,
    squad_pass_accuracy_pct REAL,
    squad_prog_passes INTEGER,
    squad_tackles INTEGER,
    squad_tackles_won INTEGER,
    squad_interceptions INTEGER,
    squad_blocks INTEGER,
    squad_clearances INTEGER,
    squad_errors INTEGER,
    squad_touches INTEGER,
    squad_carries INTEGER,
    squad_prg_carries INTEGER,
    squad_shot_creating_actions INTEGER,
    squad_goal_creating_actions INTEGER,
    last_updated TEXT DEFAULT (datetime('now')),
    UNIQUE(league, season_year, team_name)
);
"""

# ── Scraper ──────────────────────────────────────────────────────────────
class FBrefSquadScraper:
    """Scrape squad stats from FBref for a league/season."""

    BASE_URL = "https://fbref.com"
    DELAY = 6  # Rate limit: FBref blocks fast scrapers

    def __init__(self, use_selenium: bool = False):
        self.use_selenium = use_selenium and HAS_SELENIUM
        self.driver = None
        self.scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "linux", "mobile": False}
        )
        self.headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }

    def _init_selenium(self):
        if self.driver:
            return
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"user-agent={self.headers['User-Agent']}")
        self.driver = webdriver.Chrome(
            service=ChromeService(executable_path="/usr/bin/chromedriver"),
            options=chrome_options,
        )

    def _fetch_bs4(self, url: str) -> Optional[BeautifulSoup]:
        try:
            resp = self.scraper.get(url, headers=self.headers, timeout=30)
            if resp.status_code == 200:
                return BeautifulSoup(resp.text, "html.parser")
        except Exception:
            pass
        return None

    def _fetch_selenium(self, url: str, table_id: str) -> Optional[BeautifulSoup]:
        if not self.use_selenium:
            return None
        self._init_selenium()
        try:
            self.driver.get(url)
            wait = WebDriverWait(self.driver, 15)
            el = wait.until(EC.presence_of_element_located((By.ID, table_id)))
            html = el.get_attribute("outerHTML")
            return BeautifulSoup(html, "html.parser")
        except Exception:
            return None

    def _find_table(self, soup: BeautifulSoup, table_id_pattern: str) -> Optional[BeautifulSoup]:
        """Find a table by ID pattern."""
        table = soup.find("table", id=re.compile(table_id_pattern))
        if table:
            return table
        for t in soup.find_all("table"):
            tid = t.get("id", "")
            if table_id_pattern in tid:
                return t
        return None

    def _parse_numeric(self, text: str) -> Optional[float]:
        text = text.strip().replace(",", "").replace("%", "")
        try:
            return float(text)
        except (ValueError, TypeError):
            return None

    def scrape_league(self, league_id: str, season_year: int = None) -> list:
        """Scrape all team stats for a league/season.
        
        Returns list of dicts matching fbref_squad_stats schema.
        """
        if season_year is None:
            today = datetime.now()
            season_year = today.year if today.month >= 8 else today.year - 1

        league_info = FBREF_LEAGUES.get(league_id)
        if not league_info:
            print(f"  Unknown league ID: {league_id}")
            return []

        standings_url = f"{self.BASE_URL}/en/comps/{league_info['slug']}"
        print(f"  Fetching standings from: {standings_url}")

        soup = self._fetch_bs4(standings_url)
        if not soup:
            print(f"  Failed to fetch standings for {league_info['name']}")
            return []

        # Find team links from standings table
        stats_table = self._find_table(soup, "results")
        if not stats_table:
            # Try any table with squad links
            for t in soup.find_all("table"):
                links = t.find_all("a", href=re.compile(r"/squads/"))
                if links:
                    stats_table = t
                    break

        if not stats_table:
            print(f"  Could not find standings table for {league_info['name']}")
            return []

        team_links = []
        for a in stats_table.find_all("a", href=re.compile(r"/squads/")):
            href = a.get("href", "")
            if "/squads/" in href and href not in [tl for _, tl in team_links]:
                team_name = a.get_text(strip=True)
                team_links.append((team_name, href))

        if not team_links:
            print(f"  No team links found for {league_info['name']}")
            return []

        print(f"  Found {len(team_links)} teams, scraping stats...")
        results = []

        for i, (team_name, href) in enumerate(team_links, 1):
            team_url = f"{self.BASE_URL}{href}"
            if not team_url.endswith("Stats"):
                team_url = team_url.rstrip("/") + "/Stats"

            print(f"    [{i}/{len(team_links)}] {team_name}...", end=" ", flush=True)

            stats = self._scrape_team(team_url, team_name, league_id, season_year)
            if stats:
                results.append(stats)
                print("OK")
            else:
                print("FAIL")

            time.sleep(self.DELAY)

        if self.driver:
            self.driver.quit()
            self.driver = None

        return results

    def _scrape_team(self, url: str, team_name: str, league_id: str, season_year: int) -> Optional[dict]:
        """Scrape stats for a single team."""
        # Try BeautifulSoup first
        soup = self._fetch_bs4(url)
        
        # Look for standard stats table
        table = None
        for pattern in ["stats_standard", "stats_squads_standard", "matchlogs_for"]:
            table = self._find_table(soup, pattern) if soup else None
            if table:
                break

        # Fall back to Selenium if table not found
        if not table and self.use_selenium:
            for pattern in ["stats_standard", "stats_squads_standard"]:
                table = self._fetch_selenium(url, pattern)
                if table:
                    break

        if not table:
            return None

        # Parse the table
        stats = {
            "league": league_id,
            "season_year": season_year,
            "team_name": team_name,
            "fbref_slug": url.split("/")[-2] if "/" in url else "",
        }

        tbody = table.find("tbody")
        if not tbody:
            return None

        # Get first row (summary stats for all competitions or league)
        rows = tbody.find_all("tr")
        for row in rows:
            if "thead" in (row.get("class") or []):
                continue
            cells = row.find_all(["th", "td"])
            if len(cells) < 15:
                continue

            try:
                # Parse all cells
                nums = []
                for cell in cells:
                    text = cell.get_text(strip=True)
                    nums.append(self._parse_numeric(text))

                # Find xG/xGA (typically after card columns)
                xg = None
                xga = None
                xgd = None
                poss = None
                
                for i, val in enumerate(nums):
                    if val is None:
                        continue
                    # Possession: 30-70%
                    if poss is None and 30 <= val <= 70 and i < 10:
                        poss = val
                    # xG: typically 20-100
                    if xg is None and 20 < val < 120 and i > 12:
                        xg = val
                    elif xg and xga is None and 20 < val < 120:
                        xga = val
                    elif xga and xgd is None and -60 < val < 60:
                        xgd = val

                stats["squad_xg"] = xg
                stats["squad_xga"] = xga
                stats["squad_xgd"] = xgd
                stats["squad_xgd_per90"] = (xgd / 34.0) if xgd else None
                stats["possession_pct"] = poss

                # Find shots, SOT, passes
                for i, val in enumerate(nums):
                    if val is None:
                        continue
                    if 200 <= val <= 800 and i > 8:
                        stats["squad_shots"] = int(val)
                    elif 80 <= val <= 300 and i > 10:
                        stats["squad_shots_on_target"] = int(val)
                    elif 70 <= val <= 95 and i > 15:
                        stats["squad_pass_accuracy_pct"] = val
                    elif 200 <= val <= 1200 and i > 20:
                        stats["squad_prog_passes"] = int(val)
                    elif 300 <= val <= 800 and i > 25:
                        stats["squad_tackles"] = int(val)
                    elif 100 <= val <= 400 and i > 25:
                        stats["squad_interceptions"] = int(val)

                break  # Only need first data row

            except Exception:
                continue

        return stats if stats.get("squad_xg") else None


def normalize_team_name(name: str) -> str:
    """Normalize team name for FBref lookup.
    
    Returns the FBref slug fragment for use in URLs.
    """
    if name in TEAM_NAME_MAP:
        return TEAM_NAME_MAP[name]
    
    # Try fuzzy match
    name_lower = name.lower().strip()
    for fbref_name, slug in TEAM_NAME_MAP.items():
        if fbref_name.lower() == name_lower:
            return slug
    
    # Default: replace spaces with hyphens
    return name.replace(" ", "-")


def save_fbref_stats(stats_list: list):
    """Save scraped FBref stats to DB."""
    if not stats_list:
        return
    
    from database import get_db
    conn = get_db()
    conn.execute(CREATE_FBREF_TABLE)
    
    for stats in stats_list:
        conn.execute("""
            INSERT OR REPLACE INTO fbref_squad_stats (
                league, season_year, team_name, fbref_slug,
                possession_pct, squad_xg, squad_xga, squad_xgd, squad_xgd_per90,
                squad_shots, squad_shots_on_target,
                squad_pass_accuracy_pct, squad_prog_passes,
                squad_tackles, squad_interceptions,
                last_updated
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))
        """, (
            stats.get("league"),
            stats.get("season_year"),
            stats.get("team_name"),
            stats.get("fbref_slug"),
            stats.get("possession_pct"),
            stats.get("squad_xg"),
            stats.get("squad_xga"),
            stats.get("squad_xgd"),
            stats.get("squad_xgd_per90"),
            stats.get("squad_shots"),
            stats.get("squad_shots_on_target"),
            stats.get("squad_pass_accuracy_pct"),
            stats.get("squad_prog_passes"),
            stats.get("squad_tackles"),
            stats.get("squad_interceptions"),
        ))
    
    conn.commit()
    conn.close()
    print(f"  Saved {len(stats_list)} team stats to DB")


def update_matches_with_fbref(league_id: str = None):
    """Update pending matches with FBref stats from the DB table."""
    from database import get_db
    
    conn = get_db()
    if league_id:
        rows = conn.execute("""
            SELECT m.id, m.home_team, m.away_team, f.team_name, 
                   f.squad_xg, f.squad_xga, f.squad_xgd
            FROM matches m
            LEFT JOIN fbref_squad_stats f ON f.team_name = m.home_team
            WHERE m.reviewed = 0 AND m.home_squad_xg IS NULL
            AND f.league = ?
        """, (league_id,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT m.id, m.home_team, m.away_team
            FROM matches m
            WHERE m.reviewed = 0 AND m.home_squad_xg IS NULL
        """).fetchall()
    
    updated = 0
    for row in rows:
        match_id = row[0]
        home_team = row[1]
        away_team = row[2]
        
        # Look up FBref stats for both teams
        home_stats = conn.execute(
            "SELECT squad_xg, squad_xga, squad_xgd FROM fbref_squad_stats WHERE team_name = ?",
            (home_team,)
        ).fetchone()
        
        away_stats = conn.execute(
            "SELECT squad_xg, squad_xga, squad_xgd FROM fbref_squad_stats WHERE team_name = ?",
            (away_team,)
        ).fetchone()
        
        if home_stats or away_stats:
            conn.execute("""
                UPDATE matches SET
                    home_squad_xg = ?,
                    home_squad_xga = ?,
                    home_squad_xgd = ?,
                    away_squad_xg = ?,
                    away_squad_xga = ?,
                    away_squad_xgd = ?
                WHERE id = ?
            """, (
                home_stats[0] if home_stats else None,
                home_stats[1] if home_stats else None,
                home_stats[2] if home_stats else None,
                away_stats[0] if away_stats else None,
                away_stats[1] if away_stats else None,
                away_stats[2] if away_stats else None,
                match_id,
            ))
            updated += 1
    
    conn.commit()
    conn.close()
    print(f"  Updated {updated} matches with FBref stats")
    return updated


if __name__ == "__main__":
    import sys
    
    league_id = sys.argv[1] if len(sys.argv) > 1 else "9"
    use_selenium = "--selenium" in sys.argv
    
    print(f"Scraping FBref for league {league_id}...")
    scraper = FBrefSquadScraper(use_selenium=use_selenium)
    stats = scraper.scrape_league(league_id)
    
    if stats:
        save_fbref_stats(stats)
        update_matches_with_fbref(league_id)
    else:
        print("No stats scraped")
