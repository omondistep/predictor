
import requests
import re
from bs4 import BeautifulSoup

url = "https://www.forebet.com/en/football-predictions-from-yesterday/by-league"
HEADERS = {"User-Agent": "Mozilla/5.0"}
r = requests.get(url, headers=HEADERS, timeout=20)
soup = BeautifulSoup(r.text, "html.parser")

rcnt = soup.find_all("div", {"class": "rcnt"})
print(f"rcnt count: {len(rcnt)}")

predict_rows = soup.find_all(["tr", "div"], {"class": re.compile(r"(tr_\d+|predict-row)")})
print(f"predict-row/tr count: {len(predict_rows)}")

# Check for tnmscn links
links = soup.find_all("a", {"class": "tnmscn"})
print(f"tnmscn links count: {len(links)}")

# Check for l_scr scores
scores = soup.find_all(["b", "span"], {"class": re.compile(r"(l_scr|l_score|lscr_main|res_sc)")})
print(f"l_scr scores count: {len(scores)}")
