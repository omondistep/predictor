
import sys
from forebet_scraper import scrape_results_list

url = "https://www.forebet.com/en/football-predictions-from-yesterday/by-league"
results = scrape_results_list(url)
print(f"Found {len(results)} results")
for r in results[:5]:
    print(r)
