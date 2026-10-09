import time
import pandas as pd
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
print("=" * 80)
print("          CRYPTOCURRENCY PRICE TRACKER") 
print("=" * 80)
print("\nStarting Chrome Browser...")
options = Options() 
options.add_argument("--start-maximized")
options.add_argument("--disable-notifications")
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(
    service=service,
    options=options
)
try:
    print("Opening CoinMarketCap...")
    driver.get("https://coinmarketcap.com/")
    time.sleep(8)
    print("CoinMarketCap opened successfully!")
    rows = driver.find_elements(
        "css selector",
        "table tbody tr"
    )

    crypto_data = []
    for number, row in enumerate(rows[:10], start=1):

        cells = row.find_elements(
            "tag name",
            "td"
        )

        if len(cells) >= 7:

            try:
                coin_name = cells[2].text.strip()
                price = cells[3].text.strip()
                change_24h = cells[5].text.strip()
                market_cap = cells[6].text.strip()

                crypto_data.append({
                    "Rank": number,
                    "Cryptocurrency": coin_name,
                    "Price": price,
                    "24h Change": change_24h,
                    "Market Cap": market_cap
                })

            except Exception:
                continue
    if crypto_data:

        df = pd.DataFrame(crypto_data)

        print("\n")
        print("=" * 100)
        print("                 TOP 10 CRYPTOCURRENCIES")
        print("=" * 100)
        print(df.to_string(index=False))
        print("=" * 100)
        df.to_csv( 
            "crypto_prices.csv",
            index=False
        )

        print("\n✓ Top 10 coins collected successfully")
        print("✓ Rank: 1 to 10")
        print("✓ CSV file saved: crypto_prices.csv")

    else:

        print("\nNo cryptocurrency data found.")

except Exception as e:

    print("\nERROR:")
    print(e)

finally:

    print("\nBrowser will remain open.")
    print("Press Ctrl+C in Terminal when you want to stop the program.")
    while True:
        time.sleep  