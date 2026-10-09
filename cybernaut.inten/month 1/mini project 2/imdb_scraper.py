import csv
import re
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager

URL = "https://www.imdb.com/chart/top/"
OUTPUT_FILE = "imdb_top_250_movies.csv"



options = webdriver.ChromeOptions()

options.add_argument("--start-maximized")
options.add_argument("--disable-notifications")
options.add_argument("--disable-popup-blocking")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

wait = WebDriverWait(driver, 30)
print("=" * 70)
print("IMDb TOP 250 MOVIE SCRAPER")
print("=" * 70)

print("\nOpening IMDb...")

driver.get(URL)

wait.until(
    EC.presence_of_element_located(
        (By.TAG_NAME, "body")
    )
)

time.sleep(5)

print("IMDb opened successfully.")

print("\nLoading Top 250 movies...\n")

for i in range(40):

    driver.execute_script(
        "window.scrollTo(0, document.body.scrollHeight);"
    )

    time.sleep(1)

    movie_elements = driver.find_elements(
        By.CSS_SELECTOR,
        "li.ipc-metadata-list-summary-item"
    )

    count = len(movie_elements)

    print(
        f"Scroll {i + 1:02d} -> {count} movies loaded"
    )

    if count >= 250:
        break


movie_elements = driver.find_elements(
    By.CSS_SELECTOR,
    "li.ipc-metadata-list-summary-item"
)

print()
print("Movies found:", len(movie_elements))

movies = []

print()
print("=" * 100)
print("SCRAPING IMDb TOP 250")
print("=" * 100)
print()


for rank, movie in enumerate(
    movie_elements[:250],
    start=1
):

    try:


        movie_name = "N/A"

        try:

            title_link = movie.find_element(
                By.CSS_SELECTOR,
                "a.ipc-title-link-wrapper"
            )

            movie_name = title_link.text.strip()

            movie_name = re.sub(
                r"^\d+\.\s*",
                "",
                movie_name
            )

        except Exception:
            pass


        
        movie_url = "N/A"

        try:

            title_link = movie.find_element(
                By.CSS_SELECTOR,
                "a.ipc-title-link-wrapper"
            )

            href = title_link.get_attribute("href")

            if href:

                match = re.search(
                    r"(https://www\.imdb\.com/title/tt\d+/)",
                    href
                )

                if match:
                    movie_url = match.group(1)
                else:
                    movie_url = href.split("?")[0]

        except Exception:
            pass


        year = ""

        try:

            metadata_items = movie.find_elements(
                By.CSS_SELECTOR,
                "span.cli-title-metadata-item"
            )

            for item in metadata_items:

                text = item.text.strip()

                year_match = re.search(
                    r"\b(18\d{2}|19\d{2}|20\d{2})\b",
                    text
                )

                if year_match:

                    year = year_match.group(0)

                    break

        except Exception:
            pass


        
        rating = ""

        try:

            rating_element = movie.find_element(
                By.CSS_SELECTOR,
                "span.ipc-rating-star--rating"
            )

            rating = rating_element.text.strip()

        except Exception:

            try:

                rating_element = movie.find_element(
                    By.XPATH,
                    ".//*[contains(@class,'ipc-rating-star')]"
                )

                rating_text = rating_element.text.strip()

                rating_match = re.search(
                    r"\b([0-9]\.[0-9])\b",
                    rating_text
                )

                if rating_match:
                    rating = rating_match.group(1)

            except Exception:
                pass


    

        if year == "" and movie_url != "N/A":

            try:

                original_window = driver.current_window_handle

                driver.execute_script(
                    "window.open(arguments[0], '_blank');",
                    movie_url
                )

                driver.switch_to.window(
                    driver.window_handles[-1]
                )

                time.sleep(2)


               

                page_text = driver.find_element(
                    By.TAG_NAME,
                    "body"
                ).text


                year_match = re.search(
                    r"\b(18\d{2}|19\d{2}|20\d{2})\b",
                    page_text
                )

                if year_match:

                    year = year_match.group(0)


                driver.close()

                driver.switch_to.window(
                    original_window
                )

            except Exception:

                try:

                    driver.switch_to.window(
                        driver.window_handles[0]
                    )

                except Exception:
                    pass


        

        if year == "":
            year = "N/A"


        

        if rating == "":
            rating = "N/A"


        

        movies.append({

            "Rank": rank,

            "Movie Name": movie_name,

            "Year": year,

            "IMDb Rating": rating,

            "URL": movie_url
        })


        

        print(
            f"{rank:3d} | "
            f"{movie_name:<45} | "
            f"{year:<6} | "
            f"{rating:<5} | "
            f"{movie_url}"
        )


    except Exception as error:

        print(
            f"{rank:3d} | ERROR: {type(error).__name__}"
        )




driver.quit()



print("\nCreating CSV file...")


with open(
    OUTPUT_FILE,
    "w",
    newline="",
    encoding="utf-8-sig"
) as csv_file:

    fieldnames = [
        "Rank",
        "Movie Name",
        "Year",
        "IMDb Rating",
        "URL"
    ]

    writer = csv.DictWriter(
        csv_file,
        fieldnames=fieldnames,
        delimiter=",",
        quoting=csv.QUOTE_MINIMAL
    )


    # C
    writer.writeheader()

    for movie in movies:

        writer.writerow(movie)




print()
print("=" * 70)
print("IMDb SCRAPING COMPLETED")
print("=" * 70)

print(
    "Total movies scraped:",
    len(movies)
)

print(
    "CSV file created:",
    OUTPUT_FILE
)

print()
print("Columns:")
print(
    "Rank | Movie Name | Year | IMDb Rating | URL"
)

print()
print("Open the CSV using EDIT CSV in VS Code.")
print()
print("DONE!")