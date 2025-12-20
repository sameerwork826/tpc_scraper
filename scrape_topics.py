import json
import os
import time
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.placement.iitbhu.ac.in/"   # 🔴 CHANGE DOMAIN ONLY
FORUM_2025_URL = BASE_URL + "/forum/c/notice-board/2025-26/"

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw_topics")
TOPICS_FILE = os.path.join(DATA_DIR, "topics.json")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

def scrape():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state="auth.json")
        page = context.new_page()

        # 1️⃣ Open first page
        page.goto(FORUM_2025_URL)
        page.wait_for_load_state("networkidle")

        # 2️⃣ Find total number of pages
        pages = page.locator("ul.pagination a.page")
        total_pages = int(pages.last.inner_text())

        print(f"✅ Total pages detected: {total_pages}")

        all_topics = {}   # url -> title

        # 3️⃣ Loop pages explicitly (NO clicking)
        for page_no in range(1, total_pages + 1):
            print(f"📄 Scraping page {page_no}")
            page.goto(f"{FORUM_2025_URL}?page={page_no}")
            page.wait_for_load_state("networkidle")

            rows = page.locator("tr.topic-row:not(.sticky)")
            count = rows.count()

            for i in range(count):
                row = rows.nth(i)
                link = row.locator("td.topic-name a").first

                title = link.inner_text().strip()
                href = link.get_attribute("href")
                url = urljoin(BASE_URL, href)

                all_topics[url] = title

            time.sleep(0.5)

        print(f"✅ Collected {len(all_topics)} topics")

        # Save topic index
        with open(TOPICS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                [{"title": t, "url": u} for u, t in all_topics.items()],
                f,
                indent=2
            )

        # 4️⃣ Visit each topic and extract students
        for idx, (url, title) in enumerate(all_topics.items(), start=1):
            print(f"🧵 ({idx}/{len(all_topics)}) {title}")
            page.goto(url)
            page.wait_for_load_state("networkidle")

            content = page.locator("td.post-content").inner_text()

            data = {
                "topic_title": title,
                "url": url,
                "raw_text": content
            }

            safe = title[:40].replace(" ", "_").replace("/", "_")
            fname = f"{idx:03d}_{safe}.json"

            with open(os.path.join(RAW_DIR, fname), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)

            time.sleep(0.7)

        browser.close()
        print("🎉 DONE — scraping completed successfully")

if __name__ == "__main__":
    scrape()
