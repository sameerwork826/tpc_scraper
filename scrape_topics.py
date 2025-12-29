import json
import os
import time
from urllib.parse import urljoin
from playwright.sync_api import sync_playwright

BASE_URL = "https://www.placement.iitbhu.ac.in/"   
FORUM_2025_URL = BASE_URL + "/forum/c/notice-board/2025-26/"

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw_topics")
TOPICS_FILE = os.path.join(DATA_DIR, "topics.json")

os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

import process_data

def load_existing_topics():
    if os.path.exists(TOPICS_FILE):
        with open(TOPICS_FILE, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                # Return a set of URLs for fast lookup
                return {item["url"] for item in data}, data
            except json.JSONDecodeError:
                pass
    return set(), []

def scrape():
    existing_urls, existing_data = load_existing_topics()
    print(f"🔍 Loaded {len(existing_urls)} existing topics.")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context(storage_state="auth.json")
        page = context.new_page()

        # 1️⃣ Open first page
        page.goto(FORUM_2025_URL)
        page.wait_for_load_state("networkidle")

        # 2️⃣ Find total number of pages
        pages = page.locator("ul.pagination a.page")
        if pages.count() > 0:
            total_pages = int(pages.last.inner_text())
        else:
            total_pages = 1

        print(f"✅ Total pages detected: {total_pages}")

        new_topics = []
        stop_scraping = False

        # 3️⃣ Loop pages explicitly
        for page_no in range(1, total_pages + 1):
            if stop_scraping:
                break

            print(f"📄 Scraping page {page_no}")
            if page_no > 1:
                page.goto(f"{FORUM_2025_URL}?page={page_no}")
                page.wait_for_load_state("networkidle")

            # Improved selector: ensure we are in tbody to avoid header row
            rows = page.locator("tbody tr.topic-row:not(.sticky)")
            count = rows.count()

            for i in range(count):
                row = rows.nth(i)
                link = row.locator("td.topic-name a").first
                
                title = link.inner_text().strip()
                href = link.get_attribute("href")
                url = urljoin(BASE_URL, href)

                if url in existing_urls:
                    print(f"🛑 Found existing topic: {title}. Stopping search.")
                    stop_scraping = True
                    break
                
                new_topics.append({"title": title, "url": url})
                print(f"🆕 Found new topic: {title}")

            time.sleep(0.5)

        if not new_topics:
            print("✅ No new topics found.")
            browser.close()
            return

        print(f"✅ Collected {len(new_topics)} NEW topics. Starting extraction...")

        # Update topic index (Newest first)
        updated_topics = new_topics + existing_data
        with open(TOPICS_FILE, "w", encoding="utf-8") as f:
            json.dump(updated_topics, f, indent=2)

        # 4️⃣ Visit each NEW topic and extract content
        # We need to determine the starting index for filenames.
        # existing_data has N items. Filenames are 1-based index?
        # Actually, let's look at how they were named: f"{idx:03d}_{safe}.json"
        # The previous code did `enumerate(all_topics.items(), start=1)`.
        # So providing we maintain order in topics.json, we might want to rename everything?
        # User said "rebuild the db".
        # If we just name the new files with higher numbers, or just append?
        # The previous code overwrote EVERYTHING.
        # If we want to be incremental, we should probably name files based on something stable or just increment.
        # BUT: The previous code named 001 as the FIRST one found (which is the LATEST topic on page 1).
        # So 001 is the NEWEST.
        # If we find 3 new topics, they should be 001, 002, 003? And the old 001 becomes 004?
        # That would require renaming ALL files. That's messy.
        
        # ALTERNATIVE: Use the topic ID or Hash in filename?
        # The user's code used `idx` from enumerate.
        # If I want to avoid renaming all files, I should change the naming scheme or accept that I might overwrite?
        # Wait, if I have new topics, `topics.json` gets updated.
        # If I want to keep it simple and consistent with previous "scrape all", I probably should just accept that new files = new IDs?
        # Actually, let's look at `process_data.py`. It just reads ALL json files in raw_topics.
        # It doesn't care about the filename prefix number, except maybe for sorting order?
        # `files = sorted([f for f in os.listdir(RAW_DIR) if f.endswith(".json")])`
        # It sorts by filename. So 001_A.json comes before 002_B.json.
        # If 001 is the NEWEST, then `process_files` processes newest first?
        # It loops `for fname in files:`.
        # It creates events. The ID in DB is auto-increment.
        # So 1st file processed = ID 1.
        # If 001 is main list top, then ID 1 is newest date?
        # Usually we want ID 1 to be OLDEST? Or doesn't matter?
        # User didn't specify.
        
        # SCENARIO:
        # Old topics: A, B, C. (A is newest).
        # topics.json: [A, B, C]
        # Files: 001_A.json, 002_B.json, 003_C.json
        
        # New topics: X, Y. (X is newest).
        # New order: X, Y, A, B, C.
        # Desired Files: 001_X, 002_Y, 003_A, 004_B...
        # THIS REQUIRES RENAMING OLD FILES.
        
        # EASIER APPROACH:
        # Don't rely on 001 prefix for valid ordering if it's too hard to maintain, OR RENAME.
        # Since I am ONLY scraping new topics, I can't easily rename old files without listing them all.
        # But I DO have existing_data.
        
        # Let's try to just name new files using a timestamp or a hash?
        # Or just use the NEXT available index?
        # If I use next index, say 004_X, 005_Y.
        # Then `sorted()` puts them at the END.
        # So `process_files` processes old ones first, then new ones.
        # That seems fine?
        # Let's count existing files to find starting index.
        
        existing_files_count = len([f for f in os.listdir(RAW_DIR) if f.endswith(".json")])
        start_index = existing_files_count + 1
        
        for idx, item in enumerate(new_topics, start=start_index):
            title = item["title"]
            url = item["url"]
            print(f" ({idx-start_index+1}/{len(new_topics)}) Scraping content: {title}")
            
            page.goto(url)
            page.wait_for_load_state("networkidle")

            posts = page.locator("td.post-content").all_inner_texts()
            content = "\n\n---SEPARATOR---\n\n".join(posts)

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
        print("🎉 Scraping of NEW topics completed.")
        
        print("🔄 Triggering Database Rebuild...")
        process_data.process_files()

if __name__ == "__main__":
    scrape()
