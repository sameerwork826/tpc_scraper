from playwright.sync_api import sync_playwright

LOGIN_URL = "https://www.placement.iitbhu.ac.in/"  

def main():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        context = browser.new_context()
        page = context.new_page()

        page.goto(LOGIN_URL)
        print("Login manually using Google + institute ID")
        input("After login, press ENTER...")

        context.storage_state(path="auth.json")
        print("Session saved to auth.json")
        browser.close()

if __name__ == "__main__":
    main()
