# IIT BHU Placement Portal Scraper & Query Bot 🎓

A complete system to scrape placement data, store it in a structured database, and query it using natural language powered by Google Gemini.

### 🌟 Live Demo: [https://sameer-tpciitbhu-assistant.streamlit.app/](https://sameer-tpciitbhu-assistant.streamlit.app/)
*(Student Explorer works without an API key!)*

## 🚀 Features

*   **Automated Scraper**: Logs in to the placement portal and scrapes all forum topics, offers, and shortlists.
*   **Data Processing**: Robust parsing of names, roll numbers, and companies into a SQLite database.
*   **AI-Powered Query Bot**: Ask questions like *"Who got placed in database?"* or *"Count offers for CSE"* using a modern Streamlit UI.
*   **Student Explorer**: Browse student profiles and placement history manually (No API Key required).
*   **Smart Matching**: Handles partial names and fuzzy queries.

## 🛠️ Project Structure

*   `scrape_topics.py`: The main scraper using Playwright.
*   `process_data.py`: ETL script to parse JSONs -> SQLite (`data/placement.db`).
*   `app.py`: Streamlit web application with Gemini integration.
*   `login_and_save_session.py`: Utility to creates the initial `auth.json`.

## 📦 Installation

### 1. Get a Gemini API Key (Optional but Recommended)
To use the **AI Chat** feature, you need a free API key:
1.  Go to [Google AI Studio](https://aistudio.google.com/app/apikey).
2.  Click **Create API key**.
3.  Copy the key. You can enter it in the app's sidebar later.

### 2. Installation Steps

1.  **Clone the repository**
2.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Setup Environment**
    Create a `.env` file in the root directory:
    ```env
    GOOGLE_API_KEY="your_gemini_api_key_here"  # Optional: Can also be entered in UI
    ```

## 🏃 Usage

### 1. Authentication (First Run Only)
The portal requires login (Google OAuth). Run this script to open a browser, log in manually, and save your session cookies.
```bash
python login_and_save_session.py
```
*   Log in inside the popup browser.
*   Press **Enter** in the terminal once logged in.
*   This creates `auth.json`.

### 2. Scrape Data
Run the scraper to fetch all forum topics.
```bash
python scrape_topics.py
```
*   Data is saved to `data/raw_topics/`.

### 3. Process Data
Parse the raw JSON files into the database.
```bash
python process_data.py
```
*   Creates/Updates `data/placement.db`.

### 4. Run the Query App
Launch the AI interface.
```bash
streamlit run app.py
```

## 🤖 Example Queries
*   "Where was Sameer Wanjari placed?"
*   "Show all shortlists for roll number 21174028"
*   "How many offers did Oracle give?"
*   "List all companies visiting for FT"

## 📝 Technologies
*   **Python 3.10+**
*   **Playwright**: For robust scraping.
*   **Streamlit**: For the frontend.
*   **Google Gemini 2.0 Flash**: For Natural Language to SQL generation.
*   **SQLite**: Lightweight database.

## ⚠️ Disclaimer
This tool is for educational purposes and personal productivity. Please respect the scraping policies of the website.
