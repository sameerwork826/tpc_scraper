# IIT BHU Placement Portal Scraper & Query Bot 🎓

A comprehensive system to scrape placement data, analyze student performance, and query placement information using AI-powered natural language processing.

### 🌟 Live Demo: [https://sameer-tpciitbhu-assistant.streamlit.app/](https://sameer-tpciitbhu-assistant.streamlit.app/)

## 🚀 Features

### 💬 AI Chat Assistant
- **Natural Language Queries**: Ask questions like *"Who got placed in Google?"* or *"Show CSE students with CPI > 9.0"*
- **Multi-Provider Support**: Choose between Google Gemini, Groq, or local Ollama models
- **Context-Aware**: Maintains conversation history for follow-up questions
- **CPI-Based Queries**: Filter and analyze students by academic performance

### 🔍 Student Explorer
- **Fast Profile Lookup**: Optimized denormalized database for instant student searches
- **Placement Status**: Shows if student is placed with company name and CTC
- **Comprehensive Timeline**: View all offers, interviews, and test shortlists
- **CTC Information**: Displays package details for offers (CTC and in-hand salary)
- **Academic Info**: Branch, year, and CPI displayed prominently

### 🏢 Company Explorer
- **Company-wise Results**: Browse all students shortlisted/placed by each company
- **CTC Display**: Shows CTC and in-hand salary for each student
- **Full-Time & Internship**: Separate sections for FT and intern positions
- **Direct Links**: Access original forum posts for each event

### 📊 CPI Visualizer
- **Overall Statistics**: View aggregate CPI data across all students
- **Branch-wise Analysis**: Compare CPI distributions by department
- **Interactive Plots**: Histograms and boxplots for visual analysis
- **Top Performers**: Identify highest CPI students per branch
- **Distribution Analysis**: Statistical analysis including skewness and normality tests

### 📅 Company Calendar
- **Upcoming Visits**: Browse all companies visiting for placements (2025-26)
- **Search & Filter**: Find companies by name, CTC range, or location
- **Detailed Info**: View role, CTC, in-hand salary, eligibility, and JD links
- **Placement Integration**: See which companies have placed students from your database

## 🛠️ Project Structure

- `scrape_topics.py`: Forum scraper using Playwright with authentication
- `process_data.py`: ETL pipeline to parse JSONs → SQLite database
- `app.py`: Streamlit web application with 5 feature tabs
- `login_and_save_session.py`: Authentication utility to create `auth.json`

## 📦 Installation

### 1. Get API Keys (Optional for AI Chat)

**Google Gemini** (Recommended):
1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in and click **Create API key**
3. Copy the key (can be entered in app sidebar)

**Groq** (Ultra-fast alternative):
1. Visit [Groq Console](https://console.groq.com)
2. Create an account and generate API key

**Ollama** (Local, no API key needed):
1. Install from [ollama.com](https://ollama.com)
2. Run `ollama run llama3.2`

### 2. Installation Steps

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd tpc_scraper
   ```

2. **Install dependencies** (using uv - recommended)
   ```bash
   uv sync
   ```
   Or using pip:
   ```bash
   pip install -r requirements.txt
   ```

3. **Setup Environment** (Optional)
   Create a `.env` file:
   ```env
   GOOGLE_API_KEY="your_gemini_api_key_here"
   GROQ_API_KEY="your_groq_api_key_here"  # Optional
   ```

## 🏃 Usage

### 1. Authentication (First Run Only)
```bash
python login_and_save_session.py
```
- Log in via the popup browser
- Press **Enter** after successful login
- Creates `auth.json` with session cookies

### 2. Scrape Placement Data
```bash
python scrape_topics.py
```
- Fetches all forum topics from the placement portal
- Saves to `data/raw_topics/`

### 3. Process Data
```bash
python process_data.py
```
- Parses JSON files into SQLite database
- Creates/updates `data/placement.db`
- Normalizes student names and company names
- Extracts branch and year from roll numbers

### 4. Link CPI Data (Optional)
If you have CPI data in CSV format:
```bash
# Place your CSV file as 'alot_LM (2).csv'
# CPI data will be automatically integrated
```

### 5. Scrape Company Calendar (Optional)
To get upcoming company visit information:
```bash
python scrape_company_calendar.py
```
- Requires valid `auth.json`
- Fetches company calendar with CTC, roles, and locations

### 6. Run the Application
```bash
streamlit run app.py
```
- Opens at `http://localhost:8501`
- All features work without API keys except AI Chat

## 🤖 Example Queries (AI Chat)

- "Where was Sameer Wanjari placed?"
- "Show all CSE students with CPI greater than 9.5"
- "How many offers did Microsoft give?"
- "List students placed with CTC above 20 LPA"
- "Compare placement statistics for ECE and CSE"
- "Which companies offered the highest packages?"

## 📊 Database Schema

### Core Tables
- `students`: Student profiles (roll_no, name, branch, year, cpi)
- `events`: Company events (company_name, event_type, topic_url)
- `event_students`: Junction table linking students to events

### Optimized Tables
- `company_ctc`: Company CTC reference (company_name, ctc_lpa, inhand_lpa)
- `company_visits`: Company calendar data (role, location, eligibility, jd_link)
- `student_placement_summary`: Denormalized table for fast student lookups

## 📝 Technologies

- **Python 3.10+**
- **Streamlit**: Interactive web interface
- **LangChain**: LLM orchestration framework
- **Google Gemini / Groq / Ollama**: AI models for natural language processing
- **Playwright**: Robust web scraping with authentication
- **SQLite**: Lightweight relational database
- **Pandas**: Data manipulation and analysis
- **BeautifulSoup4**: HTML parsing for company calendar scraper

## 🎨 Features Highlights

### Performance Optimizations
- ✅ Denormalized `student_placement_summary` table for instant student lookups
- ✅ Indexed queries on roll numbers and company names
- ✅ Single-query architecture eliminates runtime joins
- ✅ Pre-computed CTC data for fast display

### Data Quality
- ✅ Robust name extraction from emails
- ✅ Company name normalization with comprehensive mappings
- ✅ Branch and year extraction from roll numbers
- ✅ Duplicate student detection and merging
- ✅ CPI data integration from external sources

### User Experience
- ✅ Modern chat-like interface with conversation history
- ✅ Placement status banners for placed students
- ✅ CTC and in-hand salary display throughout the app
- ✅ Search functionality in company calendar
- ✅ Interactive visualizations for CPI analysis

## 🔄 Updating Data

To refresh the database with latest placement data:

1. Re-run the scraper:
   ```bash
   python scrape_topics.py
   ```

2. Process new data:
   ```bash
   python process_data.py
   ```

3. Update company calendar (optional):
   ```bash
   python scrape_company_calendar.py
   ```

4. Rebuild summary tables:
   ```bash
   python -c "from create_student_summary_table import *; main()"
   ```

## ⚠️ Disclaimer

This tool is for educational purposes and personal productivity. Please respect the scraping policies and terms of service of the IIT BHU placement portal.

## 📄 License

MIT License - feel free to use and modify for your needs.

---

**Made with ❤️ for IIT BHU students**
