import streamlit as st
import sqlite3
import os
import json
import uuid
from datetime import datetime
from dotenv import load_dotenv
import pandas as pd
import requests
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

# Load data for CPI Visualizer
@st.cache_data
def load_cpi_data():
    try:
        # Use skipinitialspace=True to handle CSV formatting issues
        df = pd.read_csv('alot_LM (2).csv', skipinitialspace=True)
        df = df[['email', 'rollno', 'cpi']]
        
        with open('branch_mapping.json', 'r') as f:
            mapping = json.load(f)
        email_codes = mapping['email_codes']
        
        def extract_branch(email):
            import re
            user_part = str(email).split('@')[0]
            match = re.search(r'\.([a-z]{2,4})(\d{2})$', user_part)
            if match:
                b_code = match.group(1).lower()
                return email_codes.get(b_code, b_code.upper())
            return 'Unknown'

        df['branch'] = df['email'].apply(extract_branch)
        return df
    except Exception as e:
        return pd.DataFrame()

# Load environment variables
load_dotenv()

# Configure API
# Database Configuration
DB_PATH = "data/placement.db"

def load_chat_history():
    """Returns an empty dict to ensure per-session isolation on Streamlit Cloud."""
    return {}

def save_chat_history(history):
    """No-op to avoid global data persistence and protect user privacy."""
    pass

def is_valid_api_key(key):
    """Check if the provided key looks like a valid Gemini API key."""
    if not key:
        return False
    # Common placeholders and length check
    placeholders = ["your_gemini_api_key_here", "INSERT_KEY_HERE", "ENTER_KEY"]
    if any(p in key for p in placeholders):
        return False
    # Gemini keys usually start with AIza and are ~39-40 chars
    return len(key) >= 30 and key.startswith("AIza")

def get_llm(provider, model_name):
    """Factory to return a LangChain LLM instance."""
    if provider == "Gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return None
        return ChatGoogleGenerativeAI(model=model_name, google_api_key=api_key)
    
    elif provider == "Groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            return None
        return ChatGroq(model=model_name, groq_api_key=api_key)
    
    elif provider == "Ollama":
        ollama_base_url = st.session_state.get('ollama_url', "http://localhost:11434")
        return ChatOllama(model=model_name, base_url=ollama_base_url)
    
    return None

def format_history(history):
    """Convert session state messages to LangChain history."""
    lc_messages = []
    if history:
        for msg in history:
            if msg["role"] == "user":
                lc_messages.append(HumanMessage(content=msg["content"]))
            else:
                lc_messages.append(AIMessage(content=msg["content"]))
    return lc_messages

@st.cache_resource
def get_connection():
    """Cached database connection for better performance."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    # Enable WAL mode for better concurrent access
    conn.execute('PRAGMA journal_mode=WAL')
    return conn

@st.cache_data(ttl=300)  # Cache for 5 minutes
def run_query(query, params=None):
    """Cached query execution for frequently accessed data."""
    conn = get_connection()
    try:
        if params:
            # Convert params to tuple for hashability
            params_tuple = tuple(params) if isinstance(params, list) else params
            df = pd.read_sql_query(query, conn, params=params_tuple)
        else:
            df = pd.read_sql_query(query, conn)
        return df
    except Exception as e:
        return f"Error: {e}"

# ... (run_query ends)

# Cached helper functions for frequently accessed data
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_branches():
    """Get list of all branches."""
    conn = get_connection()
    branches = pd.read_sql("SELECT DISTINCT branch FROM students WHERE branch IS NOT NULL ORDER BY branch", conn)['branch'].tolist()
    return branches

@st.cache_data(ttl=600)
def get_years():
    """Get list of all years."""
    conn = get_connection()
    years = pd.read_sql("SELECT DISTINCT year FROM students WHERE year IS NOT NULL ORDER BY year", conn)['year'].tolist()
    return years

@st.cache_data(ttl=600)
def get_companies():
    """Get list of all companies."""
    conn = get_connection()
    companies = pd.read_sql("SELECT DISTINCT company_name FROM events ORDER BY company_name", conn)['company_name'].tolist()
    return companies

@st.cache_data(ttl=300)
def get_students_by_filters(branch=None, year=None):
    """Get students filtered by branch and/or year."""
    query = "SELECT DISTINCT name, roll_no FROM students WHERE 1=1"
    params = []
    if branch and branch != "All":
        query += " AND branch = ?"
        params.append(branch)
    if year and year != "All":
        query += " AND year = ?"
        params.append(year)
    query += " ORDER BY name"
    return run_query(query, params if params else None)

def generate_sql(question, model_name, provider, history=None):
    # Schema Definition for the LLM
    schema = """
    Table: events
    Columns: id (INTEGER), company_name (TEXT), event_type (TEXT), raw_filename (TEXT), topic_url (TEXT)
    
    Table: students
    Columns: id (INTEGER), roll_no (TEXT), email (TEXT), name (TEXT), branch (TEXT), year (TEXT), cpi (REAL)
    
    Table: event_students
    Columns: id (INTEGER), student_id (INTEGER), event_id (INTEGER), raw_line (TEXT)
    Foreign Keys: student_id -> students.id, event_id -> events.id
    """
    
    prompt_text = f"""
    You are a SQL Expert for SQLite. Convert the natural language question into a SQL query.
    
    {schema}
    
    Rules:
    1. Return ONLY the SQL. No markdown.
    2. JOIN logic: students -> event_students -> events.
    3. Multi-part name matching: `s.name LIKE '%Part1%' AND s.name LIKE '%Part2%'`.
    4. Offer types: 'Offer', 'PPO', 'Pre-Placement'.
    5. Columns: `s.name, s.roll_no, s.branch, s.cpi, e.company_name, e.event_type`.
    6. CPI queries: Use `s.cpi` for CPI-based filtering (e.g., `WHERE s.cpi > 9.0`).
    
    Question: {question}
    SQL:
    """
    
    llm = get_llm(provider, model_name)
    if not llm:
        return f"Error: {provider} LLM not initialized."
    
    messages = format_history(history)
    messages.append(HumanMessage(content=prompt_text))
    
    try:
        response = llm.invoke(messages)
        sql = response.content.replace("```sql", "").replace("```", "").replace("sql", "").strip()
        # Basic cleanup if model includes reasoning/text
        if "SELECT" in sql.upper():
            start = sql.upper().find("SELECT")
            sql = sql[start:]
        return sql
    except Exception as e:
        return f"Error generating SQL: {e}"

def generate_natural_answer(question, sql, df, model_name, provider, history=None):
    # safe-guard for large results
    if len(df) > 50:
        data_context = df.head(50).to_markdown(index=False) + f"\n...(and {len(df)-50} more rows)"
    else:
        data_context = df.to_markdown(index=False)

    prompt_text = f"""
    You are a helpful assistant for the IIT BHU Placement Cell.
    
    User Question: {question}
    Executed SQL: {sql}
    Result Data:
    {data_context}
    
    Task: Answer the user's question naturally based ONLY on the result data.
    
    Rules:
    - No Hallucinations: Use only given data.
    - If empty result, say "I couldn't find any records."
    - Use bullet points and bold text.
    - Do NOT mention "SQL" or "dataframe".
    """
    
    llm = get_llm(provider, model_name)
    if not llm:
        return f"Error: {provider} LLM not initialized."
    
    messages = format_history(history)
    messages.append(HumanMessage(content=prompt_text))
    
    try:
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        return f"Error generating answer: {e}"

# Streamlit UI
st.set_page_config(page_title="Placement Query Bot", page_icon="🎓", layout="wide")

# Custom CSS for GPT-like UI
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #171923 !important;
        border-right: 1px solid #2d3748;
    }
    
    /* Premium button styling for New Chat */
    .new-chat-btn {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        width: 100%;
        padding: 10px 14px;
        background-color: transparent;
        border: 1px solid #4a5568 !important;
        border-radius: 6px !important;
        color: #e2e8f0 !important;
        font-weight: 500;
        margin-bottom: 20px;
        transition: all 0.2s ease;
        text-decoration: none;
    }
    .new-chat-btn:hover {
        background-color: #2d3748 !important;
        border-color: #718096 !important;
    }
    
    /* History Button styling */
    .stButton>button {
        width: 100%;
        text-align: left;
        background-color: transparent !important;
        border: none !important;
        color: #a0aec0 !important;
        padding: 8px 12px !important;
        font-size: 14px !important;
        border-radius: 6px !important;
        transition: background-color 0.2s;
        margin-bottom: 2px !important;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        display: block !important;
    }
    
    .stButton>button:hover {
        background-color: #2d3748 !important;
        color: #edf2f7 !important;
    }
    
    .stButton>button:active {
        background-color: #4a5568 !important;
    }

    /* Active Chat Highlight (Simulation) */
    .active-chat {
        background-color: #2d3748 !important;
        color: #ffffff !important;
    }

    /* Hide default streamlit decoration */
    div[data-testid="stStatusWidget"] {
        visibility: hidden;
    }
    
    /* Header optimization */
    .sidebar-header {
        font-size: 0.8rem;
        font-weight: 600;
        color: #718096;
        margin: 20px 0 10px 0;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.title("🎓 TPC Bot")
    
    # New Chat Button (Top)
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.current_chat_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()

    # --- CHAT HISTORY SECTION ---
    st.markdown('<div class="sidebar-header">Recent Chats</div>', unsafe_allow_html=True)
    
    # Initialize chat history in session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = load_chat_history()
    
    if "current_chat_id" not in st.session_state:
        st.session_state.current_chat_id = str(uuid.uuid4())

    # Sort chat history by timestamp (decending)
    sorted_chats = sorted(
        st.session_state.chat_history.items(), 
        key=lambda x: x[1].get("timestamp", ""), 
        reverse=True
    )

    if not sorted_chats:
        st.info("No recent chats yet.")
    else:
        for chat_id, chat_data in sorted_chats:
            first_msg = chat_data.get("messages", [{}])[0].get("content", "Empty Chat")
            # Truncate title nicely
            title = (first_msg[:24] + "..") if len(first_msg) > 24 else first_msg
            
            # Use key to identify selected chat
            is_active = st.session_state.current_chat_id == chat_id
            btn_label = f"💬 {title}"
            
            # Note: We can't easily change button style per-button in Streamlit without hacky CSS
            # But we can use the key to keep it distinct.
            if st.button(btn_label, key=f"hist_{chat_id}", use_container_width=True):
                st.session_state.current_chat_id = chat_id
                st.session_state.messages = chat_data.get("messages", [])
                st.rerun()

    st.markdown("---")
    st.header("🤖 AI Brain")
    
    provider = st.radio("Select Provider", ["Gemini", "Groq", "Ollama"], index=0)
    
    selected_model = None
    if provider == "Gemini":
        # API Key Handling (Nested inside Gemini block)
        api_key = os.getenv("GOOGLE_API_KEY")
        if not is_valid_api_key(api_key):
            st.warning("⚠️ API Key Missing")
            user_api_key = st.text_input("Enter Gemini API Key", type="password")
            if user_api_key:
                if is_valid_api_key(user_api_key):
                    os.environ["GOOGLE_API_KEY"] = user_api_key
                    st.success("Key set!")
                    st.rerun()
        else:
            st.success("✅ API Key Active")
            if st.button("🗑️ Clear Key"):
                os.environ["GOOGLE_API_KEY"] = ""
                st.rerun()

        available_models = [
            "gemini-2.0-flash",
            "gemini-1.5-flash",
            "gemini-1.5-pro"
        ]
        selected_model = st.selectbox("Choose Gemini Model", available_models)
        st.info("☁️ Powered by Google AI")
        
    elif provider == "Groq":
        # Groq API Key
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            st.warning("⚠️ Groq API Key Missing")
            user_groq_key = st.text_input("Enter Groq API Key", type="password")
            if user_groq_key:
                os.environ["GROQ_API_KEY"] = user_groq_key
                st.success("Key set!")
                st.rerun()
        else:
            st.success("✅ Groq Key Active")
            if st.button("🗑️ Clear Groq Key"):
                os.environ["GROQ_API_KEY"] = ""
                st.rerun()
        
        groq_models = [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it"
        ]
        selected_model = st.selectbox("Choose Groq Model", groq_models)
        st.info("⚡ Powered by Groq (Ultra-fast)")

    else:
        # Ollama
        st.info("🏠 Local LLM via Ollama")
        ollama_url = st.text_input("Ollama URL", value="http://localhost:11434")
        st.session_state.ollama_url = ollama_url
        
        def get_ollama_models_list():
            try:
                response = requests.get(f"{ollama_url}/api/tags", timeout=2)
                if response.status_code == 200:
                    return [m['name'] for m in response.json().get('models', [])]
            except:
                pass
            return []

        if st.button("🔄 Refresh Ollama Models"):
            with st.spinner("Fetching..."):
                models = get_ollama_models_list()
                if models:
                    st.session_state.ollama_models = models
                    st.success(f"Found {len(models)} models!")
                else:
                    st.error("No models found. Is Ollama running?")
        
        ollama_models = st.session_state.get('ollama_models', [])
        if not ollama_models:
            ollama_models = get_ollama_models_list()
            st.session_state.ollama_models = ollama_models
            
        if ollama_models:
            selected_model = st.selectbox("Choose Local Model", ollama_models)
        else:
            st.warning("No Ollama models detected.")
            st.markdown("[Install Ollama](https://ollama.com) & run `ollama run llama3.2`")
    
    st.markdown("---")

# Database Stats - Cached
@st.cache_data(ttl=600)  # Cache for 10 minutes
def get_db_stats():
    """Get database statistics with caching."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("SELECT COUNT(DISTINCT roll_no) FROM students")
    total_students = c.fetchone()[0]
    c.execute("SELECT COUNT(DISTINCT company_name) FROM events")
    total_companies = c.fetchone()[0]
    return total_students, total_companies

total_students, total_companies = get_db_stats()

# with st.sidebar:
#     st.markdown("---")
#     st.header("⚙️ Data")
#     st.write(f"📊 **Students:** {total_students}")
#     st.write(f"🏢 **Companies:** {total_companies}")
    
#     if st.button("🔄 Refresh DB"):
#         with st.spinner("Processing..."):
#             try:
#                 import process_data
#                 process_data.process_files()
#                 st.success("Done! Reloading...")
#                 st.rerun()
#             except Exception as e:
#                 st.error(f"Error: {e}")

# Main Interface Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(["💬 Chat Assistant", "🔍 Student Explorer", "🏢 Company Explorer", "📊 CPI Visualizer", "📅 Company Calendar"])

# --- TAB 1: CHAT ---
with tab1:
    st.header("Ask anything about placements")
    st.markdown("Examples: *'Analysis of Sameer Wanjari'*, *'How many Physics students got offers?'*")

    # Chat History logic
    if "messages" not in st.session_state:
        # Try to load existing chat if current_chat_id is in history
        if "current_chat_id" in st.session_state and st.session_state.current_chat_id in st.session_state.chat_history:
            st.session_state.messages = st.session_state.chat_history[st.session_state.current_chat_id].get("messages", [])
        else:
            st.session_state.messages = []

    # Display Chat History
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if provider == "Gemini" and not os.getenv("GOOGLE_API_KEY"):
        st.warning("⚠️ **Gemini API Key is missing!**")
        st.info("Please select 'Gemini' and enter it in the sidebar.")
    elif provider == "Groq" and not os.getenv("GROQ_API_KEY"):
        st.warning("⚠️ **Groq API Key is missing!**")
        st.info("Please select 'Groq' and enter it in the sidebar.")
    st.markdown("---")
    # Chat Input
    if prompt := st.chat_input("Ask a question..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")
            
            try:
                # 1. Generate SQL
                sql_query = generate_sql(prompt, selected_model, provider, st.session_state.messages[:-1])
                
                # 2. Execute SQL
                result = run_query(sql_query)
                
                if isinstance(result, pd.DataFrame):
                    # 3. Generate Natural Language Answer
                    nl_response = generate_natural_answer(prompt, sql_query, result, selected_model, provider, st.session_state.messages[:-1])
                    message_placeholder.markdown(nl_response)
                    
                    # Save to history
                    st.session_state.messages.append({"role": "assistant", "content": nl_response})
                    
                    with st.expander("View Technical Details (SQL & Data)"):
                        st.code(sql_query, language="sql")
                        st.dataframe(result)
                else:
                    message_placeholder.error(result)
                    st.session_state.messages.append({"role": "assistant", "content": f"Error: {result}"})
                    
            except Exception as e:
                message_placeholder.error(f"An error occurred: {e}")
                st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})
            
            # Save to persistent history after assistant responds
            st.session_state.chat_history[st.session_state.current_chat_id] = {
                "timestamp": datetime.now().isoformat(),
                "messages": st.session_state.messages
            }
            save_chat_history(st.session_state.chat_history)
        
        # Use rerun to ensure the history loop takes over and pins the input box to the bottom
        st.rerun()

# --- TAB 2: EXPLORER ---
with tab2:
    st.header("Student Profile Explorer")
    
    # 1. Filters - Use cached functions
    col1, col2 = st.columns(2)
    with col1:
        branches = get_branches()
        selected_branch = st.selectbox("Filter by Branch", ["All"] + branches)
    
    with col2:
        years = get_years()
        selected_year = st.selectbox("Filter by Year", ["All"] + years)
    
    # 2. Student Selector - Use cached function
    students_df = get_students_by_filters(selected_branch, selected_year)
    
    if isinstance(students_df, str) or students_df.empty:
        st.warning("No students found with filters.")
    else:
        # Create display label "Name (Roll)"
        student_options = [f"{row['name']} ({row['roll_no']})" for _, row in students_df.iterrows()]
        selected_student_str = st.selectbox("Select Student", student_options, index=None, placeholder="Type to search...")
        
        if selected_student_str:
            # Extract Roll
            roll_no = selected_student_str.split("(")[-1].strip(")")
            
            st.markdown("---")
            st.subheader(f"Profile: {selected_student_str}")
            
            # Fetch ALL student data using primary tables and JOINs
            summary_query = """
                SELECT s.name, s.roll_no, s.branch, s.year, 
                       e.company_name, e.event_type, e.topic_url,
                       cc.ctc_lpa, cc.ctc_inr, cc.inhand_lpa, cc.inhand_inr
                FROM students s
                LEFT JOIN event_students es ON s.id = es.student_id
                LEFT JOIN events e ON es.event_id = e.id
                LEFT JOIN company_ctc cc ON LOWER(e.company_name) = LOWER(cc.company_name)
                WHERE s.roll_no = ?
                ORDER BY e.event_type, e.company_name
            """
            student_data = run_query(summary_query, [roll_no])
            
            if isinstance(student_data, pd.DataFrame) and not student_data.empty:
                # Merge with CPI data from CSV
                cpi_data = load_cpi_data()
                if not cpi_data.empty:
                    student_data = student_data.merge(cpi_data[['rollno', 'cpi']], left_on='roll_no', right_on='rollno', how='left')
                else:
                    student_data['cpi'] = None
                    
                # Get student info from first row
                student_info = student_data.iloc[0]
                
                # Check for offers and show placement banner
                offers_df = student_data[student_data['event_type'].str.contains('Offer', case=False)]
                if not offers_df.empty:
                    for _, offer in offers_df.iterrows():
                        company = offer['company_name']
                        ctc_display = ""
                        if pd.notna(offer['ctc_lpa']):
                            ctc_display = f" | 💰 **CTC: {offer['ctc_lpa']:.2f} LPA**"
                        elif pd.notna(offer['ctc_inr']):
                            ctc_display = f" | 💰 **CTC: ₹{offer['ctc_inr']:,}**"
                        st.success(f"🎉 **PLACED at {company}**{ctc_display}")
                
                # Display student info as metrics
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if pd.notna(student_info['branch']):
                        st.metric("Branch", student_info['branch'])
                
                with col2:
                    if pd.notna(student_info['year']):
                        st.metric("Year", student_info['year'])
                
                with col3:
                    if pd.notna(student_info['cpi']):
                        st.metric("CPI", f"{student_info['cpi']:.2f}")
                
                # Event Timeline
                st.write("#### 📅 Event Timeline")
                
                # Summary Metrics
                offers = student_data[student_data['event_type'].str.contains('Offer', case=False)]
                interviews = student_data[student_data['event_type'].str.contains('Interview', case=False)]
                tests = student_data[student_data['event_type'].str.contains('Test', case=False)]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Offers", len(offers))
                m2.metric("Interviews", len(interviews))
                m3.metric("Tests", len(tests))
                
                # Group by event type
                for etype in student_data['event_type'].unique():
                    with st.expander(f"{etype} ({len(student_data[student_data['event_type']==etype])})", expanded=True):
                        subset = student_data[student_data['event_type'] == etype]
                        for _, row in subset.iterrows():
                            # Build CTC info if it's an offer
                            ctc_info = ""
                            if 'Offer' in row['event_type']:
                                if pd.notna(row['ctc_lpa']):
                                    ctc_info = f" | 💰 CTC: {row['ctc_lpa']:.2f} LPA"
                                    if pd.notna(row['inhand_lpa']):
                                        ctc_info += f" | In-hand: {row['inhand_lpa']:.2f} LPA"
                                elif pd.notna(row['ctc_inr']):
                                    ctc_val = row['ctc_inr'] / 100000
                                    ctc_info = f" | 💰 CTC: {ctc_val:.2f} LPA"
                                    if pd.notna(row['inhand_inr']):
                                        inhand_val = row['inhand_inr'] / 100000
                                        ctc_info += f" | In-hand: {inhand_val:.2f} LPA"
                            
                            # Display with link and CTC
                            if row['topic_url']:
                                st.markdown(f"- [{row['company_name']}]({row['topic_url']}){ctc_info}")
                            else:
                                st.markdown(f"- {row['company_name']}{ctc_info}")
            else:
                st.info("No recorded events for this student.")

# --- TAB 3: COMPANY EXPLORER ---
with tab3:
    st.header("🏢 Company Explorer")

    # 1. Company Selector - Use cached function
    companies = get_companies()
    if not companies:
        st.warning("No companies found.")
    else:
        selected_company = st.selectbox("Select Company", companies, index=None, placeholder="Choose a company...")

        if selected_company:
            st.markdown("---")
            st.subheader(f"Results for: {selected_company}")

            # Fetch relevant events and students
            # We need to distinguish between FT and Intern
            
            # Get IDs of events for this company
            events_df = run_query("SELECT id, event_type, topic_url FROM events WHERE company_name = ?", [selected_company])
            
            if events_df.empty:
                st.info("No events found for this company.")
            else:
                # Separate Full-Time and Internship Events
                ft_events_df = events_df[~events_df['event_type'].str.contains("Internship|Intern", case=False, regex=True)]
                intern_events_df = events_df[events_df['event_type'].str.contains("Internship|Intern", case=False, regex=True)]
                
                def display_events_table(events_subset, section_title):
                    if events_subset.empty:
                        return
                        
                    st.subheader(section_title)
                    # Get unique event types in this subset
                    unique_types = events_subset['event_type'].unique()
                    
                    for etype in sorted(unique_types):
                        # Filter events for this specific type
                        matched_ids = events_subset[events_subset['event_type'] == etype]['id'].tolist()
                        
                        # Query students with CTC info
                        placeholders = ','.join(['?'] * len(matched_ids))
                        q = f"""
                            SELECT DISTINCT s.name, s.roll_no, s.branch, s.year,
                                   cc.ctc_lpa, cc.ctc_inr, cc.inhand_lpa, cc.inhand_inr,
                                   e.event_type, e.topic_url
                            FROM event_students es
                            JOIN students s ON es.student_id = s.id
                            JOIN events e ON es.event_id = e.id
                            LEFT JOIN company_ctc cc ON LOWER(e.company_name) = LOWER(cc.company_name)
                            WHERE es.event_id IN ({placeholders})
                            ORDER BY s.name
                        """
                        
                        results = run_query(q, matched_ids)
                        
                        if isinstance(results, str):
                            st.error(f"Query Error: {results}")
                        elif not results.empty:
                            # Merge with CPI data from CSV
                            cpi_data = load_cpi_data()
                            if not cpi_data.empty:
                                results = results.merge(cpi_data[['rollno', 'cpi']], left_on='roll_no', right_on='rollno', how='left')
                            else:
                                results['cpi'] = None
                                
                            with st.expander(f"{etype} ({len(results)})", expanded=False):
                                # Show Source Link if available
                                links = events_subset[events_subset['event_type'] == etype]['topic_url'].unique()
                                if len(links) > 0 and links[0]:
                                    st.markdown(f"🔗 **[View Original Forum Post]({links[0]})**")
                                
                                # Add CTC and In-hand columns
                                display_df = results[['name', 'roll_no', 'branch', 'year', 'cpi', 'ctc_lpa', 'ctc_inr', 'inhand_lpa', 'inhand_inr']].copy()
                                
                                # Format CTC column
                                display_df['CTC'] = display_df.apply(
                                    lambda row: f"{row['ctc_lpa']:.2f} LPA" if pd.notna(row['ctc_lpa']) 
                                    else f"₹{row['ctc_inr']:,}" if pd.notna(row['ctc_inr']) 
                                    else "N/A", axis=1
                                )
                                
                                # Format In-hand column
                                display_df['In-hand'] = display_df.apply(
                                    lambda row: f"{row['inhand_lpa']:.2f} LPA" if pd.notna(row['inhand_lpa']) 
                                    else f"₹{row['inhand_inr']:,}" if pd.notna(row['inhand_inr']) 
                                    else "N/A", axis=1
                                )
                                
                                # Format CPI
                                display_df['CPI'] = display_df['cpi'].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                                
                                # Select and rename columns for display
                                display_df = display_df[['name', 'roll_no', 'branch', 'year', 'CPI', 'CTC', 'In-hand']]
                                display_df.columns = ["Name", "Roll No", "Branch", "Year", "CPI", "CTC", "In-hand"]
                                
                                st.dataframe(display_df, hide_index=True, use_container_width=True)

                display_events_table(ft_events_df, "🎓 Full-Time")
                display_events_table(intern_events_df, "💼 Internship")


# --- TAB 4: CPI VISUALIZER ---
with tab4:
    st.header("📊 CPI Data Visualization")
    cpi_df = load_cpi_data()
    
    if cpi_df is None or (isinstance(cpi_df, pd.DataFrame) and cpi_df.empty):
        st.error("Could not load CPI data. Please ensure 'alot_LM (2).csv' and 'branch_mapping.json' exist.")
    else:
        # Local navigation for the tab
        cpi_page = st.selectbox('Select Visualization Feature', [
            'Overall Stats', 
            'Branch-wise Stats', 
            'CPI Plots', 
            'CPI Range Filter', 
            'Top Students per Branch', 
            'Branch Comparisons',
            'Distribution Analysis',
            'Student Search'
        ])
        
        st.markdown("---")
        
        if cpi_page == 'Student Search':
            st.subheader('🔍 Search Students by Name or Roll Number')
            
            # Search input
            search_query = st.text_input('Enter student name or roll number', placeholder='e.g., "sameer" or "21051"')
            
            # Branch filter (optional)
            col1, col2 = st.columns(2)
            with col1:
                branches = sorted(cpi_df['branch'].unique())
                selected_branches = st.multiselect('Filter by Branch (optional)', branches)
            
            with col2:
                # CPI range filter (optional)
                min_cpi = float(cpi_df['cpi'].min())
                max_cpi = float(cpi_df['cpi'].max())
                cpi_range = st.slider('Filter by CPI Range (optional)', min_cpi, max_cpi, (min_cpi, max_cpi))
            
            if search_query and len(search_query) >= 2:  # Require at least 2 characters
                # Fetch names from students table and merge with cpi_df for searching
                students_names_df = run_query("SELECT DISTINCT roll_no, name FROM students")
                
                search_scope_df = cpi_df.copy()
                if isinstance(students_names_df, pd.DataFrame) and not students_names_df.empty:
                    search_scope_df = search_scope_df.merge(students_names_df, left_on='rollno', right_on='roll_no', how='left', suffixes=('_cpi', '_students'))
                    # Use the name from the students table, if available
                    search_scope_df['name'] = search_scope_df['name_students'].fillna(search_scope_df['email'].apply(lambda x: x.split('@')[0] if pd.notna(x) else ''))
                else:
                    search_scope_df['name'] = search_scope_df['email'].apply(lambda x: x.split('@')[0] if pd.notna(x) else '') # Fallback to email prefix
                
                search_lower = search_query.lower().strip()
                
                # Filter by search query - more efficient filtering
                mask = (search_scope_df['email'].str.lower().str.contains(search_lower, na=False, regex=False) | 
                        search_scope_df['rollno'].astype(str).str.lower().str.contains(search_lower, na=False, regex=False) |
                        search_scope_df['name'].str.lower().str.contains(search_lower, na=False, regex=False))
                filtered_df = search_scope_df[mask].copy()
                
                # Apply branch filter if selected
                if selected_branches:
                    filtered_df = filtered_df[filtered_df['branch'].isin(selected_branches)]
                
                # Apply CPI range filter
                filtered_df = filtered_df[(filtered_df['cpi'] >= cpi_range[0]) & (filtered_df['cpi'] <= cpi_range[1])]
                
                # Sort by CPI descending
                filtered_df = filtered_df.sort_values('cpi', ascending=False)
                
                if not filtered_df.empty:
                    st.success(f"✅ Found **{len(filtered_df)}** student(s) matching your search")
                    
                    # Limit display to prevent performance issues
                    max_display = 50
                    display_limit = min(len(filtered_df), max_display)
                    
                    if len(filtered_df) > max_display:
                        st.warning(f"⚠️ Showing top {max_display} results out of {len(filtered_df)}. Use filters to narrow down.")
                    
                    # Show detailed table first (more efficient)
                    display_df = filtered_df.head(display_limit)[['rollno', 'name', 'email', 'branch', 'cpi']].copy()
                    display_df.columns = ['Roll No', 'Name', 'Email', 'Branch', 'CPI']
                    st.dataframe(display_df, hide_index=True, use_container_width=True)
                    
                    # Statistics for search results
                    if len(filtered_df) > 1:
                        st.markdown("### 📈 Statistics for Search Results")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Total Students", len(filtered_df))
                        col2.metric("Average CPI", f"{filtered_df['cpi'].mean():.2f}")
                        col3.metric("Highest CPI", f"{filtered_df['cpi'].max():.2f}")
                        col4.metric("Lowest CPI", f"{filtered_df['cpi'].min():.2f}")
                else:
                    st.warning(f"⚠️ No students found matching '{search_query}' with the selected filters")
                    st.info("💡 Try adjusting your search query or removing some filters")
            elif search_query and len(search_query) < 2:
                st.info("👆 Please enter at least 2 characters to search")
            else:
                st.info("👆 Enter a student name or roll number to search")
                
                # Show some example searches
                with st.expander("💡 Search Tips"):
                    st.markdown("""
                    - **Search by name**: Enter any part of the student's email (e.g., "sameer", "wanjari")
                    - **Search by roll number**: Enter the full or partial roll number (e.g., "21051234" or "2105")
                    - **Use filters**: Combine search with branch and CPI filters for more precise results
                    - **Case insensitive**: Search is not case-sensitive
                    - **Minimum 2 characters**: Enter at least 2 characters to start searching
                    """)
        
        elif cpi_page == 'Overall Stats':
            st.subheader('Overall Statistics')
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Students", len(cpi_df))
            col2.metric("Average CPI", round(cpi_df['cpi'].mean(), 2))
            col3.metric("Max CPI", cpi_df['cpi'].max())
            
            st.write("#### Branch-wise Student Distribution")
            branch_counts = cpi_df['branch'].value_counts().reset_index()
            branch_counts.columns = ['Branch', 'Count']
            st.dataframe(branch_counts, hide_index=True, use_container_width=True)

        elif cpi_page == 'Branch-wise Stats':
            st.subheader('Branch-wise Statistics')
            branches = sorted(cpi_df['branch'].unique())
            selected_branch = st.selectbox('Select Branch', branches)
            branch_df = cpi_df[cpi_df['branch'] == selected_branch]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Students", len(branch_df))
            col2.metric("Avg CPI", round(branch_df['cpi'].mean(), 2))
            col3.metric("Max CPI", branch_df['cpi'].max())
            
            st.dataframe(branch_df[['rollno', 'email', 'cpi']], hide_index=True, use_container_width=True)

        elif cpi_page == 'CPI Plots':
            st.subheader('CPI Distribution Plots')
            plot_type = st.radio('Select Plot Type', ['Histogram', 'Boxplot'], horizontal=True)
            branches = st.multiselect('Select Branches', sorted(cpi_df['branch'].unique()), default=sorted(cpi_df['branch'].unique())[:3])
            
            if branches:
                plot_df = cpi_df[cpi_df['branch'].isin(branches)]
                fig, ax = plt.subplots(figsize=(10, 6))
                if plot_type == 'Histogram':
                    sns.histplot(data=plot_df, x='cpi', hue='branch', multiple='stack', ax=ax, palette='viridis')
                else:
                    sns.boxplot(data=plot_df, x='branch', y='cpi', ax=ax, palette='Set2')
                    plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.warning("Please select at least one branch.")

        elif cpi_page == 'CPI Range Filter':
            st.subheader('Filter Students by CPI Range')
            min_val = float(cpi_df['cpi'].min())
            max_val = float(cpi_df['cpi'].max())
            range_val = st.slider('Select Range', min_val, max_val, (7.0, 9.0))
            
            branches = st.multiselect('Filter by Branches (optional)', sorted(cpi_df['branch'].unique()))
            filtered = cpi_df[(cpi_df['cpi'] >= range_val[0]) & (cpi_df['cpi'] <= range_val[1])]
            
            if branches:
                filtered = filtered[filtered['branch'].isin(branches)]
            
            st.write(f"**Found {len(filtered)} students in this range.**")
            st.dataframe(filtered[['rollno', 'email', 'branch', 'cpi']], hide_index=True, use_container_width=True)

        elif cpi_page == 'Top Students per Branch':
            st.subheader('Top Academic Performers')
            n = st.number_input('Show Top N Students', min_value=1, value=5)
            selected_branches = st.multiselect('Select Branches', sorted(cpi_df['branch'].unique()), default=sorted(cpi_df['branch'].unique())[:2])
            
            for branch in selected_branches:
                with st.expander(f"Top performers in {branch}"):
                    top = cpi_df[cpi_df['branch'] == branch].nlargest(int(n), 'cpi')[['rollno', 'email', 'cpi']]
                    st.dataframe(top, hide_index=True, use_container_width=True)

        elif cpi_page == 'Branch Comparisons':
            st.subheader('Cross-Branch Analysis')
            branches = st.multiselect('Select Branches to Compare', sorted(cpi_df['branch'].unique()), default=sorted(cpi_df['branch'].unique())[:3])
            
            if len(branches) >= 2:
                # Include Standard Deviation as requested
                comp_df = cpi_df[cpi_df['branch'].isin(branches)].groupby('branch')['cpi'].agg(['mean', 'std', 'max', 'min', 'count']).reset_index()
                comp_df.columns = ['Branch', 'Avg CPI', 'Std Dev', 'Max CPI', 'Min CPI', 'Students']
                
                st.write("#### Statistical Comparison")
                st.dataframe(comp_df.round(2), hide_index=True, use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(data=comp_df, x='Branch', y='Avg CPI', ax=ax, palette='coolwarm')
                ax.set_title("Average CPI Comparison")
                plt.xticks(rotation=45)
                st.pyplot(fig)
            else:
                st.info('Select at least 2 branches to compare.')

        elif cpi_page == 'Distribution Analysis':
            st.subheader('CPI Distribution & Skewness Analysis')
            branches = st.multiselect('Select Branches for Analysis', sorted(cpi_df['branch'].unique()), default=sorted(cpi_df['branch'].unique())[:1])
            
            if branches:
                analysis_df = cpi_df[cpi_df['branch'].isin(branches)]
                
                # Distribution Plot: Jittered Scatter + Histogram + KDE
                fig, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)}, figsize=(10, 7))
                
                # Boxplot with jittered scatter on top
                sns.boxplot(x=analysis_df["cpi"], ax=ax_box, color='lightgray')
                sns.stripplot(x=analysis_df["cpi"], ax=ax_box, alpha=0.5, color='blue', jitter=True)
                ax_box.set(xlabel='')
                
                # Main distribution plot
                sns.histplot(data=analysis_df, x="cpi", kde=True, ax=ax_hist, color='skyblue', stat="density")
                
                st.pyplot(fig)
                
                # Stat Calculations
                skewness = analysis_df['cpi'].skew()
                kurtosis = analysis_df['cpi'].kurt()
                
                col1, col2 = st.columns(2)
                col1.metric("Skewness", round(skewness, 4))
                col2.metric("Kurtosis", round(kurtosis, 4))
                
                # Explanation
                with st.expander("Interpretation of Results"):
                    st.write("**Skewness:**")
                    if abs(skewness) < 0.5:
                        st.write("The distribution is **fairly symmetrical**.")
                    elif 0.5 <= abs(skewness) < 1:
                        st.write("The distribution is **moderately skewed**.")
                    else:
                        st.write("The distribution is **highly skewed**.")
                        
                    st.write("**Distribution Type Estimation:**")
                    # Simple normality check
                    if len(analysis_df) >= 8:
                        stat, p = stats.normaltest(analysis_df['cpi'])
                        if p > 0.05:
                            st.success("The distribution appears to be **Gaussian (Normal)** based on D'Agostino's K^2 test (p > 0.05).")
                        else:
                            st.warning("The distribution is **not normally distributed** (p < 0.05).")
                    else:
                        st.info("Sample size too small for statistical normality testing.")
            else:
                st.warning("Please select at least one branch.")


# --- TAB 5: COMPANY CALENDAR ---
with tab5:
    st.header("📅 Company Calendar 2025-26")
    st.markdown("View upcoming company visits with CTC, role, and location details")
    
    # Search bar
    search_query = st.text_input("🔍 Search Company", placeholder="Type company name...")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_ctc = st.number_input("Min CTC (LPA)", min_value=0.0, value=0.0, step=0.5)
    
    with col2:
        # Optimized location fetching
        @st.cache_data(ttl=600)
        def get_visit_locations():
            conn = get_connection()
            locations_df = pd.read_sql("SELECT DISTINCT location FROM company_visits WHERE location IS NOT NULL", conn)
            return sorted([loc for loc in locations_df['location'].tolist() if loc])
            
        all_locations = ["All"] + get_visit_locations()
        selected_location = st.selectbox("Location", all_locations)
    
    with col3:
        sort_by = st.selectbox("Sort by", ["Company Name", "CTC (High to Low)", "CTC (Low to High)"])
    
    # Optimized query with placement counts
    query = """
        SELECT cv.*, 
               (SELECT COUNT(DISTINCT es.student_id) 
                FROM events e 
                JOIN event_students es ON e.id = es.event_id 
                WHERE LOWER(e.company_name) = LOWER(cv.company_name) 
                AND e.event_type LIKE '%Offer%') as placement_count
        FROM company_visits cv 
        WHERE 1=1
    """
    params = []
    
    # Add search filter
    if search_query:
        query += " AND company_name LIKE ?"
        params.append(f"%{search_query}%")
    
    if min_ctc > 0:
        query += " AND (ctc_lpa >= ? OR ctc_inr >= ?)"
        params.extend([min_ctc, min_ctc * 100000])
    
    if selected_location != "All":
        query += " AND cv.location LIKE ?"
        params.append(f"%{selected_location}%")
    
    # Add sorting
    if sort_by == "Company Name":
        query += " ORDER BY cv.company_name"
    elif sort_by == "CTC (High to Low)":
        query += " ORDER BY COALESCE(cv.ctc_lpa, cv.ctc_inr/100000.0) DESC"
    else:
        query += " ORDER BY COALESCE(cv.ctc_lpa, cv.ctc_inr/100000.0) ASC"
    
    # Fetch data using cached run_query
    companies_df = run_query(query, params if params else None)
    
    if isinstance(companies_df, str):
        st.error(companies_df)
    else:
        st.markdown(f"**Found {len(companies_df)} companies**")
        
        if not companies_df.empty:
            # Display companies
            for _, company in companies_df.iterrows():
                with st.expander(f"**{company['company_name']}** - {company['role'] if company['role'] else 'Multiple Roles'}"):
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if company['ctc_lpa']:
                            st.metric("CTC", f"{company['ctc_lpa']} LPA")
                        elif company['ctc_inr']:
                            st.metric("CTC", f"₹{company['ctc_inr']:,}")
                        else:
                            st.metric("CTC", "Refer JD")
                    
                    with col2:
                        if company['inhand_lpa']:
                            st.metric("In-hand", f"{company['inhand_lpa']} LPA")
                        elif company['inhand_inr']:
                            st.metric("In-hand", f"₹{company['inhand_inr']:,}")
                        else:
                            st.metric("In-hand", "Refer JD")
                    
                    with col3:
                        if company['eligibility_cgpa']:
                            st.metric("Min CGPA", f"{company['eligibility_cgpa']}")
                    
                    # Role and Location
                    if company['role']:
                        st.write(f"**Role:** {company['role']}")
                    
                    if company['location']:
                        st.write(f"**Location:** {company['location']}")
                    
                    if company['eligible_departments']:
                        st.write(f"**Eligible Departments:** {company['eligible_departments']}")
                    
                    # JD Link
                    if company['jd_link']:
                        st.markdown(f"📄 [View Job Description]({company['jd_link']})")
                    
                    if company['placement_count'] > 0:
                        st.success(f"✅ {int(company['placement_count'])} students placed from our database")
        else:
            st.info("No companies found matching the filters.")
    

