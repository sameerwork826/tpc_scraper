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

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    return conn

def run_query(query, params=None):
    conn = get_db_connection()
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        return f"Error: {e}"

# ... (run_query ends)

def generate_sql(question, model_name, provider, history=None):
    # Schema Definition for the LLM
    schema = """
    Table: events
    Columns: id (INTEGER), company_name (TEXT), event_type (TEXT), raw_filename (TEXT), topic_url (TEXT)
    
    Table: students
    Columns: id (INTEGER), roll_no (TEXT), email (TEXT), name (TEXT), branch (TEXT), year (TEXT)
    
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
    5. Columns: `s.name, s.roll_no, s.branch, e.company_name, e.event_type`.
    
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

# Database Stats
conn = get_db_connection()
c = conn.cursor()
c.execute("SELECT COUNT(DISTINCT roll_no) FROM students")
total_students = c.fetchone()[0]
c.execute("SELECT COUNT(DISTINCT company_name) FROM events")
total_companies = c.fetchone()[0]
conn.close()

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
tab1, tab2, tab3 = st.tabs(["💬 Chat Assistant", "🔍 Student Explorer", "🏢 Company Explorer"])

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
    
    conn = get_db_connection()
    
    # 1. Filters
    col1, col2 = st.columns(2)
    with col1:
        branches = pd.read_sql("SELECT DISTINCT branch FROM students WHERE branch IS NOT NULL ORDER BY branch", conn)['branch'].tolist()
        selected_branch = st.selectbox("Filter by Branch", ["All"] + branches)
    
    with col2:
        years = pd.read_sql("SELECT DISTINCT year FROM students WHERE year IS NOT NULL ORDER BY year", conn)['year'].tolist()
        selected_year = st.selectbox("Filter by Year", ["All"] + years)
    
    # 2. Student Selector
    query = "SELECT DISTINCT name, roll_no FROM students WHERE 1=1"
    params = []
    if selected_branch != "All":
        query += " AND branch = ?"
        params.append(selected_branch)
    if selected_year != "All":
        query += " AND year = ?"
        params.append(selected_year)
    
    query += " ORDER BY name"
    
    students_df = pd.read_sql(query, conn, params=params)
    
    if students_df.empty:
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
            
            # Fetch History
            history_query = """
                SELECT e.company_name, e.event_type, e.topic_url
                FROM event_students es
                JOIN students s ON es.student_id = s.id
                JOIN events e ON es.event_id = e.id
                WHERE s.roll_no = ?
                ORDER BY e.event_type, e.company_name
            """
            history = pd.read_sql(history_query, conn, params=[roll_no])
            
            if not history.empty:
                # Summary Metrics
                offers = history[history['event_type'].str.contains('Offer', case=False)]
                interviews = history[history['event_type'].str.contains('Interview', case=False)]
                tests = history[history['event_type'].str.contains('Test', case=False)]
                
                m1, m2, m3 = st.columns(3)
                m1.metric("Offers", len(offers))
                m2.metric("Interviews", len(interviews))
                m3.metric("Tests", len(tests))
                
                # Detailed Timeline
                st.write("#### 📅 Event Timeline")
                
                # Group by type for cleaner view
                for etype in history['event_type'].unique():
                    with st.expander(f"{etype} ({len(history[history['event_type']==etype])})", expanded=True):
                        subset = history[history['event_type'] == etype]
                        for _, row in subset.iterrows():
                            # Markdown list with link
                            if row['topic_url']:
                                st.markdown(f"- [{row['company_name']}]({row['topic_url']})")
                            else:
                                st.markdown(f"- {row['company_name']}")
            else:
                st.info("No recorded events for this student.")
    
    conn.close()

# --- TAB 3: COMPANY EXPLORER ---
with tab3:
    st.header("🏢 Company Explorer")
    conn = get_db_connection()

    # 1. Company Selector
    companies = pd.read_sql("SELECT DISTINCT company_name FROM events ORDER BY company_name", conn)['company_name'].tolist()
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
            events_df = pd.read_sql("SELECT id, event_type, topic_url FROM events WHERE company_name = ?", conn, params=[selected_company])
            
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
                        
                        # Query students
                        placeholders = ','.join(['?'] * len(matched_ids))
                        q = f"""
                            SELECT DISTINCT s.name, s.roll_no, s.branch, s.year, e.event_type, e.topic_url
                            FROM event_students es
                            JOIN students s ON es.student_id = s.id
                            JOIN events e ON es.event_id = e.id
                            WHERE es.event_id IN ({placeholders})
                            ORDER BY s.name
                        """
                        
                        results = pd.read_sql(q, conn, params=matched_ids)
                        
                        if not results.empty:
                            with st.expander(f"{etype} ({len(results)})", expanded=False):
                                # Show Source Link if available
                                links = events_subset[events_subset['event_type'] == etype]['topic_url'].unique()
                                if len(links) > 0 and links[0]:
                                    st.markdown(f"🔗 **[View Original Forum Post]({links[0]})**")
                                
                                display_df = results[['name', 'roll_no', 'branch', 'year']].copy()
                                display_df.columns = ["Name", "Roll No", "Branch", "Year"]
                                st.dataframe(display_df, hide_index=True, use_container_width=True)

                display_events_table(ft_events_df, "🎓 Full-Time")
                display_events_table(intern_events_df, "💼 Internship")

    conn.close()
