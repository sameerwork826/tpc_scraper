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
        
        # Filter for 2021 and 2022 only as requested
        df['year_code'] = df['rollno'].astype(str).str[:2]
        df['year'] = df['year_code'].apply(lambda x: "20" + x if x.isdigit() and len(x) == 2 else "Unknown")
        df = df[df['year'].isin(['2021', '2022'])]
        
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

def generate_ai_response(question, model_name, provider, history=None):
    # Schema Definition for the LLM
    schema = """
    Table: students
    Columns: id (INTEGER), roll_no (TEXT), email (TEXT), name (TEXT), branch (TEXT), year (TEXT), cpi (REAL)
    
    Table: events
    Columns: id (INTEGER), company_name (TEXT), event_type (TEXT), topic_url (TEXT)
    
    Table: event_students
    Columns: id (INTEGER), student_id (INTEGER), event_id (INTEGER)
    Foreign Keys: student_id -> students.id, event_id -> events.id
    
    Table: company_ctc
    Columns: id (INTEGER), company_name (TEXT), ctc_lpa (REAL), ctc_inr (REAL), inhand_lpa (REAL), inhand_inr (REAL)
    
    Table: company_visits
    Columns: id (INTEGER), company_name (TEXT), role (TEXT), placement_year (TEXT), ctc_lpa (REAL), ctc_inr (REAL), location (TEXT), jd_link (TEXT), eligibility_cgpa (REAL)

    Table: student_placement_summary (VIEW - PREFERRED FOR QUERYING)
    Columns: id (INTEGER), student_id (INTEGER), roll_no (TEXT), email (TEXT), name (TEXT), branch (TEXT), year (TEXT), cpi (REAL), company_name (TEXT), event_type (TEXT), ctc_lpa (REAL), ctc_inr (REAL), inhand_lpa (REAL), inhand_inr (REAL), topic_url (TEXT)
    """
    
    prompt_text = f"""
    You are a premium AI Assistant for the IIT BHU Placement Cell. 
    
    Available Data (Schema):
    {schema}
    
    YOUR GOAL: ALWAYS query the database for facts. Only answer DIRECTLY if the user greets you or asks for general advice (non-data questions).
    
    STRICT DECISION RULES:
    - Does the question imply looking up a student, company, placement statistic, or count? -> MUST generates SQL.
    - Does the question refer to "him", "her", "it", "they" (context)? -> Look at the chat history, identify the entity, and generate SQL for that entity.
    
    FORMATTING:
    - 'SQL:' followed by the query.
    - 'DIRECT:' followed by the answer.
    
    SQL RULES:
    1. **Primary Source**: `FROM student_placement_summary` (It has everything).
    2. **Context**: If user asks "What is his CPI?", find the distinct student mentioned in history and query `SELECT cpi FROM student_placement_summary WHERE name LIKE ...`.
    3. **Name Matching**: `name LIKE '%Part1%Part2%'` (e.g. `'%Sameer%Wanjari%'`).
    4. **Roll No**: `roll_no = '21174028'`.
    5. **Clean Output**: Just the SQL. No markdown.
    
    DIRECT TALK RULES (Only for greetings/advice):
    - Do NOT make up data.
    - If you can't find data in DB, say "No record found".
    
    User Question: {question}
    Response:
    """
    
    llm = get_llm(provider, model_name)
    if not llm:
        return f"Error: {provider} LLM not initialized."
    
    messages = format_history(history)
    messages.append(HumanMessage(content=prompt_text))
    
    try:
        response = llm.invoke(messages)
        content = response.content.strip()
        return content
    except Exception as e:
        return f"Error generating strategy: {e}"

def generate_natural_answer(question, sql, df, model_name, provider, history=None):
    # safe-guard for large results
    if len(df) > 50:
        data_context = df.head(50).to_markdown(index=False) + f"\n...(and {len(df)-50} more rows)"
    else:
        data_context = df.to_markdown(index=False)

    prompt_text = f"""
    You are a premium AI Assistant for the IIT BHU Placement Cell. 
    You have access to detailed placement statistics, student profiles, and company visit records for 2021-2022.
    
    User Question: {question}
    {f"Executed Query: {sql}" if sql else ""}
    Result Data:
    {data_context}
    
    Task: Answer the user's question clearly and concisely based on the data.
    
    Rules for response:
    - CONCISENESS IS KEY: Direct answers are better. Only provide summary/trends if explicitly asked or if the data is complex.
    - MAINTAIN CONTEXT: If the user refers to previous topics (like "what about him?"), use the chat history to understand.
    - NO FLUFF: Avoid "Based on the provided data" or "Here is the analysis". Just give the answer.
    - FALLBACK KNOWLEDGE: If the table is empty, politely use your internal knowledge about the company/topic if possible.
    - PRISTINE FORMATTING: Use Markdown tables/bullets only when helpful for readability.
    - NO TECHNICAL JARGON: Do NOT mention "SQL", "dataframe", or "query".
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
    if st.button("➕ New Chat", width='stretch'):
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
            if st.button(btn_label, key=f"hist_{chat_id}", width='stretch'):
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
c.execute("SELECT COUNT(DISTINCT roll_no) FROM students WHERE year IN ('2021', '2022')")
total_students = c.fetchone()[0]
c.execute("""
    SELECT COUNT(DISTINCT company_name) 
    FROM events e 
    JOIN event_students es ON e.id = es.event_id 
    JOIN students s ON es.student_id = s.id 
    WHERE s.year IN ('2021', '2022')
""")
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
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["💬 Chat Assistant", "🔍 Student Explorer", "🏢 Company Explorer", "📊 CPI Visualizer", "📅 Company Calendar", "📈 Branch Statistics"])

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
        if "messages" not in st.session_state:
            st.session_state.messages = []
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            message_placeholder.markdown("Thinking...")

            # Basic provider/model validation
            if provider == "Gemini" and not os.getenv("GOOGLE_API_KEY"):
                err = "Gemini API key is missing. Please set it in the sidebar."
                message_placeholder.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            elif provider == "Groq" and not os.getenv("GROQ_API_KEY"):
                err = "Groq API key is missing. Please set it in the sidebar."
                message_placeholder.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            elif not selected_model:
                err = "No model selected. Choose a model in the sidebar."
                message_placeholder.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            else:
                try:
                    # 1. Generate Strategy and SQL/DIRECT Response
                    ai_raw_response = generate_ai_response(prompt, selected_model, provider, st.session_state.messages[:-1])
                    
                    if ai_raw_response.startswith("DIRECT:"):
                        answer = ai_raw_response.replace("DIRECT:", "").strip()
                        message_placeholder.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    elif ai_raw_response.startswith("SQL:"):
                        sql_query = ai_raw_response.replace("SQL:", "").strip()
                        # Basic cleanup
                        sql_query = sql_query.replace("```sql", "").replace("```", "").replace("sql", "").strip()
                        if "SELECT" in sql_query.upper():
                            start = sql_query.upper().find("SELECT")
                            sql_query = sql_query[start:]
                        
                        # 2. Execute SQL
                        result = run_query(sql_query)

                        if isinstance(result, str) and result.startswith("Error"):
                            message_placeholder.error(result)
                            st.session_state.messages.append({"role": "assistant", "content": result})
                        elif isinstance(result, pd.DataFrame):
                            # 3. Generate Natural Language Answer (even if DF is empty, to allow fallback knowledge)
                            nl_response = generate_natural_answer(prompt, sql_query, result, selected_model, provider, st.session_state.messages[:-1])
                            if isinstance(nl_response, str) and nl_response.startswith("Error"):
                                message_placeholder.error(nl_response)
                                st.session_state.messages.append({"role": "assistant", "content": nl_response})
                            else:
                                message_placeholder.markdown(nl_response)
                                st.session_state.messages.append({"role": "assistant", "content": nl_response})
                                if not result.empty:
                                    with st.expander("View Technical Details (SQL & Data)"):
                                        st.code(sql_query, language="sql")
                                        st.dataframe(result, width='stretch')
                    
                    else:
                        # Fallback for unexpected response format
                        message_placeholder.markdown(ai_raw_response)
                        st.session_state.messages.append({"role": "assistant", "content": ai_raw_response})

                except Exception as e:
                    err = f"An unexpected error occurred: {e}"
                    message_placeholder.error(err)
                    st.session_state.messages.append({"role": "assistant", "content": err})

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
        years = pd.read_sql("SELECT DISTINCT year FROM students WHERE year IN ('2021', '2022') ORDER BY year", conn)['year'].tolist()
        selected_year = st.selectbox("Filter by Year", ["All"] + years)
    
    # 2. Student Selector
    query = "SELECT DISTINCT name, roll_no FROM students WHERE year IN ('2021', '2022')"
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
            
            # Fetch ALL student data from denormalized table in ONE query
            summary_query = """
                SELECT name, roll_no, branch, year, cpi, 
                       company_name, event_type, topic_url,
                       ctc_lpa, ctc_inr, inhand_lpa, inhand_inr
                FROM student_placement_summary
                WHERE roll_no = ?
                ORDER BY event_type, company_name
            """
            student_data = pd.read_sql(summary_query, conn, params=[roll_no])
            
            if not student_data.empty:
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
                        
                        # Query students with CTC info
                        placeholders = ','.join(['?'] * len(matched_ids))
                        q = f"""
                            SELECT DISTINCT s.name, s.roll_no, s.branch, s.year, s.cpi, 
                                   cc.ctc_lpa, cc.ctc_inr, cc.inhand_lpa, cc.inhand_inr,
                                   e.event_type, e.topic_url
                            FROM event_students es
                            JOIN students s ON es.student_id = s.id
                            JOIN events e ON es.event_id = e.id
                            LEFT JOIN company_ctc cc ON LOWER(e.company_name) = LOWER(cc.company_name)
                            WHERE es.event_id IN ({placeholders}) AND s.year IN ('2021', '2022')
                            ORDER BY s.name
                        """
                        
                        results = pd.read_sql(q, conn, params=matched_ids)
                        
                        if not results.empty:
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
                                
                                st.dataframe(display_df, hide_index=True, width='stretch')

                display_events_table(ft_events_df, "🎓 Full-Time")
                display_events_table(intern_events_df, "💼 Internship")

    conn.close()

# --- TAB 4: CPI VISUALIZER ---
with tab4:
    st.header("📊 CPI Data Visualization")
    cpi_df = load_cpi_data()
    
    if cpi_df.empty:
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
            'Distribution Analysis'
        ])
        
        st.markdown("---")
        
        if cpi_page == 'Overall Stats':
            st.subheader('Overall Statistics')
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Students", len(cpi_df))
            col2.metric("Average CPI", round(cpi_df['cpi'].mean(), 2))
            col3.metric("Max CPI", cpi_df['cpi'].max())
            
            st.write("#### Branch-wise Student Distribution")
            branch_counts = cpi_df['branch'].value_counts().reset_index()
            branch_counts.columns = ['Branch', 'Count']
            st.dataframe(branch_counts, hide_index=True, width='stretch')

        elif cpi_page == 'Branch-wise Stats':
            st.subheader('Branch-wise Statistics')
            branches = sorted(cpi_df['branch'].unique())
            selected_branch = st.selectbox('Select Branch', branches)
            branch_df = cpi_df[cpi_df['branch'] == selected_branch]
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Students", len(branch_df))
            col2.metric("Avg CPI", round(branch_df['cpi'].mean(), 2))
            col3.metric("Max CPI", branch_df['cpi'].max())
            
            st.dataframe(branch_df[['rollno', 'email', 'cpi']], hide_index=True, width='stretch')

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
            st.dataframe(filtered[['rollno', 'email', 'branch', 'cpi']], hide_index=True, width='stretch')

        elif cpi_page == 'Top Students per Branch':
            st.subheader('Top Academic Performers')
            n = st.number_input('Show Top N Students', min_value=1, value=5)
            selected_branches = st.multiselect('Select Branches', sorted(cpi_df['branch'].unique()), default=sorted(cpi_df['branch'].unique())[:2])
            
            for branch in selected_branches:
                with st.expander(f"Top performers in {branch}"):
                    top = cpi_df[cpi_df['branch'] == branch].nlargest(int(n), 'cpi')[['rollno', 'email', 'cpi']]
                    st.dataframe(top, hide_index=True, width='stretch')

        elif cpi_page == 'Branch Comparisons':
            st.subheader('Cross-Branch Analysis')
            branches = st.multiselect('Select Branches to Compare', sorted(cpi_df['branch'].unique()), default=sorted(cpi_df['branch'].unique())[:3])
            
            if len(branches) >= 2:
                # Include Standard Deviation as requested
                comp_df = cpi_df[cpi_df['branch'].isin(branches)].groupby('branch')['cpi'].agg(['mean', 'std', 'max', 'min', 'count']).reset_index()
                comp_df.columns = ['Branch', 'Avg CPI', 'Std Dev', 'Max CPI', 'Min CPI', 'Students']
                
                st.write("#### Statistical Comparison")
                st.dataframe(comp_df.round(2), hide_index=True, width='stretch')
                
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

    conn.close()

# --- TAB 5: COMPANY CALENDAR ---
with tab5:
    st.header("📅 Company Calendar (2021-2022)")
    st.markdown("View company visits with CTC, role, and location details")
    
    conn = get_db_connection()
    
    # Company selector (searchable dropdown) - Restrict to 2021/2022
    companies_list = pd.read_sql("SELECT DISTINCT company_name FROM company_visits WHERE placement_year IN ('2021', '2022', '2021-22', '2022-23') ORDER BY company_name", conn)
    company_options = ["All"] + companies_list['company_name'].dropna().tolist()
    selected_company = st.selectbox("🔍 Select Company", company_options, index=0)
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        min_ctc = st.number_input("Min CTC (LPA)", min_value=0.0, value=0.0, step=0.5)
    
    with col2:
        # Get unique locations
        locations_df = pd.read_sql("SELECT DISTINCT location FROM company_visits WHERE location IS NOT NULL", conn)
        all_locations = ["All"] + sorted([loc for loc in locations_df['location'].tolist() if loc])
        selected_location = st.selectbox("Location", all_locations)
    
    with col3:
        sort_by = st.selectbox("Sort by", ["Company Name", "CTC (High to Low)", "CTC (Low to High)"])
    
    # Build query
    query = "SELECT * FROM company_visits WHERE placement_year IN ('2021', '2022', '2021-22', '2022-23')"
    params = []
    
    # Add company filter from selectbox
    if selected_company != "All":
        query += " AND LOWER(company_name) = LOWER(?)"
        params.append(selected_company)
    
    if min_ctc > 0:
        query += " AND (ctc_lpa >= ? OR ctc_inr >= ?)"
        params.extend([min_ctc, min_ctc * 100000])
    
    if selected_location != "All":
        query += " AND location LIKE ?"
        params.append(f"%{selected_location}%")
    
    # Add sorting
    if sort_by == "Company Name":
        query += " ORDER BY company_name"
    elif sort_by == "CTC (High to Low)":
        query += " ORDER BY COALESCE(ctc_lpa, ctc_inr/100000.0) DESC"
    else:
        query += " ORDER BY COALESCE(ctc_lpa, ctc_inr/100000.0) ASC"
    
    # Fetch data
    companies_df = pd.read_sql(query, conn, params=params)
    
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
                
                # Check if students got placed
                placement_query = """
                    SELECT COUNT(DISTINCT s.id) as count
                    FROM events e
                    JOIN event_students es ON e.id = es.event_id
                    JOIN students s ON es.student_id = s.id
                    WHERE LOWER(e.company_name) = LOWER(?)
                    AND e.event_type LIKE '%Offer%'
                """
                placement_df = pd.read_sql(placement_query, conn, params=[company['company_name']])
                if placement_df['count'].iloc[0] > 0:
                    st.success(f"✅ {placement_df['count'].iloc[0]} students placed from our database")
    else:
        st.info("No companies found matching the filters.")
    
    conn.close()

# --- TAB 6: BRANCH STATISTICS ---
with tab6:
    st.header("📈 Branch-wise Placement Statistics")
    
    conn = get_db_connection()
    
    # Options to address warnings
    pd.set_option('future.no_silent_downcasting', True)
    
    # 1. Branch and Year Selection
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        branches = pd.read_sql("SELECT DISTINCT branch FROM students WHERE branch IS NOT NULL ORDER BY branch", conn)['branch'].tolist()
        selected_branch = st.selectbox("Select Branch", branches, key="branch_stats_select")
    
    with col_f2:
        years = pd.read_sql("SELECT DISTINCT year FROM students WHERE year IN ('2021', '2022') ORDER BY year DESC", conn)['year'].tolist()
        selected_year = st.selectbox("Select Year", ["All"] + years, key="branch_stats_year")
    
    if selected_branch:
        year_suffix = f" for {selected_year}" if selected_year != "All" else " (2021-2022)"
        st.markdown(f"### Statistics for {selected_branch}{year_suffix}")
        
        # 2. Fetch Aggregated Stats
        # Build filters - Always restrict to 2021/2022
        filters = "s.branch = ? AND s.year IN ('2021', '2022')"
        params = [selected_branch]
        if selected_year != "All":
            filters = "s.branch = ? AND s.year = ?"
            params = [selected_branch, selected_year]
            
        # Total Students in Branch/Year
        total_students = pd.read_sql(f"SELECT COUNT(*) as count FROM students s WHERE {filters}", conn, params=params).iloc[0]['count']
        
        # Placement Data (Direct Join)
        offer_query = f"""
            SELECT s.id, s.name, e.event_type, e.company_name, 
                   cc.ctc_lpa, cc.ctc_inr
            FROM students s
            JOIN event_students es ON s.id = es.student_id
            JOIN events e ON es.event_id = e.id
            LEFT JOIN company_ctc cc ON LOWER(e.company_name) = LOWER(cc.company_name)
            WHERE {filters} AND (e.event_type LIKE '%Offer%' OR e.event_type LIKE '%PPO%' OR e.event_type LIKE '%Pre-Placement%')
        """
        offers_df = pd.read_sql(offer_query, conn, params=params)
        
        # Interview Shortlists (Direct Join)
        shortlist_query = f"""
            SELECT s.id, e.event_type
            FROM students s
            JOIN event_students es ON s.id = es.student_id
            JOIN events e ON es.event_id = e.id
            WHERE {filters} AND e.event_type LIKE '%Interview%'
        """
        shortlists_df = pd.read_sql(shortlist_query, conn, params=params)
        
        # Calculate Metrics
        offers_df['lpa_equiv'] = offers_df['ctc_lpa'].fillna(offers_df['ctc_inr'] / 100000.0)
        
        placed_students_ids = offers_df['id'].unique()
        no_placed = len(placed_students_ids)
        
        # Count unique students for PPO and FT
        ppo_students = offers_df[offers_df['event_type'].str.contains('PPO|Pre-Placement', case=False, na=False)]['id'].unique()
        ppo_count = len(ppo_students)
        
        ft_students = offers_df[offers_df['event_type'].str.contains('FT Offer', case=False, na=False)]['id'].unique()
        ft_count = len(ft_students)
        
        placement_percentage = (no_placed / total_students * 100) if total_students > 0 else 0
        
        # Average Package Calculation: One (highest) offer per student
        best_offers_per_student = offers_df.sort_values('lpa_equiv', ascending=False).drop_duplicates('id')
        avg_package = best_offers_per_student['lpa_equiv'].mean() if not best_offers_per_student['lpa_equiv'].dropna().empty else 0
        
        # Display Metrics
        col1, col2, col3 = st.columns(3)
        col1.metric("Total Students", total_students)
        col2.metric("Placed Students", no_placed)
        col3.metric("Placement %", f"{placement_percentage:.2f}%")
        
        col4, col5, col6 = st.columns(3)
        col4.metric("PPO Offers", ppo_count)
        col5.metric("FT Offers", ft_count)
        col6.metric("Avg Package", f"{avg_package:.2f} LPA" if avg_package > 0 else "N/A")
        
        st.markdown("---")
        st.subheader("Student Details")
        
        # 3. Build Detailed Student Table
        student_details = []
        
        # Get students in this branch/year
        all_students_in_branch = pd.read_sql(f"SELECT id, name, cpi FROM students s WHERE {filters} ORDER BY name", conn, params=params)
        
        for _, student in all_students_in_branch.iterrows():
            sid = student['id']
            sname = student['name']
            scpi = student['cpi']
            
            # Shortlists
            s_shortlists = len(shortlists_df[shortlists_df['id'] == sid])
            
            # Offers
            s_offers = offers_df[offers_df['id'] == sid]
            
            offer_types = []
            offer_companies = []
            offer_ctcs = []
            
            if not s_offers.empty:
                for _, row in s_offers.iterrows():
                    etype = row['event_type'].lower()
                    if 'ppo' in etype or 'pre-placement' in etype:
                        offer_types.append("PPO")
                    else:
                        offer_types.append("FT")
                    
                    offer_companies.append(row['company_name'])
                    
                    ctc_val = row['ctc_lpa']
                    if pd.isna(ctc_val) and pd.notna(row['ctc_inr']):
                        ctc_val = row['ctc_inr'] / 100000.0
                    
                    if pd.notna(ctc_val):
                        offer_ctcs.append(f"{ctc_val:.2f} LPA")
                    else:
                        offer_ctcs.append("N/A")
            
            student_details.append({
                "Student Name": sname,
                "CPI": scpi if pd.notna(scpi) else "-",
                "Shortlists": s_shortlists,
                "Offer Type": ", ".join(list(set(offer_types))) if offer_types else "-",
                "Offer Company": ", ".join(list(set(offer_companies))) if offer_companies else "-",
                "Offer CTC": ", ".join(offer_ctcs) if offer_ctcs else "-"
            })
            
        if student_details:
            details_df = pd.DataFrame(student_details)
            st.dataframe(details_df, hide_index=True, width='stretch')
        else:
            st.info("No student data found for this branch.")
            
    conn.close()
