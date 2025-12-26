import streamlit as st
import sqlite3
from google import genai
import os
from dotenv import load_dotenv
import pandas as pd

# Load environment variables
load_dotenv()

# Configure API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found in .env file")
    st.stop()

client = genai.Client(api_key=GOOGLE_API_KEY)

# Database Configuration
DB_PATH = "data/placement.db"

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

# Streamlit UI
st.set_page_config(page_title="Placement Query Bot", page_icon="🎓", layout="wide")

st.title("🎓 IIT BHU Placement Query Bot")
st.markdown("Ask questions about student placements, offers, and shortlists naturally.")

# Sidebar Stats & Tools
conn = get_db_connection()
c = conn.cursor()
c.execute("SELECT COUNT(*) FROM students")
total_students = c.fetchone()[0]
c.execute("SELECT COUNT(DISTINCT company_name) FROM events")
total_companies = c.fetchone()[0]
conn.close()

with st.sidebar:
    st.header("📊 Database Stats")
    st.metric("Total Records", total_students)
    st.metric("Companies", total_companies)
    st.markdown("---")
    
    # Data Management
    st.header("⚙️ Data Management")
    if st.button("🔄 Refresh Database"):
        with st.spinner("Processing new files..."):
            try:
                # Run the process_data script logic
                import process_data
                process_data.process_files()
                st.success("Database updated successfully! Please reload the page.")
            except Exception as e:
                st.error(f"Error updating database: {e}")
                
    st.markdown("---")
    st.write("Using **Gemini 2.5 Flash Lite** for NL2SQL")

def generate_sql(question):
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
    
    prompt = f"""
    You are a SQL Expert. Convert the following natural language question into a SQL query for a SQLite database.
    
    Database Schema:
    {schema}
    
    CRITICAL RULES:
    1. Return ONLY the SQL query. No markdown, no explanation.
    2. **Joins are Usage**: 
       To find a student's events: `JOIN event_students es ON s.id = es.student_id JOIN events e ON es.event_id = e.id`
    3. **ROBUST NAME MATCHING (IMPORTANT)**: 
       - Users might provide only part of a name (e.g., "Sameer Wanjari" for "Sameer Nandesh Wanjari").
       - NEVER use `name LIKE '%First Last%'`.
       - ALWAYS split the name into parts and match each part separately using AND.
       - Example: For "Sameer Wanjari", use: `s.name LIKE '%Sameer%' AND s.name LIKE '%Wanjari%'`.
    4. Case Insensitive: `LIKE` in SQLite is case-insensitive for ASCII, but ensure logic holds.
    5. "Placed" = e.event_type contains 'Offer' or 'PPO' or 'Pre-Placement'.
    6. "Interview Shortlist" = e.event_type contains 'Interview'.
    7. "Test Shortlist" = e.event_type contains 'Test'.
    8. **Branches**: 'branch' column in `students` table contains values like 'CSE', 'Physics'.
    9. **Counts vs Lists**: 
       - If asked "How many" ONLY, use `COUNT(DISTINCT s.roll_no)`.
       - If asked "How many" AND "Names/Who/List", use `SELECT DISTINCT s.name, e.company_name...`.
    10. Select columns: `students.name`, `students.roll_no`, `students.branch`, `events.company_name`, `events.event_type`.
    
    Question: {question}
    SQL:
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt
    )
    sql = response.text.replace("```sql", "").replace("```", "").strip()
    return sql

def generate_natural_answer(question, sql, df):
    # safe-guard for large results
    if len(df) > 50:
        data_context = df.head(50).to_markdown(index=False) + f"\n...(and {len(df)-50} more rows)"
    else:
        data_context = df.to_markdown(index=False)

    prompt = f"""
    You are a helpful assistant for the IIT BHU Placement Cell.
    
    User Question: {question}
    Executed SQL: {sql}
    Result Data:
    {data_context}
    
    Task: Answer the user's question naturally based ONLY on the result data.
    
    SPECIAL FORMAT FOR "ANALYSIS" REQUESTS:
    If the user asks for an "analysis" or "overview" of a student, you MUST follow this format:
    1. **Interview Shortlist Summary**: "Lastname was shortlisted for X companies for Interviews." (Count unique companies from events where event_type contains 'Interview').
       - IGNORE 'Test Shortlists' for this count unless explicitly asked.
    2. **Interview Shortlist Details**: List the companies (e.g., "• Company A \n • Company B").
    3. **Offers**: Clearly state any 'FT Offers' or 'Internship Offers' received.
    
    General Rules:
    - If the result is a specific status, state it clearly.
    - If the dataframe is empty, say "I couldn't find any records matching that."
    - Use formatting like bold or bullet points for readability.
    - Do NOT mention "SQL" or "dataframe" in the final answer.
    """
    
    response = client.models.generate_content(
        model='gemini-2.5-flash-lite',
        contents=prompt
    )
    return response.text

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question (e.g., 'Where was sameer wanjari placed?')"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        
        try:
            # 1. Generate SQL
            sql_query = generate_sql(prompt)
            
            # 2. Execute SQL
            result = run_query(sql_query)
            
            if isinstance(result, pd.DataFrame):
                # 3. Generate Natural Language Answer
                nl_response = generate_natural_answer(prompt, sql_query, result)
                message_placeholder.markdown(nl_response)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": nl_response})
                
                # Optional: Show details
                with st.expander("View Technical Details (SQL & Data)"):
                    st.code(sql_query, language="sql")
                    st.dataframe(result)
            else:
                message_placeholder.error(result)
                st.session_state.messages.append({"role": "assistant", "content": f"Error: {result}"})
                
        except Exception as e:
            message_placeholder.error(f"An error occurred: {e}")
            st.session_state.messages.append({"role": "assistant", "content": f"An error occurred: {e}"})
