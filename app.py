import os
import gc
import uuid
import time
import streamlit as st

from openai import OpenAI # type: ignore
from langchain_openai import ChatOpenAI
from langchain_core.runnables import ConfigurableFieldSpec
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.tools import tool
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain import hub
from langchain_community.utilities import SQLDatabase
from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool
from pydantic import BaseModel, Field
import time
import ast
from tavily import TavilyClient
import requests
from typing import Optional
# from dotenv import load_dotenv

# Load environment variables
# load_dotenv()
openai_api_key= st.secrets["api_keys"]["OPENAI_API_KEY"]
tavily_api_key = st.secrets["api_keys"]["TAVILY_API_KEY"]
# --- Page Configuration ---
st.set_page_config(
    page_title="InsightAgentBot: Talk to Our Data & the Web",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Global Variables & Clients ---
chat_model = ChatOpenAI(model="gpt-4.1", temperature=0.0, streaming=True, api_key=openai_api_key) # Using gpt-4o as a modern choice
try:
    db = SQLDatabase.from_uri("sqlite:///our_sql_database.db")
except Exception as e:
    st.error(f"Failed to connect to the database: {e}")
    st.stop()

tavily_client= TavilyClient(api_key=tavily_api_key)

# In-memory store for chat histories
store = {}
# --- Helper Functions ---
def query_as_list(db_conn, query_str):
    """Executes a SQL query and returns a cleaned list of unique results."""
    res = db_conn.run(query_str)
    try:
        evaluated_res = ast.literal_eval(res)
    except (SyntaxError, ValueError):
        # st.warning(f"Warning: Could not parse SQL query result string: {res}")
        # If it's already a list of tuples or similar, direct processing might be better.
        # For now, keeping original logic but this part might need adjustment based on actual db.run output.
        if isinstance(res, str) and not (res.startswith("[") and res.endswith("]")):
             # If it's a plain string not looking like a list, treat as single item list
            evaluated_res = [(res,)]
        else: # Attempt to handle simple string representations of lists if ast.literal_eval fails
            st.warning(f"Could not parse SQL query result string with ast.literal_eval: {res}. Attempting basic processing.")
            return [str(r).strip() for r in res.split(',') if r.strip()] if isinstance(res, str) else []


    # Flatten list of tuples and remove empty strings
    processed_res = [str(el).strip() for sub in evaluated_res for el in sub if str(el).strip()]
    # Remove trailing digits (if any, as per original logic) - this might be too aggressive
    # processed_res = [re.sub(r"\b\d+\b$", "", string).strip() for string in processed_res] # Made it trailing digits only
    return list(set(processed_res))


class SQLQUERY(BaseModel):
    sql_query: str = Field(..., description="Syntactically valid SQL query to execute.")

# Tools
@tool
def execute_sql_query(query: SQLQUERY) -> str:
    """
    Useful tool to execute a SQL query against the database and get results. You must call this tool after each generated SQL query.
    Input must be a syntactically valid SQL query.
    """
    print(f"CONSOLE: Executing SQL query:\n{query.sql_query}")
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    result = execute_query_tool.invoke(query.sql_query)
    print(f"CONSOLE: SQL query result: {result}")
    return str(result) # Ensure result is always a string

@tool
def web_search_tool(query: str) -> str:
    """A search engine optimized for comprehensive, accurate, and trusted results. Useful for when you need to answer questions about current prices of products. This tool delivers recent, accurate results
    query: a full complete target search query
    """
    response = tavily_client.search(query= query,  search_depth="advanced", topic="general", include_raw_content= True)
    response_formatted= [f"Result number {idx}:\nURL: {res['url']}.\n Content:{res['content']}.\n\n\n" for idx, res in enumerate(response["results"])]
    output = "\n\n".join(response_formatted)
    return output

def check_urls(urls, timeout=5):
    """
    Given a list of URLs, check each one and return a list of
    (url, bool) indicating whether the URL is alive (HTTP status 2xx/3xx).
    """
    results = []
    for url in urls:
        try:
            # Use HEAD for efficiency; fallback to GET if HEAD not allowed
            resp = requests.head(url, timeout=timeout)
            is_up = resp.status_code < 400
        except requests.exceptions.RequestException:
            is_up = False
        results.append((url, is_up))
    return results

@tool
def check_urls_status(urls: Optional[list[str]]):
    """Useful tool to check whether the urls are working or not. Use this tool only after `web_search_tool` to verify if there are URLs returned by the search before responding to the user.
    Args:
        urls: A list of URLs to check.
    """
    if urls is None:
        return "No URLs provided for checking. Please, try again and provide the list of urls returned by the `web_search_tool` tool"
    results=[]
    for url, ok in check_urls(urls,):
        status_text = "✅ OK" if ok else "❌ Not working"
        results.append(f"{url}: {status_text}")

    return results

def get_session_history(session_id_str: str) -> BaseChatMessageHistory:
    """Retrieves or creates a chat history for a given session ID."""
    if session_id_str not in store:
        store[session_id_str] = ChatMessageHistory()
    return store[session_id_str]

def reset_chat_history():
    """Resets the chat history and messages for the current session."""
    st.session_state.messages = []
    session_id = str(st.session_state.id)
    if session_id in store:
        del store[session_id]
    if "_final_output_yielded" in st.session_state:
        del st.session_state._final_output_yielded
    gc.collect()
    st.success("Chat history cleared!")
    st.rerun()


def stream_agent_responses(final_response_string: str, delay_seconds: float = 0.01):
    """
    Streams a given final response string character by character.
    Escapes '$' for Markdown compatibility.

    Args:
        final_response_string: The complete string output from the agent.
        delay_seconds: The delay between yielding each character.
    """
    if not isinstance(final_response_string, str):
        yield str(final_response_string)
        return

    for char_idx, char in enumerate(final_response_string):
        if char == '$':
            yield '\\'
            yield '$'
        else:
            yield char
        
        if char_idx < 50:
            time.sleep(delay_seconds)
        else:
            time.sleep(delay_seconds * 1.5)

# --- Session State Initialization ---
if "id" not in st.session_state:
    st.session_state.id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain_runnable" not in st.session_state:
    st.session_state.chain_runnable = None
if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False

# --- Agent and Chain Initialization ---
if not st.session_state.chain_initialized:
    with st.spinner("Initializing AI Agent... Please wait."):
        try:
            # Fetch dynamic data for the prompt
            companies_list = query_as_list(db, "SELECT DISTINCT CompanyName FROM data_table")
            eqbrand_list = query_as_list(db, "SELECT DISTINCT EQBrand FROM data_table")

            # Define the system prompt for the agent
            system_message_str=f"""You are an **Intelligent SQL Query & Web Search Agent**. Your mission is to be a user-friendly interface to the `data_table` SQL database and provide current market pricing through web search when needed.
**Core Workflow:**
1.  **Understand User Query:** Interpret the user's natural language request.
2.  **Database Search First:** Create a precise SQL query for the `data_table` and execute via `execute_sql_query` tool.
3.  **Web Search Enhancement:** If database results are empty OR user specifically asks for "current", "recent", "latest market", or "updated" prices, use `web_search_tool` to find current market prices.
4.  **URL Verification:** ALWAYS use `check_urls_status` tool after `web_search_tool` to verify that the URLs returned are working.
5.  **Re-search if Needed:** If `check_urls_status` shows all URLs are not working, filter them out and use `web_search_tool` again with different search terms to find working URLs.
6.  **Synthesize & Respond:** Analyze results from all tools and provide a comprehensive, natural language answer combining database and verified web findings.

**1. `data_table` Schema & Key Values:**
*   **Prioritize Matching with Known Values:**
    *   For `CompanyName`, use: `<Known_Company_Names>{companies_list if companies_list else 'Not available'}</Known_Company_Names>`
    *   For `EQBrand`, use: `<Known_EQ_Brands>{eqbrand_list if eqbrand_list else 'Not available'}</Known_EQ_Brands>`
    *   If a user's term (e.g., "Agilent") is part of a known value (e.g., "Agilent HP Keysight"), use the full known value in your SQL.

*   **Columns:**
    *   `QID` (INT): Quote ID. For "latest" or "newest", use `ORDER BY QID DESC`.
    *   `id` (INT): General ID.
    *   `CompanyName` (TEXT): Company name.
    *   `CreatedDate` (TEXT): Creation time ('HH:MM:SS.s'). Filter using `LOWER()`.
    *   `price_range` (TEXT): Nullable.
    *   `Price` (NUMERIC): Equipment price.
    *   `price_string` (TEXT): Nullable.
    *   `EQBrand` (TEXT): Equipment brand.
    *   `EQModel` (TEXT): Equipment model.
    *   `Options` (TEXT): Nullable. For "no options", use `(Options IS NULL OR LOWER(Options) = '')`.
    *   `Date Quoted` (TEXT): Quote date. Filter using `LOWER()`.
    *   `Dealer Quoted` (TEXT): Nullable.
    *   `Sales Comments` (TEXT): Nullable.
    *   `SellerType` (TEXT): E.g., 'Dealer', 'EU'.
    *   `eqBrandConsolidated` (TEXT): Nullable.
    *   `listType` (TEXT): E.g., 'WTB', 'WTS'.
    *   `record_id` (INT): Record ID.

**2. SQL Generation Essentials (for `execute_sql_query` tool):**
*   **Case-Insensitive Matching:** CRITICAL for all string comparisons in `WHERE` clauses (e.g., `CompanyName`, `EQBrand`, `Options`). Use `LOWER(column_name) = LOWER('user_value')` for exact or `LOWER(column_name) LIKE LOWER('%user_value%')` for partial matches.
*   **Column Selection:** `SELECT *` for general queries. Select specific columns if requested (e.g., `SELECT Price, EQModel ...`).
*   **Filtering:** Use `WHERE` with `AND`, `OR`, `NOT`. Use standard operators (`=`, `>`, `<`, `BETWEEN`) for `Price`.
*   **User Intent:** Infer `ORDER BY` (e.g., `Price ASC` for "cheapest", `QID DESC` for "latest") and `LIMIT` (e.g., "top 3") from the query.

**3. Web Search Integration (for `web_search_tool`):**
*   **When to Use Web Search:**
    *   Database returns no results for price queries
    *   User explicitly asks for "current", "recent", "latest market", "updated", or "new" prices
    *   User wants to compare database prices with current market rates
*   **Search Query Construction:** Create focused search terms like "[Brand] [Model] price buy sell market" or "[Brand] [Model] current pricing dealers"
*   **Results Processing:** Extract company names and their respective prices from web search results

**4. URL Status Verification (for `check_urls_status` tool):**
*   **Mandatory Usage:** ALWAYS use `check_urls_status` immediately after `web_search_tool` to verify URL accessibility.
*   **Input Format:** Pass all URLs returned by the web search tool for verification.
*   **Filtering Logic:** 
    *   If some URLs are working and some are not, use only the working URLs and their associated data in your response.
    *   If ALL URLs are not working, discard the current web search results and perform a new `web_search_tool` query with alternative search terms.
*   **Re-search Strategy:** When all URLs fail, try:
    *   Different brand name variations (e.g., "HP" instead of "Hewlett Packard")
    *   Alternative search terms (e.g., "used equipment dealers" or "test equipment marketplace")
    *   Broader product categories if specific model searches fail

**5. Response Synthesis (Your Final Output to User):**
*   **Natural Language ONLY:** Your response to the user MUST be a helpful, human-readable sentence or paragraph. Do NOT output raw data, lists of numbers, or SQL queries.
*   **Database + Web Results:** When combining both sources:
    *   Clearly distinguish between "database records" and "current market prices"
    *   Present database findings first, then verified web search results
    *   Compare ranges and highlight differences if significant
*   **Price Information:** 
    *   For database results: State the range and mention distinct prices found
    *   For web results: Only include information from **verified working URLs**, listing each company/seller with their respective prices (e.g., "Current market prices from verified sources include: Company A at $X, Company B at $Y, Company C at $Z")
    *   Use currency symbols and clear formatting
*   **URL Disclosure:** If web search was used, **always** append a **Sources:** section listing each **validated** URL returned by `web_search_tool` and confirmed by `check_urls_status`, so the user can verify your market data.
*   **No Results:** If neither tool returns data or all URLs are non-working after re-search attempts, inform the user politely and suggest alternative search terms

**6. Enhanced Few-Shot Examples:**

*   **User Query:** "what is the current price of Agilent HP Keysight E4980A?"
    *   **Internal SQL:** `SELECT Price FROM data_table WHERE LOWER(EQBrand) = 'agilent hp keysight' AND LOWER(EQModel) = 'e4980a' ORDER BY Price ASC;`
    *   **Database Result:** `['Price': 5560, 'Price': 7000]`
    *   **Web Search Query:** "Agilent HP Keysight E4980A current price market dealers"
    *   **Web Search Result:** URLs and pricing from various sellers
    *   **URL Status Check:** Verify all returned URLs
    *   **URL Status Result:** 2 working URLs, 1 non-working URL
    *   **Your Response:** "Based on our database records, the Agilent HP Keysight E4980A has been listed at prices ranging from $5,560 to $7,000. Current market prices from verified sources show: TestMart at $8,200 and EquipNet at $7,500. The current market prices appear to be higher than our historical database records."

*   **User Query:** "Price for Keysight 34970A?"
    *   **Internal SQL:** `SELECT Price FROM data_table WHERE LOWER(EQBrand) LIKE '%keysight%' AND LOWER(EQModel) = '34970a';`
    *   **Database Result:** `[]` (empty)
    *   **Web Search Query:** "Keysight 34970A price buy market dealers"
    *   **Web Search Result:** Multiple URLs returned
    *   **URL Status Check:** All URLs are not working
    *   **Re-search Query:** "Keysight 34970A used test equipment marketplace"
    *   **Second URL Status Check:** 2 working URLs found
    *   **Your Response:** "I didn't find any records for the Keysight 34970A in our database. However, current market prices from verified sources show: Electro Rent at $2,450 and Test Equipment Connection at $2,150."

*   **User Query:** "Show me recent Tektronix oscilloscope prices"
    *   **Internal SQL:** `SELECT EQModel, Price FROM data_table WHERE LOWER(EQBrand) LIKE '%tektronix%' AND LOWER(EQModel) LIKE '%scope%' ORDER BY QID DESC LIMIT 5;`
    *   **Database Result:** Historical records found
    *   **Web Search Query:** "Tektronix oscilloscope current prices 2024"
    *   **URL Status Check:** Mixed results - some working, some not
    *   **Your Response:** "From our database, recent Tektronix oscilloscope listings include [database results]. Current market prices from verified dealers show [working URLs' pricing information with company names and prices]."""            # Pull the base agent prompt from Langchain Hub
            agent_prompt_template = hub.pull("hwchase17/openai-functions-agent")

            if hasattr(agent_prompt_template, 'messages') and \
               len(agent_prompt_template.messages) > 0 and \
               hasattr(agent_prompt_template.messages[0], 'prompt') and \
               hasattr(agent_prompt_template.messages[0].prompt, 'template'):
                agent_prompt_template.messages[0].prompt.template = system_message_str
            else:
                st.error("Critical Error: Could not customize the agent's system prompt. The Langchain Hub prompt structure might have changed.")
                st.stop()

            tools_list = [execute_sql_query, web_search_tool, check_urls_status]
            openai_tools_agent = create_openai_tools_agent(
                llm=chat_model,
                tools=tools_list,
                prompt=agent_prompt_template
            )
            agent_executor_instance = AgentExecutor(
                agent=openai_tools_agent,
                tools=tools_list,
                verbose=True,
                handle_parsing_errors=True # Add robust error handling
            )

            st.session_state.chain_runnable = RunnableWithMessageHistory(
                runnable=agent_executor_instance,
                get_session_history=get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                history_factory_config=[
                    ConfigurableFieldSpec(
                        id="session_id",
                        annotation=str,
                        name="Session ID",
                        description="Unique identifier for the chat session.",
                        default="",
                        is_shared=True,
                    )
                ],
            )
            st.session_state.chain_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize the AI agent: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            st.stop()


# --- Sidebar UI ---
with st.sidebar:
    st.title("🤖 SQL Agent Controls")
    st.markdown("---")
    st.info(
        "This AI agent interacts with a SQL database. "
        "Ask questions about the data, and the agent will attempt to answer them by generating and executing SQL queries."
    )

    if st.button("Clear Chat History ↺", on_click=reset_chat_history, use_container_width=True, type="primary"):
        pass # Action handled by on_click

    st.markdown("---")
    st.markdown("### Database Information")
    st.caption("The agent is connected to a SQLite database (`our_sql_database.db`). It can query tables like `data_table`.")
    # You could add more dynamic info here if needed, e.g., table names if fetched.
    # Example:
    # with st.expander("Show known tables"):
    # st.write(db.get_table_names())


# --- Main Chat Interface ---
st.title("Chat with Updated Products Database 📊")
st.markdown("<sub>Powered by Langchain & OpenAI</sub>", unsafe_allow_html=True)

# Display welcome message if chat is empty
if not st.session_state.messages:
    st.info("Welcome! I'm ready to help you query the database. Try asking something like: 'What are the distinct company names?' or 'How many records are in data_table?'")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask a question about the database..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        if not st.session_state.chain_initialized or not st.session_state.chain_runnable:
            st.error("The AI Agent is not initialized. Please refresh or check the logs.")
            st.stop()

        current_chain = st.session_state.chain_runnable
        full_response_content = ""

        try:
            with st.spinner("Thinking..."):
                # Each interaction uses the same session ID for history
                chat_session_id = str(st.session_state.id)
                input_payload = {"input": prompt}

                response_stream_container = current_chain.invoke(
                    input_payload,
                    config={"configurable": {"session_id": chat_session_id}}
                )

                print(f"CONSOLE: Response stream container \n: {response_stream_container}")
                
                # The output of invoke with RunnableWithMessageHistory is typically the agent's final response dictionary.
                # If 'output' key contains a generator (due to streaming=True in ChatOpenAI and how AgentExecutor handles it),
                # then stream_agent_responses should handle that generator.
                # The structure of response_stream_container needs to be handled carefully.
                # If response_stream_container['output'] is the stream:
                full_response_content = st.write_stream(stream_agent_responses(response_stream_container['output'], delay_seconds=0.01))

        except Exception as e:
            st.error(f"Error during chat generation: {e}")
            import traceback
            st.error(f"Traceback: {traceback.format_exc()}")
            full_response_content = "Sorry, I encountered an error while processing your request. Please try again."
            st.markdown(full_response_content)

    st.session_state.messages.append({"role": "assistant", "content": full_response_content})
    st.rerun() # Rerun to ensure message list is updated correctly before next input.
