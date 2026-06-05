import streamlit as st
import app as demo  # Assuming your backend file is named app_backend.py

st.set_page_config(
    page_title="Jan Aushadhi Chatbot",
    page_icon="💊",
    layout="centered"
)

st.header("Medicine Recommendation Assistant")

# =========================================================================
# 1. GLOBAL INITIALIZATION (Runs once at startup and caches the database)
# =========================================================================
@st.cache_resource
def load_rag_pipeline():
    """Triggers the backend setup to load the PDF and prepare the Chroma database."""
    return demo.initialize_medical_rag()

# This holds your master compiled LangChain retrieval chain pipeline
rag_chain = load_rag_pipeline()



# 2. SIDEBAR CONTROLS
# =========================================================================



with st.sidebar:
    col1, col2 = st.columns([1, 4], vertical_alignment="center")
    with col1:
        st.image("logo.png", width=80)
    with col2:
        st.markdown("### Jan Aushadhi")
    # st.divider()
    st.markdown("## Chatbot")
    st.caption("Powered by Gemini & LangChain")
    st.header("Controls")
    
    if st.button("Clear Conversation", use_container_width=True):
        # We clear the history array to completely reset the chat display context
        st.session_state.chat_history = []
        st.rerun()

    st.divider()
    with st.expander("About this bot"):
        st.markdown(
        
        "- Framework: Modern LangChain\n"
        "- Vector Store: Chroma DB\n"
        "- Document: `Product_Portfolio_JA.pdf`"
    )

# =========================================================================
# 3. CONVERSATION STATE STORAGE
# =========================================================================
# Modern LangChain chains track memory via a clean history list of tuples/messages
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# Display previous conversational turns from session memory
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================================================================
# 4. CHAT INPUT RUNTIME INTERFACE
# =========================================================================
input_text = st.chat_input("Ask about generic names, drug codes, or pricing...")

if input_text:
    
    # 1. Render and append the human message
    with st.chat_message("user"):
        st.markdown(input_text)
    st.session_state.chat_history.append({"role": "user", "content": input_text})

    # 2. Process query through the backend pipeline function
    with st.chat_message("assistant"):
        with st.spinner("Searching medical document portfolio..."):
            
            # Call your mentor-style function, passing the chain we loaded globally at the top
            response = demo.run_chat_turn(
                user_question=input_text,
                chat_history_list=st.session_state.chat_history,
                rag_chain=rag_chain
            )
            
        st.markdown(response)

    # 3. Cache the assistant's reply into the UI history
    st.session_state.chat_history.append({"role": "assistant", "content": response})