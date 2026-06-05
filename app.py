import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

# ==========================================
# 1. SETUP FUNCTION (Runs Once)
# ==========================================
def initialize_medical_rag():
    """Sets up the PDF database and returns the master execution pipeline."""
    # Load document
    loader = PyPDFLoader('Product_Portfolio_JA.pdf')
    docs = loader.load()
    
    # Split text

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
        )


    # text_splitter = CharacterTextSplitter(
    #     separator= '\n', 
    #     chunk_size=300, 
    #     chunk_overlap=0
    #     )
    splits = text_splitter.split_documents(docs)
    
    # vectorstore = Chroma.from_documents(
    #     documents=splits, 
    #     embedding=GoogleGenerativeAIEmbeddings(model='gemini-embedding-2')
    #     )
    valid_splits = [s for s in splits if s.page_content.strip()]
    # Vector store
    vectorstore = Chroma.from_documents(
        documents=valid_splits, 
        embedding=GoogleGenerativeAIEmbeddings(model='gemini-embedding-2')
        )
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}
        )
    
    # Brain
    # llm = ChatGoogleGenerativeAI(
    #     model='gemini-2.5-flash-lite', 
    #     temperature=0.4, 
    #     max_tokens=300, 
    #     verbose=False)

    llm = ChatGroq(
        model="qwen/qwen3-32b",
        temperature=0.4,
        reasoning_format="hidden",
        max_tokens=1000,
        verbose=False
    )
    
    # History integration prompt
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Formulate a standalone question based on history. Do NOT answer it."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    
    # System instructions
    system_prompt = (
       "Role: Medical Catalog Assistant.\n"
    "Task: Extract details from retrieved context. Format: [Sr.no][Code][Name/Strength][Pack][Price].\n"
    "Rules:\n"
    "- Strictly use context for drug info. NEVER hallucinate drug data.\n"
    "- Keep answers concise and structured.\n"
    "- Use external knowledge ONLY for terminology, symptoms, or diseases.\n"
    "- Only recommend OTC medicines present in the document.\n"
    "- If symptoms require expert attention, advise consulting a doctor.\n"
    "- If medicine is missing, say 'I don't know'.\n"
    "- Output: Generic Name, Drug Code, Price/unit, MRP.\n"
    "- Ignore minor discrepancies in strength/packaging during search.\n\n"
        "Context: \n{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    # Final operational chain
    return create_retrieval_chain(history_aware_retriever, create_stuff_documents_chain(llm, qa_prompt))


# ==========================================
# 2. RUNTIME FUNCTION (Called by Frontend)
# ==========================================
def run_chat_turn(user_question, chat_history_list, rag_chain):
    """
    Handles a single turn of conversation.
    Returns the answer text.
    """
    try:
        response = rag_chain.invoke({
            "input": user_question,
            "chat_history": chat_history_list
        })
        return response["answer"]
    except Exception as e:
        return f"Error executing request: {str(e)}"