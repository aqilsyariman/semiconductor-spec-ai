import streamlit as st
from app import read_pdf, split_text, embed_chunks, store_vectors, search, get_answer,rerank

st.set_page_config(
    page_title="Semiconductor AI Intelligence Platform",
    layout="wide"
)

if "collection" not in st.session_state:
    st.session_state.collection = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Sidebar
with st.sidebar:
    st.title("Document Upload")
    st.write("Upload your semiconductor datasheets")
    
    uploaded_files = st.file_uploader(
        "Select PDF files",
        type="pdf",
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        for file in uploaded_files:
            st.write(f"• {file.name}")
    
    if st.button("Process Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                all_chunks = []
                for file in uploaded_files:
                    with open(file.name, "wb") as f:
                        f.write(file.read())
                    text = read_pdf(file.name)
                    chunks = split_text(text)
                    all_chunks.extend(chunks)
                
                vector = embed_chunks(all_chunks)
                collection = store_vectors(all_chunks, vector.tolist())
                st.session_state.collection = collection
                st.success("Documents processed!")
        else:
            st.warning("Please upload at least one PDF.")

# Main area
st.title("Semiconductor AI Intelligence Platform")
st.write("Ask technical questions about your uploaded datasheets. Get precise answers with source and page citation.")

st.divider()

# Chat history
for chat in st.session_state.chat_history:
    with st.chat_message("user"):
        st.write(chat["question"])
    with st.chat_message("assistant"):
        st.write(chat["answer"])

# Question input
question = st.chat_input("Ask a technical question about your datasheets...")

if question:
    if st.session_state.collection is None:
        st.warning("Please upload and process your documents first.")
    else:
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Searching..."):
                results = search(question, st.session_state.collection)
                if results is None:
                    st.warning("Sorry, could not find relevant information.")
                else:
                    reranked = rerank(question, results)
                    answer = get_answer(question, reranked)
                    st.write(answer)
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer
                    })