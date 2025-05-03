import json
import os
import streamlit as st
import boto3
from datetime import datetime
from langdetect import detect
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import tempfile
import numpy as np

from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_aws.embeddings import BedrockEmbeddings
from langchain_aws import BedrockLLM

# ---------------------- AWS Clients ----------------------
bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")
translate_client = boto3.client("translate", region_name="us-east-1")

# ---------------------- Multilingual Helpers ----------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

def translate_text(text, source_lang, target_lang):
    if source_lang == target_lang:
        return text
    response = translate_client.translate_text(
        Text=text,
        SourceLanguageCode=source_lang,
        TargetLanguageCode=target_lang
    )
    return response['TranslatedText']

# ---------------------- Whisper Recorder ----------------------
def record_audio(duration=5, sample_rate=44100):
    st.info("🎙 Recording... Speak now!")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    return sample_rate, np.squeeze(recording)

def transcribe_with_whisper(sample_rate, audio_data):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav.write(tmp.name, sample_rate, audio_data)
        model = whisper.load_model("base")
        result = model.transcribe(tmp.name)
        return result['text']

# ---------------------- Reliability Scoring ----------------------
st_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")

def lexical_support_score(answer, context):
    answer_words = answer.lower().split()
    count = sum(1 for word in answer_words if word in context.lower())
    return round(count / len(answer_words), 2) if answer_words else 0

def semantic_similarity_score(answer, context):
    a_vec = st_model.encode([answer])[0]
    c_vec = st_model.encode([context])[0]
    return round(float(cosine_similarity([a_vec], [c_vec])[0][0]), 2)

# ---------------------- Titan Embeddings ----------------------
bedrock_embeddings = BedrockEmbeddings(
    model_id="amazon.titan-embed-text-v2:0",
    client=bedrock
)

# ---------------------- PDF Ingestion ----------------------
def data_ingestion_from_upload(uploaded_files):
    all_docs = []
    os.makedirs("temp", exist_ok=True)
    for uploaded_file in uploaded_files:
        temp_path = f"temp/{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        loader = PyPDFLoader(temp_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        split_docs = text_splitter.split_documents(documents)
        all_docs.extend(split_docs)
    return all_docs

# ---------------------- FAISS Index ----------------------
def get_vector_store(docs):
    vectorstore_faiss = FAISS.from_documents(docs, bedrock_embeddings)
    vectorstore_faiss.save_local("faiss_index")

# ---------------------- Cohere Model ----------------------
def invoke_cohere(prompt: str) -> str:
    body = {
        "chat_history": [
            {
                "role": "USER",
                "message": prompt
            }
        ],
        "message": "Sure, please generate the content.",
        "temperature": 0.7,
        "max_tokens": 512
    }
    try:
        response = bedrock.invoke_model(
            modelId="cohere.command-r-plus-v1:0",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )
        response_content = response['body'].read().decode("utf-8")
        response_data = json.loads(response_content)
        return response_data.get("text", "⚠️ No response generated.")
    except Exception as e:
        return f"❌ Error from Cohere model: {str(e)}"

# ---------------------- LLaMA Model ----------------------
def get_llama_llm():
    return BedrockLLM(
        model_id="meta.llama3-8b-instruct-v1:0",
        client=bedrock,
        model_kwargs={"max_gen_len": 512}
    )

# ---------------------- LangChain QA ----------------------
def get_response_llm(llm, vectorstore_faiss, query, chat_history):
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore_faiss.as_retriever(search_type="similarity", search_kwargs={"k": 3}),
        return_source_documents=True
    )
    response = qa_chain.invoke({"question": query, "chat_history": chat_history})
    return response["answer"]

# ---------------------- Streamlit UI ----------------------
def main():
    st.set_page_config("🌍 Multilingual PDF QA", layout="centered")
    st.title("🤖 Multilingual PDF QA with Whisper & Bedrock")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    uploaded_files = st.file_uploader("📁 Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

    with st.sidebar:
        st.header("⚙️ Controls")
        if uploaded_files and st.button("📤 Ingest PDFs"):
            with st.spinner("Processing PDFs..."):
                docs = data_ingestion_from_upload(uploaded_files)
                get_vector_store(docs)
                st.success("✅ PDF Vector Index Created!")

        selected_model = st.radio("🤖 Choose LLM:", ["Cohere (command-r+)", "LLaMA3"])
        high_conf_mode = st.checkbox("🛡️ Enable High Confidence Mode")
        if st.button("🧼 Clear Memory"):
            st.session_state.chat_history = []
            st.success("🧠 Memory Cleared!")

    input_type = st.radio("Choose Input Mode:", ["🧠 Type", "🎙 Speak"])
    user_question = ""

    if input_type == "🧠 Type":
        user_question = st.text_input("💬 Ask your question in any language")
    else:
        if st.button("🎙 Record & Transcribe"):
            with st.spinner("Recording and transcribing..."):
                sr, audio = record_audio(duration=7)
                user_question = transcribe_with_whisper(sr, audio)
                st.success(f"📝 Transcribed: {user_question}")

    if st.button("🚀 Get Answer") and user_question.strip():
        with st.spinner("Thinking..."):
            faiss_index = FAISS.load_local("faiss_index", bedrock_embeddings, allow_dangerous_deserialization=True)
            user_lang = detect_language(user_question)
            translated_question = translate_text(user_question, user_lang, "en") if user_lang != "en" else user_question

            if selected_model.startswith("Cohere"):
                retriever = faiss_index.as_retriever(search_type="similarity", search_kwargs={"k": 3})
                docs_with_scores = retriever.get_relevant_documents(translated_question)
                st.markdown("### 📚 Retrieved Context")
                for i, doc in enumerate(docs_with_scores):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.code(doc.page_content[:1000])
                    st.markdown("---")
                context = "\n".join([doc.page_content for doc in docs_with_scores])
                prompt = f"""
Use the following context to answer the question below in at least 250 words:
<context>
{context}
</context>

Question: {translated_question}
"""
                answer = invoke_cohere(prompt)
            else:
                llm = get_llama_llm()
                answer = get_response_llm(llm, faiss_index, translated_question, st.session_state.chat_history)

            answer_final = translate_text(answer, "en", user_lang) if user_lang != "en" else answer
            combined_context = "\n".join([doc.page_content for doc in docs_with_scores])
            lex_score = lexical_support_score(answer, combined_context)
            sem_score = semantic_similarity_score(answer, combined_context)
            st.markdown("### ✅ Reliability Summary")
            st.info(f"- 🧠 Lexical Support Score: `{lex_score*100:.1f}%`\n- 🤖 Semantic Similarity Score: `{sem_score*100:.1f}%`")

            if high_conf_mode and (lex_score < 0.5 or sem_score < 0.6):
                st.warning("⚠️ Low confidence. Try rephrasing your question or check PDF quality.")
            else:
                st.session_state.chat_history.append((user_question, answer_final))
                with st.chat_message("user"):
                    st.markdown(f"**You:** {user_question}")
                with st.chat_message("assistant"):
                    st.markdown(f"**Bot:** {answer_final}")

                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                st.download_button(
                    label="💾 Download Answer",
                    data=answer_final,
                    file_name=f"answer_{timestamp}.txt",
                    mime="text/plain"
                )

    if st.session_state.chat_history:
        st.markdown("### 🕒 Chat History")
        for q, a in reversed(st.session_state.chat_history):
            with st.chat_message("user"):
                st.markdown(f"**You:** {q}")
            with st.chat_message("assistant"):
                st.markdown(f"**Bot:** {a}")

if __name__ == "__main__":
    main()
