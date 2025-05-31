""
import os
import json
import pandas as pd
import streamlit as st

from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import docx2txt
from openpyxl import load_workbook
from dotenv import load_dotenv

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
import io

# -------------------------
# Initialisation et config
# -------------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ Clé API OpenAI manquante. Ajoutez-la dans Streamlit Cloud (Settings > Secrets)")
    st.stop()

# -------------------------
# Lecture de fichiers
# -------------------------
dataframes_excels = {}
MAX_CHARS = 50000

def lire_contenu_fichier(uploaded_file, nom=None):
    try:
        nom = nom or uploaded_file.name
        if nom.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
            dataframes_excels[nom] = df
            contenu = df.to_string()
        elif nom.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
            dataframes_excels[nom] = df
            contenu = df.to_string()
        elif nom.endswith(".txt"):
            contenu = uploaded_file.read().decode("utf-8")
        elif nom.endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            contenu = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif nom.endswith(".docx"):
            contenu = docx2txt.process(uploaded_file)
        elif nom.endswith(".json"):
            data = json.load(uploaded_file)
            contenu = json.dumps(data, indent=2)
        elif nom.endswith((".md", ".html", ".htm")):
            soup = BeautifulSoup(uploaded_file.read(), "html.parser")
            contenu = soup.get_text()
        else:
            return None

        if len(contenu) > MAX_CHARS:
            contenu = contenu[:MAX_CHARS]
        return contenu
    except Exception as e:
        return f"Erreur de lecture : {e}"

# Lecture récursive locale

def charger_fichiers_recursif(dossier):
    documents = []
    for racine, _, fichiers in os.walk(dossier):
        for fichier in fichiers:
            chemin = os.path.join(racine, fichier)
            try:
                with open(chemin, "rb") as f:
                    contenu = lire_contenu_fichier(f, nom=fichier)
                    if contenu:
                        documents.append(Document(
                            page_content=f"Nom du fichier : {fichier}\n\n{contenu}",
                            metadata={"source": chemin}
                        ))
            except Exception as e:
                st.warning(f"Erreur avec {chemin} : {e}")
    return documents

# -------------------------
# Analyse IA
# -------------------------
def creer_qa_conversation(documents):
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = splitter.split_documents(documents[:50])

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        output_key="answer"
    )
    return qa_chain

# -------------------------
# Interface Streamlit
# -------------------------
st.set_page_config(page_title="Assistant IA", page_icon="🧠")
st.title(" 🧠 Assistant IA – Analyse de fichiers")

uploaded_files = st.file_uploader("📁 Importez vos fichiers", type=["csv", "xlsx", "txt", "pdf", "docx", "json", "html"], accept_multiple_files=True)
chemin_local = st.text_input("📂 Ou entrez le chemin d'un dossier local à indexer :")

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "historique" not in st.session_state:
    st.session_state.historique = []
if "archives" not in st.session_state:
    st.session_state.archives = []

# Traitement des fichiers
if uploaded_files:
    documents = []
    for f in uploaded_files:
        contenu = lire_contenu_fichier(f)
        if contenu:
            documents.append(Document(page_content=f"Nom du fichier : {f.name}\n\n{contenu}", metadata={"source": f.name}))
    if documents:
        st.session_state.qa_chain = creer_qa_conversation(documents)
        st.success("✅ Fichiers téléversés traités")

elif chemin_local and os.path.isdir(chemin_local):
    with st.spinner("📄 Indexation du dossier..."):
        documents = charger_fichiers_recursif(chemin_local)
        if documents:
            st.session_state.qa_chain = creer_qa_conversation(documents)
            st.success("✅ Fichiers du dossier chargés et indexés")
        else:
            st.error("❌ Aucun fichier valide trouvé dans ce dossier.")

# Bouton pour nouveau chat
if st.button("🔄 Nouveau chat"):
    if st.session_state.historique:
        st.session_state.archives.append(list(st.session_state.historique))
        st.session_state.historique = []
        st.success("🗃️ Ancien chat archivé et nouveau chat démarré.")

if st.session_state.qa_chain:
    with st.form("formulaire_question", clear_on_submit=True):
        question = st.text_input("💬 Posez votre question sur les fichiers :")
        submit = st.form_submit_button("Envoyer")

    if submit and question:
        with st.spinner("🤖 Traitement par l'IA..."):
            extrait_csv = ""
            for df in dataframes_excels.values():
                buffer = io.StringIO()
                df.head(10).to_csv(buffer, index=False)
                extrait_csv += buffer.getvalue() + "\n"

            prompt = PromptTemplate.from_template(
                "Voici un extrait des fichiers :\n{data}\n\nQuestion : {question}"
            )

            chain = LLMChain(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
                prompt=prompt
            )
            reponse = chain.run(data=extrait_csv, question=question)
            st.session_state.historique.append((question, reponse))

            suggestion_prompt = PromptTemplate.from_template(
                "Tu es un assistant. Voici une réponse :\n{reponse}\n\nSuggère 3 questions pertinentes que l'utilisateur pourrait poser ensuite."
            )
            suggestion_chain = LLMChain(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
                prompt=suggestion_prompt
            )
            suggestions = suggestion_chain.run(reponse=reponse)
            st.markdown("### 💡 Suggestions de questions :")
            st.markdown(suggestions)

if st.session_state.historique:
    st.markdown("### 📜 Historique")
    for q, r in reversed(st.session_state.historique):
        st.markdown(f"**🗣️ Vous :** {q}")
        st.markdown(f"**🤖 IA :** {r}")
        st.markdown("---")

if st.session_state.archives:
    st.markdown("### 📂 Archives de conversations précédentes")
    for idx, hist in enumerate(reversed(st.session_state.archives)):
        with st.expander(f"Chat précédent #{len(st.session_state.archives) - idx}"):
            for q, r in hist:
                st.markdown(f"**🗣️ Vous :** {q}")
                st.markdown(f"**🤖 IA :** {r}")
                st.markdown("---")
""
