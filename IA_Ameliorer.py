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
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.docstore.document import Document
# Dictionnaire pour stocker les DataFrames Excel
dataframes_excels = {}


# 🔐 Charger la clé API depuis le fichier .env
load_dotenv(dotenv_path="Cle.env")  # Charge ton fichier Cle.env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("❌ Clé API OpenAI manquante. Vérifie ton fichier .env.")
    st.stop()

# -------------------------
# Lecture de contenu selon le type de fichier
# -------------------------
MAX_CHARS = 50000

def lire_contenu_fichier(chemin):
    try:
        if chemin.endswith(".csv"):
            df = pd.read_csv(chemin)
            dataframes_excels[chemin] = df
            contenu = df.to_string()
        elif chemin.endswith(".xlsx"):
            df = lire_excel_proprement(chemin)
            dataframes_excels[chemin] = df
            contenu = df.to_string()
        elif chemin.endswith(".txt"):
            with open(chemin, "r", encoding="utf-8") as f:
                contenu = f.read()
        elif chemin.endswith(".pdf"):
            reader = PdfReader(chemin)
            contenu = "\n".join([p.extract_text() for p in reader.pages if p.extract_text()])
        elif chemin.endswith(".docx"):
            contenu = docx2txt.process(chemin)
        elif chemin.endswith(".json"):
            with open(chemin, "r", encoding="utf-8") as f:
                contenu = json.dumps(json.load(f), indent=2)
        elif chemin.endswith((".md", ".html", ".htm")):
            with open(chemin, "r", encoding="utf-8") as f:
                soup = BeautifulSoup(f.read(), "html.parser")
                contenu = soup.get_text()
        else:
            return None

        if len(contenu) > MAX_CHARS:
            contenu = contenu[:MAX_CHARS]
        return contenu
    except Exception as e:
        return f"Erreur de lecture : {e}"


# -------------------------
# Création des documents indexables
# -------------------------
def charger_fichiers_recursif(dossier):
    documents = []
    for racine, _, fichiers in os.walk(dossier):
        for fichier in fichiers:
            chemin = os.path.join(racine, fichier)
            contenu = lire_contenu_fichier(chemin)
            if contenu:
                doc = Document(
                    page_content=f"Nom du fichier : {fichier}\n\n{contenu}",
                    metadata={"source": chemin}
                )
                documents.append(doc)
    return documents

# -------------------------
# Ajouter un document listant tous les fichiers indexés
# -------------------------
def inserer_liste_fichiers(docs):
    fichiers = [doc.metadata.get("source") for doc in docs]
    contenu = "Voici la liste complète des fichiers indexés :\n\n" + "\n".join(f"- " + os.path.basename(f) for f in fichiers)
    doc_liste = Document(page_content=contenu, metadata={"source": "📄 Liste des fichiers"})
    return docs + [doc_liste]

# -------------------------
# Chaîne IA
# -------------------------
def creer_qa_conversation(documents):
    from langchain.text_splitter import CharacterTextSplitter
    from langchain_community.embeddings import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS
    from langchain_community.chat_models import ChatOpenAI
    from langchain.chains import ConversationalRetrievalChain
    from langchain.memory import ConversationBufferWindowMemory

    # Réduction des morceaux pour éviter surcharge de tokens
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)

    # Optionnel : ne traiter qu’un sous-ensemble de documents
    docs = splitter.split_documents(documents[:50])  # Limite à 50 fichiers

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    vectordb = FAISS.from_documents(docs, embeddings)
    retriever = vectordb.as_retriever()

    # Historique limité pour réduire le contexte utilisé
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history",
        k=3,  # 3 derniers échanges seulement
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
# Explorateur interactif
# -------------------------
def afficher_arborescence(dossier, selection, prefix=""):
    fichiers_visibles = []
    for nom in sorted(os.listdir(dossier)):
        chemin = os.path.join(dossier, nom)
        if os.path.isdir(chemin):
            with st.expander(f"{prefix}📁 {nom}", expanded=False):
                fichiers_visibles += afficher_arborescence(chemin, selection, prefix + " ")
        elif os.path.isfile(chemin):
            if st.button(f"{prefix}📄 {nom}", key=chemin):
                selection["chemin"] = chemin
            fichiers_visibles.append(chemin)
    return fichiers_visibles

def lire_excel_proprement(chemin):
    try:
        xl = pd.ExcelFile(chemin, engine="openpyxl")
        feuilles = xl.sheet_names

        frames = []
        for feuille in feuilles:
            try:
                df = xl.parse(feuille, header=0)
                df = df.dropna(how="all")
                df.columns = df.columns.map(str)
                if not df.empty:
                    df["__Feuille__"] = feuille
                    frames.append(df)
            except Exception as fe:
                print(f"Erreur dans la feuille {feuille} : {fe}")

        if frames:
            df_total = pd.concat(frames, ignore_index=True)
            return df_total
        else:
            return pd.DataFrame()

    except Exception as e:
        print(f"Erreur de lecture du fichier Excel : {e}")
        return pd.DataFrame()

# -------------------------
# Interface principale
# -------------------------
st.set_page_config(page_title="Chat IA Fichiers Excel", page_icon="🧠")
st.title("🧠 Assistant IA – Analyse de fichiers Excel/CSV")

if "chemin_dossier" not in st.session_state:
    st.session_state.chemin_dossier = ""
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "historique" not in st.session_state:
    st.session_state.historique = []

# Saisie du chemin du dossier à analyser
chemin = st.text_input("📁 Chemin du dossier contenant les fichiers")

if chemin and os.path.isdir(chemin):
    st.session_state.chemin_dossier = chemin
    st.success(f"Dossier sélectionné : {chemin}")
    with st.spinner("📄 Chargement des fichiers..."):
        docs = charger_fichiers_recursif(chemin)
        if docs:
            docs = inserer_liste_fichiers(docs)
            st.session_state.qa_chain = creer_qa_conversation(docs)
            st.success("✅ Documents chargés et indexés")
        else:
            st.error("❌ Aucun document lisible trouvé.")

# Interface de chat avec analyse des fichiers Excel
if st.session_state.qa_chain:
    with st.form("chat_form", clear_on_submit=True):
        question = st.text_input("💬 Pose ta question sur les fichiers (ex : Donne-moi les produits liés à KMR)")
        submit = st.form_submit_button("Envoyer")

    if submit and question:
        with st.spinner("🤖 Analyse en cours..."):
            import io
            from langchain.chains import LLMChain
            from langchain.prompts import PromptTemplate
            from langchain_community.chat_models import ChatOpenAI

            extrait_csv = ""
            for df in dataframes_excels.values():
                csv_io = io.StringIO()
                df.head(20).to_csv(csv_io, index=False)
                extrait_csv += csv_io.getvalue() + "\n"

            prompt = PromptTemplate.from_template(
                "Voici un extrait des fichiers Excel/CSV chargés :\n{data}\n\nQuestion : {question}"
            )

            chain = LLMChain(
                llm=ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY),
                prompt=prompt
            )

            reponse = chain.run(data=extrait_csv, question=question)
            st.session_state.historique.append((question, reponse))

# Affichage historique
if st.session_state.historique:
    st.markdown("### 📜 Historique")
    for q, r in reversed(st.session_state.historique):
        st.markdown(f"**🗣️ Vous :** {q}")
        st.markdown(f"**🤖 IA :** {r}")
        st.markdown("---")
