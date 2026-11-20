import os
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

PDF_PATH = "CPSC 254_SYL.pdf"
PERSIST_DIR = "./chroma_syllabus_db"
CONFIG_PATH = os.path.join(PERSIST_DIR, "rag_config.json")

CHUNK_SIZE = 600
CHUNK_OVERLAP = 120
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
TOP_K = 4

DEFAULT_MODEL = os.environ.get("NAUTILUS_MODEL", "gpt-oss")
NAUTILUS_BASE_URL = os.environ.get("NAUTILUS_BASE_URL", "[https://ellm.nrp-nautilus.io/v1](https://ellm.nrp-nautilus.io/v1)")

RAG_PROMPT = ChatPromptTemplate.from_template(
    """You are a helpful course assistant for CPSC 254.
Answer the student's question using ONLY the syllabus excerpts provided below.
Rules:
1. Ground every statement strictly in the provided context.
2. Cite the page number(s) from the context when available.
3. If the provided context does not contain enough information to answer the question, respond clearly with: "The provided syllabus does not contain enough information to answer this question." Do not guess or invent details.

Syllabus Context:
{context}

Question:
{question}

Answer:"""
)


def get_llm():
    return ChatOpenAI(
        model=DEFAULT_MODEL,
        temperature=0,
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=NAUTILUS_BASE_URL,
    )


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def get_or_build_vectorstore(pdf_path: str = PDF_PATH, persist_dir: str = PERSIST_DIR, force_rebuild: bool = False):
    """
    Loads the persisted Chroma index if it exists; otherwise extracts text from
    CPSC 254_SYL.pdf, splits into overlapping chunks, embeds, and saves to disk.
    """
    embeddings = get_embeddings()
    index_exists = os.path.exists(persist_dir) and os.path.exists(CONFIG_PATH)

    if index_exists and not force_rebuild:
        print(f"Loading existing Chroma vector index from '{persist_dir}'...")
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embeddings,
            collection_name="cpsc254_syllabus",
        )
        return vectorstore

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(
            f"Syllabus PDF '{pdf_path}' not found in current directory ({os.getcwd()})."
        )

    print(f"Building Chroma vector index from '{pdf_path}'...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(documents)
    print(f"Loaded {len(documents)} pages and split into {len(chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}).")

    os.makedirs(persist_dir, exist_ok=True)
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
        collection_name="cpsc254_syllabus",
    )

    config_data = {
        "pdf_path": pdf_path,
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "embedding_model": EMBEDDING_MODEL_NAME,
        "top_k": TOP_K,
        "num_pages": len(documents),
        "num_chunks": len(chunks),
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    print(f"Saved Chroma vector index and config to '{persist_dir}'.")
    return vectorstore


def ask_syllabus(question: str, vectorstore=None, llm=None, k: int = TOP_K) -> dict:
    """
    Retrieves top-k relevant chunks from the syllabus and generates a grounded answer.
    """
    if vectorstore is None:
        vectorstore = get_or_build_vectorstore()
    if llm is None:
        llm = get_llm()

    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    retrieved_docs = retriever.invoke(question)

    formatted_chunks = []
    for i, doc in enumerate(retrieved_docs, start=1):
        page_num = doc.metadata.get("page", "Unknown")
        # PyPDFLoader uses 0-indexed page metadata
        display_page = page_num + 1 if isinstance(page_num, int) else page_num
        formatted_chunks.append(f"[Chunk {i} | Page {display_page}]\n{doc.page_content}")

    context = "\n\n".join(formatted_chunks)
    messages = RAG_PROMPT.format_messages(context=context, question=question)
    response = llm.invoke(messages)

    return {
        "question": question,
        "answer": response.content.strip(),
        "retrieved_docs": retrieved_docs,
    }


if __name__ == "__main__":
    vs = get_or_build_vectorstore()
    llm = get_llm()

    print("\n=== CPSC 254 Syllabus RAG Assistant (type 'exit' to quit) ===")
    while True:
        user_q = input("\nQuestion: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            break
        if not user_q:
            continue
        res = ask_syllabus(user_q, vectorstore=vs, llm=llm)
        print(f"\nAnswer:\n{res['answer']}")
