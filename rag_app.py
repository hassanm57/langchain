import os
import tempfile

import requests
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

PDF_URL = "https://arxiv.org/pdf/1706.03762"  # "Attention Is All You Need"


def download_pdf(url: str) -> str:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    tmp_file = tempfile.NamedTemporaryFile(suffix=".pdf", delete=False)
    tmp_file.write(response.content)
    tmp_file.close()
    return tmp_file.name


def build_retriever(pdf_path: str):
    documents = PyPDFLoader(pdf_path).load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 4})


def format_docs(docs) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def build_chain(retriever):
    llm_endpoint = HuggingFaceEndpoint(
        repo_id="meta-llama/Llama-3.1-8B-Instruct",
        task="text-generation",
        max_new_tokens=512,
        temperature=0.1,
        huggingfacehub_api_token=os.environ["HUGGINGFACEHUB_API_TOKEN"],
    )
    llm = ChatHuggingFace(llm=llm_endpoint)

    prompt = ChatPromptTemplate.from_template(
        "Answer the question using only the context below. "
        "If the answer isn't in the context, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {question}"
    )

    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


def main():
    if "HUGGINGFACEHUB_API_TOKEN" not in os.environ:
        raise SystemExit(
            "Set HUGGINGFACEHUB_API_TOKEN in your environment or .env file. "
            "Get a free token at https://huggingface.co/settings/tokens"
        )

    print(f"Downloading PDF: {PDF_URL}")
    pdf_path = download_pdf(PDF_URL)

    print("Building index...")
    retriever = build_retriever(pdf_path)
    chain = build_chain(retriever)

    print("Ready. Ask questions about the document (type 'exit' to quit).\n")
    while True:
        question = input("You: ").strip()
        if question.lower() in {"exit", "quit"}:
            break
        if not question:
            continue
        answer = chain.invoke(question)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()
