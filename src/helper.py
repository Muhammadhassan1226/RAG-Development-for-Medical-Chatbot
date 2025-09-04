from typing import List
from langchain.schema import Document
from langchain.embeddings  import HuggingFaceEmbeddings
from langchain_community.document_loaders  import  PyMuPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
# from langchain.embeddings import HuggingFaceHubEmbeddings

# https://github.com/langchain-ai/langchain/issues/29768





def extract_data(data):
    loader = DirectoryLoader(data,
                             glob="*.pdf",
                             loader_cls=PyMuPDFLoader)

    documents = loader.load()
    return documents


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Filters the input list of Document objects to retain only those with the minimal content length.

    Args:
        docs (List[Document]): A list of Document objects.

    Returns:
        List[Document]: A list of Document objects with the minimal content length.
    """
    if not docs:
        return []
    
    minimal_docs : List[Document] = []
    for doc in docs:
        src  = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metdata={"source":src}
            )
        )
    return minimal_docs


def text_split(filtered_data):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
    text_chunks = text_splitter.split_documents(filtered_data)
    return text_chunks

def download_embeddings():
    '''
    Downloads and initializes the HuggingFaceHubEmbeddings model.
    '''
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    return embeddings