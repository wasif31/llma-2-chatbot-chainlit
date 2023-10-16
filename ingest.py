from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader,DirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
Data_Path = "Data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

def create_vector_db():
    print('1')
    loader = DirectoryLoader(Data_Path,glob='*.pdf',loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
    texts = text_splitter.split_documents(documents)
    print('2')
    embeddings = HuggingFaceEmbeddings(model_name = 'sentence-transformers/all-miniLM-L6-v2',
                                       model_kwargs = {'device': 'cpu'})
    db = FAISS.from_documents(texts,embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
    print('1')
    create_vector_db()
