import re
from googlesearch import search
from utils import *
from urllib.request import Request, urlopen
from typing import Dict, List, Union

# from langchain_core.documents import Document
from langchain.docstore.document import Document

from langchain.storage import InMemoryStore
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

class ParentDocumentRetriver:
    def __init__(
        self,
        child_chunk_size,
        parent_chunk_size,
        hf_key,
        model_name='sentence-transformers/all-mpnet-base-v2',
    ) -> None:
        
        self.child_chunk_size = child_chunk_size
        self.parent_chunk_size = parent_chunk_size
        self.hf_key = hf_key
        self.model_name = model_name
    
    def retrieve(self, query, num_sites, return_child_document=False):
        # Embedding
        embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=self.hf_key, model_name=self.model_name
        )

        # This text splitter is used to create the parent documents
        parent_splitter = RecursiveCharacterTextSplitter(chunk_size=self.parent_chunk_size)

        # This text splitter is used to create the child documents
        child_splitter = RecursiveCharacterTextSplitter(chunk_size=self.child_chunk_size)

        # The vectorstore to use to index the child chunks
        vectorstore = Chroma(
            collection_name="full_documents", embedding_function=embeddings
        )

        # The storage layer for the parent documents
        store = InMemoryStore()
        parent_retriever = ParentDocumentRetriever(
            vectorstore=vectorstore,
            docstore=store,
            child_splitter=child_splitter,
            parent_splitter=parent_splitter
        )

        # Get docs 
        docs = self.get_docs(query, num_sites=num_sites)
        for doc in docs:
            if doc.page_content == '':
                continue
            parent_retriever.add_documents([doc], ids=None) # Add documents    

        # The storage layer for the child documents
        return vectorstore.as_retriever()

    def get_docs(self, query, num_sites, lang='en'):
        links = search(query, sleep_interval=1, num_results=num_sites, lang=lang)
        page_contents = []
        for path in links:        
            loader = BSHTMLLoader(path)
            page_contents.extend(loader.load())
        return page_contents


class BSHTMLLoader(BaseLoader):
    """Loader that uses beautiful soup to parse HTML files."""

    def __init__(
        self,
        path: str,
        open_encoding: Union[str, None] = None,
        bs_kwargs: Union[dict, None] = None,
        get_text_separator: str = "",
    ) -> None:
        """Initialise with path, and optionally, file encoding to use, and any kwargs
        to pass to the BeautifulSoup object.
        """
        self.file_path = path
        self.open_encoding = open_encoding
        if bs_kwargs is None:
            bs_kwargs = {"features": "lxml"}
        self.bs_kwargs = bs_kwargs
        self.get_text_separator = get_text_separator

    def load(self) -> List[Document]:
        """Load HTML document into document objects."""
        from bs4 import BeautifulSoup
        
        if is_file_path(self.file_path):
            with open(self.file_path, "r", encoding=self.open_encoding) as f:
                soup = BeautifulSoup(f, **self.bs_kwargs)
        elif is_http_path(self.file_path):
            try:
                req = Request(self.file_path, headers={'User-Agent': 'Mozilla/5.0'})
                html_page = urlopen(req).read()
            except:
                return [Document(page_content='')]
            soup = BeautifulSoup(markup=html_page, features='html.parser') # , **self.bs_kwargs

        text = soup.get_text(self.get_text_separator)
        text = re.sub(r'\s+', ' ', text)

        if soup.title:
            title = str(soup.title.string)
        else:
            title = ""

        metadata: Dict[str, Union[str, None]] = {
            "source": self.file_path,
            "title": title,
        }
        return [Document(page_content=text, metadata=metadata)]
    