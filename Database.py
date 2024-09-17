# Import necessary libraries and modules
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredMarkdownLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders import JSONLoader, PyPDFLoader
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
import os
from documentLoader import RapidOCRPDFLoader
from config import EMBEDDING_MODEL_PATH, VECTOR_DB_PATH, TARGET_DIRS

class TextProcessor:
    """
    A class for processing and indexing various types of documents.
    It can handle multiple file formats, split documents, create embeddings,
    and store them in a vector database for efficient retrieval.
    """

    def __init__(self, target_dirs):
        """
        Initialize the TextProcessor with target directories.

        :param target_dirs: List of directory paths to process
        """
        self.target_dirs = target_dirs
        self.docs = []  # Store all loaded documents
        self.split_docs = None  # Store split documents
        self.embeddings = None  # Store document embeddings
        self.vectordb = None  # Vector database for storing embeddings

    def get_files(self, dir_path):
        """
        Recursively get all files from a directory.

        :param dir_path: Path to the directory
        :return: List of file paths
        """
        file_list = []
        for filepath, dirnames, filenames in os.walk(dir_path):
            for filename in filenames:
                file_list.append(os.path.join(filepath, filename))
        return file_list

    def get_text(self, dir_path):
        """
        Load and process all supported files in a directory.

        :param dir_path: Path to the directory
        :return: List of loaded documents
        """
        file_lst = self.get_files(dir_path)
        docs = []
        for one_file in tqdm(file_lst):
            file_type = one_file.split('.')[-1]
            # Select appropriate loader based on file type
            if file_type == 'md':
                loader = UnstructuredMarkdownLoader(one_file)
            elif file_type == 'txt':
                loader = UnstructuredFileLoader(one_file)
            elif file_type == 'csv':
                loader = CSVLoader(one_file) 
            elif file_type == 'pdf':
                loader = PyPDFLoader(one_file) 
                # Uncomment the following line to use RapidOCRPDFLoader instead
                #loader = RapidOCRPDFLoader(one_file)
            elif file_type in ['png', 'jpg']:
                loader = UnstructuredImageLoader(one_file)
            else:
                continue  # Skip unsupported file types
            docs.extend(loader.load())
        return docs

    def process_text(self):
        """
        Main method to process all documents in the target directories.
        This method loads documents, splits them, creates embeddings,
        and stores them in a vector database.
        """
        # Load all documents from target directories
        for dir_path in self.target_dirs:
            self.docs.extend(self.get_text(dir_path))

        # Split documents into smaller chunks for better processing
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=150)
        self.split_docs = text_splitter.split_documents(self.docs)

        # Create embeddings using a pre-trained HuggingFace model
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

        # Define the directory to persist the vector database
        persist_directory = VECTOR_DB_PATH
        
        # Create and persist the Chroma vector database
        self.vectordb = Chroma.from_documents(documents=self.split_docs, embedding=embeddings, persist_directory=persist_directory)
        self.vectordb.persist()
        print("Vector database persisted successfully.")

# Define the target directories to process
target_dirs = [TARGET_DIRS,]

# Create an instance of TextProcessor
processor = TextProcessor(target_dirs)

# Execute the text processing pipeline
processor.process_text()