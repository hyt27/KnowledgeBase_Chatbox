import gradio as gr
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from LLM import QwenLM, ChatGLM4_LLM
from langchain.prompts import PromptTemplate
import shutil
import tempfile
from Database import TextProcessor
from langchain.chains import RetrievalQA
from langchain.memory import ConversationBufferMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
import torch
from contextlib import contextmanager
from config import EMBEDDING_MODEL_PATH, VECTOR_DB_PATH, TARGET_DIRS,LLM_MODEL_PATH,LOGO_PATH

def load_chain():
    """
    Initialize and load the question-answering chain.
    This function sets up the embedding model, vector database, language model, and QA chain.
    """
    # Define embeddings using a pre-trained model
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_PATH)

    # Set the path for the persistent vector database
    persist_directory = VECTOR_DB_PATH

    # Load or create the vector database
    vectordb = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )

    # Configure generation parameters for the language model
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    
    # Initialize the ChatGLM4 language model
    llm = ChatGLM4_LLM(mode_name_or_path=LLM_MODEL_PATH, gen_kwargs=gen_kwargs)

    # Define the prompt template for question-answering
    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。尽量使答案简明扼要。"。
    {context}
    问题: {question}
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=template)

    # Set up conversation memory
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key='answer',
        k=3  # Remember last 3 interactions
    )

    # Create the conversational retrieval chain
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT}
    )
    
    return qa_chain

@contextmanager
def gpu_memory_manager():
    """
    Context manager for handling GPU memory.
    Ensures that GPU memory is cleared after each operation.
    """
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class Model_center():
    """
    Central class for managing the question-answering model and its operations.
    """
    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        Process a question through the QA chain and return the answer.
        Also handles the chat history and source documentation.
        """
        print(question)
        print(chat_history)
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            with gpu_memory_manager():
                result_chain = self.chain({"question": question})
                result = result_chain["answer"]
            
            # Process and format source documents
            source_documents = result_chain["source_documents"]
            unique_source = set()
            source = "\n\nSources:\n"
            for i, doc in enumerate(source_documents, 1):
                source_key = (doc.metadata['source'], doc.page_content[:200])
                if source_key not in unique_source:
                    unique_source.add(source_key)
                    source += f"\n{len(unique_source)}. Content excerpt: \n"
                    source += f"{doc.page_content[:200]}...\n"
                    source += f"From: {doc.metadata['source']}\n"
            
            # Update chat history
            chat_history.append((question, result + source))
            print(chat_history)
            return "", chat_history
        except Exception as e:
            return e, chat_history

    def clear_history(self):
        """
        Clear the conversation history and free up GPU memory.
        """
        self.chain.clear_history()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def generate_file(file_obj):
    # global tmpdir
    print('tmp dir：{}'.format('tmpdir'))
    FilePath=file_obj.name
    print('upload document path：{}'.format(file_obj.name)) # output the abusolute path of uploaded doc saved in Gradio

    #get the absolute path of doc, other operation same...
    
    # copy the document into tmp dictionary 
    shutil.copy(file_obj.name, 'tmpdir')

    # get the Gradio Doc name uploaded
    FileName=os.path.basename(file_obj.name)
    print(FilePath)
    # get the copy path in temp dir 

    # open the copy document
    with open(FilePath, 'rb') as file_obj:

        #open a new document in local document, and write context in new documentd
        outputPath=os.path.join(knowledgebase_path,FileName)
        with open(outputPath,'wb') as w:
            w.write(file_obj.read())

    # notice: return new documents path
    target_dirs = [knowledgebase_path]
    # create TextProcessor instance
    processor = TextProcessor(target_dirs)

    # execute text process
    processor.process_text()

    print("Files vectorized successfully!")
    #app.show("Files vectorized successfully!")
    gr.Info("Files vectorized successfully!")
    
    files = os.listdir(knowledgebase_path)
    
    files_table = "<tr><th>File Name</th></tr>"
    for file in files:
        files_table += f"<tr><td>{file}</td></tr>"
    
    files_list_output.html = f"<table>{files_table}</table>"  # show the list into HTML table
    print(files_list_output.html)
    #print(files_table) 
    return files_list_output.html
    #return outputPath

def vectorize_files():
    target_dirs = [knowledgebase_path]
    # create TextProcessor instance
    processor = TextProcessor(target_dirs)

    # text process pogresss
    processor.process_text()

    print("Files vectorized successfully!")
    #app.show("Files vectorized successfully!")
    gr.Info("Files vectorized successfully!")
    
    #list_files()
    
def list_files():
    
    files = os.listdir(knowledgebase_path)
    
    files_table = "<tr><th>File Name</th></tr>"
    for file in files:
        files_table += f"<tr><td>{file}</td></tr>"
    
    files_list_output.html = f"<table>{files_table}</table>"
    print(files_list_output.html)
    #print(files_table) 
    return files_list_output.html        

# Initialize the model center
model_center = Model_center()

# Set up paths for logo and knowledge base
knowledgebase_path = VECTOR_DB_PATH

# Initialize the files list display
files_table = "<tr><th></th></tr>"
files_list_output = gr.HTML(f"<table id='files_table'>{files_table}</table>")
list_files()

# Define the chat interface
with gr.Blocks() as chat:
    # Create a row for the header
    with gr.Row(equal_height=True):   
        # Main title column
        with gr.Column(scale=15):
            gr.Markdown("""                    
            <h1><center>LR Chatbox Demo</center></h1>
            <center>v1.0.0</center>
                """)
        # Logo image
        gr.Image(value=LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False)

    # Create a row for the main content
    with gr.Row():
        # Left column for model selection
        with gr.Column(scale=1):
            # Dropdown for selecting the LLM model
            llm_mode = gr.Dropdown(choices=["glm-4", "qwen"], label="Choose LLM Model")
            # Commented out logo (can be uncommented if needed)
            #logo = gr.Image(value=LOGO_PATH, scale=1, min_width=5, show_label=False, show_download_button=False)
            
        # Right column for chat interface
        with gr.Column(scale=4):
            # Chatbot component to display conversation
            chatbot = gr.Chatbot(height=450, show_copy_button=True, bubble_full_width=False)
            # Text input for user prompts
            msg = gr.Textbox(label="Prompt-try")

            # Row for chat button
            with gr.Row():
                db_wo_his_btn = gr.Button("Sent Chat Prompt")
            # Row for clear button
            with gr.Row():
                clear = gr.ClearButton(components=[chatbot], value="Clear console")
                
        # Set up click event for the chat button
        # When clicked, it calls qa_chain_self_answer function with user's message and chat history
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[msg, chatbot], outputs=[msg, chatbot])
        print(msg)
        print(chatbot)
        
        # Set up click event for the clear button
        # When clicked, it clears the backend chat history
        clear.click(model_center.clear_history)

    # Add reminder text at the bottom of the chat interface
    gr.Markdown("""Reminder:<br>
    1. It may take a long time to initialize the database, please wait patiently.
    2. If an exception occurs during use, it will be displayed in the text input box, please do not worry.<br>
    """)

# Define the knowledge base management interface
with gr.Blocks() as knowledge:
    # Create a row for the header (similar to chat interface)
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            gr.Markdown("""                    
            <h1><center>LR Chatbox Demo</center></h1>
            <center>v1.0.0</center>
                """)
        gr.Image(value=LOGO_PATH, scale=1, min_width=10, show_label=False, show_download_button=False)
    
    # Create a row for file upload and display
    with gr.Row():
        with gr.Column(scale=1):
            # File upload component
            inputs = gr.components.File(label="Upload File")
            # HTML component to display the list of files
            outputs = gr.HTML(files_list_output.html)
            # Create an interface for file generation
            app = gr.Interface(fn=generate_file, inputs=inputs, outputs=outputs,
                   description="Support document format: txt, md, csv, pdf, png, jpg")
        
# Close all existing Gradio threads
gr.close_all()
                  
# Create a tabbed interface combining chat and knowledge base management
demo = gr.TabbedInterface([chat, knowledge], ["Chat", "Manage Knowledge Base"])

# Launch the Gradio app
demo.launch(server_name="0.0.0.0", server_port=8000, debug=True) 

