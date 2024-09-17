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

def load_chain():
    # load question chain
    # define Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/home/lrctadmin/Documents/LLM/self-llm/models/Qwen/Langchain/embedding_model/bge-m3")

    # persist vector database path
    persist_directory = '/home/lrctadmin/Documents/LLM/self-llm/models/Qwen/Langchain/data_base/vector_db/bge-m3'

    # load database
    vectordb = Chroma(
        persist_directory=persist_directory,  # allow save persist_directory in to disk
        embedding_function=embeddings
    )

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    llm = ChatGLM4_LLM(mode_name_or_path="/home/lrctadmin/Documents/LLM/self-llm/models/GLM-4/glm-4-9b-chat", gen_kwargs=gen_kwargs)

    template = """使用以下上下文来回答最后的问题。如果你不知道答案，就说你不知道，不要试图编造答
    案。尽量使答案简明扼要。”。
    {context}
    问题: {question}
    有用的回答:"""

    QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context","question"],
                                    template=template)

    # 运行 chain
    
    memory = ConversationBufferWindowMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key='answer',
        k=3
    )
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        retriever=vectordb.as_retriever(),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={'prompt': QA_CHAIN_PROMPT}
    )
    #qa_chain = RetrievalQA.from_chain_type(llm,retriever=vectordb.as_retriever(),return_source_documents=True,chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    
    return qa_chain

@contextmanager
def gpu_memory_manager():
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class Model_center():
    """
    save question Chain object
    """
    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        call question chain to answer without history
        """
        print(question)
        print(chat_history)
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            with gpu_memory_manager():
                result_chain=self.chain({"question": question})
                result=result_chain["answer"]
            source_documents=result_chain["source_documents"]
            
            unique_source = set()
            source = "\n\nSources:\n"
            for i,doc in enumerate(source_documents,1):
                source_key = (doc.metadata['source'],doc.page_content[:200])
                if source_key not in unique_source:
                    unique_source.add(source_key)
                    source += f"\n{len(unique_source)}. Content excerpt: \n"
                    source += f"{doc.page_content[:200]}...\n"
                    source += f"From: {doc.metadata['source']}\n"
                
            chat_history.append(
                (question, result+source))
            print(chat_history)
            return "", chat_history
        except Exception as e:
            return e, chat_history

    def clear_history(self):
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

model_center = Model_center()
LOGO_PATH = "/home/lrctadmin/Documents/LLM/self-llm/models/Qwen/Langchain/knowledgebase/lr.png"
knowledgebase_path = "/home/lrctadmin/Documents/LLM/self-llm/models/Qwen/Langchain/knowledgebase"
  
files_table = "<tr><th></th></tr>"
files_list_output = gr.HTML(f"<table id='files_table'>{files_table}</table>")
list_files()
    
with gr.Blocks() as chat:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            gr.Markdown("""                    
            <h1><center>LR Chatbox Demo</center></h1>
            <center>v1.0.0</center>
                """)
        gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)

    with gr.Row():
        with gr.Column(scale=1):
            llm_mode = gr.Dropdown(choices=["glm-4", "qwen"], label="Choose LLM Model")#, value="glm-4"
            #logo = gr.Image(value=LOGO_PATH, scale=1, min_width=5, show_label=False, show_download_button=False)
            
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(height=450, show_copy_button=True, bubble_full_width=False)
            # text box for prompt。
            msg = gr.Textbox(label="Prompt-try")

            with gr.Row():
                # chat
                db_wo_his_btn = gr.Button("Sent Chat Prompt")
            with gr.Row():
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])
        print(msg)
        print(chatbot)
        
        # 点击后清空后端存储的聊天记录
        clear.click(model_center.clear_history)
    gr.Markdown("""Reminder:<br>
    1. It may take a long time to initialize the database, please wait patiently.
    2. If an exception occurs during use, it will be displayed in the text input box, please do not worry.<br>
    """)
with gr.Blocks() as knowledge:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            gr.Markdown("""                    
            <h1><center>LR Chatbox Demo</center></h1>
            <center>v1.0.0</center>
                """)
        gr.Image(value=LOGO_PATH, scale=1, min_width=10,show_label=False, show_download_button=False)
    with gr.Row():
        with gr.Column(scale=1):
            inputs = gr.components.File(label="Upload File")
            outputs = gr.HTML(files_list_output.html)
            app = gr.Interface(fn=generate_file, inputs=inputs, outputs=outputs,
                   description="Support document format: txt, md, csv, pdf, png, jpg")
        
# threads to consume the request
gr.close_all()
                  
demo = gr.TabbedInterface([chat, knowledge], ["Chat", "Manage Knowledge Base"])
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
#demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
#demo.launch()
#demo.launch(inbrowser=True, inline=False, share=True) 

demo.launch(server_name="0.0.0.0", server_port=8000, debug=True) 
