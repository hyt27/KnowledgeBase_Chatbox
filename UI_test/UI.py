# 导入必要的库
import gradio as gr
from langchain.vectorstores import Chroma
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
import os
from LLM import QwenLM, ChatGLM4_LLM
from langchain.prompts import PromptTemplate

def load_chain():
    # 加载问答链
    # 定义 Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="/home/lrctadmin/Documents/LLM/self-llm/models/Qwen/Langchain/embedding_model/bge-m3")

    # 向量数据库持久化路径
    persist_directory = '/home/lrctadmin/Documents/LLM/self-llm/models/Qwen/Langchain/data_base/vector_db/bge-m3'

    # 加载数据库
    vectordb = Chroma(
        persist_directory=persist_directory,  # 允许我们将persist_directory目录保存到磁盘上
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
    from langchain.chains import RetrievalQA

    qa_chain = RetrievalQA.from_chain_type(llm,
                                        retriever=vectordb.as_retriever(),
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt":QA_CHAIN_PROMPT})
    
    return qa_chain

class Model_center():
    """
    存储问答 Chain 的对象 
    """
    def __init__(self):
        self.chain = load_chain()

    def qa_chain_self_answer(self, question: str, chat_history: list = []):
        """
        调用不带历史记录的问答链进行回答
        """
        if question == None or len(question) < 1:
            return "", chat_history
        try:
            chat_history.append(
                (question, self.chain({"query": question})["result"]))
            return "", chat_history
        except Exception as e:
            return e, chat_history

    def clear_history(self):
        self.chain.clear_history()


model_center = Model_center()
LOGO_PATH = "/home/lrctadmin/Documents/LLM/self-llm/models/Qwen/Langchain/knowledgebase/lr.png"
                  
block = gr.Blocks()
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
            chatbot = gr.Chatbot(height=450, show_copy_button=True)
            # 创建一个文本框组件，用于输入 prompt。
            msg = gr.Textbox(label="Prompt-try")

            with gr.Row():
                # 创建提交按钮。
                db_wo_his_btn = gr.Button("Sent Chat Prompt")
            with gr.Row():
                # 创建一个清除按钮，用于清除聊天机器人组件的内容。
                clear = gr.ClearButton(
                    components=[chatbot], value="Clear console")
                
        # 设置按钮的点击事件。当点击时，调用上面定义的 qa_chain_self_answer 函数，并传入用户的消息和聊天历史记录，然后更新文本框和聊天机器人组件。
        db_wo_his_btn.click(model_center.qa_chain_self_answer, inputs=[
                            msg, chatbot], outputs=[msg, chatbot])
        
        # 点击后清空后端存储的聊天记录
        clear.click(model_center.clear_history)
    gr.Markdown("""Reminder:<br>
    1. It may take a long time to initialize the database, please wait patiently.
    2. If an exception occurs during use, it will be displayed in the text input box, please do not worry.<br>
    """)
with gr.Blocks() as knowledge:
    with gr.Row(equal_height=True):   
        with gr.Column(scale=15):
            page1_content = gr.Markdown("""
                    <h1><center>glm4</center></h1>
                    <center>trytrytry</center>
                """)
        
# threads to consume the request
gr.close_all()
                  
demo = gr.TabbedInterface([chat, knowledge], ["Chat", "Manage Knowledge Base"])
# 启动新的 Gradio 应用，设置分享功能为 True，并使用环境变量 PORT1 指定服务器端口。
#demo.launch(share=True, server_port=int(os.environ['PORT1']))
# 直接启动
demo.launch()
#demo.launch(inbrowser=True, inline=False, share=True) "top_k": 1}

#demo.launch(server_name="0.0.0.0", server_port=8000, debug=True) 
