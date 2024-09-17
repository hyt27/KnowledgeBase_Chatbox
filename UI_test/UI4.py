import os
import gradio as gr
import shutil
import tempfile
from Database import TextProcessor

LOGO_PATH = "/home/lrctadmin/Documents/LLM/self-llm/models/Qwen/Langchain/knowledgebase/lr.png"
knowledgebase_path = "/home/lrctadmin/Documents/LLM/self-llm/models/Qwen/Langchain/knowledgebase"

def generate_file(file_obj):
    # global tmpdir
    print('tmp dir：{}'.format('tmpdir'))
    FilePath=file_obj.name
    print('upload document path：{}'.format(file_obj.name)) # 输出上传后的文件在gradio中保存的绝对地址

    #获取到上传后的文件的绝对路径后，其余的操作就和平常一致了

    # 将文件复制到临时目录中
    shutil.copy(file_obj.name, 'tmpdir')

    # 获取上传Gradio的文件名称
    FileName=os.path.basename(file_obj.name)
    print(FilePath)
    # 获取拷贝在临时目录的新的文件地址

    # 打开复制到新路径后的文件
    with open(FilePath, 'rb') as file_obj:

        #在本地电脑打开一个新的文件，并且将上传文件内容写入到新文件
        outputPath=os.path.join(knowledgebase_path,FileName)
        with open(outputPath,'wb') as w:
            w.write(file_obj.read())

    # 返回新文件的的地址（注意这里）
    target_dirs = [knowledgebase_path]
    # 创建 TextProcessor 实例
    processor = TextProcessor(target_dirs)

    # 执行文本处理流程
    processor.process_text()

    print("Files vectorized successfully!")
    #app.show("Files vectorized successfully!")
    gr.Info("Files vectorized successfully!")
    
    files = os.listdir(knowledgebase_path)
    
    files_table = "<tr><th>File Name</th></tr>"
    for file in files:
        files_table += f"<tr><td>{file}</td></tr>"
    
    files_list_output.html = f"<table>{files_table}</table>"  # 将文件列表设置到HTML表格中显示
    print(files_list_output.html)
    #print(files_table) 
    return files_list_output.html
    #return outputPath


def vectorize_files():
    target_dirs = [knowledgebase_path]
    # 创建 TextProcessor 实例
    processor = TextProcessor(target_dirs)

    # 执行文本处理流程
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

files_table = "<tr><th></th></tr>"
files_list_output = gr.HTML(f"<table id='files_table'>{files_table}</table>")
list_files() 
with gr.Blocks() as demo:
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
            #logo = gr.Image(value=LOGO_PATH, scale=1, min_width=5, show_label=False, show_download_button=False)

    #with gr.Row():                
    #    vectorize_button = gr.Button("Click to Add the document into Vector Database")
    #with gr.Row():
        #know_button = gr.Button("knowledge") 
    #    files_list_output = gr.Interface(fn=list_files, inputs=None, outputs=gr.HTML(files_list_output.html),description="Click Generate to upload documents list")
        #gr.Markdown("""<table>{file_table}</table>""")
        #
    #vectorize_button.click(vectorize_files)
    #know_button.click(list_files)
    #live=True
demo.launch()