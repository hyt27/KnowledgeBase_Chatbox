# Knowledgebase_Chatbox

## 项目描述
这是一个基于大语言模型和向量数据库的智能问答系统。它能够处理多种文档格式，构建知识库，并提供智能对话和文档管理功能。

## 功能特点
- 支持多种文档格式（txt, md, csv, pdf, png, jpg）的处理和索引
- 使用向量数据库进行高效的文档检索
- 集成多个大语言模型（如QwenLM和ChatGLM4）
- 提供用户友好的聊天界面和知识库管理界面
- 实现对话历史记忆功能
- 支持GPU内存管理和清理

## 系统架构
![系统架构图](/jupyter_test/whiteboard_exported_image.png)

## 安装说明
1. 克隆项目仓库
2. 安装依赖项
```   
pip install -r requirements.txt
```
3. 配置路径
在config.py中，將路徑改成本地路徑

## 使用方法
运行以下命令启动系统：
```
python startup.py
```

## 文件结构
- `data_base/`: 存储创建的向量数据库，注意使用不同的embedding model的時候要創建不同的向量數據庫。
- `embedding_model/`: 存放embedding模型
- `LLM_model/`: 存放使用的LLM模型
- `knowledgebase/`: 存放知识数据文件
- `version_update/`: 存放每个版本的UI界面（仅用于记录）
- `jupyter_test/`: 存放功能测试代码和日志文件

## 主要代码文件
- `Database.py`: 构建向量数据库的脚本。創建了一個 TextProcessor 實例，並指定了多個ducument loader來處理文檔，目前只能支持Support document format: txt, md, csv, pdf, png, jpg。這個過程建立一個可搜索的文檔庫，用於後續的文本檢索和問答系統。
- `LLM.py`: 将 InternLM 封装为自定义 LLM 的脚本。這個 Python 文件定義了兩個自定義的語言模型類：QwenLM 和 ChatGLM4_LLM，它們都繼承自 LangChain 的 LLM 基類。
- `Startup.py`: 启动Gradio服务的脚本。提供兩個主要界面，聊天界面:用戶可以與AI進行對話；知識庫管理界面:用戶可以上傳和管理文檔。
- `documentLoader.py`: 实现文档加载和处理的脚本，這個代碼實現了一個名為 RapidOCRPDFLoader 的類，用於從 PDF 文件中提取文本和圖像。


## 作者
[Patty Hao]

## 许可证
[]

## 项目状态
[积极开发中]
