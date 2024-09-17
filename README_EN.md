# LR Chatbox

## Project Description
This is an intelligent Q&A system based on large language models and vector databases. It can process multiple document formats, build knowledge bases, and provide intelligent dialogue.

## Features
- Support for processing and indexing multiple document formats (txt, md, csv, pdf, png, jpg)
- Efficient document retrieval using vector databases
- Integration of multiple large language models (such as Qwen7B and ChatGLM4)
- User-friendly chat interface and knowledge base management interface
- Dialogue history memory function
- GPU memory management and cleanup support

## System Architecture
![System Architecture Diagram](/jupyter_test/whiteboard_exported_image.png)

## Installation Instructions
1. Clone the project repository
2. Install dependencies
```   
pip install -r requirements.txt
```
3. Configure paths

In `config.py`, change the paths to your local paths

## Usage
Run the following command to start the system:
```
python startup.py
```

## File Structure
- `data_base/`: Stores created vector databases. Note that different vector databases should be created when using different embedding models.
- `embedding_model/`: Stores embedding models
- `LLM_model/`: Stores the LLM models used
- `knowledgebase/`: Stores knowledge data files
- `version_update/`: Stores UI interfaces for each version (for record-keeping only)
- `jupyter_test/`: Stores functional test code and log files

## Main Code Files
- `Database.py`: Script for building vector databases. Creates a TextProcessor instance and specifies multiple document loaders to process documents. Currently supports document formats: txt, md, csv, pdf, png, jpg. This process builds a searchable document library for subsequent text retrieval and Q&A systems.
- `LLM.py`: Script that encapsulates InternLM as custom LLMs. This Python file defines two custom language model classes: QwenLM and ChatGLM4_LLM, both inheriting from LangChain's LLM base class.
- `Startup.py`: Script to start the Gradio service. Provides two main interfaces, a chat interface where users can converse with AI, and a knowledge base management interface where users can upload and manage documents.
- `documentLoader.py`: Script implementing document loading and processing. This code implements a class called RapidOCRPDFLoader for extracting text and images from PDF files.

## Author
@patty.hao

## License

## Project Status
Active Development


