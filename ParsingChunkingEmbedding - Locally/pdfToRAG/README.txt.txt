Clean Architecture 

Use two scripts.

project/
│
├── embed_pdf.py
├── query_rag.py
├── your_document.pdf
└── faiss_index/

Workflow:

embed_pdf.py → run once
query_rag.py → run many times
--------------------------------------------------
Installation (VERY IMPORTANT)

Run these first.

pip install langchain langchain-community langchain-ollama
pip install pypdf faiss-cpu sentence-transformers
pip install ollama

Then pull local model:

ollama pull llama3


-----------------------------------------
Optional:

ollama pull mistral