# RAG-Integra
Basic RAG based chatbot which uses a custom knowledge base to improve the results.

* Add all 'text' based files into **data** folder
* Add 'csv/xlsx' files into **structuredData** folder [Make sure to use the copy of the file]
* Run `conda create --name <env> --file reqs.txt` to install all required libraries
* To run **csvChat.py** use this `streamlit run csvChat.py`
* To run **UnstructuredRAG.py** use this `python UnstructuredRAG.py`
* Provide 'path' to your data at the respective parts of the code.
* Add you OPENAI_API_KEY in the **.env** file and in **csvChat.py**
