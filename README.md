# RAG-Integra
Basic RAG based chatbot which uses a custom knowledge base to improve the results.

* Create three new folders, one for `images`, one for `audio_files` and another for `pdf`
* Add all 'text' based files into **data** folder
* Add 'csv/xlsx' files into **structuredData** folder [Make sure to use the copy of the file]
* Run `conda create -n <environment-name> --file reqs.txt` to install all required libraries [Having installed **anaconda** or **miniconda** on your PC]
* If the above code doesn't work for you, then use this -> `pip install pandas llama-index langchain python-dotenv pytesseract whisper openai-whisper streamlit langchain_openai langchain_experimental tabulate`
* To run **csvChat.py** use this `streamlit run csvChat.py`
* To run **UnstructuredRAG.py** use this `python UnstructuredRAG.py`
* Provide 'path' to your all the required data in the `.env` file. 
* Add your OPENAI_API_KEY in the **.env** file
* To run the **Chatbot App** which works with both structured and unstructured data, run `streamlit run Chatbot.py`
