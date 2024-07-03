import os
import pickle
import shutil
import warnings
import streamlit as st
import csv_app_helper as cs
from dotenv import load_dotenv
from openai import OpenAI
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from langchain.prompts import PromptTemplate
import pytesseract
from PIL import Image
import whisper
from whisper.utils import get_writer

# Suppress warnings
warnings.filterwarnings("ignore")

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
os.environ['FINAL_DATA_PATH'] = os.getenv("FINAL_DATA_PATH")
os.environ['STRUCTURED_DATA_PATH'] = os.getenv("STRUCTURED_DATA_PATH")
os.environ['TESSERACT_PATH'] = os.getenv("TESSERACT_PATH")
os.environ['PDF_PATH'] = os.getenv("PDF_PATH")

# Load Whisper model
model = whisper.load_model("base")
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

INDEX_FILE_PATH = 'imageLlama.pkl'

def load_data_from_directory(directory_path):
    reader = SimpleDirectoryReader(directory_path)
    documents = reader.load_data()
    return documents

pytesseract.pytesseract.tesseract_cmd = os.getenv('TESSERACT_PATH')

def process_image(imagePath):
        imagePath = imagePath.replace("\\", '/')
        image = Image.open(imagePath)
        text = pytesseract.image_to_string(image)
        with open(f"{os.getenv('FINAL_DATA_PATH')}/image_updated_{(imagePath.split('/')[-1]).split('.')[0]}.txt", "w") as file:
            file.write(text)
        print('image saved successfully')

def process_images_in_folder(folderPath):
    for filename in os.listdir(folderPath):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            imagePath = os.path.join(folderPath, filename)
            process_image(imagePath)

def process_audio_files_in_folder(audio_directory, output_directory):
    for filename in os.listdir(audio_directory):
        if filename.endswith((".wav", ".mp3", ".m4a")):
            audio_path = os.path.join(audio_directory, filename)
            audio_path = audio_path.replace("\\", '/')
            print(audio_path)
            # Transcribe the audio file
            result = model.transcribe(audio_path, fp16=False)
            
            # Save transcription without line breaks
            txt_path = os.path.join(output_directory, f"{os.path.splitext(filename)[0]}_nolinebreaks.txt")
            txt_path = txt_path.replace("\\", '/')
            with open(txt_path, "w", encoding="utf-8") as txt:
                txt.write(result["text"])
            
            # Save transcription with hard line breaks
            txt_writer = get_writer("txt", output_directory)
            txt_writer(result, audio_path)
            
            print(f"Transcription saved for {filename}")

def processPdf(pdf_directory):
    for filename in os.listdir(pdf_directory):
        if filename.endswith((".pdf")):
            total_path = f'{pdf_directory}/{filename}'
            shutil.copy(total_path, os.getenv('FINAL_DATA_PATH'))

def create_llama_index(documents):
    index = VectorStoreIndex.from_documents(documents)
    return index

# Save LlamaIndex to a file
def save_index(index, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(index, file)

def load_index(file_path):
    with open(file_path, 'rb') as file:
        index = pickle.load(file)
    return index

def generate_custom_response(index, query, chat_history_arr):
    #chat_history = "".join(chat_history_arr)
    retriever = index.as_retriever()
    
    prompt_template = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="Based on the following context, {context}, and the conversation history with the current agent, {chat_history}, answer the question: {question}."
    )
    
    context = retriever.retrieve(query)
    
    formatted_query = prompt_template.format(context=context, question=query, chat_history=chat_history)

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant, so answer the question with chat history included."},
            {"role": "user", "content": formatted_query}
        ],
        model="gpt-3.5-turbo",
    )
    return response.choices[0].message.content

num=0

def unstructuredChat(directory_path, query, chat_history, indexPath, reload):
    if os.path.exists(indexPath):
        if reload=='No':
            index = load_index(indexPath)
            st.sidebar.write("Loaded index from file.")
    elif ((os.path.exists(indexPath) and (reload=="Yes")) or (not os.path.exists(indexPath))):
        process_images_in_folder(os.getenv('IMAGE_PATH'))
        #process_audio_files_in_folder(os.getenv('AUDIO_PATH'), os.getenv('FINAL_DATA_PATH'))
        processPdf(os.getenv('PDF_PATH'))
        documents = load_data_from_directory(directory_path)
        index = create_llama_index(documents)
        save_index(index, indexPath)
        st.sidebar.write("Created new index and saved successfully")
    else: st.sidebar.write("Indexing Error")
    response = generate_custom_response(index, query, chat_history)
    #chat_history.append(f"user: {query}\nAI assistant: {response}\n")
    chat_history.append({"user": query, "AI": response})
    st.session_state["chat_history"] = chat_history
    #st.write(response)

def csvChat(query, csv_file):
    agent = cs.ChatWithCSVAgent(csv_file)
    response = agent.run(query)['output']
    #chat_history.append(f"user: {query}\nAI assistant: {response}\n")
    chat_history.append({"user": query, "AI": response})
    st.session_state["chat_history"] = chat_history
    #st.write(response)

def main():
    # Custom CSS for chat layout
    st.markdown("""
            <style>
                .user-message {
                    background-color: #DCF8C6;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 10px;
                    text-align: right;
                    color: black;
                }
                .bot-message {
                    background-color: #C2DFFF;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 10px;
                    text-align: left;
                    color: black;
                }
            </style>
        """, unsafe_allow_html=True)
    st.sidebar.title("Chatbot")

    if st.sidebar.button("Refresh"):
        st.session_state["chat_history"] = []
        st.write("App refreshed!")

    chattype = st.sidebar.selectbox("Choose the chat type: ", ("RAG", "CSV_Chat"), index=None, placeholder="No Selection")

    if chattype == 'CSV_Chat':
        #csv.main()
        csv_file = cs.chooseFile(os.getenv('STRUCTURED_DATA_PATH'))
        if csv_file:
            query = st.chat_input("Enter a query...")
            if query:
                    csvChat(query, csv_file)

    elif chattype == 'RAG':
        reload = st.sidebar.selectbox('Want to reload index?', ("No", "Yes"), index=None, placeholder="No Selection")
        query = st.chat_input("Enter a query...")
        if query:
                indexPath = 'llama_final.pkl'  # Modify as needed
                unstructuredChat(os.getenv('FINAL_DATA_PATH'), query, chat_history, indexPath, reload)
    
    # # Retrieve chat history from session state
    # chat_history = st.session_state.get("chat_history", [])
    # Display chat history
    for interaction in chat_history:
            st.markdown(f'<div class="user-message">{interaction["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message">{interaction["AI"]}</div>', unsafe_allow_html=True)

# Initialize chat history as an empty list
chat_history = st.session_state.get("chat_history", [])

if __name__ == "__main__":
    main()
