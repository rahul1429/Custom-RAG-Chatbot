import os
import pandas as pd
import pickle
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from openai import OpenAI
import pytesseract
from PIL import Image
import whisper
from whisper.utils import get_writer

# Load Whisper model
model = whisper.load_model("base")

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

INDEX_FILE_PATH = 'imageLlama.pkl'

# Load data from a directory
def load_data_from_directory(directory_path):
    reader = SimpleDirectoryReader(directory_path)
    documents = reader.load_data()
    return documents

pytesseract.pytesseract.tesseract_cmd = "D:/pytesseract/tesseract.exe"

def process_image(imagePath):
        imagePath = imagePath.replace("\\", '/')
        image = Image.open(imagePath)
        text = pytesseract.image_to_string(image)
        with open(f"PATH TO YOUR DATA/image_updated_{(imagePath.split('/')[-1]).split('.')[0]}.txt", "w") as file:
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

# Create LlamaIndex and index documents
def create_llama_index(documents):
    index = VectorStoreIndex.from_documents(documents)
    return index

# Save LlamaIndex to a file
def save_index(index, file_path):
    with open(file_path, 'wb') as file:
        pickle.dump(index, file)

# Load LlamaIndex from a file
def load_index(file_path):
    with open(file_path, 'rb') as file:
        index = pickle.load(file)
    return index

def generate_custom_response(index, query, chat_history_arr):
    chat_history = ""
    for val in chat_history_arr:
        chat_history += val
    # Configure LangChain with OpenAI
    retriever = index.as_retriever()
    
    # Define a custom prompt
    prompt_template = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template="Based on the following context, {context}, and the conversation history with the current agent, {chat_history}, answer the question: {question}."
    )
    
    # Retrieve context using the retriever
    context = retriever.retrieve(query)
    
    # Use the prompt template to create a formatted query
    formatted_query = prompt_template.format(context=context, question=query, chat_history=chat_history)

    # Generate the response using ChatCompletion
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant, so answer the question with chat history included."},
            {"role": "user", "content": formatted_query}
        ],
        model="gpt-3.5-turbo",
    )
    return response.choices[0].message.content

def main(directory_path, query, chat_history, indexPath):
    # Load or create index
    if os.path.exists(indexPath):
        index = load_index(indexPath)
        print("Loaded index from file.")
    else:
        documents = load_data_from_directory(directory_path)
        index = create_llama_index(documents)
        save_index(index, indexPath)
        print("Created new index and saved to file.")
    
    # Generate and print custom response
    #print('chat history:', chat_history, sep=' ')
    response = generate_custom_response(index, query, chat_history)
    chat_history.append(f"user: {query}\nAI assistant: {response}\n")
    #print('updated chat history: ', chat_history)
    print(response)

if __name__ == "__main__":
    process_images_in_folder("PATH TO YOUR IMAGES")
    process_audio_files_in_folder("PATH TO YOUR AUDIO FILES", "PATH TO YOUR DATA")
    chat_history = []
    exit_keywords = ['Q', 'q', 'exit', 'EXIT', 'quit', 'QUIT']
    while True:
        query = ''
        query = input("Enter a query\n")
        if query in exit_keywords:
            print("Exiting...")
            break
        else:
            indexPath = 'llama_final.pkl'
            main("PATH TO YOUR DATA", query, chat_history, indexPath)

