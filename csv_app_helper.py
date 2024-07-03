import os
import warnings
import streamlit as st
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain.agents.agent_types import AgentType

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

# Suppress warnings
warnings.filterwarnings("ignore")

# Initialize OpenAI
llm = OpenAI(temperature=0, openai_api_key=os.environ.get("OPENAI_API_KEY"))

def csvFilesPopulation(path_to_csv_files):
    csvFiles = []
    for filename in os.listdir(path_to_csv_files):
        tmp = os.path.join(path_to_csv_files, filename)
        tmp = tmp.replace("\\", '/')
        csvFiles.append(tmp)
    return csvFiles

def chooseFile(path):
    csvFiles = csvFilesPopulation(path)
    if 'selected_csv_file' not in st.session_state:
        st.session_state.selected_csv_file = csvFiles[0] if csvFiles else None
    chosen = st.sidebar.selectbox("Choose the File: ", [file.split('/')[-1] for file in csvFiles], index=csvFiles.index(st.session_state.selected_csv_file) if csvFiles else 0)
    st.sidebar.write(f'Chosen: {chosen}')
    for file in csvFiles:
        if file.split('/')[-1] == chosen:
            st.session_state.selected_csv_file = file  # Store in session state
            return file

# Define the ChatWithCSVAgent class
class ChatWithCSVAgent:
    def __init__(self, csv_file):
        self.memory_x = ConversationBufferMemory()
        self.agent = create_csv_agent(
            llm,
            csv_file,
            verbose=False,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            allow_dangerous_code=True,
            memory=self.memory_x
        )

    def run(self, user_input):
        response = self.agent.invoke(user_input)
        self.memory_x.save_context({"input": user_input}, {"output": response['output']})
        return response
