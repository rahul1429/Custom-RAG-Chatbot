import streamlit as st
import sys
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Import necessary modules
from langchain_openai import OpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent

from langchain.agents.agent_types import AgentType
#Add OPEN_API_KEY below
llm = OpenAI(temperature=0, openai_api_key='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')
# Define the ChatWithCSVAgent class
class ChatWithCSVAgent:
    def __init__(self):
        self.agent = create_csv_agent(
            llm,
            #Provide path to your data below
            "updated_modifData.csv",
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        )

    def run(self, user_input):
        response = self.agent.invoke(user_input)
        return response

# Main function to run the Streamlit app
def main():
    st.sidebar.title("Helper-Bot")  # Move title and refresh button to sidebar

    if st.sidebar.button("Refresh"):
        st.session_state["chat_history"] = []
        st.write("App refreshed!")

    file = 'updated_modifData.csv'
    #file = st.file_uploader("Upload your csv file", type=["csv"])
    if file is not None:
        conversational_agent = ChatWithCSVAgent()

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

        # Retrieve chat history from session state
        chat_history = st.session_state.get("chat_history", [])

        query = st.chat_input("Ask me anything!")

        if query:
                response = conversational_agent.run(query)['output']
                # Update chat history and store in session state
                chat_history.append({"user": query, "AI": response})
                st.session_state["chat_history"] = chat_history

        # Display chat history
        for interaction in chat_history:
            st.markdown(f'<div class="user-message">{interaction["user"]}</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="bot-message">{interaction["AI"]}</div>', unsafe_allow_html=True)


# Run the Streamlit app
if __name__ == "__main__":
    main()
