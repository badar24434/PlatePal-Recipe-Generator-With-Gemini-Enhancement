import os
import streamlit as st
from dotenv import load_dotenv
from gemini_api import GeminiAPI
from ml_model import MLModel
import google.generativeai as gen_ai

# Print TensorFlow version
import tensorflow as tf

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Recipe Generator ChatBot",
    page_icon=":alien:",  # Favicon emoji
    layout="centered",  # Page layout option
)

# Sidebar to input Google API Key
st.sidebar.title("PlatePal ChatBot Configuration")
GOOGLE_API_KEY = st.sidebar.text_input("Enter your Password", type="password")

# Check if API key is provided
if not GOOGLE_API_KEY:
    st.error("Please enter your password.")
    st.stop()

# Set up Google Gemini-Pro AI model
try:
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    model = gen_ai.GenerativeModel('gemini-1.0-pro')
    st.write("Connected to Gemini-Pro successfully.")
except Exception as e:
    st.error(f"Failed to connect to Gemini-Pro: {e}")
    st.stop()

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Initialize ML model and Gemini API
ml_model = MLModel('model/recipe_generation_rnn.h5')
gemini_api = GeminiAPI(model)

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = {"history": []}

# Display the chatbot's title on the page
st.title("🤖 Recipe Generator ChatBot - PlatePal")

# Display the chat history
for message in st.session_state.chat_session["history"]:
    with st.chat_message(translate_role_for_streamlit(message.role)):
        st.markdown(message.parts[0].text)

# Input field for user's message
user_prompt = st.chat_input("Ask ✨PlatePal✨ for a recipe! 🍽️")
if user_prompt:
    # Add user's message to chat and display it
    st.chat_message("user").markdown(user_prompt)

    # Get initial recipe recommendation from Gemini API
    gemini_response = gemini_api.get_recommendation(user_prompt)
    initial_recipe = gemini_response.text  # Assuming response text contains the recipe

    # ML model refinement process
    refined_recipe = ml_model.process(initial_recipe)

    # Display the final output as if it's refined by ML and presented by Gemini
    with st.chat_message("assistant"):
        st.markdown(f"""
                <div style='font-size:20px;'>
                    {refined_recipe}
                </div>
                """, unsafe_allow_html=True)
        st.markdown(refined_recipe)
