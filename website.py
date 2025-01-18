import os
import pandas as pd
import streamlit as st
from detection import predict_image, MODEL_PATH, CATEGORIES
from dotenv import load_dotenv
import google.generativeai as gen_ai

# Load environment variables
load_dotenv()

# Configure Streamlit page settings
st.set_page_config(
    page_title="Pill.AI",
    page_icon=":pill:",  # Favicon emoji
    layout="centered",  # Page layout option
)

# Load Google API key from .env file
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Set up Google Gemini-Pro AI model
gen_ai.configure(api_key=GOOGLE_API_KEY)
model = gen_ai.GenerativeModel('gemini-pro')

# Function to translate roles between Gemini-Pro and Streamlit terminology
def translate_role_for_streamlit(user_role):
    if user_role == "model":
        return "assistant"
    else:
        return user_role

# Initialize chat session in Streamlit if not already present
if "chat_session" not in st.session_state:
    st.session_state.chat_session = model.start_chat(history=[])

# Title of the app
st.title("Pill.AI")

# Initialize session state for detected pill and messages
if "detected_pill" not in st.session_state:
    st.session_state["detected_pill"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Upload image for pill detection
uploaded_file = st.file_uploader("Upload a pill image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded file temporarily
    temp_path = "temp_image.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict using the model
    label, confidence = predict_image(MODEL_PATH, temp_path)

    if confidence > 0.5:
        st.session_state["detected_pill"] = label
        st.success(f"Detected: **{label}** (Confidence: {confidence:.2f}). You can now ask me about this pill.")
    else:
        st.session_state["detected_pill"] = None
        st.error("Confidence is too low to identify the pill accurately.")

    # Clean up
    os.remove(temp_path)

# Display alert below the chat (if any)
if st.session_state["detected_pill"] is None and len(st.session_state['messages']) == 0:
    st.warning("Please upload an image of a pill to get started.")

# Load the pill dataset
df = pd.read_csv('data.csv')
df.dropna(subset=["effects"], axis=0, inplace=True)

# Display previous messages in chat history
for message in st.session_state['messages']:
    with st.chat_message(translate_role_for_streamlit(message['role'])):
        st.markdown(message['content'])

# Input and response processing
if st.session_state["detected_pill"] != None:
    if prompt := st.chat_input("Enter your query (e.g., effects, usage, indications):"):
        st.session_state['messages'].append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message('user'):
            st.markdown(prompt)

        # Handle query processing
        if st.session_state["detected_pill"] is None:
            response = "No pill has been detected yet. Please upload an image of a pill first."
        else:
            detected_pill = st.session_state["detected_pill"]
            header = ['effects', 'usage', 'indications', 'ingredients', 'type', 'administration', 'precautions']
            found = False

            # Check if the query contains 'price'
            if 'price' in prompt.lower():
                response = "Sorry, please visit your nearest pharmacy for the price."
            else:
                # Find the appropriate service based on the query
                for field in header:
                    if field in prompt.lower():
                        found = True
                        service = field
                        break
                if found:
                    try:
                        # Find the detected pill in the dataset
                        class_index = df[df["Medicine"].str.contains(detected_pill, case=False)].index[0]
                        output = df.loc[class_index, service]
                        response = f"### {service.capitalize()} for the pill **{detected_pill}**\n\n**Details:** {output}"
                        gemini_response = st.session_state.chat_session.send_message(
                            f"Can you provide additional details or recommendations for {detected_pill} regarding its {service}?")

                        response += f"\n\n{gemini_response.text}"
                    except Exception as e:
                        response = f"An error occurred while fetching the data: {e}"
                else:
                    # If no keyword matched, send the query to Gemini for an AI-generated response
                    try:
                        # Generate a response using Gemini's AI model
                        gemini_response = st.session_state.chat_session.send_message(
                            f"Please provide relevant information about {detected_pill}, such as its effects, usage, or precautions."
                        )

                        response = gemini_response.text
                    except Exception as e:
                        response = f"An error occurred while processing the pill data: {e}"

            # Append the assistant's response to the chat
            st.session_state['messages'].append({"role": "assistant", "content": response})

            # Display assistant message
            with st.chat_message('assistant'):
                st.markdown(response)
