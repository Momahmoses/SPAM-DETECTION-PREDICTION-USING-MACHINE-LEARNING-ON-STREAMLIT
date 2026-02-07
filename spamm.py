# Import the Streamlit library for building web apps
import streamlit as st

# Import pandas for data loading and manipulation
import pandas as pd

# Import CountVectorizer to convert text into numerical features
from sklearn.feature_extraction.text import CountVectorizer

# Import the Naive Bayes classifier for text classification
from sklearn.naive_bayes import MultinomialNB


# Configure the Streamlit page (title shown in browser tab)
st.set_page_config(page_title='SPAM DETECTOR')

# Display the main title of the app
st.title('SPAM MESSAGE DETECTION APP')

# Display a short description under the title
st.write('Enter a message below to check whether it is **Spam** or **Normal**')


# Define a function to load and preprocess the dataset
def load_data():
    # Read the CSV file containing spam data
    # encoding='latin-1' avoids Unicode decoding errors
    data = pd.read_csv("spam.csv", encoding='latin-1')
    
    # Replace the label "ham" with "normal" for clarity
    data['class'] = data['class'].replace('ham', 'normal')
    
    # Return the cleaned dataset
    return data


# Load the dataset by calling the function
data = load_data()


# Define a function to train the machine learning model
def train_model(data):
    # Create a CountVectorizer object to convert text into word counts
    vectorizer = CountVectorizer()
    
    # Transform the message column into numerical feature vectors
    X = vectorizer.fit_transform(data['message'])
    
    # Store the target labels (spam or normal)
    y = data['class']
    
    # Create a Multinomial Naive Bayes model
    model = MultinomialNB()
    
    # Train the model using the vectorized text and labels
    model.fit(X, y)
    
    # Return the trained vectorizer and model
    return vectorizer, model


# Train the model using the dataset
vectorizer, model = train_model(data)

# Show a success message in the Streamlit app
st.success('MODEL TRAINED SUCCESSFULLY')


# Create a text area for user input
user_message = st.text_area(
    'Enter a message to check:',  # Label shown above the text area
    height=150,                   # Height of the input box
    placeholder='Type your message here...'  # Placeholder text
)


# Check if the user clicked the "check message" button
if st.button('check message'):
    
    # Check if the input message is empty or only spaces
    if user_message.strip() == "":
        # Show a warning if no message was entered
        st.warning('Please enter a message')
    
    else:
        # Convert the user message into numerical features
        # and make a prediction using the trained model
        prediction = model.predict(
            vectorizer.transform([user_message])
        )[0]
        
        # If the prediction is spam, show an error message
        if prediction == 'spam':
            st.error('This message is **SPAM**')
        
        # Otherwise, show a success message
        else:
            st.success('This message is **NORMAL**')
            
        # Display a label for the entered message
        st.write('**Message**')
        
        # Display the user's original message
        st.write(user_message)
