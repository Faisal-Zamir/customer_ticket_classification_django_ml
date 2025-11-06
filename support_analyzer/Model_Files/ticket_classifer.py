from django.conf import settings
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from scipy.special import softmax
import os

# Base directory of current file
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Correct nltk_data path (inside Model_Files)
nltk_data_path = os.path.join(BASE_DIR, "nltk_data")
nltk.data.path.append(nltk_data_path)

# Download stopwords only if missing
try:
    stop_words = set(stopwords.words('english'))
except LookupError:
    nltk.download('stopwords', download_dir=nltk_data_path)
    stop_words = set(stopwords.words('english'))

# Load model and TF-IDF vectorizer
model = joblib.load(os.path.join(BASE_DIR, "customer_ticket_classification_model.pkl"))
tfidf = joblib.load(os.path.join(BASE_DIR, "tfidf_vectorizer.pkl"))
le = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# Initialize preprocessing components
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    """
    Preprocess ticket text in the same way as training:
    - Lowercase
    - Remove punctuation / special characters
    - Remove stopwords
    - Lemmatization
    """
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
    return ' '.join(words)

def predict_ticket_category(ticket_data):
    """
    Predict the category for a new ticket
    
    Parameters:
        ticket_data (dict): {'Document': 'ticket text here'}
    
    Returns:
        dict: prediction results
    """
    ticket_text = ticket_data.get('Document', '')
    
    # Preprocess
    cleaned_text = preprocess_text(ticket_text)
    
    # Transform using TF-IDF
    text_tfidf = tfidf.transform([cleaned_text])
    
    # Predict category code
    prediction_encoded = model.predict(text_tfidf)[0]
    
    # Decode label
    prediction_label = le.inverse_transform([prediction_encoded])[0]
    
    # Compute confidence using softmax of decision function
    try:
        decision_scores = model.decision_function(text_tfidf)[0]
        confidence_scores = softmax(decision_scores)
        confidence = max(confidence_scores) * 100  # Convert to friendly %
    except:
        confidence = None
    
    return {
        'input_text': ticket_text,
        'predicted_category': prediction_label,
        'confidence': float(confidence) if confidence else "N/A",
        'category_code': int(prediction_encoded)
    }

# ------------------------------
# Test the function with a sample ticket
# ------------------------------
if __name__ == "__main__":
    test_ticket = {
        'Document': (
            "I need to install specialized statistical analysis software for my research work, "
            "but the installation keeps failing due to permission restrictions. "
            "The software requires access to system directories that my current account can't modify. "
            "Can someone help me get this working?"
        )
    }
    
    result = predict_ticket_category(test_ticket)
    print("🔎 Ticket Classification Result")
    print("="*50)
    print(f"Input Ticket: {result['input_text']}")
    print(f"Predicted Category: {result['predicted_category']}")
    print(f"Category Code: {result['category_code']}")
    print(f"Confidence: {result['confidence']:.2f}%")
