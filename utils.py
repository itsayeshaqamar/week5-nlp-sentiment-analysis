import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("stopwords")
nltk.download("wordnet")

# Initialize objects
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))
def clean_text(text):
    """
    Clean and preprocess input review text.
    """

    # Handle empty input
    if text is None:
        return ""

    # Convert to lowercase
    text = text.lower()

    # Replace smart quotes with standard ones
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')

    # Remove apostrophes
    text = text.replace("'", "")

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", "", text)

    # Remove punctuation and numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenize
    words = text.split()

    # Remove stopwords and lemmatize
    cleaned_words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word not in stop_words
    ]

    # Return cleaned text
    return " ".join(cleaned_words)
if __name__ == "__main__":
    sample = "I absolutely LOVE my Alexa!!! It works amazingly well."
    print(clean_text(sample))