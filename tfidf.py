from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import string
from tqdm import tqdm

class TFIDF():

    def __init__(self):

        self.vectorizer = TfidfVectorizer(max_df=0.95, min_df=2)
   
        self.stopwords = list(set(list(stopwords.words('english')) + list(ENGLISH_STOP_WORDS)))
        self.lemmatizer = WordNetLemmatizer()

    def _additonal_tfidf_preprocessing(self,doc):

        # Lowercase
        doc = doc.lower()
        
        # Remove punctuation
        doc = doc.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(doc)
        
        # Remove stop words
        tokens = [word for word in tokens if word not in self.stopwords]

        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        # Remove short words
        tokens = [word for word in tokens if len(word) > 1]
        
        return ' '.join(tokens)

    def encode(self, texts):

        print('Preprocessing')
        texts = [self._additonal_tfidf_preprocessing(text) for text in tqdm(texts)]
        print('Training')
        embeddings = self.vectorizer.fit_transform(texts)

        return embeddings