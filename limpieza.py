import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import requests

nltk.download("stopwords")
nltk.download("wordnet")

# Descargar libro
url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
texto = requests.get(url).text.lower()

# Tokenización simple
tokens = texto.split()
tokens = [t.strip('.,!?()[]:;"\'') for t in tokens if t.isalpha()]

# Stopwords y lematización
stop_words = set(stopwords.words("english"))
tokens = [t for t in tokens if t not in stop_words]

lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(t) for t in tokens]

# Guardar resultado
with open("libro_limpio.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(tokens))

print("Limpieza completada. Archivo 'libro_limpio.txt' generado.")