import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

# Descargar recursos necesarios
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")

# Cargar texto
with open("libro.txt", "r", encoding="utf-8") as f:
    texto = f.read()

# Normalización: pasar a minúsculas
texto = texto.lower()

# Tokenización
tokens = nltk.word_tokenize(texto)

# Eliminación de puntuación y stopwords
stop_words = set(stopwords.words("english"))  # Cambia a 'spanish' si tu libro es en español
tokens = [t for t in tokens if t.isalpha() and t not in stop_words]

# Lematización
lemmatizer = WordNetLemmatizer()
tokens_lemmatizados = [lemmatizer.lemmatize(t) for t in tokens]

# Guardar resultado limpio
with open("libro_limpio.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(tokens_lemmatizados))

print("Limpieza completada. Archivo 'libro_limpio.txt' generado.")