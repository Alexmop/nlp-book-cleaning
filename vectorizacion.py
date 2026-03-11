import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from gensim.models import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

nltk.download("stopwords")
nltk.download("wordnet")

# -----------------------
# 1 Cargar libro
# -----------------------

url = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
texto = requests.get(url).text.lower()

# -----------------------
# 2 Limpieza
# -----------------------

tokens = texto.split()
tokens = [t.strip('.,!?()[]:;"\'') for t in tokens if t.isalpha()]

stop_words = set(stopwords.words("english"))
tokens = [t for t in tokens if t not in stop_words]

lemmatizer = WordNetLemmatizer()
tokens = [lemmatizer.lemmatize(t) for t in tokens]

# guardar texto limpio
with open("texto_procesado.txt","w",encoding="utf-8") as f:
    f.write(" ".join(tokens))

# -----------------------
# 3 Preparar corpus
# -----------------------

sentences = [tokens[i:i+20] for i in range(0,len(tokens),20)]

# -----------------------
# 4 Word2Vec
# -----------------------

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=2,
    workers=2
)

words = list(model.wv.index_to_key)
vectors = model.wv[words]

# -----------------------
# 5 PCA para visualización
# -----------------------

pca = PCA(n_components=2)
result = pca.fit_transform(vectors)

# -----------------------
# 6 Gráfica 1
# -----------------------

plt.figure(figsize=(10,8))
plt.scatter(result[:,0], result[:,1])

for i,word in enumerate(words[:50]):
    plt.annotate(word,(result[i,0],result[i,1]))

plt.title("Espacio vectorial Word2Vec (PCA)")
plt.savefig("grafica_vectores_1.png")
plt.close()

# -----------------------
# 7 Gráfica 2 (subset)
# -----------------------

subset = vectors[:100]

pca2 = PCA(n_components=2)
result2 = pca2.fit_transform(subset)

plt.figure(figsize=(10,8))
plt.scatter(result2[:,0], result2[:,1])

for i,word in enumerate(words[:30]):
    plt.annotate(word,(result2[i,0],result2[i,1]))

plt.title("Espacio semántico del libro")
plt.savefig("grafica_vectores_2.png")
plt.close()

print("Proceso terminado")