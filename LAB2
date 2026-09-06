from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer

# Input text corpus
corpus = [
    "Machine learning enables computers to learn from data",
    "Artificial intelligence allows machines to perform intelligent tasks",
    "Deep learning uses neural networks to learn complex patterns",
    "Supervised learning trains models using labeled data",
    "Natural language processing helps computers understand human language"
]

# ------------------------------------------------
# Bag of Words Representation
# ------------------------------------------------

# Create CountVectorizer object
vectorizer = CountVectorizer()

# Convert the corpus into Bag of Words matrix
matrix_bagofwords = vectorizer.fit_transform(corpus)

print("Bag of Words\n")

# Display the vocabulary
print("Vocabulary:")
print(vectorizer.get_feature_names_out())

# Display the Bag of Words matrix
print("\nMatrix Bag of Words:")
print(matrix_bagofwords.toarray())


# ------------------------------------------------
# Dense Vector Embeddings
# ------------------------------------------------

# Load the pre-trained Sentence Transformer model
model_dense = SentenceTransformer("all-MiniLM-L6-v2")

# Convert each sentence into a dense vector
matrix_dense = model_dense.encode(corpus)

print("\nDense Vector Embeddings")

# Display the shape of the dense embedding matrix
print("\nDense Embedding Shape:")
print(matrix_dense.shape)

# Display the embedding of the first sentence
print("\nDense First Sentence Embeddings:")
print(matrix_dense[0])


# ------------------------------------------------
# Comparison
# ------------------------------------------------

print("\nComparison")

# Display dimensions of Bag of Words
print("Bag of Words dimensions:", matrix_bagofwords.shape)

# Display dimensions of Dense Embeddings
print("Dense Vector Embeddings dimensions:", matrix_dense.shape)


# ------------------------------------------------
# First Sentence Comparison
# ------------------------------------------------

print("\nBag of Words Representation of 1st sentence:")
print(matrix_bagofwords.toarray()[0])

print("\nDense Embeddings of 1st sentence:")
print(matrix_dense[0])
