import pandas as pd
import faiss
import numpy as np

pd.set_option("display.max_colwidth", 100)

df = pd.read_csv("data/sample_text.csv")
df.shape

df

from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-mpnet-base-v2")
vectors = encoder.encode(df.text)

dim = vectors.shape[1]

index = faiss.IndexFlatL2(dim)

index.add(vectors)

search_query = "How good is squats?"

vec = encoder.encode(search_query)
vec.shape

svec = np.array(vec).reshape(1, -1)
svec.shape

distances, I = index.search(svec, k=2)
I

df.loc[I[0]]
