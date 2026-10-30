import numpy as np


def cosine_similarity(v1, v2):
  dot_product = np.dot(v1, v2)
  norm_v1 = np.linalg.norm(v1)
  norm_v2 = np.linalg.norm(v2)

  if norm_v1 == 0 or norm_v2 == 0:
    return 0.0

  return dot_product / (norm_v1 * norm_v2)


def euclidean_distance(v1, v2):
  return np.linalg.norm(v1 - v2)


vec_a = np.array([1, 2, 3])
vec_b = np.array([4, 5, 6])
manual_cosine = 32 / (np.sqrt(14) * np.sqrt(77))
calculated_cosine = cosine_similarity(vec_a, vec_b)

print("Manual Cosine:", manual_cosine)
print("Function Cosine:", calculated_cosine)
print("-" * 40)

vec_a_scaled = vec_a * 10
print(
    "Cosine between A and B:",
    cosine_similarity(vec_a, vec_b),
)
print(
    "Cosine between scaled A and B:",
    cosine_similarity(vec_a_scaled, vec_b),
)
print(
    "Euclidean Distance between A and B:",
    euclidean_distance(vec_a, vec_b),
)
print(
    "Euclidean Distance between scaled A and B:",
    euclidean_distance(vec_a_scaled, vec_b),
)
print("-" * 40)

sentences = [
    "الطقس جميل وممشمس اليوم",
    "احب تناول القهوة في الصباح",
    "برمجة الذكاء الاصطناعي ممتعة جدا",
    "التعلم العميق يغير العالم",
    "كرة القدم هي الرياضة المفضلة لدي",
]

np.random.seed(42)
embeddings = np.random.rand(5, 3)

query = "كيف أتعلم الذكاء الاصطناعي؟"
query_embedding = np.array([0.5, 0.6, 0.7])

similarities = []
for i, emb in enumerate(embeddings):
  sim = cosine_similarity(query_embedding, emb)
  similarities.append((sim, sentences[i]))

similarities.sort(key=lambda x: x[0], reverse=True)

print("Query:", query)
print("Top 3 sentences:")
for rank, (sim, sent) in enumerate(similarities[:3], 1):
  print(f"{rank}. [Similarity: {sim:.4f}] {sent}")
