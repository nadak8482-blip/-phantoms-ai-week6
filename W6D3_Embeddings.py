import os
import numpy as np
from google import genai

if "GEMINI_API_KEY" not in os.environ:
  raise ValueError("برجاء ضبط متغير البيئة بالـ API Key الخاص بك أولاً.")

client = genai.Client()

sentences = [
    "The team won the football match after a great goal.",
    "Our squad secured the soccer victory following a brilliant strike.",
    "Python is an amazing programming language for data science.",
    "I love writing code in Python for software development.",
    "I went to the river bank to sit on the grass and relax.",
    "I need to deposit my money at the local bank tomorrow.",
    "Pizza and pasta are my favorite Italian meals.",
    "Delicious burgers and fries make a great lunch.",
    "The weather is very cold and rainy today.",
    "A Python snake slithered silently through the tall grass.",
]

result = client.models.embed_content(
    model="text-embedding-004",
    contents=sentences,
)

embeddings = [np.array(e.values) for e in result.embeddings]

print("=" * 60)
print(f"عدد الجمل الكلي: {len(embeddings)}")
print(f"عدد الأبعاد (Dimensions) للـ Embedding الواحد: {len(embeddings[0])}")
print(f"أول 5 أرقام من أول جملة: {embeddings[0][:5]}")
print("=" * 60)


def cosine_similarity(a, b):
  return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


pairs_to_compare = [
    (0, 1, "كرة القدم (كلمات مختلفة، نفس المعنى)"),
    (2, 3, "برمجة بايثون (تشابه في المعنى والكلمات)"),
    (4, 5, "كلمة bank (نهر vs مال)"),
    (6, 7, "أكل وطعام"),
    (8, 9, "حالة الطقس vs ثعبان بايثون"),
]

print("\n--- نتائج مقار
