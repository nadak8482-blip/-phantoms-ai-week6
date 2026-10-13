import os
from google import genai
from google.genai import types

if "GEMINI_API_KEY" not in os.environ:
  raise ValueError("برجاء ضبط متغير البيئة بالـ API Key الخاص بك أولاً.")

client = genai.Client()

user_question = "اشرح لي مفهوم الذكاء الاصطناعي ببساطة شديدة في سطرين."

print("=" * 50)
print(f"السؤال الأساسي: {user_question}\n")

print("--- 1. رد بـ System Prompt: (مدرس صبور) ---")
response_patient = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_question,
    config=types.GenerateContentConfig(
        system_instruction="أنت معلم صبور وودود، تشرح بحماس وبساطة تامة للمبتدئين.",
        temperature=0.7,
    ),
)
print(response_patient.text)
print(f"عدد الـ Tokens المستخدمة:")
print(f"   - Input Tokens: {response_patient.usage_metadata.prompt_token_count}")
print(f"   - Output Tokens: {response_patient.usage_metadata.candidates_token_count}")
print(f"   - Total Tokens: {response_patient.usage_metadata.total_token_count}\n")

print("-" * 50)

print("--- 2. رد بـ System Prompt: (خبير مختصر جداً) ---")
response_concise = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=user_question,
    config=types.GenerateContentConfig(
        system_instruction="أنت خبير تقني دقيق، وتعطي الإجابات باختصار شديد ومباشر دون مقدمات.",
        temperature=0.2,
    ),
)
print(response_concise.text)
print(f"عدد الـ Tokens المستخدمة:")
print(f"   - Input Tokens: {response_concise.usage_metadata.prompt_token_count}")
print(f"   - Output Tokens: {response_concise.usage_metadata.candidates_token_count}")
print(f"   - Total Tokens: {response_concise.usage_metadata.total_token_count}")
print("=" * 50)
