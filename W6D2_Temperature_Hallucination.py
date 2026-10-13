import os
from google import genai
from google.genai import types

if "GEMINI_API_KEY" not in os.environ:
  raise ValueError("برجاء ضبط متغير البيئة بالـ API Key الخاص بك أولاً.")

client = genai.Client()

text_to_summarize = (
    "الذكاء الاصطناعي هو فرع من علوم الحاسوب يهدف إلى إنشاء أنظمة قادرة على محاكاة "
    "القدرات البشرية مثل التعلم، الاستنتاج، وحل المشكلات. في السنوات الأخيرة، شهد المجال "
    "طفرة هائلة بفضل النماذج اللغوية الكبيرة (LLMs) التي تعتمد على شبكات عصبية عميقة "
    "تسمى محولات (Transformers). هذه النماذج قادرة على توليد نصوص وفهم السياق بشكل مذهل، "
    "مما فتح باباً واسعاً لتطبيقات في مختلف القطاعات مثل التعليم، الصحة، وبرمجة الحاسوب. "
    "رغم هذه التطورات، تظل هناك تحديات كبرى تواجه المطورين، أبرزها ظاهرة الهلوسة الناتجة "
    "عن التنبؤ الإحصائي، بالإضافة إلى استهلاك الطاقة الهائل والحاجة إلى ضبط دقيق للأمان."
)

print("=" * 60)
print("1. تجربة الـ Temperature (0.1 مقابل 0.9)")
print("=" * 60)

print("\n--- تكرار 3 مرات بـ Temperature = 0.1 (إجابات ثابتة وأكثر موثوقية) ---")
for i in range(1, 4):
    response_low = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=f"اكتب تلخيصاً موجزاً للنص التالي: {text_to_summarize}",
        config=types.GenerateContentConfig(temperature=0.1),
    )
    print(f"التجربة {i}: {response_low.text.strip()}")

print("\n
