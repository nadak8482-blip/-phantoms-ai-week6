import os
import json
from google import genai

# إعداد الـ API Key
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
MODEL_ID = "gemini-2.5-flash"

reviews = [
    "اشتريت سماعة Anker من أحمد في المعادي، الصوت ممتاز والبطارية بتعدي اليوم كله. التوصيل جه في معاده. أنصح بيها.",
    "موبايل Samsung A54 كاميرته حلوة جداً بس البطارية بتخلص قبل نص اليوم والشاحن بطيء. مش عارف أنصح بيه ولا لأ.",
    "التوصيل اتأخر أسبوع كامل ومحدش رد على الرسايل. الخدمة وحشة جداً.",
    "برافو عليهم، طلبت الشنطة تيجي في يومين فجت بعد تلات أسابيع، وكمان الزرار اتكسر من أول استخدام. خدمة تحفة فعلاً 👏",
    "لابتوب Lenovo IdeaPad اللي استلمته من فرع سيتي ستارز شغال تمام والسعر مناسب، بس مفيش كرتونة أصلية والتغليف كان ضعيف."
]

# --- 1. اختبار البرومبت القديم (الغير منظم) ---
def run_old_prompts():
    success_count = 0
    total_tests = len(reviews) * 3
    
    print("جاري اختبار البرومبت القديم...")
    for review in reviews:
        for _ in range(3):
            prompt = f"قولي رأيك في الريفيو ده وهل هو كويس ولا وحش، ومتطولش، بس اشرح ليه، وطلعلي أي أسماء أو منتجات مذكورة. الريفيو: {review}"
            try:
                response = client.models.generate_content(model=MODEL_ID, contents=prompt)
                cleaned_text = response.text.strip().removeprefix("```json").removesuffix("```").strip()
                json.loads(cleaned_text)
                success_count += 1
            except Exception:
                pass # فشل في القراءة كـ JSON
    return success_count

# --- 2. اختبار البرومبت الجديد (المصلح والمهيكل) ---
def run_new_prompts():
    success_count = 0
    
    system_instruction = """You are an expert sentiment analysis and named entity extraction engine. You must strictly output valid JSON only, with no markdown formatting or extra text. If information is missing, use null."""
    
    print("جاري اختبار البرومبت الجديد والمصلح...")
    for review in reviews:
        for _ in range(3):
            prompt = f"""
[Context]
Review text: "{review}"

[Instruction]
1. Classify sentiment strictly as: "Positive", "Negative", or "Neutral".
2. Provide a very concise explanation (max 1 sentence).
3. Extract names, products, locations (use null if not found).

[Output Format]
Return ONLY a valid JSON object with these keys:
{{"sentiment": "...", "explanation": "...", "extracted_entities": {{"person": "...", "product": "...", "location": "..."}}}}
"""
            try:
                response = client.models.generate_content(
                    model=MODEL_ID, 
                    contents=prompt,
                    config={"system_instruction": system_instruction}
                )
                text = response.text.strip()
                if text.startswith("```json"):
                    text = text[7:]
                if text.endswith("```"):
                    text = text[:-3]
                
                json.loads(text.strip())
                success_count += 1
            except Exception:
                pass
    return success_count

if __name__ == "__main__":
    print("=" * 50)
    print("بدء تجربة قياس الثبات والـ Debugging (15 محاولة لكل حالة)")
    print("=" * 50)
    
    old_success = run_old_prompts()
    new_success = run_new_prompts()
    
    print("\n" + "=" * 50)
    print("نتائج المقارنة النهائية (من إجمالي 15 محاولة)")
    print("=" * 50)
    print(f"عدد الردود المقروءة صحيحة (قبل الإصلاح): {old_success} / 15")
    print(f"عدد الردود المقروءة صحيحة (بعد الإصلاح): {new_success} / 15")
    print("=" * 50)
