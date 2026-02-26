from src.logging_config import setup_logging
setup_logging()
from src.agent import run_agent

# أمر (Prompt) صريح يمنع الموديل من استخدام الويب ويحدد كلمات البحث
prompt = """
CRITICAL INSTRUCTION: 
1. DO NOT use web_search or wikipedia. 
2. Use ONLY the 'rag_retrieval' tool.
3. Search specifically for these keywords: 'Quiz', 'Exercises', 'Review Questions', 'Test'.

GOAL:
Find the questions located at the very end of the book. 
- List the questions.
- Provide the answer for each one based on the book text.
"""

print("\n🚀 Running the Agent in 'Strict Book Mode'...\n")

# تشغيل العميل مع إجبار الموديل على الالتزام بالكتاب فقط
answer = run_agent(prompt)

print("\n" + "="*50)
print("✨ FINAL ANSWERS FROM THE BOOK ✨")
print("="*50)
print(answer)