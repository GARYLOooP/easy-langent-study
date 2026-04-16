import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

print("1. 开始加载 .env")
load_dotenv()

API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")
print(f"2. API_KEY 存在: {bool(API_KEY)}, BASE_URL: {BASE_URL}")

if not API_KEY:
    raise ValueError("未找到 API_KEY")

print("3. 初始化 ChatOpenAI")
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

prompt = "请写一段50字左右的 AI 学习建议，语言简洁、实用，适合初学者。"
print("4. 开始调用模型...")
response = llm.invoke(prompt)
print("5. 调用完成，输出结果：")
print(response.content)