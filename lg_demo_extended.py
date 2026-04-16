# 1. 导入需要的模块
import os 
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from dotenv import load_dotenv

# 2. 加载 .env 环境变量
load_dotenv()

# 3. 读取 API 配置
API_KEY = os.getenv("API_KEY")
BASE_URL = os.getenv("BASE_URL")

if not API_KEY:
    raise ValueError("未检测到 API_KEY，请检查 .env 文件是否配置正确")

# 4. 初始化大模型（与之前相同）
llm = ChatOpenAI(
    api_key=API_KEY,
    base_url=BASE_URL,
    model="deepseek-chat",
    temperature=0.3
)

# 5. 扩展 State：增加一个字段存储英文翻译
class WorkflowState(TypedDict, total=False):
    user_role: str
    original_advice: str
    simplified_advice: str
    english_advice: str  # 新增字段

# 6. 节点函数（前两个保持不变）

def generate_advice(state: WorkflowState):
    prompt = f"给{state['user_role']}写一段50字左右的 AI 学习建议。"
    result = llm.invoke(prompt)
    return {"original_advice": result.content}

def simplify_advice(state: WorkflowState):
    prompt = f"把下面的学习建议精简到30字以内：{state['original_advice']}"
    result = llm.invoke(prompt)
    return {"simplified_advice": result.content}

# 新增节点：翻译成英文
def translate_to_english(state: WorkflowState):
    prompt = f"把下面的中文建议翻译成英文：{state['simplified_advice']}"
    result = llm.invoke(prompt)
    return {"english_advice": result.content}

# 7. 构建工作流（边的关系改为三步）
workflow = StateGraph(WorkflowState)

workflow.add_node("generate", generate_advice)
workflow.add_node("simplify", simplify_advice)
workflow.add_node("translate", translate_to_english)  # 新节点

# 定义流程：START → generate → simplify → translate → END
workflow.add_edge(START, "generate")
workflow.add_edge("generate", "simplify")
workflow.add_edge("simplify", "translate")
workflow.add_edge("translate", END)

app = workflow.compile()

# 8. 执行工作流
result = app.invoke({"user_role": "高校学生"})

# 9. 输出所有结果
print("原始学习建议：")
print(result["original_advice"])
print("\n精简后学习建议：")
print(result["simplified_advice"])
print("\n英文翻译：")
print(result["english_advice"])