from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import BaseOutputParser, StrOutputParser
from langchain_core.output_parsers.json import JsonOutputParser  # 注意正确导入路径
from dotenv import load_dotenv
import os

# 加载环境变量（API Key 等）
load_dotenv()

# ------------------- 通用调用函数 -------------------
def ask_and_parse(prompt: str, parser: BaseOutputParser):
    llm = ChatOpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
        model="deepseek-chat",
        temperature=0.3
    )
    # 安全获取格式指令：如果解析器没有实现该方法，则跳过
    try:
        instructions = parser.get_format_instructions()
    except NotImplementedError:
        instructions = ""

    if instructions:
        formatted_prompt = prompt + "\n" + instructions
    else:
        formatted_prompt = prompt

    response = llm.invoke(formatted_prompt)
    return parser.parse(response.content)
# ------------------- 自定义解析器：按 @ 分割 -------------------
class CustomToolParser(BaseOutputParser):
    """按 '工具名@核心功能@学习难度' 格式解析为字典"""

    def parse(self, text: str) -> dict:
        parts = text.strip().split("@")
        if len(parts) != 3:
            raise ValueError(f"格式错误：期望 3 个字段，实际收到 {len(parts)} 个。内容：{text}")
        return {
            "工具名": parts[0].strip(),
            "核心功能": parts[1].strip(),
            "学习难度": parts[2].strip()
        }

    def get_format_instructions(self) -> str:
        return "请严格按照格式输出：工具名@核心功能@学习难度。不要添加任何其他内容。"

    @property
    def _type(self) -> str:
        return "custom_tool_parser"


# ------------------- 测试调用 -------------------
if __name__ == "__main__":
    # 场景1：纯文本解析
    str_parser = StrOutputParser()
    result1 = ask_and_parse("请用一句话介绍 Python", str_parser)
    print("【纯文本结果】\n", result1, "\n")

    # 场景2：JSON 字典解析
    json_parser = JsonOutputParser()
    result2 = ask_and_parse(
        "返回一个包含 name 和 age 的 JSON 对象，例如：{'name': 'Alice', 'age': 20}",
        json_parser
    )
    print("【JSON 解析结果】\n", result2, "\n")

    # 场景3：自定义 @ 分割解析器
    custom_parser = CustomToolParser()
    result3 = ask_and_parse("推荐一个适合初学者的 AI 开发工具", custom_parser)
    print("【自定义 @ 解析结果】\n", result3, "\n")

    # 额外：直接测试解析器的静态文本（不调用模型）
    test_text = "LangChain@简化LLM应用开发@中等"
    parsed = custom_parser.parse(test_text)
    print("【直接解析测试】\n", parsed)