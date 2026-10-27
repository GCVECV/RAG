# 公共 LLM 客户端模块
import os
from openai import OpenAI

# ========== 公共 LLM 客户端 ==========
#作用：把创建大模型客户端的代码封装起来，后面不管哪里要用，直接调用 get_llm_client() 就行，避免写重复代码。
def get_llm_client():
    """统一创建 DeepSeek 客户端，避免重复代码"""
    return OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )
# ========================================
