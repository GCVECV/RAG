# 公共 LLM 客户端模块
import os
from openai import OpenAI

def get_llm_client():
    """统一创建 DeepSeek 客户端，避免重复代码"""
    return OpenAI(
        api_key=os.getenv("API_KEY"),
        base_url=os.getenv("BASE_URL"),
    )
# ========================================
