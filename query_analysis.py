# 查询分析模块：多轮对话改写 / 多跳分解 / 查询变体生成
import json
from llm import get_llm_client

# ========== 多轮对话查询改写 ==========
def rewrite_query(query, history_messages):
    """结合历史对话，把当前问题改写成一句独立、完整的话。"""
    if not history_messages:
        return query
    recent_history = history_messages[-6:]
    history_text = "\n".join([
        f"{m['role']}: {m['content'][:200]}" for m in recent_history
    ])
    prompt = f"""请根据以下对话历史，将用户最新的问题改写成一句独立、完整、便于检索的话。
只返回改写后的问题，不要添加任何解释、引号或前缀。
对话历史：
{history_text}
最新问题：{query}
改写后的问题："""
    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
        )
        rewritten = response.choices[0].message.content.strip()
        print(f"🔄 查询改写：{query} -> {rewritten}")
        return rewritten if rewritten else query
    except Exception as e:
        print(f"⚠️ 改写失败，使用原问题：{e}")
        return query
# =====================================================

# ========== 【新增：多跳分解函数】 ==========
def decompose_multi_hop(query):
    """判断问题是否跨文件/多步，如果是，返回拆解后的列表，否则返回空列表"""
    prompt = f"""你是一个逻辑推理专家。请判断用户的问题是否需要经过两步或两步以上的逻辑推导才能回答（即多跳问题）。
如果是多跳问题（例如：“小王的上司是谁？”，需要先查“小王是哪个部门”，再查“该部门负责人是谁”），请将其拆解为 2 个依次执行的子查询。
如果不是多跳问题，请返回空列表。
严格输出 JSON 格式，不要其他文字：
{{"sub_queries": ["子查询1", "子查询2"]}}

问题：{query}"""
    try:
        client = get_llm_client()
        resp = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"}
        )
        data = json.loads(resp.choices[0].message.content)
        subs = data.get("sub_queries", [])
        # 【修复】子查询最多取 3 步：LLM 一次可能拆出 4 个以上子查询，不设上限会导致检索次数与耗时膨胀
        return subs[:3] if isinstance(subs, list) and len(subs) >= 2 else []
    except Exception as e:
        print(f"⚠️ 多跳分解失败：{e}")
        return []
# ============================================

# ========== 多查询检索——生成多个查询变体 ==========
def generate_query_variants(query, n=3):
    """让 LLM 生成多个查询变体，用于多路检索。"""
    prompt = f"""请为以下问题生成 {n} 个不同的查询变体，用于多路检索。
变体应该：
1. 保持原问题含义不变
2. 使用不同的表达方式或关键词
3. 每个变体独立成行，不要编号，不要解释
原问题：{query}
生成 {n} 个变体："""
    try:
        client = get_llm_client()
        response = client.chat.completions.create(
            model="deepseek-chat",
            temperature=0.7, #稍微提高一点随机性，让大模型生成不同的问法
            messages=[{"role": "user", "content": prompt}],
        )
        variants_text = response.choices[0].message.content.strip()
        variants = [v.strip() for v in variants_text.split("\n") if v.strip()]
        #variants_text = "什么是RAG？\n\n RAG的定义是什么？ \n\n"
        # variants_text.split("\n"):把大模型返回的多行文字，按换行符切开，变成列表。
        #得到["什么是RAG？", "", " RAG的定义是什么？ ", ""]
        #if v.strip()：判断处理之后有没有值，有保留（只是判断，不处理，保留是是没有剪掉的原值），没有去除
        #v.strip() for v ...：把保留下来的内容，去掉两头的空格，存进新列表。
        #最终结果：最终结果：["什么是RAG？", "RAG的定义是什么？"]
        # 把原问题也加进去，一起去检索
        all_queries = [query] + variants[:n]
        #把原问题和生成的 3 个变体拼在一起，返回一个包含 4 个问题的列表。
        print(f"🔍 多查询生成：{all_queries}")
        return all_queries
    except Exception as e:
        print(f"⚠️ 多查询生成失败，只用原问题：{e}")
        return [query]
# =====================================================
