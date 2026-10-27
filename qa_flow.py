# 提问处理主流程模块：改写 -> 多跳分解/变体生成 -> 混合检索 -> RRF 融合 -> 子块重排 -> 父块回溯 -> 拒答判断 -> 流式生成
import streamlit as st
from sentence_transformers import CrossEncoder
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from config import RERANK_THRESHOLD
from llm import get_llm_client
from query_analysis import rewrite_query, decompose_multi_hop, generate_query_variants
from document_loader import save_history


def handle_query(query):
    """处理用户一次提问的完整流程（含 UI 渲染、拒答分支与历史保存）。"""
    if st.session_state.collection is None:
        st.warning("请先上传并解析文件")
        answer = None          # 【修复 1】确保 answer 有定义
    else:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        # 多轮对话查询改写
        rewritten_query = rewrite_query(query, st.session_state.messages[:-1])
        
        with st.chat_message("assistant"):
            with st.spinner("正在检索中..."):
                vector_docs = []
                bm25_docs = []
                bm25_retriever = st.session_state.get('bm25_retriever', None)

                # ========== 【新增：多跳分解与迭代检索判断】 ==========
                sub_queries = decompose_multi_hop(rewritten_query)
                
                if sub_queries:
                    st.info(f"🧠 检测到跨文件/多跳问题，Agent 已自动拆解步骤：{' -> '.join(sub_queries)}")
                    current_context = ""
                    for idx, sub_q in enumerate(sub_queries):
                        # 如果是第二步及以后，把上一步的检索结果作为线索拼进去
                        search_q = sub_q
                        if idx > 0 and current_context:
                            search_q = f"{sub_q} (已知前置信息: {current_context[:150]})"
                            
                        print(f"🔍 多跳迭代检索 第 {idx+1} 步：{search_q}")
                        q_emb = st.session_state.model.encode([search_q]).tolist()
                        v_res = st.session_state.collection.query(
                            query_embeddings=q_emb,
                            n_results=5,
                        )
                        for i, doc in enumerate(v_res['documents'][0]):
                            vector_docs.append(Document(
                                page_content=doc,
                                metadata=v_res['metadatas'][0][i],
                            ))
                        if bm25_retriever is not None:
                            bm25_docs.extend(bm25_retriever.invoke(search_q))
                            
                        # 将当前步骤的检索结果提取为下一步的线索
                        if v_res['documents'][0]:
                            current_context = " ".join([d[:150] for d in v_res['documents'][0][:2]])
                else:
                    # 如果不是多跳，走原有的单步多查询逻辑
                    query_variants = generate_query_variants(rewritten_query, n=3)
                    #生成包含原问题在内的 4 个变体列表
                    
                    # 每个变体都做向量检索 + BM25 检索，结果合并
                    for q in query_variants:
                        q_emb = st.session_state.model.encode([q]).tolist()
                        v_res = st.session_state.collection.query(
                            query_embeddings=q_emb,
                            n_results=5,
                        )
                        for i, doc in enumerate(v_res['documents'][0]):
                            vector_docs.append(Document(
                                page_content=doc,
                                metadata=v_res['metadatas'][0][i],
                            ))
                        if bm25_retriever is not None:
                            bm25_docs.extend(bm25_retriever.invoke(q))
                # =======================================================

                # RRF 融合
                scores = {}
                rrf_meta = {}

                def rrf_fusion(docs_a, docs_b, k=60):
                    for i, doc in enumerate(docs_a):
                        # 【修复】RRF 键从"全文"改为元组 (来源, 页码, 父块ID, 内容前80字)，
                        # 避免不同块因内容相同/重叠发生碰撞、metadata 互相覆盖
                        key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0),
                               doc.metadata.get("parent_id", ""), doc.page_content[:80])
                        scores[key] = scores.get(key, 0) + 1 / (k + i + 1)
                        rrf_meta[key] = doc
                    for i, doc in enumerate(docs_b):
                        # 【修复】RRF 键从"全文"改为元组 (来源, 页码, 父块ID, 内容前80字)，
                        # 避免不同块因内容相同/重叠发生碰撞、metadata 互相覆盖
                        key = (doc.metadata.get("source", ""), doc.metadata.get("page", 0),
                               doc.metadata.get("parent_id", ""), doc.page_content[:80])
                        scores[key] = scores.get(key, 0) + 1 / (k + i + 1)
                        rrf_meta[key] = doc
                    sorted_keys = sorted(scores.keys(), key=lambda k: scores[k], reverse=True)[:10]
                    # 【修复】key 改为元组后直接返回原 doc，避免 page_content 被截断为 80 字或把元组当正文
                    return [rrf_meta[k] for k in sorted_keys]

                rrf_docs = rrf_fusion(vector_docs, bm25_docs)

                # ========== 【核心优化：先对子块做 Rerank，速度极快，精度极高】 ==========
                top_score = 0.0
                top_children = []
                try:
                    if rrf_docs:
                        if "reranker" not in st.session_state:
                            with st.spinner("正在加载Rerank模型，请稍等..."):
                                st.session_state.reranker = CrossEncoder('BAAI/bge-reranker-base')
                        reranker = st.session_state.reranker
                        pairs = [[rewritten_query, doc.page_content] for doc in rrf_docs] # 对 250 字的子块打分
                        
                        # 【修复 OOM】分批计算 Rerank，防止瞬间内存撑爆
                        reranker_scores = []
                        batch_size = 4
                        for i in range(0, len(pairs), batch_size):
                            batch = pairs[i:i+batch_size]
                            batch_scores = reranker.predict(batch)
                            reranker_scores.extend(batch_scores)
                        
                        print(f"📊 子块 Rerank 分数范围：{min(reranker_scores):.4f} ~ {max(reranker_scores):.4f}，top1={max(reranker_scores):.4f}")
                        
                        ranked = sorted(zip(reranker_scores, rrf_docs), key=lambda x: x[0], reverse=True)
                        top_score = float(ranked[0][0])
                        top_children = [doc for _, doc in ranked[:3]] # 选出 Top 3 子块
                except Exception as e:
                    print(f"⚠️ Rerank 失败，降级使用 RRF 结果：{e}")
                    # 【修复】降级时置 -1.0 而非 1.0：1.0 永远大于拒答阈值 0.2，
                    # 会让无关文档绕过阈值被送进 LLM；置 -1.0 必走拒答分支，宁可不答也不错答
                    top_score = -1.0
                    top_children = rrf_docs[:3]

                # ========== 【用 Top 3 子块去回溯父块】 ==========
                final_docs = []
                if top_children and st.session_state.parents_collection is not None:
                    # 去重获取 parent_id
                    parent_ids = list(dict.fromkeys([
                        d.metadata.get("parent_id") for d in top_children
                        if d.metadata and d.metadata.get("parent_id")
                    ]))
                    if parent_ids:
                        try:
                            p_res = st.session_state.parents_collection.get(ids=parent_ids)
                            # 【修复】按 id2doc 映射取回父块，严格保持 parent_ids（即子块 Rerank 分数降序）的顺序；
                            # 原先直接 zip 依赖 chromadb get() 的返回顺序，乱序时最相关父块可能排不到 context 最前
                            id2doc = {
                                pid: (p_text, p_meta)
                                for pid, p_text, p_meta in zip(
                                    p_res["ids"], p_res["documents"], p_res["metadatas"]
                                )
                            }
                            final_docs = [
                                Document(
                                    page_content=id2doc[pid][0],
                                    metadata={**id2doc[pid][1], "parent_id": pid},
                                )
                                for pid in parent_ids if pid in id2doc
                            ]
                            print(f"🧩 父子分块：Top 3 子块 -> 回溯出 {len(final_docs)} 个父块")
                        except Exception as e:
                            print(f"⚠️ 回溯父块失败，直接使用子块：{e}")
                            final_docs = top_children
                else:
                    final_docs = top_children # 兜底

                reranker_docs = [doc.page_content for doc in final_docs]
                reranker_metas = [doc.metadata for doc in final_docs]

            # ========== 【修改：退出 spinner 作用域，解决 DOM removeChild 报错】 ==========
            # 注意：从这里开始，代码已经退出了 with st.spinner 块
            # ========== 【修复 5】用 Rerank 分数判断是否真的找到资料 ==========
            if not reranker_docs or top_score < RERANK_THRESHOLD:
                # ========== 【修改：换业务场景 4：空检索降级提示与转人工按钮】 ==========
                answer = "抱歉，知识库暂未收录该问题的解决方案。"
                st.markdown(answer)
                st.warning("您可以尝试换个问法，或者直接联系人工客服。")
                if st.button("📩 一键提交人工工单"):
                    st.success("已为您生成工单，请留意企业微信通知。")
                # ==============================================================================
            else:
                # 拼接上下文
                context = ""
                for i, doc in enumerate(reranker_docs):
                    page_info = f"[来源: {reranker_metas[i]['source']} 第{reranker_metas[i]['page']}页]"
                    context += f"{doc}\n{page_info}\n\n"

                # ========== 【修改：换业务场景 5：修改 IT/HR 专家人设与 Prompt 模板】 ==========
                prompt_template = PromptTemplate(
                    template="""你是一个专业、耐心、严谨的企业 IT 与 HR 专家。
请基于以下内部资料回答员工的问题。
资料：
{context}
问题：{question}
要求：
1. 语气要职业、温和。
2. 涉及操作步骤时，请使用 1. 2. 3. 分点列出。
3. 如果资料中找不到答案，请回答“未找到相关信息”，并引导员工转人工。
4. 请在回答末尾标注引用的来源，格式如 [来源: 文件名 第x页]。
5. 如果资料中有对比性、多项数据或列表信息，请用 Markdown 表格呈现。
6. 请完整、逐点覆盖与问题相关的全部要点，宁可多列也不要遗漏；资料中没有的内容不要编造。
回答：""",
                    input_variables=["context", "question"],
                )
                # ==============================================================================

                final_prompt = prompt_template.format(context=context, question=rewritten_query)

                # ========== 【修改：加入流式输出】主回答加异常保护 ==========
                try:
                    client = get_llm_client()
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        temperature=0.1,
                        messages=[{"role": "user", "content": final_prompt}],
                        stream=True,  # 开启流式
                    )
                    # 使用 st.write_stream 展示打字机效果，并接收完整字符串
                    answer = st.write_stream(
                        chunk.choices[0].delta.content
                        for chunk in response
                        if chunk.choices[0].delta.content is not None
                    )
                except Exception as e:
                    answer = f"⚠️ 生成回答失败，请稍后重试（{e}）"
                    st.markdown(answer)
                # ================================================
                
                # 注意：原本的 st.markdown(answer) 必须注释掉，因为 st.write_stream 已经渲染了！

    # ========== 【修复 1】只在有答案时追加并保存 ==========
    if answer is not None:
        st.session_state.messages.append({"role": "assistant", "content": answer})
        save_history()
    # =====================================================
