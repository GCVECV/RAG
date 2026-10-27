# 入库模块：文件解析 -> 父子分块 -> 向量化 -> 增量入库 -> 重建 BM25 检索器
import hashlib   
import streamlit as st
import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
import jieba
from config import CHROMA_PATH, PARENT_CHUNK_SIZE, PARENT_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHILD_CHUNK_OVERLAP
from document_loader import parse_uploaded_file


def run_ingest(uploaded_files):
    """执行上传文件的父子分块、向量化与增量入库（对应侧边栏“开始解析并存入数据库”按钮）。"""
    with st.spinner("正在解析文件并向量化（父子分块），请稍后..."):
        all_chunks = []
        all_metadatas = []
        all_chunk_ids = []         
        all_parents = []            
        all_parent_metadatas = []   
        all_parent_ids = []         
        all_chunk = 0


        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP
        )
        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP
        )

        for uploaded_file in uploaded_files:
            page_texts = parse_uploaded_file(uploaded_file)
            for pt in page_texts:
             
                for p_idx, p_text in enumerate(parent_splitter.split_text(pt['text'])):
                    pid = hashlib.md5(
                        f"P:{uploaded_file.name}:{pt['page']}:{p_idx}:{p_text[:50]}".encode("utf-8")
                    ).hexdigest()
                    all_parents.append(p_text)
                    all_parent_metadatas.append({"source": uploaded_file.name, "page": pt["page"]})
                    all_parent_ids.append(pid)


                    for c_idx, c_text in enumerate(child_splitter.split_text(p_text)):
                        cid = hashlib.md5(
                            f"C:{pid}:{c_idx}:{c_text[:50]}".encode("utf-8")
                        ).hexdigest()
                        all_chunks.append(c_text)
                        all_metadatas.append({
                            "source": uploaded_file.name,
                            "page": pt["page"],
                            "parent_id": pid,
                            "is_parent": False,
                        })
                        all_chunk_ids.append(cid)
                        all_chunk += 1
            st.success(f"{uploaded_file.name}解析完成，共{len(page_texts)}页")
        st.success(f"所有可解析文件解析完成，一共{all_chunk}块（父子分块）")
    if not all_chunks:
        st.error("解析失败")
        st.stop()


    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_or_create_collection(
        name="collection",
        metadata={"hnsw:space": "cosine"},
    )
    embeddings = st.session_state.model.encode(all_chunks,batch_size=32).tolist()

    collection.upsert(
        documents=all_chunks,
        metadatas=all_metadatas,
        embeddings=embeddings,
        ids=all_chunk_ids,
    )

    parent_collection = client.get_or_create_collection(name="collection_parents")
    _dim = st.session_state.model.get_sentence_embedding_dimension()
    parent_embeddings = [[0.0] * _dim for _ in all_parents]
    parent_collection.upsert(
        documents=all_parents,
        metadatas=all_parent_metadatas,
        embeddings=parent_embeddings,
        ids=all_parent_ids,
    )

    stored = collection.get(include=["documents", "metadatas"])
    st.session_state.all_docs = [
        Document(page_content=d, metadata=m)
        for d, m in zip(stored["documents"], stored["metadatas"])
    ]
    bm25_retriever = BM25Retriever.from_documents(
        st.session_state.all_docs,
        preprocess_func=lambda text: list(jieba.cut(text)),
    )
    bm25_retriever.k = 5
    st.session_state.bm25_retriever = bm25_retriever
    st.session_state.collection = collection
    st.session_state.parents_collection = parent_collection   # 【新增】
    st.success(f"入库完成：子块 {collection.count()} 个，父块 {parent_collection.count()} 个")
    # =================================================
