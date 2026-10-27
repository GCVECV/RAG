import json
import streamlit as st
import chromadb
import jieba
import config  # 必须在导入 sentence_transformers 之前（config 内部设置 HF_ENDPOINT）
from config import HISTORY_FILE, CHROMA_PATH
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from ingest import run_ingest
from qa_flow import handle_query

st.set_page_config(page_title="企业智能服务台", page_icon="👔")
st.title("👔 企业智能服务台")

if 'messages' not in st.session_state:
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.messages = json.load(f)
        except Exception:
            st.session_state.messages = []
    else:
        st.session_state.messages = []
if 'collection' not in st.session_state:
    st.session_state.collection = None

if 'parents_collection' not in st.session_state:
    st.session_state.parents_collection = None

if 'all_docs' not in st.session_state:
    st.session_state.all_docs = []

if 'model' not in st.session_state:
    with st.spinner("正在加载模型..."):
        st.session_state.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None


if st.session_state.collection is None:
    try:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        if any(c.name == "collection" for c in _client.list_collections()):
            _col = _client.get_collection(name="collection")
            if _col.count() > 0:
                st.session_state.collection = _col
                if any(c.name == "collection_parents" for c in _client.list_collections()):
                    st.session_state.parents_collection = _client.get_collection(name="collection_parents")
                _stored = _col.get(include=["documents", "metadatas"])
                st.session_state.all_docs = [
                    Document(page_content=d, metadata=m)
                    for d, m in zip(_stored["documents"], _stored["metadatas"])
                ]
                _bm25 = BM25Retriever.from_documents(
                    st.session_state.all_docs,
                    preprocess_func=lambda text: list(jieba.cut(text)),
                )
                _bm25.k = 5
                st.session_state.bm25_retriever = _bm25
                st.sidebar.info(f"已加载本地知识库（{_col.count()} 块文档），可直接提问。")
    except Exception as e:
        print(f"⚠️ 加载持久化知识库失败：{e}")


with st.sidebar:
    st.header("📄 文档管理")
    uploaded_files = st.file_uploader(
        "请上传文件", type=['pdf', 'docx', 'txt'], accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("开始解析并存入数据库"):
            run_ingest(uploaded_files)

    if st.button("🧹 清空知识库"):
        try:
            _c = chromadb.PersistentClient(path=CHROMA_PATH)
            _c.delete_collection(name="collection")
        except Exception:
            pass

        try:
            _c = chromadb.PersistentClient(path=CHROMA_PATH)
            _c.delete_collection(name="collection_parents")
        except Exception:
            pass
        st.session_state.collection = None
        st.session_state.parents_collection = None
        st.session_state.all_docs = []
        st.session_state.bm25_retriever = None
        st.success("知识库已清空，请重新上传文件")
        st.rerun()

    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink()
        st.success("已清空对话历史")
        st.rerun()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

st.markdown("💡 **常见问题快速提问：**")
col1, col2, col3 = st.columns(3)
quick_query = None
with col1:
    if st.button("✈️ 出差报销标准是多少？"):
        quick_query = "出差报销标准是多少？"
with col2:
    if st.button("💻 怎么连公司 VPN？"):
        quick_query = "怎么连公司 VPN？"
with col3:
    if st.button("🏖️ 年假怎么计算？"):
        quick_query = "年假怎么计算？"


query_input = st.chat_input("请输入问题：")
query = quick_query or query_input

if query:
    handle_query(query)
