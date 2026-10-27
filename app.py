# 应用入口：页面配置 / 会话状态初始化 / 持久化知识库加载 / 侧边栏 / 快捷提问 / 提问入口
import json  #读写 JSON 文件。
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

# ========== 【修改：换业务场景 1：修改页面标题和图标】 ==========
st.set_page_config(page_title="企业智能服务台", page_icon="👔")
st.title("👔 企业智能服务台")
# ==============================================================

# ========== 会话状态初始化 ==========
#先看网页内存有没有记录。没有的话，去硬盘找。
# 找到就读出来，找不到就两手空空。这就叫持久化（关了网页记录还在）
if 'messages' not in st.session_state:
    # 启动时加载历史对话
    if HISTORY_FILE.exists():
        #.exists()：检查硬盘上有没有 chat_history.json 这个文件。
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                st.session_state.messages = json.load(f)
                #json.load(f)：把文件里的文字（JSON格式）转换成 Python 认识的列表。
        except Exception:
            st.session_state.messages = []
    else:
        st.session_state.messages = []
#如果里面没有聊天记录，就去检查本地磁盘有没有 chat_history.json。有就读进来，没有就建个空列表。
if 'collection' not in st.session_state:
    st.session_state.collection = None

# 【新增】父块 collection 单独存
if 'parents_collection' not in st.session_state:
    st.session_state.parents_collection = None

if 'all_docs' not in st.session_state:
    st.session_state.all_docs = []

if 'model' not in st.session_state:
    with st.spinner("正在加载模型..."):
        st.session_state.model = SentenceTransformer('BAAI/bge-small-zh-v1.5')

if "bm25_retriever" not in st.session_state:
    st.session_state.bm25_retriever = None

# ========== 【修复 2】启动时加载磁盘上已有的持久化知识库 ==========
if st.session_state.collection is None:  #网页内存里没有数据库集合
    try:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
        #尝试去本地磁盘（CHROMA_PATH 路径）找
        #chromadb.PersistentClient：打开本地硬盘上的数据库。
        #下划线 _：程序员的一个习惯，意思是“这是个临时用一下的内部变量”
        if any(c.name == "collection" for c in _client.list_collections()):
            #any(...)：只要里面有一个表叫 "collection"，就返回 True
            #.list_collections()：客户端的方法，意思是“列出数据库里所有的表（集合）”。
            _col = _client.get_collection(name="collection")
            #把硬盘里的那个表，重新接回网页内存。
            if _col.count() > 0:
                #看看这个表里有没有存数据（大于0条）
                st.session_state.collection = _col
                # 【新增】同时加载父块 collection
                if any(c.name == "collection_parents" for c in _client.list_collections()):
                    st.session_state.parents_collection = _client.get_collection(name="collection_parents")
                # 从库中取回全部文档，恢复 BM25 检索器
                _stored = _col.get(include=["documents", "metadatas"]) #include:包含，不能改，意思是取什么值
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
# ================================================================

with st.sidebar:
    st.header("📄 文档管理")
    uploaded_files = st.file_uploader(
        "请上传文件", type=['pdf', 'docx', 'txt'], accept_multiple_files=True
    )
    if uploaded_files:
        if st.button("开始解析并存入数据库"):
            run_ingest(uploaded_files)

    # ========== 新增：清空知识库按钮 ==========
    if st.button("🧹 清空知识库"):
        try:
            _c = chromadb.PersistentClient(path=CHROMA_PATH)
            _c.delete_collection(name="collection")
        except Exception:
            pass
        # 【新增】同时删除父块 collection
        try:
            _c = chromadb.PersistentClient(path=CHROMA_PATH)
            _c.delete_collection(name="collection_parents")
        except Exception:
            pass
        st.session_state.collection = None
        st.session_state.parents_collection = None   # 【新增】
        st.session_state.all_docs = []
        st.session_state.bm25_retriever = None
        st.success("知识库已清空，请重新上传文件")
        st.rerun()  #刷新网页，让页面立刻变成初始状态
    # ============================================

    # 清空对话历史按钮
    if st.button("🗑️ 清空对话历史"):
        st.session_state.messages = []
        if HISTORY_FILE.exists():
            HISTORY_FILE.unlink() #把硬盘上的 chat_history.json 文件彻底删除。
        st.success("已清空对话历史")
        st.rerun()

# 渲染历史消息
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ========== 【修改：换业务场景 2：增加快捷提问按钮】 ==========
# 利用 columns 布局，在输入框上方增加几个常见问题的快捷按钮
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
# ==============================================================

# ========== 提问处理 ==========
# 【修改：换业务场景 3：将快捷按钮的点击与输入框合并判断】
query_input = st.chat_input("请输入问题：")
query = quick_query or query_input

if query:
    handle_query(query)
