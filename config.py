# 全局配置模块：集中管理环境变量、路径、阈值与分块参数
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path  


HISTORY_FILE = Path("./chat_history.json")
CHROMA_PATH = "./chroma_db"

RERANK_THRESHOLD = 0.2
VECTOR_DISTANCE_THRESHOLD = 0.6  # cosine 距离，越小越相关

PARENT_CHUNK_SIZE = 1200
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 250
CHILD_CHUNK_OVERLAP = 50

