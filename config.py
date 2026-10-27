# 全局配置模块：集中管理环境变量、路径、阈值与分块参数
# 注意：HF_ENDPOINT 必须在导入 sentence_transformers 之前设置，config 需作为第一个被导入的模块
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from dotenv import load_dotenv
load_dotenv()

from pathlib import Path  #处理文件路径。

# ========== 对话历史保存的路径 ==========
#存聊天记录的文件路径，可以改
HISTORY_FILE = Path("./chat_history.json")
# 存向量数据库的文件夹路径。
CHROMA_PATH = "./chroma_db"
#重排分数阈值
#重排模型会给找到的文档打分。如果最高分低于这个及格线，系统就判定“没找到有用信息”，直接回复未找到，不让大模型瞎编。
# 【修改】将阈值从 -10.0 提高到 0.2，真正发挥防幻觉拦截作用（根据测试，相关块得分通常在0.5以上）
RERANK_THRESHOLD = 0.2
# 【新增】父子分块参数：父块给上下文，子块做检索
PARENT_CHUNK_SIZE = 1200
PARENT_CHUNK_OVERLAP = 100
CHILD_CHUNK_SIZE = 250
CHILD_CHUNK_OVERLAP = 50
# =================================================================
