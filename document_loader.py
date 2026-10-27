# 文档解析与对话历史保存模块
import io
import json
import pdfplumber
import docx2txt
import streamlit as st
from config import HISTORY_FILE

# ========== 保存对话历史到本地 ==========
def save_history():
    """把当前对话历史保存到本地 JSON 文件。"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(st.session_state.messages, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"⚠️ 保存历史失败：{e}")
        
# "w"：代表 Write（写）。打开文件准备清空重写。
# json.dump(数据, 文件, ...)：把内存里的聊天记录列表，转成文字存进硬盘。
# ensure_ascii=False：保住中文。如果设成 True，中文“你好”会变成 \u4f60\u597d 这种乱码。
# indent=2：存进去的文字，每行缩进 2 个空格，看起来排版好看。


# =====================================================

def parse_uploaded_file(uploaded_file):
    page_texts = []
    try:
        if uploaded_file.name.endswith('.pdf'):
            with pdfplumber.open(uploaded_file) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    if text:
                        page_texts.append({"page": page_num + 1, "text": text})
        elif uploaded_file.name.endswith(".docx"):
            text = docx2txt.process(io.BytesIO(uploaded_file.read()))
            if text:
                page_texts.append({"page": 1, "text": text})
        elif uploaded_file.name.endswith(".txt"):
            text = uploaded_file.read().decode("utf-8")
            if text:
                page_texts.append({"page": 1, "text": text})
    except Exception as e:
        st.warning(f"{uploaded_file.name}解析失败")
    return page_texts
