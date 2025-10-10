import os
import shutil
import zipfile
from io import BytesIO

import requests
import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# ------------------- Streamlit 配置 -------------------
st.set_page_config(
    page_title="锐瞳智能科技公司 - 小雨智能体",
    page_icon="💡",
    layout="centered"
)

# ------------------- DeepSeek API 配置 -------------------
DEEPSEEK_API_KEY = "sk-8213b5bbd5054511aa940116e7e421dc"  # 替换为你的 DeepSeek Key
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# ------------------- 路径配置 -------------------
CHROMA_DIR = "./ruitongkeji"
GITHUB_REPO = "https://github.com/zebinlu7-a11y/ruitong-chat-app"

# ------------------- 下载 GitHub 仓库 -------------------
def download_github_repo(repo_url, extract_to="."):
    """下载 GitHub 仓库 zip 并解压"""
    try:
        zip_url = repo_url.rstrip("/") + "/archive/refs/heads/main.zip"
        r = requests.get(zip_url, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall(extract_to)
        st.success(f"仓库 {repo_url} 下载完成！")
    except Exception as e:
        st.error(f"下载 GitHub 仓库失败: {str(e)}")

# ------------------- 准备 Chroma 文件夹 -------------------
def prepare_chroma_dir(raw_dir, target_dir=CHROMA_DIR):
    os.makedirs(target_dir, exist_ok=True)
    for root, _, files in os.walk(raw_dir):
        for f in files:
            if f.endswith((".bin", ".sqlite3")):
                shutil.copy(os.path.join(root, f), os.path.join(target_dir, f))
    return target_dir

# ------------------- 加载知识库 -------------------
@st.cache_resource
def load_vectorstore():
    # 如果 Chroma 文件夹不存在，自动下载
    if not os.path.exists(CHROMA_DIR):
        st.info("知识库不存在，正在自动下载...")
        download_github_repo(GITHUB_REPO)
        raw_chroma_dir = "./ruitong-chat-app-main/ruitongkeji"
        prepare_chroma_dir(raw_chroma_dir, CHROMA_DIR)

    try:
        # HuggingFace 自动下载 MiniLM 模型
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"知识库加载失败: {str(e)}")
        return None

# ------------------- 初始化向量库 -------------------
vectorstore = load_vectorstore()
if vectorstore:
    st.success("知识库加载完成！")
else:
    st.error("知识库加载失败，请检查路径或数据。")

# ------------------- 系统提示 -------------------
system_prompt = (
    "你是锐瞳智能科技公司的智能助手，名字叫小雨，跟用户对话请以第一人称方式沟通。"
    "工作流程如下："
    " - 查询知识库：我会提供相关上下文，你基于上下文回答用户查询，生成自然、简洁的回答。"
    " - 如果指令不被识别，返回 '抱歉，我不认识这个命令'。"
    " 请结合知识库内容回答用户问题，只回答相关信息。"
)

# ------------------- 初始化会话状态 -------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
    st.session_state.chat_history = [{"role": "assistant", "content": "你好，我是小雨助手，有什么需要帮助的吗？"}]

# ------------------- DeepSeek API 调用 -------------------
def call_deepseek_api(user_input, context):
    try:
        full_prompt = f"{system_prompt}\n\n用户查询：{user_input}\n知识库上下文：{context}"
        response = requests.post(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": [{"role": "user", "content": full_prompt}],
                "temperature": 0.7,
                "max_tokens": 500
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"API 调用失败: {str(e)}")
        return "API 调用失败，请稍后重试。"

# ------------------- 聊天界面 -------------------
st.title("💡 锐瞳智能科技公司 - 小雨智能体")
st.write("你好，我是小雨助手，有什么需要帮助的吗？")

# 显示历史消息
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
user_input = st.chat_input("请输入您的消息...", key="chat_input")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 检索知识库
    if vectorstore:
        results = vectorstore.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in results]) if results else "无相关知识库内容"
    else:
        context = "知识库不可用"

    # 调用 DeepSeek 并显示回复
    with st.chat_message("assistant"):
        with st.spinner("小雨正在思考..."):
            reply = call_deepseek_api(user_input, context)
            st.write(reply)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# 操作指南
if st.checkbox("操作指南"):
    st.write("会话状态: 查找锐瞳科技相关信息，请咨询我")
