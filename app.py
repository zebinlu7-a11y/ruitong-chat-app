from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st
import os
import requests
import zipfile
from io import BytesIO
import json

# ------------------- Streamlit 配置 -------------------
st.set_page_config(
    page_title="锐瞳智能科技公司 - 小雨智能体",
    page_icon="💡",
    layout="centered"
)

# ------------------- DeepSeek API 配置 -------------------
DEEPSEEK_API_KEY = "sk-8213b5bbd5054511aa940116e7e421dc"
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# ------------------- GitHub 仓库下载配置 -------------------
GITHUB_REPO_URL = "https://github.com/zebinlu7-a11y/ruitong-chat-app"
LOCAL_REPO_DIR = "ruitong-chat-app"  # 最终本地目录

MODEL_DIR = os.path.join(LOCAL_REPO_DIR, "models", "all-MiniLM-L6-v2")
CHROMA_DIR = os.path.join(LOCAL_REPO_DIR, "models", "ruitongkeji")

# ------------------- 下载 GitHub 仓库 -------------------
def download_github_repo(repo_url, local_dir):
    if os.path.exists(local_dir):
        return  # 已存在，不下载
    st.info("知识库或模型文件不存在，正在自动下载，请稍等...")
    try:
        zip_url = repo_url.rstrip("/") + "/archive/refs/heads/main.zip"
        r = requests.get(zip_url, timeout=120)
        r.raise_for_status()
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall(".")
        # 解压后默认是 ruitong-chat-app-main，需要改名
        extracted_dir = repo_url.split("/")[-1] + "-main"
        if os.path.exists(extracted_dir):
            os.rename(extracted_dir, local_dir)
        st.success(f"仓库 {repo_url} 下载完成！")
    except Exception as e:
        st.error(f"下载 GitHub 仓库失败: {str(e)}")

# ------------------- 加载向量库 -------------------
@st.cache_resource
def load_vectorstore():
    download_github_repo(GITHUB_REPO_URL, LOCAL_REPO_DIR)
    if not os.path.exists(MODEL_DIR) or not os.path.exists(CHROMA_DIR):
        st.error("模型或知识库文件不存在，请检查 GitHub 仓库是否包含对应文件。")
        return None
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_DIR)
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"知识库加载失败: {str(e)}")
        return None

vectorstore = load_vectorstore()
if vectorstore:
    st.success("知识库加载完成！")
else:
    st.warning("知识库加载失败，请检查路径或数据。")

# ------------------- 系统提示 -------------------
system_prompt = (
    "你是一个锐瞳智能科技公司的智能助手，名字叫小雨,跟用户对话请以第一人称的方式与用户沟通。"
    "工作流程如下："
    " 你可以实现对用户的问题回答，以第一人称的方式与用户沟通"
    " 此外，用户可以输入查询或指令："
    "   - 查询知识库：我会提供相关上下文，你基于上下文回答用户查询，生成自然、简洁的回答。"
    "   - 如果指令不被识别，返回 '抱歉，我不认识这个命令'。"
    " 基于提供的上下文回答问题，不要直接显示知识库内容，要结合原始知识库内容，回答用户有关你所在的锐瞳智能科技公司的信息，只用回答相关信息，不需要其他的语气词。"
)

# ------------------- 会话状态初始化 -------------------
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "system", "content": system_prompt}]
    st.session_state.chat_history = [{"role": "assistant", "content": "你好，我是小雨助手，有什么需要帮助的吗？"}]

# ------------------- 调用 DeepSeek API -------------------
def call_deepseek_api(user_input, context):
    try:
        full_prompt = (
            f"{system_prompt}\n\n用户查询：{user_input}\n"
            f"知识库上下文：{context}\n"
            "根据上下文回答用户的问题，生成自然、简洁的回答。"
        )
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
            timeout=60
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        st.error(f"API 调用失败: {str(e)}")
        return "API 调用失败，请稍后重试。"

# ------------------- 聊天界面 -------------------
st.title("💡 锐瞳智能科技公司 - 小雨智能体")
st.write("你好，我是小雨助手，有什么需要帮助的吗？")

# 显示聊天记录
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# 用户输入
user_input = st.chat_input("请输入您的消息...", key="chat_input")

if user_input:
    # 显示用户消息
    with st.chat_message("user"):
        st.write(user_input)
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    # 检索知识库
    if vectorstore:
        results = vectorstore.similarity_search(user_input, k=3)
        context = "\n".join([doc.page_content for doc in results]) if results else "无相关知识库内容"
    else:
        context = "知识库不可用"

    # 调用 API 并显示回复
    with st.chat_message("assistant"):
        with st.spinner("小雨正在思考..."):
            reply = call_deepseek_api(user_input, context)
            st.write(reply)
    st.session_state.chat_history.append({"role": "assistant", "content": reply})

# ------------------- 操作指南（可选） -------------------
if st.checkbox("操作指南"):
    st.write("会话状态:", "查找锐瞳科技相关信息，请咨询我")
