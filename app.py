# app.py
import streamlit as st
import json
import os
import re
import requests
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------- Streamlit 配置 -------------------
st.set_page_config(
    page_title="锐瞳智能科技有限公司——小锐智能体",
    page_icon="Robot",
    layout="wide"
)

# ------------------- DeepSeek API 配置 -------------------
DEEPSEEK_API_KEY = st.secrets.get("DEEPSEEK_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
if not DEEPSEEK_API_KEY:
    st.error("请配置 DEEPSEEK_API_KEY（在 Secrets 或环境变量中）")
    st.stop()

DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# ------------------- 路径配置 -------------------
CONVERSATIONS_DIR = "./conversations"
CHROMA_DIR = "./models/ruitongkeji"

os.makedirs(CONVERSATIONS_DIR, exist_ok=True)


# ------------------- 用户名验证 -------------------
def is_valid_username(username):
    return bool(re.match(r'^[a-zA-Z0-9_]+$', username))


# ------------------- 会话保存/加载 -------------------
def save_conversations(username):
    file_path = os.path.join(CONVERSATIONS_DIR, f"conv_{username}.json")
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(st.session_state.conversations, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"保存失败: {e}")

def load_conversations(username):
    file_path = os.path.join(CONVERSATIONS_DIR, f"conv_{username}.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}
    return {}


# ------------------- 删除用户 -------------------
def delete_user(username):
    file_path = os.path.join(CONVERSATIONS_DIR, f"conv_{username}.json")
    if os.path.exists(file_path):
        os.remove(file_path)
    st.session_state.username = None
    st.session_state.conversations = None
    st.rerun()


# ------------------- 加载知识库（永久缓存） -------------------
@st.cache_resource(show_spinner="正在唤醒小锐智能体...")
def load_vectorstore():
    if not os.path.exists(CHROMA_DIR):
        st.error("错误：未找到 `models/ruitongkeji` 文件夹！")
        st.info("请将知识库文件夹上传到项目根目录")
        st.stop()
    
    try:
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        st.success("锐瞳知识库已就绪！")
        return vectorstore
    except Exception as e:
        st.error(f"知识库加载失败: {e}")
        return None

vectorstore = load_vectorstore()


# ------------------- 获取知识库内容 -------------------
def get_knowledge_context():
    if not vectorstore:
        return "知识库不可用。"
    try:
        docs = vectorstore.get()
        if docs and "documents" in docs:
            text = " ".join(docs["documents"])
            return text[:4000]  # 限制长度
    except:
        pass
    return "知识库内容不可用"


# ------------------- 系统提示 -------------------
full_context = get_knowledge_context() if vectorstore else ""
system_prompt = f"""
你是一个锐瞳智能科技公司的智能助手，名字叫小锐，用第一人称与用户沟通。
你既能回答公司相关问题，也能回答任何学科问题。
当用户问及公司内容时，参考以下信息自然回答（不要引用原文）：
[内部信息：{full_context}]
其他问题直接用你的知识回答。
例如：用户问“你是谁”，回答“我是小锐，来自锐瞳智能科技。”
保持语气友好、专业、流畅。
""".strip()


# ------------------- 用户登录界面 -------------------
if "username" not in st.session_state:
    st.session_state.username = None

if not st.session_state.username:
    st.title("欢迎使用 小锐智能体")
    existing = [f.replace("conv_", "").replace(".json", "") 
                for f in os.listdir(CONVERSATIONS_DIR) if f.startswith("conv_")]
    
    col1, col2 = st.columns(2)
    with col1:
        if existing:
            user = st.selectbox("选择已有用户", existing)
            if st.button("登录"):
                st.session_state.username = user
                st.rerun()
    with col2:
        new = st.text_input("新用户名（字母数字下划线）")
        if st.button("注册"):
            if new and is_valid_username(new):
                st.session_state.username = new
                st.rerun()
            else:
                st.error("用户名无效")
else:
    # ------------------- 初始化会话 -------------------
    if "conversations" not in st.session_state:
        data = load_conversations(st.session_state.username)
        if not data:
            default_id = "default"
            data = {
                default_id: {
                    "title": "新对话",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "assistant", "content": "你好！我是小锐，有什么可以帮助你？"}
                    ]
                }
            }
        st.session_state.conversations = data
        st.session_state.current_session = list(data.keys())[0]
        save_conversations(st.session_state.username)

    # ------------------- DeepSeek API 调用 -------------------
    def call_deepseek(messages):
        try:
            resp = requests.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {DEEPSEEK_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": DEEPSEEK_MODEL,
                    "messages": messages,
                    "temperature": 0.7,
                    "max_tokens": 800
                },
                timeout=30
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            return f"API 错误: {str(e)}"

    # ------------------- 侧边栏 -------------------
    with st.sidebar:
        st.header(f"用户: {st.session_state.username}")

        if st.button("新建对话"):
            nid = f"chat_{len(st.session_state.conversations)}"
            st.session_state.conversations[nid] = {
                "title": f"对话 {len(st.session_state.conversations)+1}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": "你好！我是小锐，有什么可以帮助你？"}
                ]
            }
            st.session_state.current_session = nid
            save_conversations(st.session_state.username)
            st.rerun()

        sess = st.radio(
            "对话列表",
            options=list(st.session_state.conversations.keys()),
            format_func=lambda x: st.session_state.conversations[x]["title"],
            index=list(st.session_state.conversations.keys()).index(st.session_state.current_session)
        )
        if sess != st.session_state.current_session:
            st.session_state.current_session = sess
            st.rerun()

        # 重命名
        cur = st.session_state.conversations[st.session_state.current_session]
        new_title = st.text_input("标题", value=cur["title"], key="title")
        if new_title != cur["title"]:
            cur["title"] = new_title
            save_conversations(st.session_state.username)

        if st.button("清除所有对话"):
            st.session_state.conversations = {
                "default": {
                    "title": "新对话",
                    "messages": [{"role": "system", "content": system_prompt},
                                {"role": "assistant", "content": "你好！我是小锐，有什么可以帮助你？"}]
                }
            }
            st.session_state.current_session = "default"
            save_conversations(st.session_state.username)
            st.rerun()

        if st.button("删除当前用户"):
            if st.button("确认删除", type="primary"):
                delete_user(st.session_state.username)

        if st.button("切换用户"):
            st.session_state.username = None
            st.rerun()

    # ------------------- 主聊天界面 -------------------
    st.title("小锐智能体")
    msgs = st.session_state.conversations[st.session_state.current_session]["messages"]

    for m in msgs:
        if m["role"] != "system":
            with st.chat_message(m["role"]):
                st.write(m["content"])

    if prompt := st.chat_input("输入你的问题..."):
        with st.chat_message("user"):
            st.write(prompt)
        msgs.append({"role": "user", "content": prompt})

        with st.chat_message("assistant"):
            with st.spinner("小锐思考中..."):
                reply = call_deepseek(msgs)
                st.write(reply)
            msgs.append({"role": "assistant", "content": reply})
        
        save_conversations(st.session_state.username)
