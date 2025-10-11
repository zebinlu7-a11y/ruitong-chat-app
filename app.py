import streamlit as st
import json
import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import zipfile
from io import BytesIO
import shutil
from streamlit_cookies_manager import EncryptedCookieManager

# ------------------- Streamlit 配置 -------------------
st.set_page_config(
    page_title="锐瞳智能科技有限公司———小锐智能体",
    page_icon="🤖",
    layout="wide"
)

# ------------------- DeepSeek API 配置 -------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "sk-8213b5bbd5054511aa940116e7e421dc")  # 建议用环境变量
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# ------------------- 路径配置 -------------------
CHROMA_DIR = "./models/ruitongkeji"

# ------------------- Cookies 配置 -------------------
cookies = EncryptedCookieManager(
    prefix="ruitong_chat/",  # Cookie 前缀
    password="your-secret-password-123456"  # 用于加密 Cookies，替换为强密码
)
if not cookies.ready():
    st.spinner("初始化 Cookies...")
    st.stop()  # 等待 Cookies 初始化

# ------------------- 保存/加载对话历史到 Cookies -------------------
def save_conversations():
    """保存会话到 Cookies"""
    try:
        # 限制会话数据大小（避免 Cookies 4KB 限制）
        conversations = st.session_state.conversations
        for session_id, conv in conversations.items():
            # 每会话最多保留 10 条消息（可调整）
            conv["messages"] = conv["messages"][-10:] if len(conv["messages"]) > 10 else conv["messages"]
        cookies["conversations"] = json.dumps(conversations, ensure_ascii=False)
        cookies.save()
    except Exception as e:
        st.error(f"保存对话到 Cookies 失败: {str(e)}")

def load_conversations():
    """从 Cookies 加载会话"""
    try:
        if "conversations" in cookies:
            return json.loads(cookies["conversations"])
    except Exception as e:
        st.error(f"加载 Cookies 失败: {str(e)}")
    return {}

# ------------------- 自动下载 GitHub 仓库 -------------------
def download_github_repo(repo_url, extract_to="."):
    try:
        zip_url = repo_url.rstrip("/") + "/archive/refs/heads/main.zip"
        r = requests.get(zip_url, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall(extract_to)
        st.success(f"仓库 {repo_url} 下载完成！")
    except Exception as e:
        st.error(f"下载 GitHub 仓库失败: {str(e)}")

# ------------------- 准备 Chroma 知识库目录 -------------------
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
    if not os.path.exists(CHROMA_DIR):
        st.info("知识库不存在，正在自动下载，请稍等...")
        download_github_repo("https://github.com/zebinlu7-a11y/ruitong-chat-app")
        raw_chroma_dir = "./ruitong-chat-app-main/models/ruitongkeji"
        prepare_chroma_dir(raw_chroma_dir)
    MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"知识库加载失败: {str(e)}")
        return None

vectorstore = load_vectorstore()
if vectorstore:
    st.success("知识库加载完成！")
else:
    st.error("知识库加载失败，请检查路径或数据。")

# ------------------- 系统提示 -------------------
system_prompt = (
    " 你是一个锐瞳智能科技公司的智能助手，名字叫小锐,跟用户对话请以第一人称的方式与用户沟通。"
    " 你不仅能回答与公司有关的问题，还是个百科全书，能回答各学科的所有问题。"
    " 问你与锐瞳智能科技公司有关的问题，不要直接显示知识库内容，要结合原始知识库内容，回答用户有关你所在的锐瞳智能科技公司的信息。"
    " 问你与锐瞳智能科技公司无关的其他信息，就直接回答，不要结合知识库，发挥你自身的专业能力去回答。"
)

# ------------------- 初始化会话状态（支持多会话） -------------------
if "conversations" not in st.session_state:
    # 从 Cookies 加载历史（如果存在）
    st.session_state.conversations = load_conversations()
    # 如果没有历史，初始化默认会话
    if not st.session_state.conversations:
        default_id = "default"
        st.session_state.conversations = {
            default_id: {
                "title": "新对话",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": "你好，我是小锐助手，有什么需要帮助的吗？"}
                ]
            }
        }
    st.session_state.current_session = list(st.session_state.conversations.keys())[0]
    save_conversations()  # 初始化后保存

# ------------------- 调用 DeepSeek API -------------------
def call_deepseek_api(messages, context):
    try:
        user_messages = [msg for msg in messages if msg["role"] == "user"]
        if user_messages:
            last_user_msg = user_messages[-1]["content"]
            if vectorstore:
                results = vectorstore.similarity_search(last_user_msg, k=3)
                context_str = "\n".join([doc.page_content for doc in results]) if results else "无相关知识库内容"
                messages[-1]["content"] += f"\n\n[知识库上下文，仅供参考：{context_str}]"
        response = requests.post(
            f"{DEEPSEEK_API_BASE}/chat/completions",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {DEEPSEEK_API_KEY}"
            },
            json={
                "model": DEEPSEEK_MODEL,
                "messages": messages,
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

# ------------------- 侧边栏：会话历史 -------------------
with st.sidebar:
    st.header("💬 对话历史")
    if st.button("新建对话", key="new_chat"):
        new_id = f"chat_{len(st.session_state.conversations)}"
        st.session_state.conversations[new_id] = {
            "title": f"对话 {len(st.session_state.conversations) + 1}",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "assistant", "content": "你好，我是小锐助手，有什么需要帮助的吗？"}
            ]
        }
        st.session_state.current_session = new_id
        save_conversations()  # 保存到 Cookies
        st.rerun()

    session_options = list(st.session_state.conversations.keys())
    selected_session = st.radio(
        "选择对话：",
        options=session_options,
        index=session_options.index(st.session_state.current_session),
        format_func=lambda x: st.session_state.conversations[x]["title"]
    )
    if selected_session != st.session_state.current_session:
        st.session_state.current_session = selected_session
        st.rerun()

    current_conv = st.session_state.conversations[st.session_state.current_session]
    if len(current_conv["messages"]) > 2:
        title = st.text_input("重命名对话：", value=current_conv["title"], key="rename")
        if title != current_conv["title"]:
            current_conv["title"] = title
            save_conversations()  # 更新标题后保存

    # 添加清除历史按钮
    if st.button("清除所有对话历史", key="clear_history"):
        st.session_state.conversations = {
            "default": {
                "title": "新对话",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": "你好，我是小锐助手，有什么需要帮助的吗？"}
                ]
            }
        }
        st.session_state.current_session = "default"
        save_conversations()
        st.rerun()

# ------------------- 聊天界面 -------------------
st.title("💡锐瞳智能科技公司——小锐智能体")
current_messages = st.session_state.conversations[st.session_state.current_session]["messages"]

for msg in current_messages:
    if msg["role"] != "system":
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

user_input = st.chat_input("请输入您的问题...", key=f"chat_input_{st.session_state.current_session}")

if user_input:
    with st.chat_message("user"):
        st.write(user_input)
    current_messages.append({"role": "user", "content": user_input})
    with st.chat_message("assistant"):
        with st.spinner("小锐正在思考..."):
            reply = call_deepseek_api(current_messages, None)
            st.write(reply)
        current_messages.append({"role": "assistant", "content": reply})
    save_conversations()  # 保存对话到 Cookies

# ------------------- 操作指南 -------------------
if st.checkbox("操作指南"):
    st.write("查找锐瞳科技相关信息，请咨询小锐")
