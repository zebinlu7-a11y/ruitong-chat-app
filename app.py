import streamlit as st
import json
import os
import re
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import requests
import zipfile
from io import BytesIO
import shutil

# ------------------- Streamlit 配置 -------------------
st.set_page_config(
    page_title="锐瞳智能科技有限公司———小锐智能体",
    page_icon="🤖",
    layout="wide"
)

# ------------------- DeepSeek API 配置 -------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  # 建议用环境变量
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# ------------------- 路径配置 -------------------
CONVERSATIONS_DIR = "./conversations"  # 存储用户 JSON 文件的目录
CHROMA_DIR = "./models/ruitongkeji"     # Chroma 数据库本地路径
EMBEDDINGS_DIR = "./models/all-MiniLM-L6-v2"  # 嵌入模型本地路径

# ------------------- 确保用户目录存在 -------------------
os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(EMBEDDINGS_DIR, exist_ok=True)  # 确保嵌入模型目录存在

# ------------------- 用户名验证和保存/加载函数 -------------------
def is_valid_username(username):
    """验证用户名：只允许字母、数字、下划线"""
    return bool(re.match(r'^[a-zA-Z0-9_]+$', username))

def save_conversations(username):
    """保存会话到用户专属 JSON 文件"""
    try:
        conversations_file = os.path.join(CONVERSATIONS_DIR, f"conversations_{username}.json")
        with open(conversations_file, "w", encoding="utf-8") as f:
            json.dump(st.session_state.conversations, f, ensure_ascii=False, indent=2)
        #st.success(f"对话保存到 {conversations_file}")
    except Exception as e:
        st.error(f"保存对话失败: {str(e)}")

def load_conversations(username):
    """从用户专属 JSON 文件加载会话"""
    try:
        conversations_file = os.path.join(CONVERSATIONS_DIR, f"conversations_{username}.json")
        if os.path.exists(conversations_file):
            with open(conversations_file, "r", encoding="utf-8") as f:
                #st.info(f"加载对话从 {conversations_file}")
                return json.load(f)
        else:
            st.warning(f"未找到 {conversations_file}")
    except Exception as e:
        st.error(f"加载对话失败: {str(e)}")
    return {}

def delete_user(username):
    """删除指定用户的数据文件并重置状态"""
    conversations_file = os.path.join(CONVERSATIONS_DIR, f"conversations_{username}.json")
    try:
        if os.path.exists(conversations_file):
            os.remove(conversations_file)
            st.success(f"成功删除用户 {username} 的数据文件: {conversations_file}")
        else:
            st.warning(f"用户 {username} 的数据文件不存在: {conversations_file}")
        st.session_state.username = None
        st.session_state.conversations = None
        st.session_state.show_delete_confirmation = False  # 确保重置
        st.rerun()
    except Exception as e:
        st.error(f"删除用户 {username} 失败: {str(e)}")

# ------------------- 自动下载 GitHub 仓库（可选备份） -------------------
def download_github_repo(repo_url, extract_to="."):
    try:
        zip_url = repo_url.rstrip("/") + "/archive/refs/heads/main.zip"
        r = requests.get(zip_url, timeout=60)
        r.raise_for_status()
        z = zipfile.ZipFile(BytesIO(r.content))
        z.extractall(extract_to)
        st.success(f"仓库 {repo_url} 下载完成！")
        # 调试：列出解压后的文件
        st.write(f"解压文件列表: {os.listdir(extract_to)}")
    except Exception as e:
        st.error(f"下载 GitHub 仓库失败: {str(e)}")

# ------------------- 准备 Chroma 知识库目录 -------------------
def prepare_chroma_dir(raw_dir, target_dir=CHROMA_DIR):
    if not os.path.exists(target_dir) or not any(f.endswith((".bin", ".sqlite3")) for f in os.listdir(target_dir)):
        os.makedirs(target_dir, exist_ok=True)
        copied_files = []
        for root, _, files in os.walk(raw_dir):
            for f in files:
                if f.endswith((".bin", ".sqlite3")):
                    shutil.copy(os.path.join(root, f), os.path.join(target_dir, f))
                    copied_files.append(f)
        st.write(f"复制到 Chroma 目录的文件: {copied_files}")
    else:
        st.write(f"Chroma 目录 {target_dir} 已存在且包含文件，无需复制。")
    return target_dir

# ------------------- 准备嵌入模型目录 -------------------
def prepare_embeddings_dir(raw_dir, target_dir=EMBEDDINGS_DIR):
    if not os.path.exists(target_dir) or not any(f.endswith((".json", ".bin", ".safetensors")) for f in os.listdir(target_dir)):
        os.makedirs(target_dir, exist_ok=True)
        copied_files = []
        model_raw_path = os.path.join(raw_dir, "models/all-MiniLM-L6-v2")  # 调整为你的仓库模型路径
        if os.path.exists(model_raw_path):
            for root, _, files in os.walk(model_raw_path):
                for f in files:
                    if f.endswith((".json", ".bin", ".safetensors")):
                        shutil.copy(os.path.join(root, f), os.path.join(target_dir, f))
                        copied_files.append(f)
            st.write(f"复制到嵌入模型目录的文件: {copied_files}")
        else:
            st.error(f"嵌入模型源目录 {model_raw_path} 不存在，请检查 GitHub 仓库！")
    else:
        st.write(f"嵌入模型目录 {target_dir} 已存在且包含文件，无需复制。")
    return target_dir

# ------------------- 加载知识库 -------------------
@st.cache_resource
def load_vectorstore():
    repo_url = "https://github.com/zebinlu7-a11y/ruitong-chat-app"
    raw_repo_dir = "./ruitong-chat-app-main"

    # 优先检查本地文件是否存在
    if not os.path.exists(CHROMA_DIR) or not os.path.exists(EMBEDDINGS_DIR):
        st.info("本地知识库或模型文件缺失，尝试从 GitHub 下载...")
        download_github_repo(repo_url)
        raw_chroma_dir = os.path.join(raw_repo_dir, "models/ruitongkeji")
        raw_embeddings_dir = os.path.join(raw_repo_dir, "models/all-MiniLM-L6-v2")
        prepare_chroma_dir(raw_chroma_dir)
        prepare_embeddings_dir(raw_embeddings_dir)
    else:
        st.write(f"使用本地知识库 {CHROMA_DIR} 和嵌入模型 {EMBEDDINGS_DIR}")

    # 加载嵌入模型
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDINGS_DIR,  # 使用本地路径
            model_kwargs={'device': 'cpu'}  # 确保在 CPU 上运行
        )
        st.write("嵌入模型加载成功！")
    except Exception as e:
        st.error(f"嵌入模型加载失败: {str(e)}")
        return None

    # 加载 Chroma 向量数据库
    try:
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        st.write(f"知识库加载成功，文档数量: {vectorstore._collection.count()}")
        return vectorstore
    except Exception as e:
        st.error(f"知识库加载失败: {str(e)}")
        return None

vectorstore = load_vectorstore()
if vectorstore:
    st.success("锐瞳知识库加载完成！")
else:
    st.error("知识库加载失败，请检查路径或数据。")

# ------------------- 获取全部知识库内容 -------------------
def get_full_knowledge_context(vectorstore):
    if vectorstore:
        all_docs = vectorstore.get()
        if all_docs and "documents" in all_docs:
            full_context = " ".join(doc for doc in all_docs["documents"])
            return full_context
    return "知识库内容不可用"

# ------------------- 系统提示 -------------------
if vectorstore:
    full_context = get_full_knowledge_context(vectorstore)
    system_prompt = (
        f"你是一个锐瞳智能科技公司的智能助手，名字叫小锐，跟用户对话请以第一人称方式与用户沟通。"
        f"你不仅能回答与公司有关的问题，还是个百科全书，能回答各学科的所有问题。"
        f"当用户询问与锐瞳智能科技公司相关的内容时，结合以下内部知识库信息以自然语言回答，但不要直接引用或显示原始文本：\n[内部参考信息，仅供内部使用：{full_context}]\n"
        f"当用户询问与公司无关的问题时，基于你自身的知识直接回答，不需参考知识库。"
        f"例如，用户问'你叫什么名字'，你回答：'我叫小锐，来自锐瞳科技。'"
        f"请根据对话语境智能判断是否需要调用知识库，保持回答流畅自然。"
    )
else:
    system_prompt = (
        "你是一个锐瞳智能科技公司的智能助手，名字叫小锐，跟用户对话请以第一人称方式与用户沟通。"
        "你不仅能回答与公司有关的问题，还是个百科全书，能回答各学科的所有问题。"
        "当用户询问与锐瞳智能科技公司相关的内容时，基于你自身的知识回答，因知识库不可用。"
        "当用户询问与公司无关的问题时，基于你自身的知识直接回答。"
        "例如，用户问'你叫什么名字'，你回答：'我叫小锐，来自锐瞳科技。'"
        "请保持回答流畅自然。"
    )

# 以下代码保持不变（省略了未修改的部分），请将原代码中的剩余部分粘贴过来继续使用
# ------------------- 用户选择/输入界面 -------------------
if "username" not in st.session_state:
    st.session_state.username = None
    st.session_state.show_delete_confirmation = False  # 初始化时重置


if not st.session_state.username:
    st.title("请选择或输入用户名")
    existing_users = [f.replace("conversations_", "").replace(".json", "") for f in os.listdir(CONVERSATIONS_DIR) if f.startswith("conversations_") and f.endswith(".json")]
    if existing_users:
        selected_user = st.selectbox("已有用户：", existing_users)
        if st.button("加载已有用户"):
            st.session_state.username = selected_user
            st.session_state.show_delete_confirmation = False  # 登录时重置
            st.rerun()
    new_user = st.text_input("或输入新用户名（仅限字母、数字、下划线）：")
    if st.button("使用新用户名"):
        if new_user and is_valid_username(new_user):
            st.session_state.username = new_user
            st.session_state.show_delete_confirmation = False  # 登录时重置
            st.rerun()
        else:
            st.error("用户名无效或为空（仅限字母、数字、下划线）！")
else:
    # ------------------- 初始化会话状态（支持多会话） -------------------
    if "conversations" not in st.session_state or st.session_state.conversations is None:
        st.session_state.conversations = load_conversations(st.session_state.username)
        if not st.session_state.conversations or st.session_state.conversations == {}:
            default_id = "default"
            st.session_state.conversations = {
                default_id: {
                    "title": "新对话",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "assistant", "content": f"你好，我是小锐助手，有什么需要帮助的吗？"}
                    ]
                }
            }
            st.info("初始化新会话")
        st.session_state.current_session = list(st.session_state.conversations.keys())[0]
        save_conversations(st.session_state.username)

    # ------------------- 调用 DeepSeek API -------------------
    def call_deepseek_api(messages, context):
        try:
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
                    "max_tokens": 800
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
        st.header(f"💬 {st.session_state.username} 的对话历史")
        if st.button("新建对话", key="new_chat"):
            new_id = f"chat_{len(st.session_state.conversations)}"
            st.session_state.conversations[new_id] = {
                "title": f"对话 {len(st.session_state.conversations) + 1}",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "assistant", "content": f"你好，我是小锐助手，有什么需要帮助的吗？"}
                ]
            }
            st.session_state.current_session = new_id
            save_conversations(st.session_state.username)
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
                save_conversations(st.session_state.username)

        if st.button("清除所有对话历史", key="clear_history"):
            st.session_state.conversations = {
                "default": {
                    "title": "新对话",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "assistant", "content": f"你好，我是小锐助手，有什么需要帮助的吗？"}
                    ]
                }
            }
            st.session_state.current_session = "default"
            save_conversations(st.session_state.username)
            st.rerun()
            
        # --------------- 删除用户功能 -------------------
        if "show_delete_confirmation" not in st.session_state:
            st.session_state.show_delete_confirmation = False
        
        if st.button("删除用户", key="delete_user"):
            st.session_state.show_delete_confirmation = True
        
        if st.session_state.show_delete_confirmation:
            st.warning(f"确定要删除用户 '{st.session_state.username}' 吗？这将删除所有对话历史！")
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("确定", key="confirm_delete"):
                    delete_user(st.session_state.username)
            with col2:
                if st.button("取消", key="cancel_delete"):
                    st.session_state.show_delete_confirmation = False
                    st.rerun()
        
        if st.button("切换用户", key="switch_user"):
            st.session_state.username = None
            st.session_state.conversations = None
            st.session_state.show_delete_confirmation = False  # 确保重置
            st.rerun()

    # ------------------- 聊天界面 -------------------
    st.title(f"💡锐瞳智能科技公司——小锐智能体（欢迎，{st.session_state.username}）")
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
        save_conversations(st.session_state.username)

    # ------------------- 操作指南 -------------------
    if st.checkbox("操作指南"):
        st.write("查找锐瞳科技相关信息，请咨询小锐")
