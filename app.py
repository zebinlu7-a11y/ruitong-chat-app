import os
# 解决 protobuf 版本兼容性问题（Streamlit Cloud 部署必需）
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st
import json
import re
import time
import requests
import base64
from datetime import datetime
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
import numpy as np

# ------------------- API Key 配置文件（定义在路径配置之后） -------------------
API_KEY_FILE = None  # 稍后初始化

def init_api_key_file():
    """初始化 API Key 文件路径"""
    global API_KEY_FILE
    API_KEY_FILE = os.path.join(CONVERSATIONS_DIR, "api_keys.json")

def save_api_key(username, api_key):
    """保存用户的 API Key（简单编码存储）"""
    try:
        # 简单 base64 编码（防止明文泄露，非真正加密）
        encoded = base64.b64encode(api_key.encode()).decode()
        if os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            data = {}
        data[username] = encoded
        with open(API_KEY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        return True
    except Exception as e:
        st.error(f"保存 API Key 失败: {str(e)}")
        return False

def load_api_key(username):
    """加载用户的 API Key"""
    try:
        if API_KEY_FILE and os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if username in data:
                encoded = data[username]
                return base64.b64decode(encoded.encode()).decode()
    except Exception:
        pass
    return None

def delete_api_key(username):
    """删除用户的 API Key"""
    try:
        if API_KEY_FILE and os.path.exists(API_KEY_FILE):
            with open(API_KEY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            if username in data:
                del data[username]
                with open(API_KEY_FILE, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False)
        return True
    except Exception:
        return False

# ------------------- 通用 API 调用函数（带重试+超时） -------------------
def call_deepseek_api_retry(
    prompt,
    max_tokens=1000,
    temperature=0.1,
    max_retries=3,
    timeout=60,
    is_json=False,
    api_key=None
):
    """
    带重试和超时的 DeepSeek API 调用
    
    参数:
        prompt: 提示词
        max_tokens: 最大生成token数
        temperature: 温度参数
        max_retries: 最大重试次数
        timeout: 超时时间（秒）
        is_json: 是否返回JSON格式
        api_key: API Key（优先使用，否则使用全局变量）
    返回:
        生成的文本内容，失败返回 None
    """
    # 优先使用传入的 api_key，否则使用全局变量
    key = api_key if api_key else DEEPSEEK_API_KEY
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    
    json_data = {
        "model": DEEPSEEK_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens
    }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers=headers,
                json=json_data,
                timeout=timeout
            )
            response.raise_for_status()
            result = response.json()["choices"][0]["message"]["content"].strip()
            return result
            
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt  # 指数退避: 1, 2, 4秒
                time.sleep(wait_time)
                continue
            else:
                st.warning(f"API 调用超时（已重试{max_retries}次）")
                return None
                
        except requests.exceptions.ConnectionError as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                time.sleep(wait_time)
                continue
            else:
                st.warning(f"API 连接失败: {e}")
                return None
                
        except Exception as e:
            st.warning(f"API 调用异常: {e}")
            return None
    
    return None

# ------------------- Streamlit 配置 -------------------
st.set_page_config(
    page_title="锐瞳智能科技有限公司———小锐智能体",
    page_icon="🤖",
    layout="wide"
)

# ------------------- DeepSeek API 配置 -------------------
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_MODEL = "deepseek-chat"
DEEPSEEK_API_BASE = "https://api.deepseek.com/v1"

# ------------------- 路径配置 -------------------
CONVERSATIONS_DIR = "./conversations"
CHROMA_DIR = "./models/ruitongkeji"
HISTORY_CHROMA_DIR = "./models/history_vectorstore"  # 历史对话向量库
MEMORY_MAX_FACTS = 30
SESSION_SUMMARY_THRESHOLD = 10  # 触发摘要的对话轮数

os.makedirs(CONVERSATIONS_DIR, exist_ok=True)
init_api_key_file()  # 初始化 API Key 文件路径
os.makedirs(HISTORY_CHROMA_DIR, exist_ok=True)

# ------------------- 用户名验证 -------------------
def is_valid_username(username):
    return bool(re.match(r'^[a-zA-Z0-9_]+$', username))

# ------------------- 会话持久化 -------------------
def save_conversations(username):
    try:
        path = os.path.join(CONVERSATIONS_DIR, f"conversations_{username}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(st.session_state.conversations, f, ensure_ascii=False, indent=2)
    except Exception as e:
        st.error(f"保存对话失败: {str(e)}")

def load_conversations(username):
    try:
        path = os.path.join(CONVERSATIONS_DIR, f"conversations_{username}.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        else:
            st.warning(f"未找到 {path}")
    except Exception as e:
        st.error(f"加载对话失败: {str(e)}")
    return {}

def delete_session(username, session_id):
    """删除指定会话"""
    path = os.path.join(CONVERSATIONS_DIR, f"conversations_{username}.json")
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            if session_id in data:
                # 删除会话
                del data[session_id]
                
                # 保存更新后的数据
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # 同时删除向量库中该会话的摘要
                summary_path = os.path.join(CONVERSATIONS_DIR, f"summary_{session_id}.json")
                if os.path.exists(summary_path):
                    os.remove(summary_path)
                
                # 删除历史向量库中该会话的数据
                history_vs = load_history_vectorstore()
                if history_vs:
                    try:
                        history_vs.delete(filter={"session_id": session_id})
                    except:
                        pass
                
                return True, "删除成功"
            else:
                return False, "会话不存在"
        else:
            return False, "对话文件不存在"
    except Exception as e:
        return False, f"删除失败: {str(e)}"

def delete_user(username):
    path = os.path.join(CONVERSATIONS_DIR, f"conversations_{username}.json")
    mem_path = os.path.join(CONVERSATIONS_DIR, f"memory_{username}.json")
    try:
        # 删除基础文件
        for p in [path, mem_path]:
            if os.path.exists(p):
                os.remove(p)
        
        # 删除该用户的历史摘要文件
        for f in os.listdir(CONVERSATIONS_DIR):
            if f.startswith("summary_") and f.endswith(".json"):
                summary_path = os.path.join(CONVERSATIONS_DIR, f)
                with open(summary_path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)
                    if data.get("session_id", "").startswith(username):
                        os.remove(summary_path)
        
        # 删除历史向量库中该用户的数据
        history_vs = load_history_vectorstore()
        if history_vs:
            try:
                history_vs.delete(filter={"username": username})
            except:
                pass
        
        st.success(f"成功删除用户 {username} 的数据")
        st.session_state.username = None
        st.session_state.conversations = None
        st.session_state.show_delete_confirmation = False
        st.rerun()
    except Exception as e:
        st.error(f"删除用户 {username} 失败: {str(e)}")

# ------------------- 长期记忆 -------------------
def load_long_term_memory(username):
    path = os.path.join(CONVERSATIONS_DIR, f"memory_{username}.json")
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f).get("facts", [])
    return []

def save_long_term_memory(username, facts):
    path = os.path.join(CONVERSATIONS_DIR, f"memory_{username}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({
            "updated_at": datetime.now().isoformat(timespec="seconds"),
            "facts": facts[-MEMORY_MAX_FACTS:]
        }, f, ensure_ascii=False, indent=2)

# ------------------- 历史对话向量库 -------------------
@st.cache_resource
def load_history_vectorstore():
    """加载用户历史对话向量库"""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-zh-v1.5")
        history_vs = Chroma(
            persist_directory=HISTORY_CHROMA_DIR, 
            embedding_function=embeddings,
            collection_name="history"
        )
        return history_vs
    except Exception as e:
        st.warning(f"历史向量库加载失败: {e}")
        return None

def generate_session_summary(session_messages, session_id):
    """生成会话摘要"""
    dialogue = [m for m in session_messages if m["role"] in ("user", "assistant")]
    if len(dialogue) < SESSION_SUMMARY_THRESHOLD:
        return None
    
    dialogue_text = "\n".join(
        f"{'用户' if m['role']=='user' else '助手'}: {m['content']}"
        for m in dialogue
    )
    
    prompt = (
        f"请将以下对话压缩成一个300字以内的摘要，包含：\n"
        f"1. 讨论的主题或核心话题\n"
        f"2. 用户的核心需求或问题\n"
        f"3. 达成的结论或关键信息\n"
        f"4. 用户的偏好或关注点（如有）\n\n"
        f"对话内容：\n{dialogue_text}"
    )
    
    # 使用通用重试函数
    summary = call_deepseek_api_retry(
        prompt=prompt,
        max_tokens=500,
        timeout=30
    )
    
    if summary:
        # 保存摘要到JSON
        summary_path = os.path.join(CONVERSATIONS_DIR, f"summary_{session_id}.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump({
                "session_id": session_id,
                "summary": summary,
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "message_count": len(dialogue)
            }, f, ensure_ascii=False, indent=2)
        return summary
    
    return None

def save_to_history_vectorstore(username, texts, metadata_type="summary", session_id=None):
    """保存摘要或对话片段到向量库"""
    history_vs = load_history_vectorstore()
    if not history_vs or not texts:
        return
    
    try:
        ids = [f"{username}_{session_id or metadata_type}_{i}_{datetime.now().strftime('%Y%m%d%H%M%S')}" 
               for i in range(len(texts))]
        metadatas = [
            {"username": username, "type": metadata_type, "session_id": session_id or ""} 
            for _ in texts
        ]
        history_vs.add_texts(texts=texts, ids=ids, metadatas=metadatas)
        history_vs.persist()
    except Exception as e:
        st.warning(f"保存到历史向量库失败: {e}")

def search_history_vectorstore(query, username, k=5, return_with_score=True):
    """从历史向量库检索相关内容，返回带session_id的结果"""
    history_vs = load_history_vectorstore()
    if not history_vs:
        return []
    
    try:
        results = history_vs.similarity_search_with_score(
            query, k=k, filter={"username": username}
        )
        # 返回结果：包含内容、分数、session_id
        formatted_results = []
        for r, score in results:
            formatted_results.append({
                "content": r.page_content,
                "score": score,
                "session_id": r.metadata.get("session_id", ""),
                "type": r.metadata.get("type", "")
            })
        return formatted_results
    except Exception:
        return []

def get_user_summaries(username):
    """获取用户所有会话摘要（含session_id）"""
    summaries = []
    for f in os.listdir(CONVERSATIONS_DIR):
        if f.startswith("summary_") and f.endswith(".json"):
            path = os.path.join(CONVERSATIONS_DIR, f)
            with open(path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                summaries.append({
                    "summary": data["summary"],
                    "session_id": data.get("session_id", "")
                })
    return summaries

# ------------------- RRF融合排序 -------------------
def rrf_fusion(*ranked_lists, k=60):
    """
    Reciprocal Rank Fusion - 多结果融合排序
    参数: 多个已排序的列表，每个列表元素为 dict
    返回: 融合后的排序列表
    """
    from collections import defaultdict
    
    scores = defaultdict(float)
    item_data = {}  # 存储每个item的完整数据
    
    for ranked_list in ranked_lists:
        if not ranked_list:
            continue
        for rank, item in enumerate(ranked_list):
            item_key = f"{item.get('session_id', '')}_{item.get('content', '')[:50]}"
            item_data[item_key] = item
            # RRF公式: 1 / (k + rank)
            scores[item_key] += 1 / (k + rank + 1)
    
    # 按融合分数排序
    sorted_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    
    # 返回排序后的完整数据
    return [{"key": k, "score": scores[k], **item_data[k]} for k in sorted_keys]

# ------------------- 关键词提取与匹配 -------------------
def extract_keywords_from_query(query):
    """从查询中提取关键词"""
    # 简单分词 + 去停用词
    stopwords = {"的", "是", "在", "和", "了", "我", "你", "他", "她", "它", "这", "那", "有", "什么", "怎么", "如何", "吗", "呢", "吧"}
    
    # 简单按标点和空格分词
    words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', query)
    keywords = [w for w in words if len(w) >= 2 and w not in stopwords]
    return list(set(keywords))

def keyword_match_in_session(query, session_messages, max_matches=3):
    """在会话中用关键词匹配，返回匹配片段"""
    keywords = extract_keywords_from_query(query)
    if not keywords:
        return []
    
    matches = []
    dialogue = [m for m in session_messages if m["role"] in ("user", "assistant")]
    
    for i, msg in enumerate(dialogue):
        text = msg["content"]
        # 计算关键词匹配数
        matched_kws = [kw for kw in keywords if kw in text]
        if matched_kws:
            # 滑动窗口：取前后各1条消息
            start = max(0, i - 1)
            end = min(len(dialogue), i + 2)
            window = dialogue[start:end]
            window_text = "\n".join(
                f"{'用户' if m['role']=='user' else '助手'}: {m['content']}"
                for m in window
            )
            matches.append({
                "content": window_text,
                "matched_keywords": matched_kws,
                "match_count": len(matched_kws),
                "session_id": session_messages.get("session_id", "")
            })
    
    # 按匹配数排序
    matches.sort(key=lambda x: x["match_count"], reverse=True)
    return matches[:max_matches]

def load_session_from_json(username, session_id):
    """从JSON加载指定会话"""
    path = os.path.join(CONVERSATIONS_DIR, f"conversations_{username}.json")
    if not os.path.exists(path):
        return None
    
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if session_id in data:
            session = data[session_id].copy()
            session["session_id"] = session_id
            return session
    return None

# ------------------- 混合历史检索（核心） -------------------
def hybrid_history_search(query, username, k_summary=5, k_session=3):
    """
    混合历史检索流程：
    1. 向量检索摘要（快速定位话题）
    2. 判断摘要是否足够
    3. 如果不足，加载会话JSON，RRF融合向量+关键词匹配
    4. 返回匹配片段
    """
    # Step 1: 检索相关摘要
    summary_results = search_history_vectorstore(query, username, k=k_summary)
    
    if not summary_results:
        return {"status": "no_summary", "results": [], "keywords": extract_keywords_from_query(query)}
    
    # Step 2: 提取摘要中的关键词
    summary_keywords = extract_keywords_from_query(query)
    
    # Step 3: 检查摘要是否足够回答
    context_enough = check_summary_enough(query, summary_results)
    
    if context_enough:
        # 摘要足够，直接返回摘要结果
        return {
            "status": "summary_enough",
            "results": summary_results,
            "keywords": summary_keywords
        }
    
    # Step 4: 摘要不足，检索会话细节
    # 获取相关的session_id列表
    relevant_sessions = list(set([r.get("session_id", "") for r in summary_results if r.get("session_id")]))
    
    # 存储所有匹配结果
    all_vector_matches = []
    all_keyword_matches = []
    
    for session_id in relevant_sessions:
        # 向量检索该会话的细节
        session_vectors = search_history_vectorstore(query, username, k=k_session, return_with_score=True)
        session_vectors = [r for r in session_vectors if r.get("session_id") == session_id]
        all_vector_matches.extend(session_vectors)
        
        # 关键词匹配该会话
        session_data = load_session_from_json(username, session_id)
        if session_data:
            keyword_matches = keyword_match_in_session(query, session_data.get("messages", []))
            all_keyword_matches.extend(keyword_matches)
    
    # Step 5: RRF融合排序
    # 转换格式以便RRF处理
    vector_for_rrf = [{"session_id": r.get("session_id", ""), "content": r.get("content", ""), "rank_source": "vector"} 
                      for r in all_vector_matches]
    keyword_for_rrf = [{"session_id": m.get("session_id", ""), "content": m.get("content", ""), "rank_source": "keyword"} 
                       for m in all_keyword_matches]
    
    fused_results = rrf_fusion(vector_for_rrf, keyword_for_rrf)
    
    # 合并摘要结果
    combined = fused_results + [{"content": r.get("content", ""), "session_id": r.get("session_id", ""), 
                                 "source": "summary", "score": r.get("score", 1.0)} 
                                for r in summary_results]
    
    return {
        "status": "session_detail",
        "results": combined[:k_session * 2],
        "keywords": summary_keywords
    }

def check_summary_enough(query, summary_results):
    """判断摘要是否足够回答问题"""
    if not summary_results:
        return False
    
    # 合并所有摘要内容
    all_summary_text = "\n".join([r.get("content", "") for r in summary_results])
    
    prompt = (
        f"基于以下摘要内容，判断能否回答用户问题。\n\n"
        f"摘要内容：\n{all_summary_text[:1000]}\n\n"
        f"用户问题：{query}\n\n"
        "如果摘要内容能回答问题（即使需要推理），返回'是'；如果明显需要更多信息，返回'否'。"
        "只返回'是'或'否'，不要其他内容。"
    )
    
    result = call_deepseek_api_retry(
        prompt=prompt,
        max_tokens=50,
        timeout=30
    )
    
    # 默认认为足够，避免频繁回退
    return result == "是" if result else True

def extract_and_update_memory(username, messages):
    dialogue = [m for m in messages if m["role"] in ("user", "assistant")]
    if len(dialogue) < 4:
        return
    
    dialogue_text = "\n".join(
        f"{'用户' if m['role']=='user' else '助手'}: {m['content']}"
        for m in dialogue[-10:]
    )
    existing_facts = load_long_term_memory(username)
    existing_str = "\n".join(f"- {f}" for f in existing_facts) if existing_facts else "（暂无）"
    prompt = (
        f"以下是一段对话记录：\n{dialogue_text}\n\n"
        f"已有的用户长期记忆：\n{existing_str}\n\n"
        "请从对话中提取值得长期记住的用户偏好、关注点或重要信息（不超过3条）。"
        "如果没有新信息，返回空列表。"
        "只返回 JSON 数组，例如：[\"用户关注产品价格\"]"
    )
    
    raw = call_deepseek_api_retry(
        prompt=prompt,
        max_tokens=200,
        timeout=30
    )
    
    if raw:
        match = re.search(r'\[.*?\]', raw, re.DOTALL)
        new_facts = json.loads(match.group()) if match else []
        if new_facts:
            merged = existing_facts + [f for f in new_facts if f not in existing_facts]
            save_long_term_memory(username, merged)

# ------------------- 动态 system prompt -------------------
def build_system_prompt(username):
    base = (
        "你是锐瞳智能科技公司的智能助手，名字叫小锐，以第一人称与用户沟通。"
        "你不仅能回答公司相关问题，还能回答与机器视觉光学，大模型等计算机领域的问题。"
        "当用户询问与公司相关内容时，结合提供的知识库信息以自然语言回答，不要直接引用原始文本。"
        "当用户询问与公司无关的问题时，基于自身知识直接回答。"
    )
    facts = load_long_term_memory(username)
    if facts:
        facts_str = "\n".join(f"- {f}" for f in facts)
        base += f"\n\n【关于该用户的长期记忆，请参考但不要主动提及】\n{facts_str}"
    return base

# ------------------- 加载知识库 -------------------
@st.cache_resource
def load_vectorstore():
    """加载知识库向量库（文件已包含在 GitHub 仓库中）"""
    
    # 检查 Chroma 目录是否存在
    if not os.path.exists(CHROMA_DIR):
        st.error(f"知识库目录不存在: {CHROMA_DIR}")
        return None
    
    # 检查必要文件
    sqlite_path = os.path.join(CHROMA_DIR, "chroma.sqlite3")
    if not os.path.exists(sqlite_path):
        st.error(f"SQLite 文件不存在: {sqlite_path}")
        return None
    
    # 检查子目录
    has_subdirs = any(os.path.isdir(os.path.join(CHROMA_DIR, d)) 
                      for d in os.listdir(CHROMA_DIR) if os.path.isdir(os.path.join(CHROMA_DIR, d)))
    if not has_subdirs:
        st.error("Chroma 向量子目录不存在")
        return None
    
    # 打印文件信息用于调试
    # sqlite_size = os.path.getsize(sqlite_path)
    # st.info(f"知识库文件大小: {sqlite_size/1024:.1f} KB")
    
    MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    try:
        embeddings = HuggingFaceEmbeddings(model_name=MODEL_NAME)
        vectorstore = Chroma(persist_directory=CHROMA_DIR, embedding_function=embeddings)
        
        # 验证向量库是否可用
        count = vectorstore._collection.count()
        # st.success(f"知识库加载完成！共 {count} 条记录")
        return vectorstore
    except Exception as e:
        st.error(f"知识库加载失败: {str(e)}")
        return None
        return None


vectorstore = load_vectorstore()
# if vectorstore:
#     st.success("锐瞳知识库加载完成！")
# else:
#     st.error("知识库加载失败，请检查路径或数据。")

# ------------------- 加载 BM25 索引 -------------------
@st.cache_resource
def load_bm25_index():
    try:
        from rank_bm25 import BM25Okapi
        raw = vectorstore.get()
        docs = raw.get("documents", [])
        tokenized = [list(d) for d in docs]
        bm25 = BM25Okapi(tokenized)
        return bm25, docs
    except Exception as e:
        st.warning(f"BM25 索引构建失败: {e}")
        return None, []

# ------------------- 加载 bge-reranker -------------------
# RERANKER_MODEL_PATH = "./models/BAAI/bge-reranker-base"

# @st.cache_resource
# def load_reranker():
#     try:
#         from transformers import AutoTokenizer, AutoModelForSequenceClassification
#         
#         # 检查本地模型是否存在
#         if not os.path.exists(RERANKER_MODEL_PATH):
#             st.warning(f"Reranker 模型本地路径不存在: {RERANKER_MODEL_PATH}")
#             return None, None
#         
#         tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_PATH)
#         model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_PATH)
#         model.eval()
#         return tokenizer, model
#     except Exception as e:
#         st.warning(f"Reranker 加载失败: {e}")
#         return None, None

# BM25 加载（必要）
# with st.spinner("正在加载 BM25 索引..."):
#     bm25_index, bm25_docs = load_bm25_index() if vectorstore else (None, [])
bm25_index, bm25_docs = load_bm25_index() if vectorstore else (None, [])

# Reranker 加载（已禁用）
# with st.spinner("正在加载 Reranker 模型..."):
#     reranker_tokenizer, reranker_model = load_reranker()
#     if reranker_tokenizer is None:
#         st.info("💡 Reranker 模型未加载，将使用基础向量检索（可正常使用所有功能）")
#     else:
#         st.success("✅ Reranker 模型加载成功")

# 禁用 reranker
reranker_tokenizer, reranker_model = None, None

# ------------------- 用户选择/输入界面 -------------------
if "username" not in st.session_state:
    st.session_state.username = None
    st.session_state.show_delete_confirmation = False

if not st.session_state.username:
    st.title("请选择或输入用户名")
    existing_users = [
        f.replace("conversations_", "").replace(".json", "")
        for f in os.listdir(CONVERSATIONS_DIR)
        if f.startswith("conversations_") and f.endswith(".json")
    ]
    if existing_users:
        selected_user = st.selectbox("已有用户：", existing_users)
        if st.button("加载已有用户"):
            st.session_state.username = selected_user
            st.session_state.show_delete_confirmation = False
            st.rerun()
    new_user = st.text_input("或输入新用户名（仅限字母、数字、下划线）：")
    if st.button("使用新用户名"):
        if new_user and is_valid_username(new_user):
            st.session_state.username = new_user
            st.session_state.show_delete_confirmation = False
            st.rerun()
        else:
            st.error("用户名无效或为空（仅限字母、数字、下划线）！")
else:
    # ------------------- 初始化会话状态 -------------------
    if "conversations" not in st.session_state or st.session_state.conversations is None:
        st.session_state.conversations = load_conversations(st.session_state.username)
        if not st.session_state.conversations or st.session_state.conversations == {}:
            default_id = "default"
            st.session_state.conversations = {
                default_id: {
                    "title": "新对话",
                    "messages": [
                        {"role": "system", "content": build_system_prompt(st.session_state.username)},
                        {"role": "assistant", "content": "你好，我是小锐助手，有什么需要帮助的吗？"}
                    ]
                }
            }
            st.info("初始化新会话")
        st.session_state.current_session = list(st.session_state.conversations.keys())[0]
        save_conversations(st.session_state.username)

    # ------------------- DeepSeek API（流式输出） -------------------
    def call_deepseek_api_stream(messages, context, api_key=None):
        """流式生成回答的API调用"""
        try:
            messages_to_send = list(messages)
            if context:
                messages_to_send.insert(-1, {
                    "role": "system",
                    "content": f"[检索到的相关知识库内容，仅供参考：{context[:60000]}]"
                })
            
            # 优先使用传入的 api_key，否则使用全局变量
            key = api_key if api_key else DEEPSEEK_API_KEY
            
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {key}"
            }
            
            json_data = {
                "model": DEEPSEEK_MODEL,
                "messages": messages_to_send,
                "temperature": 0.3,
                "max_tokens": 1200,
                "stream": True  # 启用流式输出
            }
            
            response = requests.post(
                f"{DEEPSEEK_API_BASE}/chat/completions",
                headers=headers,
                json=json_data,
                stream=True,
                timeout=120
            )
            response.raise_for_status()
            
            # 流式读取响应
            full_response = ""
            for line in response.iter_lines():
                if line:
                    line_text = line.decode('utf-8')
                    if line_text.startswith("data: "):
                        data_str = line_text[6:]
                        if data_str == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                if "content" in delta:
                                    content = delta["content"]
                                    full_response += content
                                    yield content
                        except json.JSONDecodeError:
                            continue
            
            if full_response:
                yield "__DONE__"  # 标记完成
            else:
                yield "抱歉，AI 服务暂时不可用，请稍后重试。"
                yield "__DONE__"
                
        except Exception as e:
            st.error(f"API 调用失败: {str(e)}")
            yield "API 调用失败，请稍后重试。"
            yield "__DONE__"

    # ------------------- DeepSeek API（带重试，非流式） -------------------
    def call_deepseek_api(messages, context):
        """生成回答的API调用（非流式，用于摘要等）"""
        try:
            messages_to_send = list(messages)
            if context:
                messages_to_send.insert(-1, {
                    "role": "system",
                    "content": f"[检索到的相关知识库内容，仅供参考：{context[:60000]}]"
                })
            
            full_prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages_to_send])
            
            result = call_deepseek_api_retry(
                prompt=full_prompt,
                max_tokens=1200,
                temperature=0.3,
                timeout=60
            )
            
            if result:
                return result
            else:
                return "抱歉，AI 服务暂时不可用，请稍后重试。"
                
        except Exception as e:
            st.error(f"API 调用失败: {str(e)}")
            return "API 调用失败，请稍后重试。"

    # ------------------- 重排序 -------------------
    def rerank(query, candidates, top_k=5):
        if not reranker_tokenizer or not reranker_model or not candidates:
            return candidates[:top_k]
        try:
            import torch
            pairs = [[query, c] for c in candidates]
            encoded = reranker_tokenizer(pairs, padding=True, truncation=True,
                                         max_length=512, return_tensors="pt")
            with torch.no_grad():
                scores = reranker_model(**encoded).logits.squeeze(-1).tolist()
            ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
            return [c for _, c in ranked[:top_k]]
        except Exception:
            return candidates[:top_k]

    # ------------------- 混合检索（知识库 + 历史回退 + RRF融合） -------------------
    def retrieve_context(query, username, history_context="", need_full_retrieval=True):
        """
        检索上下文 - 完整流程：
        1. 历史摘要向量检索（快速定位话题）
        2. 判断摘要是否足够
        3. 如果不足，会话JSON + 关键词匹配 + RRF融合
        4. 合并知识库检索结果
        """
        results = []
        
        # Step 1: 如果有当前会话上下文，标记来源
        if history_context:
            results.append(f"[当前会话上下文]\n{history_context[:800]}")
        
        # Step 2: 使用混合检索（摘要 + 会话细节 + RRF）
        if need_full_retrieval:
            # 混合历史检索
            history_search_result = hybrid_history_search(query, username)
            
            # 添加检索结果（带来源标记）
            for item in history_search_result.get("results", []):
                content = item.get("content", "")
                source = item.get("source", item.get("type", "history"))
                session_id = item.get("session_id", "")
                
                if source == "summary":
                    results.append(f"[历史摘要] {content}")
                elif source == "vector":
                    results.append(f"[历史会话-向量] {content}")
                elif source == "keyword":
                    keywords = item.get("matched_keywords", [])
                    results.append(f"[历史会话-关键词:{','.join(keywords)}] {content}")
                else:
                    results.append(f"[历史对话] {content}")
        
        # Step 3: 知识库检索（向量 + BM25）
        vector_texts = []
        if vectorstore:
            vector_docs = vectorstore.similarity_search(query, k=6)
            vector_texts = [d.page_content for d in vector_docs]

        bm25_texts = []
        if bm25_index and bm25_docs:
            scores = bm25_index.get_scores(list(query))
            top_indices = np.argsort(scores)[::-1][:6]
            bm25_texts = [bm25_docs[i] for i in top_indices if scores[i] > 0]

        seen = set()
        merged = []
        for t in vector_texts + bm25_texts:
            key = t[:100]
            if key not in seen:
                seen.add(key)
                merged.append(t)
        
        knowledge_results = rerank(query, merged, top_k=5)
        
        # 合并知识库结果
        for text in knowledge_results:
            results.append(f"[知识库] {text}")
        
        return results

    # ------------------- 多轮感知检索：增强版 Query Rewriting -------------------
    def rewrite_query(user_input, recent_messages, username):
        """增强版Query改写：融合长期记忆 + 历史摘要 + 当前会话"""
        
        # 0. 常量定义
        MAX_TURNS_FOR_DIRECT = 5          # 轮数阈值
        MAX_TOKENS_FOR_DIRECT = 800        # Token阈值（中文约1字=1token）
        
        # 1. 获取长期记忆 facts
        user_facts = load_long_term_memory(username)
        facts_context = ""
        if user_facts:
            facts_context = "\n【用户的已知偏好/信息】：\n" + "\n".join(f"- {f}" for f in user_facts)
        
        # 2. 获取历史会话摘要
        summaries_data = get_user_summaries(username)
        summaries_context = ""
        if summaries_data:
            summary_texts = [s["summary"] for s in summaries_data[-3:]]
            summaries_context = "\n【该用户的历史会话摘要】：\n" + "\n".join(f"- {s}" for s in summary_texts)
        
        # 3. 当前会话处理：轮数<=5 且 token<800 → 用原始对话；否则用摘要
        dialogue = [m for m in recent_messages if m["role"] in ("user", "assistant")]
        history_text = ""
        history_source = "direct"  # 标记来源
        
        # 计算对话总token数（简单估算：中文约1字=1token）
        total_chars = sum(len(m.get("content", "")) for m in dialogue)
        total_tokens_est = total_chars // 2  # 中文字符转token估算
        
        # 判断：轮数<=5 且 token<阈值 → 用原始对话
        if len(dialogue) <= MAX_TURNS_FOR_DIRECT and total_tokens_est <= MAX_TOKENS_FOR_DIRECT:
            # 对话短，用原始对话
            history_text = "\n【当前对话内容】：\n" + "\n".join(
                f"{'用户' if m['role']=='user' else '助手'}: {m['content'][:500]}"
                for m in dialogue
            )
        else:
            # 对话长或token多，用摘要
            summary = generate_session_summary(recent_messages, st.session_state.current_session)
            if summary:
                history_text = "\n【当前对话摘要】：\n" + summary
                history_source = "summary"
                # 同时保存摘要到向量库
                save_to_history_vectorstore(username, [summary], "current_summary")
        
        # 4. 构造改写Prompt
        prompt = (
            f"你需要根据上下文，补全用户最新问题中的指代词和省略内容。\n\n"
            f"{facts_context}"
            f"{summaries_context}"
            f"{history_text}\n\n"
            f"【用户最新问题】：{user_input}\n\n"
            "请将问题改写为一个独立、完整的检索查询，只返回改写后的查询，不要解释。"
            "如果问题已经完整，直接返回原问题。"
        )
        
        rewritten = call_deepseek_api_retry(
            prompt=prompt,
            max_tokens=100,
            timeout=30
        )
        
        if rewritten:
            return rewritten, history_source
        return user_input, "error"

    def check_if_answered_by_history(query, username, rewritten_query, history_context):
        """检查问题是否可以被历史上下文直接回答"""
        # 如果有足够的当前会话上下文，可能不需要额外检索
        if history_context and len(history_context) > 50:
            prompt = (
                f"基于以下上下文，判断能否直接回答用户问题。\n\n"
                f"上下文：\n{history_context[:800]}\n\n"  # 限制长度
                f"用户问题：{query}\n\n"
                "如果上下文能回答问题，返回'是'；如果需要更多外部知识，返回'否'。"
                "只返回'是'或'否'，不要其他内容。"
            )
            
            result = call_deepseek_api_retry(
                prompt=prompt,
                max_tokens=50,
                timeout=30
            )
            
            return result == "是" if result else False
        return False

    # ------------------- 侧边栏：API Key 管理 -------------------
    with st.sidebar:
        st.header("🔑 API Key 设置")
        
        # 初始化 API Key 相关 session_state
        if "api_key_input" not in st.session_state:
            st.session_state.api_key_input = ""
        if "show_api_key" not in st.session_state:
            st.session_state.show_api_key = False
        
        # 尝试加载已保存的 API Key
        saved_api_key = load_api_key(st.session_state.username)
        
        if saved_api_key:
            st.success("✅ 已保存 API Key")
            # 显示 API Key（可切换）
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("显示" if not st.session_state.show_api_key else "隐藏", key="toggle_show_key"):
                    st.session_state.show_api_key = not st.session_state.show_api_key
                    st.rerun()
            with col2:
                if st.button("🗑️清除", key="clear_api_key"):
                    delete_api_key(st.session_state.username)
                    st.session_state.api_key_input = ""
                    st.session_state.show_api_key = False
                    st.rerun()
            
            # 显示/隐藏 API Key
            display_key = saved_api_key[:8] + "..." + saved_api_key[-4:] if len(saved_api_key) > 12 else saved_api_key
            if st.session_state.show_api_key:
                st.text(f"当前 Key: {saved_api_key}")
            else:
                st.text(f"当前 Key: {display_key}")
        
        # API Key 输入
        st.subheader("输入新的 API Key")
        api_key_input = st.text_input(
            "DeepSeek API Key",
            value=st.session_state.api_key_input,
            type="password" if not st.session_state.show_api_key else "default",
            placeholder="sk-...",
            key="api_key_text_input"
        )
        
        col_save, col_cancel = st.columns(2)
        with col_save:
            if st.button("💾保存", key="save_api_key_btn"):
                if api_key_input and api_key_input.strip():
                    if save_api_key(st.session_state.username, api_key_input.strip()):
                        st.session_state.api_key_input = ""
                        st.success("API Key 保存成功！")
                        st.rerun()
                else:
                    st.warning("请输入有效的 API Key")
        with col_cancel:
            if st.button("🔄重置", key="reset_api_key_btn"):
                st.session_state.api_key_input = ""
                st.rerun()
        
        st.divider()
        
        # 使用保存的或当前输入的 API Key
        current_api_key = saved_api_key if saved_api_key else api_key_input
        if current_api_key:
            # 更新全局 API Key（确保所有 API 调用都能使用）
            DEEPSEEK_API_KEY = current_api_key
        else:
            # 如果没有 API Key，设置默认值（让代码不报错，但实际调用会失败）
            DEEPSEEK_API_KEY = ""
        
        st.header(f"💬{st.session_state.username}的对话历史")
        if st.button("新建对话", key="new_chat"):
            # 对当前会话生成摘要并存入向量库
            old_session_id = st.session_state.current_session
            if old_session_id and old_session_id in st.session_state.conversations:
                old_messages = st.session_state.conversations[old_session_id]["messages"]
                summary = generate_session_summary(old_messages, old_session_id)
                if summary:
                    save_to_history_vectorstore(st.session_state.username, [summary], "summary")
            
            # 生成唯一 ID（使用时间戳避免重复）
            import uuid
            new_id = f"chat_{int(time.time() * 1000)}"
            st.session_state.conversations[new_id] = {
                "title": f"对话 {len(st.session_state.conversations) + 1}",
                "messages": [
                    {"role": "system", "content": build_system_prompt(st.session_state.username)},
                    {"role": "assistant", "content": "你好，我是小锐助手，有什么需要帮助的吗？"}
                ]
            }
            st.session_state.current_session = new_id
            save_conversations(st.session_state.username)
            st.rerun()

        st.subheader("对话列表")
        # 可点击的对话列表，每个后面带删除按钮
        for session_id in list(st.session_state.conversations.keys()):
            session_title = st.session_state.conversations[session_id]["title"]
            col1, col2 = st.columns([4, 1])
            with col1:
                # 当前选中的对话高亮显示
                is_selected = session_id == st.session_state.current_session
                btn_label = f"▶ {session_title}" if is_selected else session_title
                if st.button(btn_label, key=f"select_{session_id}", use_container_width=True):
                    st.session_state.current_session = session_id
                    st.rerun()
            with col2:
                if st.button("🗑️", key=f"delete_{session_id}", help=f"删除「{session_title}」"):
                    # 直接在内存中删除，不需要调用 delete_session
                    if session_id in st.session_state.conversations:
                        del st.session_state.conversations[session_id]
                        
                        # 切换到第一个可用会话
                        if st.session_state.conversations:
                            st.session_state.current_session = list(st.session_state.conversations.keys())[0]
                        else:
                            # 如果没有会话了，创建新的
                            new_id = "default"
                            st.session_state.conversations = {
                                new_id: {
                                    "title": "新对话",
                                    "messages": [
                                        {"role": "system", "content": build_system_prompt(st.session_state.username)},
                                        {"role": "assistant", "content": "你好，我是小锐助手，有什么需要帮助的吗？"}
                                    ]
                                }
                            }
                            st.session_state.current_session = new_id
                        
                        # 同时删除文件中的数据和向量库
                        delete_session(st.session_state.username, session_id)
                        save_conversations(st.session_state.username)
                        st.rerun()
                    else:
                        st.error("会话不存在")

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
                        {"role": "system", "content": build_system_prompt(st.session_state.username)},
                        {"role": "assistant", "content": "你好，我是小锐助手，有什么需要帮助的吗？"}
                    ]
                }
            }
            st.session_state.current_session = "default"
            save_conversations(st.session_state.username)
            st.rerun()

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
            st.session_state.show_delete_confirmation = False
            st.rerun()

    # ------------------- 聊天界面 -------------------
    st.title(f"💡锐瞳智能科技公司——小锐智能体（欢迎，{st.session_state.username}）")
    
    # 检查 API Key 是否已设置
    saved_api_key = load_api_key(st.session_state.username)
    if not saved_api_key and not st.session_state.get("api_key_input"):
        st.warning("⚠️ 请先在左侧侧边栏设置 API Key 才能使用聊天功能")
        st.stop()
    
    current_api_key = saved_api_key if saved_api_key else st.session_state.get("api_key_input", "")
    if not current_api_key:
        st.warning("⚠️ 请先在左侧侧边栏设置 API Key 才能使用聊天功能")
        st.stop()
    
    # 更新全局 API Key
    DEEPSEEK_API_KEY = current_api_key
    
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
            # 获取当前会话对话
            recent = [m for m in current_messages[:-1] if m["role"] in ("user", "assistant")]
            
            # 优化：计算一次上下文，避免重复生成摘要
            total_chars = sum(len(m.get("content", "")) for m in recent)
            total_tokens_est = total_chars // 2
            use_direct = len(recent) <= 5 and total_tokens_est <= 800
            
            if use_direct:
                # 轮数<=5 且 token<800，用原始对话
                history_context = "\n".join(
                    f"{'用户' if m['role']=='user' else '助手'}: {m['content']}"
                    for m in recent
                )
            else:
                # 轮数多或token多，生成摘要
                with st.spinner("正在生成摘要..."):
                    summary = generate_session_summary(current_messages, st.session_state.current_session)
                    history_context = summary if summary else ""
            
            # Step 1: Query改写（简化版，不重复生成摘要）
            search_query = user_input  # 默认使用原问题
            if history_context or get_user_summaries(st.session_state.username):
                # 只有在有上下文时才改写
                user_facts = load_long_term_memory(st.session_state.username)
                facts_context = "\n【用户偏好】：" + "\n".join(f"- {f}" for f in user_facts) if user_facts else ""
                
                prompt = (
                    f"{facts_context}\n\n"
                    f"对话历史：\n{history_context[:1000]}\n\n"
                    f"用户问题：{user_input}\n\n"
                    "请补全问题中的指代词，只返回改写后的问题。"
                )
                
                result = call_deepseek_api_retry(prompt=prompt, max_tokens=100, timeout=30)
                if result:
                    search_query = result
            
            # Step 2: 直接检索上下文
            with st.spinner("正在检索知识库..."):
                text_docs = retrieve_context(search_query, st.session_state.username, history_context, need_full_retrieval=True)
                context_str = "\n".join(text_docs) if text_docs else None
            
            # 流式输出回答
            reply = ""
            message_placeholder = st.empty()
            for chunk in call_deepseek_api_stream(current_messages, context_str, api_key=current_api_key):
                if chunk == "__DONE__":
                    break
                reply += chunk
                message_placeholder.write(reply + "▌")  # 闪烁光标效果
            message_placeholder.write(reply)  # 最终显示

        current_messages.append({"role": "assistant", "content": reply})
        save_conversations(st.session_state.username)

        # 每 3 轮提取一次长期记忆
        user_msg_count = sum(1 for m in current_messages if m["role"] == "user")
        if user_msg_count % 3 == 0:
            extract_and_update_memory(st.session_state.username, current_messages)
        
        # 检查是否需要生成会话摘要
        if user_msg_count == SESSION_SUMMARY_THRESHOLD:
            summary = generate_session_summary(current_messages, st.session_state.current_session)
            if summary:
                save_to_history_vectorstore(st.session_state.username, [summary], "summary")

    # ------------------- 操作指南 -------------------
    if st.checkbox("操作指南"):
        st.write("查找锐瞳科技相关信息，请咨询小锐")
