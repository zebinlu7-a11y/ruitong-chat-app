# 锐瞳智能助手 - 小锐

基于 Streamlit 和 DeepSeek API 的智能客服系统，支持知识库检索、多轮对话、历史记忆等功能。

## 功能特性

### 核心功能
- **智能问答**：基于 DeepSeek 大模型，理解并回答用户问题
- **知识库检索**：结合本地知识库，提供精准的 RAG 检索增强
- **多轮对话**：支持上下文理解，实现连贯的多轮交互
- **流式输出**：实时流式展示 AI 生成的回答

### 用户管理
- **多用户支持**：支持多个用户独立使用，数据隔离
- **会话管理**：新建、切换、删除会话
- **对话持久化**：对话自动保存到本地，重启后可继续

### AI 功能
- **历史摘要**：自动生成对话摘要，加速长对话处理
- **长期记忆**：记住用户的偏好和关注点
- **Query 改写**：智能补全指代词和省略内容
- **编辑重生成**：支持编辑问题并重新生成回答

### API Key 管理
- **独立配置**：每个用户可配置自己的 API Key
- **安全存储**：API Key 使用 base64 编码存储
- **一键保存/清除**：方便管理 API Key

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

首次使用需要在应用界面中输入 DeepSeek API Key：
1. 在侧边栏找到「API Key 设置」
2. 输入你的 DeepSeek API Key
3. 点击「保存」按钮
4. 下次启动会自动加载保存的 Key

### 3. 启动应用

```bash
streamlit run 云端app.py
```

## 项目结构

```
ruitong-app/
├── 云端app.py          # 主应用代码
├── ingest.py           # 知识库导入脚本
├── clip_embeddings.py  # CLIP 向量生成
├── requirements.txt     # Python 依赖
├── conversations/      # 对话数据存储
│   ├── api_keys.json   # API Key 存储
│   ├── conversations_*.json  # 用户对话记录
│   └── memory_*.json   # 用户长期记忆
└── models/             # 模型文件
    ├── ruitongkeji/   # 知识库向量库
    ├── BAAI/          # 嵌入模型
    └── history_vectorstore/  # 历史对话向量库
```

## 知识库导入

如需更新知识库内容，运行：

```bash
python ingest.py
```

## 技术架构

- **前端**：Streamlit
- **后端**：Python
- **向量数据库**：Chroma
- **嵌入模型**：BAAI/bge-small-zh-v1.5
- **大模型**：DeepSeek Chat API
- **检索增强**：BM25 + 向量检索混合

## 注意事项

- API Key 需要自行申请，建议前往 [DeepSeek 开放平台](https://platform.deepseek.com/)
- 知识库向量库位于 `models/ruitongkeji/` 目录
- 对话数据保存在 `conversations/` 目录
