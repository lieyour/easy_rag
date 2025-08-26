"""
多PDF文档的RAG系统
将多pdf统一加载到向量库
支持检索排序, 来源感知
"""
# 标准库导入
import os
import io
import re
import base64
import json

# 第三方库导入
from dotenv import load_dotenv
from PIL import Image
from openai import OpenAI
import fitz  # PyMuPDF
import pdfplumber

# LangChain相关导入
from typing import Annotated, TypedDict, List, Any, Dict, Optional
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# LangGraph相关导入
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages

# 检查点导入
try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
except ImportError:
    SqliteSaver = None  # type: ignore


load_dotenv()

# ===========================================
# 配置参数
# ===========================================
CONFIG = {
    "embedding_model": "text-embedding-v1",
    "chat_model": "deepseek-chat",
    "vision_model": "qwen-vl-plus",
    "checkpoint_file": "rag_sessions.db",
    "vector_store_path": "vector_store",
    "chunk_size": 1500,
    "chunk_overlap": 200,
    "max_chunk_length": 2000,
    "truncate_length": 1900,
    "retrieval_k": 10,
    "mmr_lambda": 0.25,
    "context_max_length": 4000,
    "text_threshold": 100,  # 判断是否需要OCR的文字数量阈值
    "preview_pages": 3  # 分类时检查的页数
}

# 初始化组件
embedding = DashScopeEmbeddings(model=CONFIG["embedding_model"])
checkpoint_file = CONFIG["checkpoint_file"]

# 初始化检查点存储
if SqliteSaver:
    try:
        memory = SqliteSaver.from_conn_string(f"sqlite:///{checkpoint_file}")
    except (AttributeError, TypeError):
        memory = SqliteSaver(checkpoint_file)
else:
    memory = None  # type: ignore

# ===========================================
# 工具函数
# ===========================================
def clear_session_data():
    """清理所有会话数据"""
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"✅ 已清理会话数据文件: {checkpoint_file}")
    else:
        print(f"📝 会话数据文件不存在: {checkpoint_file}")

# ===========================================
# 状态定义
# ===========================================
class RagGraph(TypedDict):
    input: str
    pdf_path: str
    pdf_content: Optional[str]
    chunks: Optional[List[Document]]
    vector_store: Optional[Any]
    retrieved_docs: Optional[List[Document]]
    output: Optional[str]
    cache_exists: bool
    pdf_classification: Dict[str, str]
    messages: Annotated[List[dict], add_messages]
    user_id: str

# ===========================================
# PDF处理辅助函数
# ===========================================
def extract_with_qwenvl(image: Image.Image) -> Dict[str, Any]:
    """使用QwenVL进行OCR和表格提取"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    completion = client.chat.completions.create(
        model=CONFIG["vision_model"],
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "请分析这张COA图片，提取检验项目、标准、结果等信息。以JSON格式返回：{\"text\": \"文本内容\", \"tables\": [{\"headers\": [...], \"rows\": [...]}]}"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
            ]
        }]
    )
    
    response_content = completion.choices[0].message.content  # type: ignore
    if response_content:
        try:
            result = json.loads(response_content)
        except json.JSONDecodeError:
            result = {"text": "", "tables": []}
    else:
        result = {"text": "", "tables": []}
    return {"text": result.get("text", ""), "tables": result.get("tables", [])}

# ===========================================
# 核心处理函数
# ===========================================
def pdf_classify(state: RagGraph) -> RagGraph:
    """分类PDF文档: 判断是否需要OCR识别"""
    print("🔍 开始PDF文档分类...")
    pdf_folder = state['pdf_path']
    classification = {}
    
    for pdf_name in os.listdir(pdf_folder):
        if not pdf_name.endswith('.pdf'):
            continue
            
        pdf_path = os.path.join(pdf_folder, pdf_name)
        print(f"📄 正在分类: {pdf_name}")
        
        try:
            with pdfplumber.open(pdf_path) as reader:
                text_content = ""
                # 只检查前几页来判断
                for page in reader.pages[:CONFIG["preview_pages"]]:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text
            
            # 根据文本量判断处理方式
            if len(text_content.strip()) > CONFIG["text_threshold"]:
                classification[pdf_name] = "text_extractable"
                print(f"  ✅ {pdf_name} - 可直接提取文字")
            else:
                classification[pdf_name] = "vision_needed"
                print(f"  🔍 {pdf_name} - 需要视觉识别")
                    
        except Exception as e:
            print(f"  ❌ {pdf_name} - 分类失败: {e}")
            classification[pdf_name] = "vision_needed"
    
    state['pdf_classification'] = classification
    print(f"📊 分类完成，共 {len(classification)} 个文档")
    return state


def process_pdfs(state: RagGraph) -> RagGraph:
    """统一处理PDF文档, 根据分类结果选择处理方式"""
    all_text = ""
    pdf_folder = state['pdf_path']
    classification = state.get('pdf_classification', {})
    
    for pdf_name in os.listdir(pdf_folder):
        if not pdf_name.endswith('.pdf'):
            continue
            
        pdf_path = os.path.join(pdf_folder, pdf_name)
        process_method = classification.get(pdf_name, "vision_needed")
        
        if process_method == "text_extractable":
            # 使用pdfplumber提取文本
            print(f"📖 使用pdfplumber处理: {pdf_name}")
            try:
                with pdfplumber.open(pdf_path) as reader:
                    for page in reader.pages:  # type: ignore
                        page_text = page.extract_text()  # type: ignore
                        if page_text:
                            all_text += f"[来源:{pdf_name}|页码:{page.page_number}]\n{page_text}\n\n"  # type: ignore
                    all_text += f"[END:{pdf_name}]\n"
            except Exception as e:
                print(f"❌ pdfplumber处理失败: {e}")
                
        elif process_method == "vision_needed":
            # 使用QwenVL进行OCR
            print(f"🔍 使用QwenVL处理: {pdf_name}")
            try:
                with fitz.open(pdf_path) as doc:  # type: ignore
                    for page_num in range(len(doc)):
                        page = doc[page_num]
                        
                        # 转换为图像并进行OCR
                        mat = fitz.Matrix(2.0, 2.0)  # type: ignore
                        pix = page.get_pixmap(matrix=mat)  # type: ignore
                        img_data = pix.tobytes("png")  # type: ignore
                        image = Image.open(io.BytesIO(img_data))
                        
                        result = extract_with_qwenvl(image)
                        if result["text"]:
                            all_text += f"[来源:{pdf_name}|页码:{page_num + 1}]\n{result['text']}\n\n"
                        
                        # 处理表格数据
                        if result["tables"]:
                            for table in result["tables"]:
                                table_text = "\n".join(["\t".join(row) for row in [table.get("headers", [])] + table.get("rows", [])])
                                all_text += f"[来源:{pdf_name}|页码:{page_num + 1}|表格]\n{table_text}\n\n"
                    all_text += f"[END:{pdf_name}]\n"
            except Exception as e:
                print(f"❌ QwenVL处理失败: {e}")
    
    state['pdf_content'] = all_text
    print(f"✅ PDF处理完成，总长度: {len(all_text)} 字符")
    return state

def check_cache(state: RagGraph) -> RagGraph:
    """检查向量存储缓存是否存在"""
    cache_path = CONFIG["vector_store_path"]
    cache_exists = os.path.exists(cache_path) and os.path.exists(f"{cache_path}/index.faiss")
    state["cache_exists"] = cache_exists
    print("✅ 发现已有向量存储缓存" if cache_exists else "🔄 未发现缓存，需要处理文档")
    return state

def get_chunks(state: RagGraph) -> RagGraph:
    """使用RecursiveCharacterTextSplitter分块, 保留元数据"""
    raw_text = state["pdf_content"]
    if not raw_text:
        state["chunks"] = []
        print("📝 没有文档内容需要分块")
        return state
    
    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CONFIG["chunk_size"],
        chunk_overlap=CONFIG["chunk_overlap"],
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ";", ":", "，", " ", ""]
    )
    
    # 按文档分割
    doc_sections = raw_text.split("[END:")
    documents = []
    
    for section in doc_sections:
        if not section.strip():
            continue
            
        # 提取来源信息
        source_matches = re.findall(r'\[来源:([^|]+)\|页码:(\d+)\]', section)
        
        if source_matches:
            pdf_name = source_matches[0][0]
            chunks = text_splitter.split_text(section)
            
            for chunk in chunks:
                # 为每个chunk提取页码信息
                page_matches = re.findall(r'\[来源:[^|]+\|页码:(\d+)\]', chunk)
                page_num = page_matches[0] if page_matches else "未知"
                
                # 创建Document对象，包含元数据
                doc = Document(
                    page_content=chunk,
                    metadata={
                        "source": pdf_name,
                        "page": page_num,
                        "chunk_id": len(documents)
                    }
                )
                documents.append(doc)
    
    state["chunks"] = documents
    print(f"📝 文档已分块，共 {len(documents)} 个块")
    return state

def vector_store_func(state: RagGraph) -> RagGraph:
    """构建向量存储, 处理Document对象"""
    documents = state.get("chunks") or []
    
    # 处理过长文档
    valid_docs = []
    for doc in documents:
        if len(doc.page_content) <= CONFIG["max_chunk_length"]:
            valid_docs.append(doc)
        else:
            valid_docs.append(Document(
                page_content=doc.page_content[:CONFIG["truncate_length"]] + "...[文本被截断]",
                metadata=doc.metadata
            ))
    
    if not valid_docs:
        print("❌ 没有有效的文档块")
        return state
    
    # 构建向量存储
    vector_store = FAISS.from_documents(
        documents=valid_docs,
        embedding=embedding
    )
    vector_store.save_local(CONFIG["vector_store_path"])
    state["vector_store"] = None  # 标记向量存储已创建
    print(f"✅ 向量索引已构建，处理了 {len(valid_docs)} 个文档块")
    
    return state

def retrieve(state: RagGraph) -> RagGraph:
    """从向量存储中检索相关文档"""
    print("🔍 正在检索相关文档...")
    
    question = state.get("input", "")
    
    try:
        # 每次都从磁盘加载向量存储，避免序列化问题
        vector_store = FAISS.load_local(
            CONFIG["vector_store_path"], 
            embedding, 
            allow_dangerous_deserialization=True
        )
        print("✅ 成功加载已有向量存储")
        
        # 执行相似性搜索
        docs = vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={'k': CONFIG["retrieval_k"], 'lambda_mult': CONFIG["mmr_lambda"]}
        ).invoke(question)
        
        state["retrieved_docs"] = docs
        print(f"✅ 检索完成，找到 {len(docs)} 个相关文档")
        
        # 显示检索到的文档来源
        sources = set()
        for doc in docs:
            if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                sources.add(doc.metadata['source'])
        
        if sources:
            print(f"📚 文档来源: {', '.join(sources)}")
            
    except Exception as e:
        print(f"❌ 检索失败: {e}")
        state["retrieved_docs"] = []
    
    return state

def ai_answer(state: RagGraph) -> RagGraph:
    """生成AI答案, 基于检索文档和对话历史"""
    question = state.get("input", "")
    docs = state.get("retrieved_docs", []) or []
    messages = state.get("messages", [])
    
    # Build conversation history
    conversation_history = ""
    if len(messages) > 1:  # If there is historical dialogue
        for msg in messages[:-1]:  # Exclude current question
            # Handle both dict format and LangChain message objects
            if isinstance(msg, dict):
                # Dict format - 优先检查字典类型
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
            elif hasattr(msg, 'content') and not isinstance(msg, dict):
                # LangChain message object - 确保不是字典类型
                role = msg.__class__.__name__.replace('Message', '').lower()
                content = msg.content
            else:
                # Fallback
                role = "unknown"
                content = str(msg)
            conversation_history += f"{role}: {content}\n"
    
    # 构建包含历史的提示词
    # 处理文档内容提取，兼容不同的文档类型
    context_parts = []
    # 确保docs不为None
    if docs:
        for doc in docs:
            if hasattr(doc, 'page_content'):
                context_parts.append(doc.page_content)
            elif isinstance(doc, dict) and 'page_content' in doc:
                context_parts.append(doc['page_content'])
            elif isinstance(doc, str):
                context_parts.append(doc)
    
    context = "\n\n".join(context_parts)
    
    prompt = f"""对话历史: {conversation_history}相关文档: {context[:CONFIG["context_max_length"]]}当前问题: {question}要求:
1. 基于上述文档内容和对话历史回答
2. 用中文回答
3. 简洁明了"""
    
    llm = init_chat_model(CONFIG["chat_model"])
    response = llm.invoke(prompt)
    
    # 确保answer始终为字符串类型
    if hasattr(response, 'content'):
        content = response.content
        if isinstance(content, list):
            # 如果content是列表，将其转换为字符串
            answer = ' '.join(str(item) for item in content)
        else:
            answer = str(content) if content is not None else ""
    else:
        answer = str(response)
    
    # Add AI answer to message history - let LangGraph handle message objects automatically
    state["output"] = answer
    
    return state
# ===========================================
# 路由函数
# ===========================================
def should_process_document(state: RagGraph) -> str:
    """根据缓存状态决定下一步"""
    return "retrieve" if state["cache_exists"] else "pdf_classify"

# ===========================================
# 工作流构建
# ===========================================
def create_smart_rag():
    """创建智能RAG工作流"""
    workflow = StateGraph(RagGraph)
    
    # 添加节点
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("pdf_classify", pdf_classify)
    workflow.add_node("process_pdfs", process_pdfs)
    workflow.add_node("get_chunks", get_chunks)
    workflow.add_node("vector_store", vector_store_func)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("ai_answer", ai_answer)
    
    # 添加边和条件边
    workflow.add_conditional_edges(
        "check_cache",
        should_process_document,
        {"retrieve": "retrieve", "pdf_classify": "pdf_classify"}
    )
    
    # 简化的线性流程
    workflow.add_edge("pdf_classify", "process_pdfs")
    workflow.add_edge("process_pdfs", "get_chunks")
    workflow.add_edge("get_chunks", "vector_store")
    workflow.add_edge("vector_store", "retrieve")
    workflow.add_edge("retrieve", "ai_answer")
    
    workflow.set_entry_point("check_cache")
    workflow.set_finish_point("ai_answer")
    
    return workflow.compile(checkpointer=memory)

# ===========================================
# 主程序
# ===========================================
if __name__ == "__main__":
    # 配置参数
    pdf_path = r"C:\Users\Yu\Desktop\coa\COA"
    user_id = "user_123"

    # 检查是否清理会话数据
    clear_sessions = input("🗄️ 是否清理旧的会话数据? (y/N): ").strip().lower()
    if clear_sessions in ['y', 'yes', '是']:
        clear_session_data()

    # 初始化系统
    smart_rag = create_smart_rag()
    print("🤖 智能RAG系统启动（输入 'quit' 退出）")
    print("💡 线程功能已启用，系统会记住对话历史")
    
    config = RunnableConfig(configurable={"thread_id": user_id})

    # 主循环
    while True:
        question = input("\n❓ 请输入您的问题: ").strip()
        if question.lower() in ['quit', 'exit', '退出', 'q']:
            break

        if not question:
            continue

        # 构建状态
        state: RagGraph = {
            "input": question,
            "messages": [{"role": "user", "content": question}],
            "user_id": user_id,
            "pdf_path": pdf_path,
            "pdf_content": None,
            "chunks": None,
            "vector_store": None,
            "retrieved_docs": None,
            "output": None,
            "cache_exists": False,
            "pdf_classification": {},
        }

        try:
            result = smart_rag.invoke(state, config=config)
            print(f"\n🤖 回答: {result['output']}")
        except Exception as e:
            print(f"❌ 出错了: {e}")
