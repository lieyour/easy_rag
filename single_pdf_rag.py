from langgraph.graph import StateGraph
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from typing import TypedDict, List, Any
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()

embedding = DashScopeEmbeddings(
            model="text-embedding-v1", 
        )

class RagGraph(TypedDict):
    input: str
    pdf_path: str
    pdf_content: Any
    chunks: list
    vector_store: Any
    retrieved_docs: List[dict]
    output: str
    cache_exists: bool  # 新增：缓存状态标记


# 检查缓存状态
def check_cache(state: RagGraph):
    """检查向量存储缓存是否存在"""
    cache_exists = os.path.exists("vector_store") and os.path.exists("vector_store/index.faiss")
    state["cache_exists"] = cache_exists
    if cache_exists:
        print("✅ 发现已有向量存储缓存")
    else:
        print("🔄 未发现缓存，需要处理文档")
    return state

def pdf_read(state: RagGraph):
    print("正在阅读PDF文档...")
    try:
        text = ""
        reader = pdfplumber.open(state["pdf_path"])
        for page in reader.pages:
            text += page.extract_text()
        reader.close()
        state["pdf_content"] = text
    except Exception as e:
        print(f"读取PDF文档时出错: {e}")
        state["pdf_content"] = None
    return state
    

def get_chunks(state: RagGraph):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200
    )
    chunks = text_splitter.split_text(state["pdf_content"]) 
    state["chunks"] = chunks
    print("文档已分块")
    return state

def vector_store_func(state: RagGraph):
    vector_store = FAISS.from_texts(
        texts=state["chunks"],
        embedding=embedding
    )
    vector_store.save_local("vector_store")
    state["vector_store"] = vector_store
    print("向量索引已构建")
    return state

def retrieve(state: RagGraph):
    """统一的检索函数"""
    new_db = FAISS.load_local("vector_store", 
    embeddings = embedding,
    allow_dangerous_deserialization=True
    ) 
    retriever = new_db.as_retriever()
    docs = retriever.invoke(state["input"])
    print(f"检索到 {len(docs)} 个相关文档片段")
    state["retrieved_docs"] = docs
    return state

def ai_answer(state: RagGraph) -> RagGraph:
    """根据检索到的文档生成AI回答"""
    print("正在生成回答...")
    
    question = state.get("input", "")
    docs = state.get("retrieved_docs", [])
    
    context = "\n\n".join(
        doc.page_content if hasattr(doc, 'page_content') 
        else doc.get('content', str(doc)) 
        for doc in docs
    ) if docs else ""

    try:
        if context:
            llm = init_chat_model("deepseek-chat")
            prompt = f"""根据以下上下文回答问题：
            
{context[:3000]}

问题：{question}
要求：
1. 基于上下文回答
2. 保持专业准确
3. 用中文回答"""
            
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, 'content') else str(response)
        else:
            answer = f"未找到与「{question}」相关的文档内容"
            
    except Exception as e:
        print(f"生成失败: {e}")
        if context:
            keywords = [w for w in question.replace('?', '').replace('？', '').split() if w]
            found = [s for s in context.split('。') if any(k in s for k in keywords)][:3]
            answer = "找到相关内容：\n" + "\n".join(set(found)) if found else "文档中未找到直接答案"
        else:
            answer = "无可用文档内容"
    
    state["output"] = answer
    return state

# 条件边函数：决定是否需要文档预处理
def should_process_document(state: RagGraph) -> str:
    """条件边：根据缓存状态决定下一步"""
    if state["cache_exists"]:
        return "retrieve"  # 有缓存，直接检索
    else:
        return "pdf_read"  # 无缓存，需要处理文档

# 创建智能RAG工作流
def create_smart_rag():
    workflow = StateGraph(RagGraph)
    
    # 添加所有节点
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("pdf_read", pdf_read)
    workflow.add_node("get_chunks", get_chunks)
    workflow.add_node("vector_store", vector_store_func)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("ai_answer", ai_answer)
    
    # 设置条件边：从缓存检查开始分支
    workflow.add_conditional_edges(
        "check_cache",
        should_process_document,
        {
            "retrieve": "retrieve",      # 有缓存直接检索
            "pdf_read": "pdf_read"       # 无缓存先处理文档
        }
    )
    
    # 文档处理流程的边
    workflow.add_edge("pdf_read", "get_chunks")
    workflow.add_edge("get_chunks", "vector_store")
    workflow.add_edge("vector_store", "retrieve")
    
    # 最终回答
    workflow.add_edge("retrieve", "ai_answer")
    
    # 设置入口和出口
    workflow.set_entry_point("check_cache")
    workflow.set_finish_point("ai_answer")
    
    return workflow.compile()

# 运行演示
if __name__ == "__main__":
    pdf_path = "D:\\Browser Download\\AppCacth\\Princess3\\princess.pdf"
    
    # 创建智能RAG代理
    smart_rag = create_smart_rag()
    
    print("🤖 智能RAG系统启动（输入 'quit' 退出）:")
    
    while True:
        question = input("\n❓ 请输入您的问题: ").strip()
        if question.lower() in ['quit', 'exit', '退出', 'q']:
            print("👋 再见！")
            break
        
        if not question:
            continue
            
        # 创建状态
        state: RagGraph = {
            "input": question,
            "pdf_path": pdf_path,
            "pdf_content": None,
            "chunks": None,
            "vector_store": None,
            "retrieved_docs": None,
            "output": None,
            "cache_exists": False
        }
        
        # 运行智能工作流
        try:
            result = smart_rag.invoke(state)
            print(f"\n🤖 回答: {result['output']}")
        except Exception as e:
            print(f"❌ 出错了: {e}")