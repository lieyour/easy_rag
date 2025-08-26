"""
多PDF文档的RAG系统
将多pdf统一加载到向量库
支持检索排序，来源感知
"""

from langgraph.graph import StateGraph
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from typing import TypedDict, List, Any, Dict
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import DashScopeEmbeddings
import pdfplumber
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import fitz  # PyMuPDF
from PIL import Image
import io
import base64
import json
from openai import OpenAI
from langchain.schema import Document
import re
import dashscope
from http import HTTPStatus
from datetime import datetime


load_dotenv()

embedding = DashScopeEmbeddings(model="text-embedding-v2")


class RagGraph(TypedDict):
    input: str
    pdf_path: str
    pdf_content: Any
    chunks: list
    vector_store: Any
    retrieved_docs: List[dict]
    output: str
    cache_exists: bool
    pdf_classification: Dict[str, str]  # PDF分类结果
    json_cache_exists: bool  # JSON缓存是否存在
    # 评估模式相关字段
    evaluation_mode: bool
    test_questions: List[Dict]
    current_question_index: int
    evaluation_results: List[Dict]
    metrics: Dict[str, float]


def convert_page_to_image(page: fitz.Page) -> Image.Image:
    """将PDF页面转换为PIL图像"""
    mat = fitz.Matrix(2.0, 2.0)
    pix = page.get_pixmap(matrix=mat)
    img_data = pix.tobytes("png")
    return Image.open(io.BytesIO(img_data))


def extract_with_qwenvl(image: Image.Image) -> Dict[str, Any]:
    """使用QwenVL进行OCR和表格提取"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen-vl-plus-latest",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": '请分析这张COA图片，提取检验项目、标准、结果等信息。以JSON格式返回：{"text": "文本内容", "tables": [{"headers": [...], "rows": [...]}]}',
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_base64}"},
                    },
                ],
            }
        ],
    )

    response_content = completion.choices[0].message.content

    result = json.loads(response_content)
    return {"text": result.get("text", ""), "tables": result.get("tables", [])}


def pdf_classify(state: RagGraph):
    """分类路由节点：遍历文件夹，对每个PDF进行分类"""
    print("🔍 开始PDF文档分类...")
    pdf_folder = state["pdf_path"]
    classification = {}

    for pdf_name in os.listdir(pdf_folder):
        if not pdf_name.endswith(".pdf"):
            continue

        pdf_path = os.path.join(pdf_folder, pdf_name)

        # 尝试使用pdfplumber提取文字
        with pdfplumber.open(pdf_path) as reader:
            text_content = ""
            for page in reader.pages[:3]:  # 只检查前3页
                page_text = page.extract_text()
                if page_text:
                    text_content += page_text

            # 判断是否能提取到足够的文字内容
        if len(text_content.strip()) > 100:  # 如果提取到超过100个字符
            classification[pdf_name] = "text_extractable"

        else:
            classification[pdf_name] = "vision_needed"
    state["pdf_classification"] = classification
    print(f"📊 分类完成，共 {len(classification)} 个文档")
    return state


def check_cache(state: RagGraph):
    """检查向量存储缓存是否存在"""
    cache_exists = os.path.exists("vector_store") and os.path.exists(
        "vector_store/index.faiss"
    )
    state["cache_exists"] = cache_exists
    print("✅ 发现已有向量存储缓存" if cache_exists else "🔄 未发现缓存，需要处理文档")
    return state


def check_json_cache(state: RagGraph):
    """检查PDF内容JSON缓存是否存在"""
    json_file_path = "pdf_content.json"
    json_exists = os.path.exists(json_file_path)

    if json_exists:
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                pdf_data = json.load(f)
                state["pdf_content"] = pdf_data.get("pdf_content", "")
                state["pdf_classification"] = pdf_data.get("pdf_classification", {})
            print(f"✅ 从JSON文件加载PDF内容，长度: {len(state['pdf_content'])} 字符")
            state["json_cache_exists"] = True
        except Exception as e:
            print(f"❌ 读取JSON文件失败: {e}")
            state["json_cache_exists"] = False
    else:
        print("🔄 未发现JSON缓存，需要重新处理PDF")
        state["json_cache_exists"] = False

    return state


def pdfread_qwen(state: RagGraph):
    """使用QwenVL读取PDF文档，直接保存到JSON文件"""
    try:
        all_text = ""
        pdf_folder = state["pdf_path"]
        classification = state.get("pdf_classification", {})

        # 先尝试从JSON文件读取现有内容
        json_file_path = "pdf_content.json"
        existing_data = {}
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    all_text = existing_data.get("pdf_content", "")
                print(f"📖 读取现有JSON文件，当前内容长度: {len(all_text)} 字符")
            except Exception as e:
                print(f"⚠️ 读取现有JSON文件失败: {e}，将创建新文件")

        for pdf_name in os.listdir(pdf_folder):
            # 只处理需要视觉识别的PDF
            if classification.get(pdf_name) != "vision_needed":
                continue

            pdf_path = os.path.join(pdf_folder, pdf_name)
            print(f"🔍 使用QwenVL处理: {pdf_name}")

            with fitz.open(pdf_path) as doc:
                for page_num in range(len(doc)):
                    page = doc[page_num]

                    # 转换为图像
                    image = convert_page_to_image(page)
                    result = extract_with_qwenvl(image)
                    if result["text"]:
                        # 添加来源标记
                        all_text += f"[来源:{pdf_name}|页码:{page_num + 1}]\n{result['text']}\n\n"

                    # 如果有表格数据，也添加进去
                    if result["tables"]:
                        for table in result["tables"]:
                            table_text = "\n".join(
                                [
                                    "\t".join(row)
                                    for row in [table.get("headers", [])]
                                    + table.get("rows", [])
                                ]
                            )
                            all_text += f"[来源:{pdf_name}|页码:{page_num + 1}|表格]\n{table_text}\n\n"

            # 添加文档结束标志
            all_text += f"[END:{pdf_name}]\n"

        # 保存到JSON文件
        from datetime import datetime

        pdf_data = {
            "pdf_content": all_text,
            "pdf_classification": classification,
            "timestamp": datetime.now().isoformat(),
            "total_length": len(all_text),
        }

        try:
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(pdf_data, f, ensure_ascii=False, indent=2)
            print(f"💾 PDF内容已保存到 {json_file_path}，总长度: {len(all_text)} 字符")
        except Exception as e:
            print(f"❌ 保存JSON文件失败: {e}")

        # 同时设置到state中
        state["pdf_content"] = all_text
        print(
            f"🔍 QwenVL处理完成，新增内容长度: {len(all_text) - len(existing_data.get('pdf_content', ''))} 字符"
        )

    except Exception as e:
        print(f"QwenVL处理PDF文档时出错: {e}")

    return state


def pdf_read(state: RagGraph):
    """读取PDF文档，直接保存到JSON文件"""
    try:
        all_text = ""
        pdf_folder = state["pdf_path"]
        classification = state.get("pdf_classification", {})

        # 先尝试从JSON文件读取现有内容
        json_file_path = "pdf_content.json"
        existing_data = {}
        if os.path.exists(json_file_path):
            try:
                with open(json_file_path, "r", encoding="utf-8") as f:
                    existing_data = json.load(f)
                    all_text = existing_data.get("pdf_content", "")
                print(f"📖 读取现有JSON文件，当前内容长度: {len(all_text)} 字符")
            except Exception as e:
                print(f"⚠️ 读取现有JSON文件失败: {e}，将创建新文件")

        for pdf_name in os.listdir(pdf_folder):
            if not pdf_name.endswith(".pdf"):
                continue

            # 只处理可直接提取文字的PDF
            if classification.get(pdf_name) != "text_extractable":
                continue

            pdf_path = os.path.join(pdf_folder, pdf_name)
            print(f"📖 使用pdfplumber处理: {pdf_name}")

            with pdfplumber.open(pdf_path) as reader:
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        # 在每个文档块前添加来源标记
                        all_text += f"[来源:{pdf_name}|页码:{page.page_number}]\n{page_text}\n\n"
                all_text += f"[END:{pdf_name}]\n"

        # 保存到JSON文件
        from datetime import datetime

        pdf_data = {
            "pdf_content": all_text,
            "pdf_classification": classification,
            "timestamp": datetime.now().isoformat(),
            "total_length": len(all_text),
        }

        try:
            with open(json_file_path, "w", encoding="utf-8") as f:
                json.dump(pdf_data, f, ensure_ascii=False, indent=2)
            print(f"💾 PDF内容已保存到 {json_file_path}，总长度: {len(all_text)} 字符")
        except Exception as e:
            print(f"❌ 保存JSON文件失败: {e}")

        state["pdf_content"] = all_text
        print(
            f"📖 pdfplumber处理完成，新增内容长度: {len(all_text) - len(existing_data.get('pdf_content', ''))} 字符"
        )
    except Exception as e:
        print(f"读取PDF文档时出错: {e}")
        state["pdf_content"] = None
    return state


def extract_product_info(text: str, chunk_text: str = None) -> Dict[str, str]:
    """从文本中提取所有元数据信息：来源、页码、产品编号、批次号"""
    import re

    # 提取来源信息
    source_matches = re.findall(r"\[来源:([^|]+)\|页码:(\d+)\]", text)
    pdf_name = source_matches[0][0] if source_matches else "未知"

    # 提取页码信息（优先从chunk_text中提取）
    page_num = "未知"
    if chunk_text:
        page_matches = re.findall(r"\[来源:[^|]+\|页码:(\d+)\]", chunk_text)
        if page_matches:
            page_num = page_matches[0]
    if page_num == "未知" and source_matches:
        page_num = source_matches[0][1]

    # 产品编号和批次号的模式
    product_patterns = [
        r"产品编号[：:]*\s*([A-Z0-9\-]+)",
        r"产品号[：:]*\s*([A-Z0-9\-]+)",
        r"Product\s*No[.：:]*\s*([A-Z0-9\-]+)",
        r"Item\s*No[.：:]*\s*([A-Z0-9\-]+)",
        r"货号[：:]*\s*([A-Z0-9\-]+)",
        r"编号[：:]*\s*([A-Z0-9\-]+)",
    ]

    batch_patterns = [
        r"批次号[：:]*\s*([A-Z0-9\-]+)",
        r"批号[：:]*\s*([A-Z0-9\-]+)",
        r"Batch\s*No[.：:]*\s*([A-Z0-9\-]+)",
        r"Lot\s*No[.：:]*\s*([A-Z0-9\-]+)",
        r"LOT[：:]*\s*([A-Z0-9\-]+)",
        r"生产批次[：:]*\s*([A-Z0-9\-]+)",
    ]

    # 提取产品编号（优先从chunk_text中提取）
    product_number = "未知"
    search_text = chunk_text if chunk_text else text
    for pattern in product_patterns:
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            product_number = match.group(1).strip()
            break

    # 如果chunk中没找到，再从整个文档中找
    if product_number == "未知" and chunk_text:
        for pattern in product_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                product_number = match.group(1).strip()
                break

    # 提取批次号（同样的策略）
    batch_number = "未知"
    for pattern in batch_patterns:
        match = re.search(pattern, search_text, re.IGNORECASE)
        if match:
            batch_number = match.group(1).strip()
            break

    if batch_number == "未知" and chunk_text:
        for pattern in batch_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                batch_number = match.group(1).strip()
                break

    return {
        "source": pdf_name,
        "page": page_num,
        "product_number": product_number,
        "batch_number": batch_number,
    }


def get_chunks(state: RagGraph):
    """从JSON文件读取内容并分块"""
    # 优先从JSON文件读取
    json_file_path = "pdf_content.json"
    raw_text = ""

    if os.path.exists(json_file_path):
        try:
            with open(json_file_path, "r", encoding="utf-8") as f:
                pdf_data = json.load(f)
                raw_text = pdf_data.get("pdf_content", "")
            print(f"📖 从JSON文件读取内容，长度: {len(raw_text)} 字符")
        except Exception as e:
            print(f"❌ 读取JSON文件失败: {e}")
            raw_text = state.get("pdf_content", "")
    else:
        raw_text = state.get("pdf_content", "")
        print("📝 使用state中的PDF内容")

    if not raw_text:
        state["chunks"] = []
        print("📝 没有文档内容需要分块")
        return state

    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", "。", "！", "？", ";", ":", "，", " ", ""],
    )

    # 按文档分割
    doc_sections = raw_text.split("[END:")
    documents = []

    for section in doc_sections:
        if not section.strip():
            continue

        # 分割文本
        chunks = text_splitter.split_text(section)

        for chunk in chunks:
            # 提取所有元数据信息
            metadata = extract_product_info(section, chunk)
            metadata["chunk_id"] = len(documents)

            # 创建Document对象
            doc = Document(page_content=chunk, metadata=metadata)
            documents.append(doc)

    state["chunks"] = documents
    print(f"📝 文档已分块，共 {len(documents)} 个块")

    # 显示提取的信息统计
    product_stats = {}
    batch_stats = {}
    for doc in documents:
        prod_num = doc.metadata.get("product_number", "未知")
        batch_num = doc.metadata.get("batch_number", "未知")
        product_stats[prod_num] = product_stats.get(prod_num, 0) + 1
        batch_stats[batch_num] = batch_stats.get(batch_num, 0) + 1

    print(
        f"📊 产品编号统计: {dict(list(product_stats.items())[:5])}{'...' if len(product_stats) > 5 else ''}"
    )
    print(
        f"📊 批次号统计: {dict(list(batch_stats.items())[:5])}{'...' if len(batch_stats) > 5 else ''}"
    )

    return state


def vector_store_func(state: RagGraph):
    """构建向量存储，处理Document对象"""
    documents = state["chunks"]

    # 处理过长文档
    valid_docs = []
    for doc in documents:
        if len(doc.page_content) <= 2000:
            valid_docs.append(doc)
        else:
            valid_docs.append(
                Document(
                    page_content=doc.page_content[:1900] + "...[文本被截断]",
                    metadata=doc.metadata,
                )
            )

    if not valid_docs:
        print("❌ 没有有效的文档块")
        return state

    # 构建向量存储
    vector_store = FAISS.from_documents(documents=valid_docs, embedding=embedding)
    vector_store.save_local("vector_store")
    state["vector_store"] = vector_store
    print(f"✅ 向量索引已构建，处理了 {len(valid_docs)} 个文档块")

    return state


def retrieve(state: RagGraph):
    """从向量存储中检索相关文档，使用交叉编码器重排"""
    print("🔍 正在检索相关文档...")

    question = state.get("input", "")
    vector_store = state.get("vector_store")

    if not vector_store:
        vector_store = FAISS.load_local(
            "vector_store", embedding, allow_dangerous_deserialization=True
        )
        state["vector_store"] = vector_store
        print("✅ 成功加载已有向量存储")

    try:
        # 第一步：使用相似性搜索获取更多候选文档
        candidate_docs = vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}  # 获取更多候选文档用于重排
        ).invoke(question)

        print(f"📋 获取到 {len(candidate_docs)} 个候选文档")

        # 第二步：使用qwen gte-rerank-v2进行重排
        if candidate_docs:
            reranked_docs = rerank_with_qwen(question, candidate_docs)
            # 取第一个重排后的文档
            state["retrieved_docs"] = reranked_docs[:3]
            print(f"✅ 重排完成，返回前 {len(state['retrieved_docs'])} 个相关文档")
        else:
            state["retrieved_docs"] = []
            print("❌ 未找到候选文档")

        # 显示检索到的文档来源
        sources = set()
        for doc in state["retrieved_docs"]:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                sources.add(doc.metadata["source"])

        if sources:
            print(f"📚 文档来源: {', '.join(sources)}")

    except Exception as e:
        print(f"❌ 检索失败: {e}")
        state["retrieved_docs"] = []

    return state


def rerank_with_qwen(
    query: str, documents: List[Document], top_k: int = 10
) -> List[Document]:
    """使用qwen gte-rerank-v2对文档进行重排"""
    try:
        # 设置API密钥
        dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")

        # 准备重排数据
        texts = []
        for doc in documents:
            content = doc.page_content if hasattr(doc, "page_content") else str(doc)
            # 限制文本长度，避免超过模型限制
            texts.append(content[:1000])

        # 调用重排API（按照官方示例）
        resp = dashscope.TextReRank.call(
            model="gte-rerank-v2",
            query=query,
            documents=texts,
            top_n=min(top_k, len(texts)),
            return_documents=True,
        )

        if resp.status_code == HTTPStatus.OK:
            # 根据重排结果重新排序文档
            reranked_docs = []
            for item in resp.output.results:
                index = item.index
                if index < len(documents):
                    reranked_docs.append(documents[index])

            print(f"🔄 重排成功，返回 {len(reranked_docs)} 个文档")
            return reranked_docs
        else:
            print(f"❌ 重排API调用失败: {resp.status_code}, {resp.message}")
            return documents[:top_k]  # 降级到原始排序

    except Exception as e:
        print(f"❌ 重排过程出错: {e}")
        return documents[:top_k]  # 降级到原始排序


def ai_answer(state: RagGraph) -> RagGraph:
    """根据检索到的文档生成AI回答，显示来源信息"""
    print("🤖 正在生成回答...")

    question = state.get("input", "")
    docs = state.get("retrieved_docs", [])

    # 构建上下文和来源信息
    context_parts = []
    sources = set()

    for doc in docs:
        content = doc.page_content if hasattr(doc, "page_content") else str(doc)
        context_parts.append(content)

        # 收集来源信息
        if hasattr(doc, "metadata"):
            source = doc.metadata.get("source", "未知")
            page = doc.metadata.get("page", "未知")
            sources.add(f"{source}(页码:{page})")

    context = "\n\n".join(context_parts)

    try:
        client = OpenAI(
            api_key=os.getenv("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        prompt = f"""根据以下文档回答问题：{context[:4000]}问题：{question}要求：1. 基于上述文档内容回答2. 用中文回答3. 简洁明了"""

        response = client.chat.completions.create(
            model="deepseek-v3",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        # 正确提取回答内容
        if (
            hasattr(response, "choices")
            and response.choices
            and len(response.choices) > 0
        ):
            answer = response.choices[0].message.content
        else:
            answer = "生成回答失败"

        # 添加来源信息
        sources_text = "\n📚 信息来源：" + "、".join(sources) if sources else ""
        answer += f"\n\n📊 检索统计：共检索到 {len(docs)} 个相关文档片段{sources_text}"

    except Exception as e:
        print(f"生成失败: {e}")
        sources_text = "、".join(sources) if sources else "未知"
        answer = f"找到相关内容来自：{sources_text}\n\n检索到 {len(docs)} 个文档片段，但AI生成失败。"

    state["output"] = answer
    return state


def load_test_questions(state: RagGraph):
    """加载测试问题集"""
    print("📋 加载测试问题集...")

    # 修复：直接使用项目目录中的文件
    current_dir = os.path.dirname(os.path.abspath(__file__))
    test_file = os.path.join(current_dir, "coa_question.json")

    with open(test_file, "r", encoding="utf-8") as f:
        data = json.load(f)
        state["test_questions"] = data["test_questions"]
        state["current_question_index"] = 0
        state["evaluation_results"] = []

    print(f"✅ 已加载 {len(state['test_questions'])} 个测试问题")
    return state


def evaluate_single_question(state: RagGraph):
    """评估单个问题"""
    questions = state["test_questions"]
    index = state["current_question_index"]

    if index >= len(questions):
        return state

    current_q = questions[index]
    question = current_q["question"]
    ground_truth = current_q["ground_truth"]
    relevant_docs = set(current_q["relevant_docs"])

    print(f"\n📝 评估问题 {index + 1}/{len(questions)}: {question[:50]}...")

    # 设置当前问题到state
    state["input"] = question

    # 执行检索
    state = retrieve(state)

    # 生成答案
    state = ai_answer(state)

    # 计算指标
    retrieved_sources = set()
    for doc in state.get("retrieved_docs", []):
        if hasattr(doc, "metadata") and "source" in doc.metadata:
            retrieved_sources.add(doc.metadata["source"])

    # 计算召回率和准确率
    if relevant_docs:
        recall = len(retrieved_sources & relevant_docs) / len(relevant_docs)
    else:
        recall = 0.0

    if retrieved_sources:
        precision = len(retrieved_sources & relevant_docs) / len(retrieved_sources)
    else:
        precision = 0.0

    # 计算F1分数
    if precision + recall > 0:
        f1 = 2 * (precision * recall) / (precision + recall)
    else:
        f1 = 0.0

    # 保存结果
    result = {
        "question": question,
        "ground_truth": ground_truth,
        "generated_answer": state["output"],
        "relevant_docs": list(relevant_docs),
        "retrieved_docs": list(retrieved_sources),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

    state["evaluation_results"].append(result)
    state["current_question_index"] += 1

    print(f"  📊 P: {precision:.3f}, R: {recall:.3f}, F1: {f1:.3f}")

    return state


def calculate_final_metrics(state: RagGraph):
    """计算最终评估指标"""
    print("\n📊 计算最终评估指标...")

    results = state["evaluation_results"]

    if not results:
        state["metrics"] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        return state

    # 计算平均指标
    avg_precision = sum(r["precision"] for r in results) / len(results)
    avg_recall = sum(r["recall"] for r in results) / len(results)
    avg_f1 = sum(r["f1"] for r in results) / len(results)

    state["metrics"] = {
        "precision": avg_precision,
        "recall": avg_recall,
        "f1": avg_f1,
        "total_questions": len(results),
    }

    # 生成评估报告
    report = f"""\n🎯 评估完成！
📈 总体指标：
   准确率 (Precision): {avg_precision:.3f}
   召回率 (Recall): {avg_recall:.3f}
   F1分数: {avg_f1:.3f}
   测试问题数: {len(results)}

📋 详细结果已保存到 evaluation_results 中"""

    state["output"] = report
    print(report)

    return state


def should_process_document(state: RagGraph) -> str:
    """条件边：根据缓存状态和评估模式决定下一步"""
    if state.get("evaluation_mode", False):
        return "load_test_questions"
    return "retrieve" if state["cache_exists"] else "pdf_classify"


def should_continue_evaluation(state: RagGraph) -> str:
    """条件边：判断是否继续评估"""
    questions = state.get("test_questions", [])
    index = state.get("current_question_index", 0)

    if index < len(questions):
        return "evaluate_single_question"
    else:
        return "calculate_final_metrics"


def should_route_pdf_processing(state: RagGraph) -> str:
    """条件边：根据分类结果路由到不同的PDF处理节点"""
    classification = state.get("pdf_classification", {})

    has_vision_needed = any(
        method == "vision_needed" for method in classification.values()
    )
    has_text_extractable = any(
        method == "text_extractable" for method in classification.values()
    )

    if has_text_extractable and has_vision_needed:
        return "pdf_read"
    elif has_text_extractable:
        return "pdf_read"
    elif has_vision_needed:
        return "pdfread_qwen"
    else:
        return "get_chunks"


def merge_pdf_content(state: RagGraph):
    """合并PDF处理结果（内容已在各自函数中保存到JSON）"""
    print("📋 PDF处理结果已保存到JSON文件")
    return state


def create_smart_rag():
    """创建智能RAG工作流"""
    workflow = StateGraph(RagGraph)

    # 原有节点
    workflow.add_node("check_cache", check_cache)
    workflow.add_node("pdf_classify", pdf_classify)
    workflow.add_node("pdf_read", pdf_read)
    workflow.add_node("pdfread_qwen", pdfread_qwen)
    workflow.add_node("merge_content", merge_pdf_content)
    workflow.add_node("get_chunks", get_chunks)
    workflow.add_node("vector_store", vector_store_func)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("ai_answer", ai_answer)

    # 评估模式节点
    workflow.add_node("load_test_questions", load_test_questions)
    workflow.add_node("evaluate_single_question", evaluate_single_question)
    workflow.add_node("calculate_final_metrics", calculate_final_metrics)

    # 原有边和条件边
    workflow.add_conditional_edges(
        "check_cache",
        should_process_document,
        {
            "retrieve": "retrieve",
            "pdf_classify": "pdf_classify",
            "load_test_questions": "load_test_questions",
        },
    )

    workflow.add_conditional_edges(
        "pdf_classify",
        should_route_pdf_processing,
        {
            "pdf_read": "pdf_read",
            "pdfread_qwen": "pdfread_qwen",
            "get_chunks": "get_chunks",
        },
    )

    # 评估模式的条件边
    workflow.add_conditional_edges(
        "load_test_questions",
        should_continue_evaluation,
        {
            "evaluate_single_question": "evaluate_single_question",
            "calculate_final_metrics": "calculate_final_metrics",
        },
    )

    workflow.add_conditional_edges(
        "evaluate_single_question",
        should_continue_evaluation,
        {
            "evaluate_single_question": "evaluate_single_question",
            "calculate_final_metrics": "calculate_final_metrics",
        },
    )

    workflow.add_edge("pdf_read", "merge_content")
    workflow.add_edge("pdfread_qwen", "get_chunks")
    workflow.add_edge("merge_content", "pdfread_qwen")

    workflow.add_edge("get_chunks", "vector_store")
    workflow.add_edge("vector_store", "retrieve")
    workflow.add_edge("retrieve", "ai_answer")

    workflow.set_entry_point("check_cache")
    workflow.set_finish_point("ai_answer")
    workflow.set_finish_point("calculate_final_metrics")

    return workflow.compile()


if __name__ == "__main__":
    pdf_path = r"C:\Users\Yu\Desktop\coa\COA"

    smart_rag = create_smart_rag()
    print("🤖 智能RAG系统启动")
    print("💡 输入 'eval' 进入评估模式，'quit' 退出")

    while True:
        question = input("\n❓ 请输入您的问题 (或 'eval' 评估): ").strip()
        if question.lower() in ["quit", "exit", "退出", "q"]:
            break

        if not question:
            continue

        # 判断是否进入评估模式
        evaluation_mode = question.lower() == "eval"

        state: RagGraph = {
            "input": question if not evaluation_mode else "",
            "pdf_path": pdf_path,
            "pdf_content": None,
            "chunks": None,
            "vector_store": None,
            "retrieved_docs": None,
            "output": None,
            "cache_exists": False,
            "pdf_classification": {},
            "evaluation_mode": evaluation_mode,
            "test_questions": [],
            "current_question_index": 0,
            "evaluation_results": [],
            "metrics": {},
        }

        # 在main函数中增加递归限制配置
        result = smart_rag.invoke(state, config={"recursion_limit": 500})

        if evaluation_mode:
            print(f"\n{result['output']}")
            # 可选：保存详细结果到文件
            with open("evaluation_detailed_results.json", "w", encoding="utf-8") as f:
                json.dump(result["evaluation_results"], f, ensure_ascii=False, indent=2)
            print("📁 详细结果已保存到 evaluation_detailed_results.json")
        else:
            print(f"\n🤖 回答: {result['output']}")
