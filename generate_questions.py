import os
import json
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

def get_cache_filename(pdf_name, content):
    """生成缓存文件名"""
    content_hash = hashlib.md5(content[:1000].encode()).hexdigest()[:8]
    return f"question_cache/{pdf_name}_{content_hash}.json"

def load_cached_questions(cache_file):
    """加载缓存的问题"""
    if os.path.exists(cache_file):
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_cached_questions(cache_file, questions):
    """保存问题到缓存"""
    os.makedirs("question_cache", exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)

def generate_questions_for_pdf(pdf_name, content):
    """为单个PDF生成测试问题"""
    # 检查缓存
    cache_file = get_cache_filename(pdf_name, content)
    cached_questions = load_cached_questions(cache_file)
    if cached_questions:
        print(f"  📋 从缓存加载 {pdf_name} 的问题")
        return cached_questions
    
    print(f"🤖 为 {pdf_name} 生成测试问题...")
    
    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    
    # 限制内容长度
    content_summary = content[:3000] if len(content) > 3000 else content
    
    prompt = f"""基于以下COA（Certificate of Analysis）文档内容，生成2-3个测试问题，用于评估RAG系统的检索和回答质量。

COA文档内容：
{content_summary}

请严格按照以下JSON格式返回，不要添加任何其他文本：
{{
    "questions": [
        {{
            "question": "具体的问题文本",
            "ground_truth": "标准答案文本",
            "relevant_docs": ["{pdf_name}"],
            "relevant_chunks": ["应该检索到的关键文本片段"]
        }}
    ]
}}

要求：
1. 问题要针对COA文档特点，如检测项目、检测标准、检测结果、合格判定等
2. 至少包含以下类型问题：
   - 检测项目查询（如"在某某产品检测了哪些项目？"）
   - 检测结果查询（如"某某产品的XX项目的检测结果是多少？"）
   - 标准对比查询（如"XX项目是否符合标准要求？"）
3. 标准答案要准确、完整，每个问题都需要有pdf文件的名称
4. 文本片段要包含关键信息
5. 只返回JSON，不要添加任何解释文字"""
    
    try:
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # 尝试提取JSON部分
        if response_text.startswith('```json'):
            response_text = response_text.replace('```json', '').replace('```', '').strip()
        elif response_text.startswith('```'):
            response_text = response_text.replace('```', '').strip()
        
        # 查找JSON开始和结束位置
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        
        if start_idx != -1 and end_idx > start_idx:
            json_text = response_text[start_idx:end_idx]
            questions_data = json.loads(json_text)
            questions = questions_data.get("questions", [])
            
            # 保存到缓存
            save_cached_questions(cache_file, questions)
            
            print(f"  ✅ 生成了 {len(questions)} 个问题")
            return questions
        else:
            print(f"  ❌ 未找到有效JSON格式，跳过 {pdf_name}")
            return []
            
    except json.JSONDecodeError as e:
        print(f"  ❌ JSON解析失败: {e}，跳过 {pdf_name}")
        return []
    except Exception as e:
        print(f"  ❌ 生成问题时出错: {e}，跳过 {pdf_name}")
        return []

def generate_coa_test_questions_parallel(pdf_contents, max_workers=5):
    """并行生成所有PDF的测试问题"""
    all_questions = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_pdf = {
            executor.submit(generate_questions_for_pdf, pdf_name, content): pdf_name
            for pdf_name, content in pdf_contents.items()
        }
        
        # 收集结果
        for future in as_completed(future_to_pdf):
            pdf_name = future_to_pdf[future]
            try:
                questions = future.result()
                all_questions.extend(questions)
            except Exception as e:
                print(f"  ❌ 处理 {pdf_name} 时出错: {e}")
    
    return all_questions

def main():
    # 配置文件路径
    input_file = "coa_pdf_contents.json"
    output_file = "coa_test_dataset.json"
    
    print("🚀 开始并行生成COA测试问题...")
    
    # 读取现有的PDF内容
    print(f"📖 读取PDF内容: {input_file}")
    with open(input_file, "r", encoding="utf-8") as f:
        pdf_contents = json.load(f)
    
    print(f"✅ 已加载 {len(pdf_contents)} 个PDF文件的内容")
    
    # 并行生成测试问题
    print("\n🤖 并行生成测试问题...")
    all_questions = generate_coa_test_questions_parallel(pdf_contents, max_workers=5)
    
    # 构建最终数据集
    test_dataset = {
        "test_questions": all_questions,
        "metadata": {
            "total_pdfs": len(pdf_contents),
            "total_questions": len(all_questions),
            "pdf_files": list(pdf_contents.keys())
        }
    }
    
    # 保存测试数据集
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(test_dataset, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 测试数据集已生成: {output_file}")
    print(f"📊 共处理 {len(pdf_contents)} 个PDF文件")
    print(f"📊 共生成 {len(all_questions)} 个测试问题")
    
    # 显示示例
    if all_questions:
        print("\n📋 示例问题:")
        example = all_questions[0]
        print(f"问题: {example['question']}")
        print(f"答案: {example['ground_truth'][:100]}...")
        print(f"相关文档: {example['relevant_docs']}")

if __name__ == "__main__":
    main()