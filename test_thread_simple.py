
"""
线程功能测试脚本
测试 RAG 系统的线程记忆功能
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入 RAG 系统

from pdfs_rag_mmr import create_smart_rag



def test_thread_memory():
    """测试线程记忆功能"""

    # 创建 RAG 系统
    smart_rag = create_smart_rag()
    print("✅ RAG系统创建成功")
    
    # 测试配置
    user_id = "test_user_001"
    config = {"configurable": {"thread_id": user_id}}
    print(f"📋 测试配置: 用户ID={user_id}")
    
    # 第一轮对话
    print("\n🔍 测试1: 第一个问题")
    state1 = {
        "input": "什么是机器学习？",
        "messages": [{"role": "user", "content": "什么是机器学习？"}],
        "user_id": user_id,
        "pdf_path": r"C:\Users\Yu\Desktop\coa\COA",
        "pdf_content": None,
        "chunks": None,
        "vector_store": None,
        "retrieved_docs": None,
        "output": None,
        "cache_exists": False,
        "pdf_classification": {},
    }
    
    result1 = smart_rag.invoke(state1, config=config)
    print(f"✅ 第一轮回答: {result1['output'][:100]}...")
    
    # 第二轮对话 - 测试上下文记忆
    print("\n🔍 测试2: 追问（测试记忆功能）")
    state2 = {
        "input": "请详细解释一下",
        "messages": [{"role": "user", "content": "请详细解释一下"}],
        "user_id": user_id,
        "pdf_path": r"C:\Users\Yu\Desktop\coa\COA",
        "pdf_content": None,
        "chunks": None,
        "vector_store": None,
        "retrieved_docs": None,
        "output": None,
        "cache_exists": False,
        "pdf_classification": {},
    }
    
    result2 = smart_rag.invoke(state2, config=config)
    print(f"✅ 第二轮回答: {result2['output'][:100]}...")
    
    # 检查消息历史
    if 'messages' in result2 and len(result2['messages']) > 2:
        print(f"✅ 消息历史累积成功，共 {len(result2['messages'])} 条消息")
        return True
    else:
        print("❌ 消息历史未正确累积")
        return False
            


if __name__ == "__main__":
    success = test_thread_memory()
    if success:
        print("\n🎉 线程功能测试通过！")
        print("💡 系统能够正确记住对话历史并在后续对话中使用")
    else:
        print("\n❌ 线程功能测试失败")
        print("💡 请检查代码实现和配置")