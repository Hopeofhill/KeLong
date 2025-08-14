#!/usr/bin/env python3
"""
调试前端API调用
"""

import requests
import json

def debug_review_topics_api():
    """调试综述选题API调用"""
    print("🔍 调试综述选题API调用...")
    
    # 模拟前端发送的数据
    test_papers = [
        {
            "title": "Diet and exercise in the prevention and treatment of type 2 diabetes mellitus",
            "authors": ["Author A", "Author B"],
            "journal": "Nature Reviews Endocrinology",
            "pub_year": "2020"
        }
    ]
    
    test_data = {
        "query": "(diabetes mellitus[MeSH] OR diabetes[Title/Abstract] OR T2DM[Title/Abstract] OR \"type 2 diabetes\"[Title/Abstract]) AND (management[Title/Abstract] OR treatment[MeSH] OR therapy[Title/Abstract] OR control[Title/Abstract])",
        "papers": test_papers
    }
    
    try:
        response = requests.post(
            'http://127.0.0.1:5001/api/review_topic_suggestion',
            json=test_data,
            headers={'Content-Type': 'application/json'},
            timeout=60
        )
        
        print(f"HTTP状态码: {response.status_code}")
        
        if response.status_code == 200:
            result = response.json()
            print("\n🔍 完整API响应结构:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
            
            print("\n📊 字段分析:")
            print(f"success: {result.get('success')}")
            print(f"analyzed_papers: {result.get('analyzed_papers')}")
            print(f"total_papers: {result.get('total_papers')}")
            
            # 检查所有可能的结果字段
            fields_to_check = ['analysis', 'suggestions', 'suggestion', 'recommendations', 'result']
            print(f"\n🔍 前端字段检查逻辑模拟:")
            
            for field in fields_to_check:
                value = result.get(field)
                if value:
                    print(f"✅ {field}: 有值 ({len(value)} 字符)")
                    break
                else:
                    print(f"❌ {field}: 无值")
            
            # 模拟前端的字段提取逻辑
            frontend_result = result.get('analysis') or result.get('suggestions') or result.get('suggestion') or result.get('recommendations') or result.get('result')
            
            if frontend_result:
                print(f"\n✅ 前端应该提取到内容: {len(frontend_result)} 字符")
                print(f"内容预览: {frontend_result[:200]}...")
            else:
                print(f"\n❌ 前端无法提取到内容！")
                print(f"可用字段: {list(result.keys())}")
            
        else:
            print(f"HTTP错误: {response.status_code}")
            print(f"响应内容: {response.text}")
            
    except Exception as e:
        print(f"请求出错: {e}")

if __name__ == "__main__":
    debug_review_topics_api()