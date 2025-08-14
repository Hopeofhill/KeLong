#!/usr/bin/env python3
"""
直接检查session中存储的keys
"""

import requests

def check_existing_sessions():
    """检查现有session中的数据"""
    base_url = 'http://localhost:5000'
    
    # 使用已知工作的session访问分析页面
    session = requests.Session()
    
    print("🧪 检查现有session数据")
    print("=" * 40)
    
    # 测试访问不同的分析页面
    analysis_types = ['full_review', 'research_topic', 'review_topic']
    
    for analysis_type in analysis_types:
        print(f"\n🔍 测试 {analysis_type}:")
        
        url = f'{base_url}/analysis/{analysis_type}'
        response = session.get(url)
        
        if response.status_code == 200:
            content = response.text
            
            # 检查显示的论文数量
            if "基于 10 篇相关文献" in content:
                print(f"  ✅ 显示10篇文献（正常）")
            elif "基于 0 篇相关文献" in content:
                print(f"  ❌ 显示0篇文献")
            elif "基于URL参数恢复" in content:
                print(f"  ⚠️  触发备用逻辑")
            else:
                print(f"  ⚠️  未知状态")
                
            # 检查是否有真实的AI内容
            if len(content) > 5000 and ("选题建议" in content or "综述" in content):
                print(f"  ✅ 包含完整AI生成内容")
            else:
                print(f"  ❌ 缺少AI生成内容")
        else:
            print(f"  ❌ 访问失败: {response.status_code}")

if __name__ == "__main__":
    check_existing_sessions()