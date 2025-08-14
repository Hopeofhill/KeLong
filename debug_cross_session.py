#!/usr/bin/env python3
"""
调试跨session访问时的内容问题
"""

import requests
import json
import time
import re

def debug_cross_session():
    """详细调试跨session内容传递"""
    base_url = 'http://localhost:5001'
    
    print("🔍 调试跨session内容传递问题")
    print("=" * 50)
    
    session1 = requests.Session()
    
    # 使用已知成功的session
    known_session_id = "365a3ce6-8c6b-4e3a-b249-3f5b460efe3b"
    
    print(f"1️⃣ 使用已知session访问: {known_session_id}")
    
    # 直接访问分析页面
    session2 = requests.Session()
    analysis_url = f"{base_url}/analysis/research_topic?sid={known_session_id}"
    
    print(f"   URL: {analysis_url}")
    
    analysis_response = session2.get(analysis_url)
    
    if analysis_response.status_code == 200:
        content = analysis_response.text
        print(f"✅ 页面访问成功")
        
        # 详细分析页面内容
        print(f"\n2️⃣ 分析页面内容结构...")
        
        # 查找论文数量
        paper_count_match = re.search(r'基于\s*(\d+)\s*篇相关文献', content)
        if paper_count_match:
            found_count = paper_count_match.group(1)
            print(f"   论文数量: {found_count} 篇")
        
        # 查找分析内容区域
        pre_matches = re.findall(r'<pre[^>]*>(.*?)</pre>', content, re.DOTALL)
        
        if pre_matches:
            for i, match in enumerate(pre_matches):
                cleaned_content = match.strip()
                print(f"   <pre>块 {i+1}: 长度={len(cleaned_content)}")
                print(f"   内容预览: {cleaned_content[:200]}...")
                
                # 检查是否是有意义的AI内容
                if len(cleaned_content) > 100:
                    if "选题建议" in cleaned_content or "研究方向" in cleaned_content:
                        print(f"   ✅ 包含AI分析内容")
                    elif "基于URL参数恢复" in cleaned_content:
                        print(f"   ❌ 显示备用恢复内容")
                    else:
                        print(f"   ⚠️  内容类型不明确")
                else:
                    print(f"   ❌ 内容过短")
        else:
            print(f"   ❌ 没有找到<pre>标签内容")
        
        # 检查页面是否触发了恢复逻辑
        if "基于URL参数恢复" in content:
            print(f"\n   ⚠️  页面触发了备用恢复逻辑")
            print(f"   这意味着session缓存中没有找到完整的分析结果")
        
        # 查找模板变量的值
        print(f"\n3️⃣ 检查模板变量...")
        
        # 查找analysis_result变量的实际值
        analysis_result_pattern = r'<pre[^>]*>([^<]*)</pre>'
        analysis_match = re.search(analysis_result_pattern, content, re.DOTALL)
        
        if analysis_match:
            actual_analysis_result = analysis_match.group(1).strip()
            print(f"   analysis_result长度: {len(actual_analysis_result)}")
            print(f"   analysis_result内容: {actual_analysis_result[:300]}...")
        
    else:
        print(f"❌ 页面访问失败: {analysis_response.status_code}")
    
    # 检查session缓存状态
    print(f"\n4️⃣ 检查session缓存状态...")
    
    try:
        from utils.session_cache import get_papers_cache
        cache = get_papers_cache()
        
        # 检查papers缓存
        papers_data = cache.get_papers(known_session_id)
        if papers_data:
            print(f"   ✅ Papers缓存存在")
            print(f"      论文数量: {len(papers_data.get('papers', []))}")
            print(f"      查询: {papers_data.get('query', 'N/A')}")
        else:
            print(f"   ❌ Papers缓存不存在")
        
        # 检查分析结果缓存
        analysis_data = cache.get_analysis_result(known_session_id, 'research_topic')
        if analysis_data:
            print(f"   ✅ 分析结果缓存存在")
            print(f"      论文数量: {analysis_data.get('paper_count')}")
            print(f"      内容长度: {len(analysis_data.get('content', ''))}")
            print(f"      内容预览: {analysis_data.get('content', '')[:200]}...")
            
            # 分析这个缓存的内容是否有效
            cached_content = analysis_data.get('content', '')
            if len(cached_content) > 500:
                print(f"   ✅ 缓存内容充足")
            else:
                print(f"   ❌ 缓存内容过短")
        else:
            print(f"   ❌ 分析结果缓存不存在")
        
        # 检查所有缓存的session
        cache_stats = cache.get_cache_stats()
        print(f"\n   缓存统计:")
        print(f"      总session数: {cache_stats.get('total_sessions', 0)}")
        print(f"      总论文数: {cache_stats.get('total_papers', 0)}")
        
        sessions_info = cache_stats.get('sessions', {})
        for sid, info in sessions_info.items():
            if sid == known_session_id:
                print(f"      ✅ 目标session存在: {sid}")
                print(f"         论文数量: {info.get('paper_count', 0)}")
                print(f"         访问次数: {info.get('access_count', 0)}")
            
    except Exception as e:
        print(f"   ❌ 检查缓存出错: {e}")
    
    # 测试手动重建分析结果
    print(f"\n5️⃣ 测试手动创建分析结果...")
    
    try:
        from utils.session_cache import get_papers_cache
        cache = get_papers_cache()
        
        # 手动存储一个测试分析结果
        test_result = {
            'content': '<h3>测试AI分析内容</h3><p>这是一个测试的AI分析结果，用于验证跨session存储和检索功能。</p><p>如果您看到这个内容，说明session缓存的分析结果存储功能正常工作。</p>',
            'paper_count': 100,
            'search_query': 'test query',
            'timestamp': '2025-07-13 18:30:00',
            'loading': False,
            'error': None,
            'analysis_type': 'research_topic'
        }
        
        success = cache.store_analysis_result(known_session_id, 'research_topic', test_result)
        
        if success:
            print(f"   ✅ 手动存储测试结果成功")
            
            # 立即尝试检索
            retrieved = cache.get_analysis_result(known_session_id, 'research_topic')
            if retrieved:
                print(f"   ✅ 立即检索成功")
                print(f"      内容长度: {len(retrieved.get('content', ''))}")
            else:
                print(f"   ❌ 立即检索失败")
        else:
            print(f"   ❌ 手动存储失败")
            
    except Exception as e:
        print(f"   ❌ 手动测试出错: {e}")

if __name__ == "__main__":
    debug_cross_session()