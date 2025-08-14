#!/usr/bin/env python3
"""
使用正确的API响应格式进行最终测试
"""

import requests
import json
import time

def final_correct_test():
    """使用正确格式的最终测试"""
    base_url = 'http://localhost:5001'
    
    print("🎯 最终正确测试：验证完整的跨session功能")
    print("=" * 50)
    
    session1 = requests.Session()
    
    # 第一步：搜索
    print("1️⃣ 搜索文献...")
    search_response = session1.post(f'{base_url}/api/search', json={
        'query': 'final correct test study',
        'mode': 'strategy'
    })
    
    if search_response.status_code == 200:
        search_data = search_response.json()
        session_id = search_data.get('session_id')
        paper_count = len(search_data.get('data', []))
        print(f"✅ 搜索成功")
        print(f"   Session ID: {session_id}")
        print(f"   找到文献: {paper_count} 篇")
    else:
        print(f"❌ 搜索失败: {search_response.status_code}")
        return False
    
    # 第二步：启动异步任务
    print(f"\n2️⃣ 启动research_topic_suggestion任务...")
    
    task_response = session1.post(f'{base_url}/api/async/research_topic_suggestion', json={
        'query': 'final correct test study'
    })
    
    if task_response.status_code == 200:
        task_data = task_response.json()
        task_id = task_data.get('task_id')
        print(f"✅ 任务启动成功: {task_id}")
    else:
        print(f"❌ 任务启动失败: {task_response.status_code}")
        return False
    
    # 第三步：等待任务完成
    print(f"\n3️⃣ 等待任务完成...")
    
    for i in range(30):
        time.sleep(2)
        
        status_response = session1.get(f'{base_url}/api/async/task/{task_id}/status')
        
        if status_response.status_code == 200:
            status_data = status_response.json()
            task_info = status_data.get('task', {})
            task_status = task_info.get('status')
            progress = task_info.get('progress', 0)
            
            if task_status == 'completed':
                print(f"✅ 任务完成 (进度: {progress}%)")
                break
            elif task_status == 'failed':
                error_msg = task_info.get('error', '未知错误')
                print(f"❌ 任务失败: {error_msg}")
                return False
            else:
                print(f"   进度: {progress}% - 状态: {task_status}")
        else:
            print(f"   获取状态失败: {status_response.status_code}")
    else:
        print(f"❌ 任务超时")
        return False
    
    # 第四步：获取结果
    print(f"\n4️⃣ 获取任务结果...")
    
    result_response = session1.get(f'{base_url}/api/async/task/{task_id}/result')
    
    if result_response.status_code == 200:
        result_data = result_response.json()
        
        if result_data.get('success'):
            result = result_data.get('result', {})
            redirect_url = result_data.get('redirect_url', '')
            
            print(f"✅ 获取结果成功")
            print(f"   分析类型: {result.get('analysis_type')}")
            print(f"   论文数量: {result.get('paper_count')}")
            print(f"   内容长度: {len(result.get('content', ''))}")
            print(f"   重定向URL: {redirect_url}")
            
            # 立即检查session缓存
            print(f"\n📋 检查结果API是否正确存储到session缓存...")
            
            try:
                from utils.session_cache import get_papers_cache
                cache = get_papers_cache()
                
                cached_analysis = cache.get_analysis_result(session_id, 'research_topic')
                
                if cached_analysis:
                    print(f"✅ 结果API成功存储到session缓存")
                    print(f"   缓存论文数量: {cached_analysis.get('paper_count')}")
                    print(f"   缓存内容长度: {len(cached_analysis.get('content', ''))}")
                else:
                    print(f"❌ 结果API未能存储到session缓存")
                    print(f"   这将导致跨session访问失败")
                    
            except Exception as e:
                print(f"❌ 检查缓存出错: {e}")
        else:
            print(f"❌ 结果获取失败: {result_data.get('error')}")
            return False
    else:
        print(f"❌ 结果API调用失败: {result_response.status_code}")
        return False
    
    # 第五步：跨session访问测试
    print(f"\n5️⃣ 跨session访问测试...")
    
    session2 = requests.Session()
    
    if redirect_url:
        analysis_url = f"{base_url}{redirect_url}"
        print(f"   访问URL: {analysis_url}")
        
        analysis_response = session2.get(analysis_url)
        
        if analysis_response.status_code == 200:
            content = analysis_response.text
            
            # 检查论文数量
            expected_count = result.get('paper_count', 0)
            if f"基于 {expected_count} 篇相关文献" in content:
                print(f"✅ 跨session论文数量显示正确: {expected_count} 篇")
                
                # 检查AI内容
                original_content = result.get('content', '')
                if original_content and original_content in content:
                    print(f"✅ 跨session AI内容完全正确")
                    print(f"🎉 完美！所有测试通过，跨session分析功能完全正常！")
                    return True
                elif "基于URL参数恢复" in content:
                    print(f"❌ 跨session触发了备用逻辑")
                    print(f"   这说明session缓存存储或检索有问题")
                    
                    # 再次检查缓存状态
                    try:
                        from utils.session_cache import get_papers_cache
                        cache = get_papers_cache()
                        
                        cached_analysis = cache.get_analysis_result(session_id, 'research_topic')
                        if cached_analysis:
                            print(f"   奇怪：缓存中有数据但页面触发了备用逻辑")
                            print(f"   可能是web路由的检索逻辑有问题")
                        else:
                            print(f"   确认：缓存中没有数据")
                    except Exception as e:
                        print(f"   检查缓存出错: {e}")
                else:
                    print(f"❌ 跨session AI内容状态未知")
                    
                    # 显示实际内容进行调试
                    import re
                    pre_match = re.search(r'<pre[^>]*>(.*?)</pre>', content, re.DOTALL)
                    if pre_match:
                        actual_content = pre_match.group(1).strip()
                        print(f"   实际显示内容长度: {len(actual_content)}")
                        print(f"   实际内容前200字符: {actual_content[:200]}...")
                        print(f"   原始内容前200字符: {original_content[:200]}...")
            else:
                print(f"❌ 跨session论文数量显示错误")
                
                # 调试信息
                import re
                count_match = re.search(r'基于\s*(\d+)\s*篇相关文献', content)
                if count_match:
                    found_count = count_match.group(1)
                    print(f"   实际显示: {found_count} 篇")
                    print(f"   期望显示: {expected_count} 篇")
        else:
            print(f"❌ 跨session访问失败: {analysis_response.status_code}")
    else:
        print(f"❌ 没有重定向URL")
    
    return False

if __name__ == "__main__":
    success = final_correct_test()
    
    if success:
        print(f"\n🎊 测试结果：完全成功！")
        print(f"跨session分析功能已经完美修复并正常工作。")
    else:
        print(f"\n🔧 测试结果：需要进一步修复")
        print(f"已定位到具体问题，可以针对性修复。")