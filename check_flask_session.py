#!/usr/bin/env python3
"""
检查Flask session中的数据
"""

import requests

def check_flask_session():
    """检查Flask session数据传递"""
    base_url = 'http://localhost:5001'
    
    # 创建一个新的session，进行一次完整的流程
    session1 = requests.Session()
    
    print("🧪 测试Flask session数据传递")
    print("=" * 40)
    
    # 1. 检索
    search_response = session1.post(f'{base_url}/api/search', json={
        'query': 'flask session test',
        'mode': 'strategy'
    })
    
    search_data = search_response.json()
    session_id = search_data.get('session_id')
    print(f"1️⃣ 检索成功，session_id: {session_id}")
    
    # 2. 启动任务
    task_response = session1.post(f'{base_url}/api/async/research_topic_suggestion', json={
        'query': 'flask session test'
    })
    
    task_data = task_response.json()
    task_id = task_data.get('task_id')
    print(f"2️⃣ 任务启动，task_id: {task_id}")
    
    # 3. 等待并获取结果
    import time
    for i in range(25):
        time.sleep(2)
        
        # 检查任务状态
        status_response = session1.get(f'{base_url}/api/async/task/{task_id}/status')
        if status_response.status_code == 200:
            status_data = status_response.json()
            task_status = status_data.get('task', {}).get('status')
            
            if task_status == 'completed':
                print(f"3️⃣ 任务完成")
                break
            elif task_status == 'failed':
                print(f"❌ 任务失败")
                return
    else:
        print(f"⏰ 任务超时")
        return
    
    # 4. 获取结果
    result_response = session1.get(f'{base_url}/api/async/task/{task_id}/result')
    result_data = result_response.json()
    result = result_data.get('result', {})
    redirect_url = result_data.get('redirect_url', '')
    
    print(f"4️⃣ 获取结果成功")
    print(f"   Paper count: {result.get('paper_count')}")
    print(f"   Content length: {len(result.get('content', ''))}")
    print(f"   Redirect URL: {redirect_url}")
    
    # 5. 使用同一session立即访问（应该从Flask session获取）
    print(f"\n5️⃣ 同一session立即访问分析页面:")
    analysis_response1 = session1.get(f'{base_url}/analysis/research_topic')
    
    if analysis_response1.status_code == 200:
        content1 = analysis_response1.text
        if result.get('content', '') in content1:
            print(f"✅ 同一session包含完整AI内容（Flask session工作）")
        else:
            print(f"❌ 同一session缺少AI内容")
    
    # 6. 新session使用URL参数访问（应该从session缓存获取）
    print(f"\n6️⃣ 新session使用URL参数访问:")
    session2 = requests.Session()
    
    analysis_url = f"{base_url}{redirect_url}"
    print(f"   URL: {analysis_url}")
    
    analysis_response2 = session2.get(analysis_url)
    
    if analysis_response2.status_code == 200:
        content2 = analysis_response2.text
        
        # 检查论文数量
        if f"基于 {result.get('paper_count', 0)} 篇相关文献" in content2:
            print(f"✅ 新session论文数量正确")
        else:
            print(f"❌ 新session论文数量错误")
        
        # 检查AI内容
        if result.get('content', '') in content2:
            print(f"✅ 新session包含完整AI内容（session缓存工作）")
        elif "基于URL参数恢复" in content2:
            print(f"❌ 新session触发了备用逻辑（session缓存失败）")
        else:
            print(f"❌ 新session内容状态未知")
    else:
        print(f"❌ 新session访问失败: {analysis_response2.status_code}")

if __name__ == "__main__":
    check_flask_session()