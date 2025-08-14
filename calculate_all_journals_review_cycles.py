#!/usr/bin/env python3
"""
计算所有期刊从收到稿件到接收的平均周期
"""

import requests
import xml.etree.ElementTree as ET
import json
import csv
from datetime import datetime, timedelta
from collections import defaultdict
from typing import Dict, List, Any, Optional
import statistics
import time

def calculate_all_journals_review_cycles():
    """计算所有期刊的审稿周期"""
    
    print("=" * 100)
    print("所有期刊审稿周期统计分析")
    print("=" * 100)
    
    # 获取高影响因子期刊列表
    journals_list = get_top_journals_list()
    
    print(f"准备分析 {len(journals_list)} 个期刊的审稿周期")
    print("-" * 100)
    
    all_results = []
    successful_count = 0
    
    for i, journal in enumerate(journals_list, 1):
        print(f"\n[{i}/{len(journals_list)}] 分析期刊: {journal}")
        print("-" * 60)
        
        try:
            result = analyze_journal_review_cycle(journal)
            if result:
                all_results.append(result)
                successful_count += 1
                print(f"✅ 成功分析，平均审稿周期: {result['avg_review_days']:.1f} 天")
            else:
                print(f"❌ 分析失败或无有效数据")
                
            # 添加延迟避免API限制
            time.sleep(1)
            
        except Exception as e:
            print(f"❌ 分析出错: {e}")
            continue
    
    print(f"\n" + "=" * 100)
    print(f"分析完成！成功分析 {successful_count}/{len(journals_list)} 个期刊")
    print("=" * 100)
    
    # 保存结果
    save_results(all_results)
    
    # 生成排行榜
    generate_ranking_report(all_results)

def get_top_journals_list() -> List[str]:
    """获取顶级期刊列表"""
    return [
        # Nature系列
        "Nature",
        "Nature Medicine",
        "Nature Biotechnology", 
        "Nature Genetics",
        "Nature Immunology",
        "Nature Neuroscience",
        "Nature Cell Biology",
        "Nature Communications",
        "Nature Reviews Molecular Cell Biology",
        "Nature Reviews Drug Discovery",
        "Nature Reviews Cancer",
        "Nature Reviews Immunology",
        "Nature Reviews Genetics",
        
        # Science系列
        "Science",
        "Science Translational Medicine",
        "Science Immunology",
        
        # Cell系列
        "Cell",
        "Cell Metabolism",
        "Cell Stem Cell",
        "Cancer Cell",
        "Molecular Cell",
        "Developmental Cell",
        "Current Biology",
        
        # 医学期刊
        "The Lancet",
        "New England Journal of Medicine",
        "JAMA",
        "BMJ",
        "Annals of Internal Medicine",
        "The Lancet Oncology",
        
        # 其他顶级期刊
        "PNAS",
        "eLife",
        "EMBO Journal",
        "Journal of Clinical Investigation",
        "Blood",
        "Immunity",
        "Neuron",
        "Cancer Research",
        "Journal of Experimental Medicine",
        "Genes & Development",
        "Molecular Biology of the Cell",
        "EMBO Reports",
        "Nature Structural & Molecular Biology",
        "Nature Chemical Biology",
        "Nature Methods",
        "Genome Research",
        "PLoS Biology",
        "Nucleic Acids Research",
        "Bioinformatics",
        "Genome Biology"
    ]

def analyze_journal_review_cycle(journal_name: str, max_papers: int = 200) -> Optional[Dict[str, Any]]:
    """分析单个期刊的审稿周期"""
    
    # 构建检索式 - 近一年的文献
    search_query = f'("{journal_name}"[Journal]) AND (("2024/01/01"[Date - Create] : "2025/12/31"[Date - Create]))'
    
    try:
        # 搜索文献
        pmids = search_papers(search_query, max_papers)
        if not pmids:
            return None
        
        print(f"   找到 {len(pmids)} 篇文献")
        
        # 分批获取详细信息
        papers_with_dates = []
        batch_size = 50
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            batch_papers = fetch_papers_batch(batch)
            papers_with_dates.extend(batch_papers)
            
            if i + batch_size < len(pmids):
                time.sleep(0.5)  # 批次间延迟
        
        if not papers_with_dates:
            return None
        
        print(f"   成功解析 {len(papers_with_dates)} 篇文献的时间信息")
        
        # 计算审稿周期
        review_cycles = calculate_review_cycles_from_papers(papers_with_dates)
        
        if not review_cycles:
            return None
        
        # 计算统计值
        result = {
            'journal_name': journal_name,
            'total_papers': len(papers_with_dates),
            'papers_with_review_data': len(review_cycles),
            'avg_review_days': round(statistics.mean(review_cycles), 1),
            'median_review_days': round(statistics.median(review_cycles), 1),
            'min_review_days': min(review_cycles),
            'max_review_days': max(review_cycles),
            'std_review_days': round(statistics.stdev(review_cycles) if len(review_cycles) > 1 else 0, 1),
            'review_cycles': review_cycles,
            'analysis_date': datetime.now().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"   分析失败: {e}")
        return None

def search_papers(query: str, max_results: int) -> List[str]:
    """搜索文献获取PMID列表"""
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'xml',
            'tool': 'NNScholar',
            'email': 'test@nnscholar.com'
        }
        
        response = requests.get(base_url, params=params, timeout=30)
        response.raise_for_status()
        
        root = ET.fromstring(response.content)
        pmids = [id_elem.text for id_elem in root.findall('.//Id')]
        
        return pmids
        
    except Exception as e:
        return []

def fetch_papers_batch(pmids: List[str]) -> List[Dict[str, Any]]:
    """批量获取文献详细信息"""
    try:
        base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
        params = {
            'db': 'pubmed',
            'id': ','.join(pmids),
            'retmode': 'xml',
            'rettype': 'abstract',
            'tool': 'NNScholar',
            'email': 'test@nnscholar.com'
        }
        
        response = requests.get(base_url, params=params, timeout=120)
        response.raise_for_status()
        
        return parse_papers_for_review_cycle(response.content)
        
    except Exception as e:
        return []

def parse_papers_for_review_cycle(xml_content: bytes) -> List[Dict[str, Any]]:
    """解析文献XML获取审稿周期相关信息"""
    papers = []
    
    try:
        root = ET.fromstring(xml_content)
        
        for article_elem in root.findall('.//PubmedArticle'):
            paper_data = extract_review_cycle_data(article_elem)
            if paper_data:
                papers.append(paper_data)
                
    except ET.ParseError:
        pass
    
    return papers

def extract_review_cycle_data(article_elem) -> Optional[Dict[str, Any]]:
    """提取单篇文献的审稿周期数据"""
    try:
        # 获取PMID
        medline_citation = article_elem.find('.//MedlineCitation')
        if medline_citation is None:
            return None
        
        pmid = get_text(medline_citation.find('.//PMID'))
        if not pmid:
            return None
        
        # 提取时间信息
        dates = {}
        
        # 从PubmedData/History中提取各种状态的日期
        pubmed_data = article_elem.find('.//PubmedData')
        if pubmed_data is not None:
            history = pubmed_data.find('.//History')
            if history is not None:
                for pub_date in history.findall('.//PubMedPubDate'):
                    status = pub_date.get('PubStatus', '')
                    date_obj = parse_date_from_element(pub_date)
                    if date_obj:
                        dates[status] = date_obj
        
        # 只返回有received和accepted日期的文献
        if 'received' in dates and 'accepted' in dates:
            return {
                'pmid': pmid,
                'received_date': dates['received'],
                'accepted_date': dates['accepted'],
                'other_dates': dates
            }
        
        return None
        
    except Exception:
        return None

def parse_date_from_element(date_elem) -> Optional[datetime]:
    """从XML元素解析日期"""
    try:
        year = get_text(date_elem.find('.//Year'))
        month = get_text(date_elem.find('.//Month'))
        day = get_text(date_elem.find('.//Day'))
        
        if not year:
            return None
        
        # 处理月份
        if month:
            try:
                month_num = int(month)
            except ValueError:
                # 处理月份名称
                month_names = {
                    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
                }
                month_num = month_names.get(month[:3], 1)
        else:
            month_num = 1
        
        day_num = int(day) if day else 1
        
        return datetime(int(year), month_num, day_num)
        
    except (ValueError, TypeError):
        return None

def calculate_review_cycles_from_papers(papers: List[Dict[str, Any]]) -> List[int]:
    """从文献数据计算审稿周期"""
    review_cycles = []
    
    for paper in papers:
        received_date = paper.get('received_date')
        accepted_date = paper.get('accepted_date')
        
        if received_date and accepted_date:
            # 计算天数差
            days = (accepted_date - received_date).days
            
            # 过滤异常值（0-730天范围内）
            if 0 <= days <= 730:
                review_cycles.append(days)
    
    return review_cycles

def save_results(results: List[Dict[str, Any]]):
    """保存结果到文件"""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 保存JSON格式
    json_filename = f"all_journals_review_cycles_{timestamp}.json"
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump({
            'analysis_time': datetime.now().isoformat(),
            'total_journals_analyzed': len(results),
            'results': results
        }, f, ensure_ascii=False, indent=2, default=str)
    
    # 保存CSV格式
    csv_filename = f"all_journals_review_cycles_{timestamp}.csv"
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            '期刊名称', '总文献数', '有审稿数据文献数', '平均审稿周期(天)', 
            '中位数(天)', '最短(天)', '最长(天)', '标准差(天)', '数据完整率(%)'
        ])
        
        for result in results:
            data_completeness = (result['papers_with_review_data'] / result['total_papers']) * 100
            writer.writerow([
                result['journal_name'],
                result['total_papers'],
                result['papers_with_review_data'],
                result['avg_review_days'],
                result['median_review_days'],
                result['min_review_days'],
                result['max_review_days'],
                result['std_review_days'],
                f"{data_completeness:.1f}%"
            ])
    
    print(f"\n✅ 结果已保存:")
    print(f"   JSON格式: {json_filename}")
    print(f"   CSV格式: {csv_filename}")

def generate_ranking_report(results: List[Dict[str, Any]]):
    """生成排行榜报告"""
    if not results:
        return
    
    # 按平均审稿周期排序
    sorted_results = sorted(results, key=lambda x: x['avg_review_days'])
    
    print(f"\n📊 期刊审稿周期排行榜 (Top {len(sorted_results)})")
    print("=" * 100)
    print(f"{'排名':<4} {'期刊名称':<40} {'平均周期':<10} {'中位数':<8} {'样本数':<8} {'完整率':<8}")
    print("-" * 100)
    
    for i, result in enumerate(sorted_results, 1):
        data_completeness = (result['papers_with_review_data'] / result['total_papers']) * 100
        
        print(f"{i:<4} {result['journal_name']:<40} {result['avg_review_days']:<10.1f} "
              f"{result['median_review_days']:<8.1f} {result['papers_with_review_data']:<8} "
              f"{data_completeness:<8.1f}%")
    
    # 统计摘要
    avg_cycles = [r['avg_review_days'] for r in results]
    print(f"\n📈 统计摘要:")
    print(f"   分析期刊数: {len(results)}")
    print(f"   平均审稿周期: {statistics.mean(avg_cycles):.1f} 天")
    print(f"   中位数审稿周期: {statistics.median(avg_cycles):.1f} 天")
    print(f"   最快期刊: {sorted_results[0]['journal_name']} ({sorted_results[0]['avg_review_days']:.1f}天)")
    print(f"   最慢期刊: {sorted_results[-1]['journal_name']} ({sorted_results[-1]['avg_review_days']:.1f}天)")

def get_text(element) -> str:
    """安全获取元素文本"""
    if element is None:
        return ''
    return ''.join(element.itertext()).strip()

if __name__ == "__main__":
    calculate_all_journals_review_cycles()
