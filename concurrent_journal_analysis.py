#!/usr/bin/env python3
"""
多线程并发分析期刊审稿周期
"""

import requests
import xml.etree.ElementTree as ET
import json
import csv
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Any, Optional
import statistics
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConcurrentJournalAnalyzer:
    """并发期刊分析器"""
    
    def __init__(self, max_workers: int = 8, rate_limit_delay: float = 0.2):
        self.max_workers = max_workers
        self.rate_limit_delay = rate_limit_delay
        self.results_queue = queue.Queue()
        self.progress_lock = threading.Lock()
        self.completed_count = 0
        self.total_journals = 0
        
    def analyze_all_journals_concurrent(self):
        """并发分析所有期刊"""
        
        print("=" * 100)
        print("多线程并发期刊审稿周期分析")
        print("=" * 100)
        
        # 获取期刊列表
        journals_list = self.get_journals_list()
        self.total_journals = len(journals_list)
        
        print(f"准备并发分析 {self.total_journals} 个期刊")
        print(f"并发线程数: {self.max_workers}")
        print(f"速率限制: {self.rate_limit_delay}秒/请求")
        print("-" * 100)
        
        start_time = time.time()
        all_results = []
        
        # 使用线程池执行器
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_journal = {
                executor.submit(self.analyze_single_journal_safe, journal): journal 
                for journal in journals_list
            }
            
            # 收集结果
            for future in as_completed(future_to_journal):
                journal = future_to_journal[future]
                try:
                    result = future.result()
                    if result:
                        all_results.append(result)
                        logger.info(f"✅ {journal}: 平均审稿周期 {result['avg_review_days']:.1f} 天")
                    else:
                        logger.warning(f"❌ {journal}: 分析失败或无数据")
                        
                except Exception as e:
                    logger.error(f"❌ {journal}: 分析出错 - {e}")
                
                # 更新进度
                with self.progress_lock:
                    self.completed_count += 1
                    progress = (self.completed_count / self.total_journals) * 100
                    print(f"\r进度: {self.completed_count}/{self.total_journals} ({progress:.1f}%)", end="", flush=True)
        
        print()  # 换行
        end_time = time.time()
        
        print(f"\n" + "=" * 100)
        print(f"并发分析完成！")
        print(f"成功分析: {len(all_results)}/{self.total_journals} 个期刊")
        print(f"总耗时: {end_time - start_time:.1f} 秒")
        print(f"平均每期刊: {(end_time - start_time) / self.total_journals:.1f} 秒")
        print("=" * 100)
        
        # 保存结果
        self.save_concurrent_results(all_results)
        
        # 生成报告
        self.generate_concurrent_report(all_results)
        
        return all_results

    def analyze_single_journal_safe(self, journal_name: str) -> Optional[Dict[str, Any]]:
        """线程安全的单期刊分析"""
        try:
            # 添加速率限制
            time.sleep(self.rate_limit_delay)
            
            return self.analyze_single_journal(journal_name)
            
        except Exception as e:
            logger.error(f"期刊 {journal_name} 分析异常: {e}")
            return None

    def analyze_single_journal(self, journal_name: str, max_papers: int = 200) -> Optional[Dict[str, Any]]:
        """分析单个期刊的审稿周期"""
        
        # 构建检索式
        search_query = f'("{journal_name}"[Journal]) AND (("2024/01/01"[Date - Create] : "2025/12/31"[Date - Create]))'
        
        try:
            # 搜索文献
            pmids = self.search_papers(search_query, max_papers)
            if not pmids:
                return None
            
            logger.debug(f"{journal_name}: 找到 {len(pmids)} 篇文献")
            
            # 分批获取详细信息
            papers_with_dates = self.fetch_all_papers_concurrent(pmids)
            
            if not papers_with_dates:
                return None
            
            logger.debug(f"{journal_name}: 解析 {len(papers_with_dates)} 篇文献的时间信息")
            
            # 计算审稿周期
            review_cycles = self.calculate_review_cycles(papers_with_dates)
            
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
                'data_completeness': round((len(review_cycles) / len(papers_with_dates)) * 100, 1),
                'analysis_thread': threading.current_thread().name,
                'analysis_time': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"{journal_name} 分析失败: {e}")
            return None

    def fetch_all_papers_concurrent(self, pmids: List[str], batch_size: int = 50) -> List[Dict[str, Any]]:
        """并发获取所有文献的详细信息"""
        all_papers = []
        
        # 分批处理
        batches = [pmids[i:i + batch_size] for i in range(0, len(pmids), batch_size)]
        
        # 对于单个期刊内的批次，使用较小的并发数避免过度请求
        max_batch_workers = min(3, len(batches))
        
        with ThreadPoolExecutor(max_workers=max_batch_workers) as executor:
            future_to_batch = {
                executor.submit(self.fetch_papers_batch, batch): batch 
                for batch in batches
            }
            
            for future in as_completed(future_to_batch):
                try:
                    batch_papers = future.result()
                    all_papers.extend(batch_papers)
                except Exception as e:
                    logger.error(f"批次获取失败: {e}")
                    continue
        
        return all_papers

    def search_papers(self, query: str, max_results: int) -> List[str]:
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
            logger.error(f"搜索失败: {e}")
            return []

    def fetch_papers_batch(self, pmids: List[str]) -> List[Dict[str, Any]]:
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
            
            return self.parse_papers_xml(response.content)
            
        except Exception as e:
            logger.error(f"批次获取失败: {e}")
            return []

    def parse_papers_xml(self, xml_content: bytes) -> List[Dict[str, Any]]:
        """解析文献XML数据"""
        papers = []
        
        try:
            root = ET.fromstring(xml_content)
            
            for article_elem in root.findall('.//PubmedArticle'):
                paper_data = self.extract_paper_dates(article_elem)
                if paper_data:
                    papers.append(paper_data)
                    
        except ET.ParseError as e:
            logger.error(f"XML解析失败: {e}")
        
        return papers

    def extract_paper_dates(self, article_elem) -> Optional[Dict[str, Any]]:
        """提取单篇文献的时间信息"""
        try:
            # 获取PMID
            medline_citation = article_elem.find('.//MedlineCitation')
            if medline_citation is None:
                return None
            
            pmid = self.get_text(medline_citation.find('.//PMID'))
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
                        date_obj = self.parse_date_element(pub_date)
                        if date_obj:
                            dates[status] = date_obj
            
            # 只返回有received和accepted日期的文献
            if 'received' in dates and 'accepted' in dates:
                return {
                    'pmid': pmid,
                    'received_date': dates['received'],
                    'accepted_date': dates['accepted'],
                    'all_dates': dates
                }
            
            return None
            
        except Exception as e:
            logger.error(f"提取文献日期失败: {e}")
            return None

    def parse_date_element(self, date_elem) -> Optional[datetime]:
        """从XML元素解析日期"""
        try:
            year = self.get_text(date_elem.find('.//Year'))
            month = self.get_text(date_elem.find('.//Month'))
            day = self.get_text(date_elem.find('.//Day'))
            
            if not year:
                return None
            
            # 处理月份
            if month:
                try:
                    month_num = int(month)
                except ValueError:
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

    def calculate_review_cycles(self, papers: List[Dict[str, Any]]) -> List[int]:
        """计算审稿周期"""
        review_cycles = []
        
        for paper in papers:
            received_date = paper.get('received_date')
            accepted_date = paper.get('accepted_date')
            
            if received_date and accepted_date:
                days = (accepted_date - received_date).days
                if 0 <= days <= 730:  # 过滤异常值
                    review_cycles.append(days)
        
        return review_cycles

    def get_journals_list(self) -> List[str]:
        """获取期刊列表"""
        return [
            # 顶级期刊
            "Nature", "Science", "Cell",
            "Nature Medicine", "Nature Biotechnology", "Nature Genetics",
            "Nature Immunology", "Nature Neuroscience", "Nature Cell Biology",
            "Nature Communications", "PNAS", "eLife",
            
            # Cell系列
            "Cell Metabolism", "Cell Stem Cell", "Cancer Cell",
            "Molecular Cell", "Developmental Cell", "Current Biology",
            
            # 医学期刊
            "The Lancet", "New England Journal of Medicine", "JAMA",
            "The Lancet Oncology", "Blood", "Immunity",
            
            # 其他重要期刊
            "EMBO Journal", "EMBO Reports", "PLoS Biology",
            "Genome Research", "Genome Biology", "Nucleic Acids Research",
            "Bioinformatics", "Nature Methods", "Nature Structural & Molecular Biology"
        ]

    def save_concurrent_results(self, results: List[Dict[str, Any]]):
        """保存并发分析结果"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON格式
        json_filename = f"concurrent_journal_analysis_{timestamp}.json"
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'analysis_method': 'concurrent_multithreading',
                'max_workers': self.max_workers,
                'rate_limit_delay': self.rate_limit_delay,
                'analysis_time': datetime.now().isoformat(),
                'total_journals_analyzed': len(results),
                'results': results
            }, f, ensure_ascii=False, indent=2, default=str)
        
        # CSV格式
        csv_filename = f"concurrent_journal_analysis_{timestamp}.csv"
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                '期刊名称', '总文献数', '有审稿数据文献数', '平均审稿周期(天)', 
                '中位数(天)', '最短(天)', '最长(天)', '标准差(天)', '数据完整率(%)', '分析线程'
            ])
            
            for result in results:
                writer.writerow([
                    result['journal_name'], result['total_papers'],
                    result['papers_with_review_data'], result['avg_review_days'],
                    result['median_review_days'], result['min_review_days'],
                    result['max_review_days'], result['std_review_days'],
                    result['data_completeness'], result['analysis_thread']
                ])
        
        print(f"\n✅ 并发分析结果已保存:")
        print(f"   JSON: {json_filename}")
        print(f"   CSV: {csv_filename}")

    def generate_concurrent_report(self, results: List[Dict[str, Any]]):
        """生成并发分析报告"""
        if not results:
            return
        
        # 按平均审稿周期排序
        sorted_results = sorted(results, key=lambda x: x['avg_review_days'])
        
        print(f"\n📊 并发分析期刊审稿周期排行榜")
        print("=" * 120)
        print(f"{'排名':<4} {'期刊名称':<35} {'平均周期':<10} {'中位数':<8} {'样本数':<8} {'完整率':<8} {'分析线程':<15}")
        print("-" * 120)
        
        for i, result in enumerate(sorted_results, 1):
            print(f"{i:<4} {result['journal_name']:<35} {result['avg_review_days']:<10.1f} "
                  f"{result['median_review_days']:<8.1f} {result['papers_with_review_data']:<8} "
                  f"{result['data_completeness']:<8.1f}% {result['analysis_thread']:<15}")
        
        # 性能统计
        avg_cycles = [r['avg_review_days'] for r in results]
        thread_distribution = {}
        for result in results:
            thread = result['analysis_thread']
            thread_distribution[thread] = thread_distribution.get(thread, 0) + 1
        
        print(f"\n📈 并发分析统计:")
        print(f"   成功分析期刊数: {len(results)}")
        print(f"   平均审稿周期: {statistics.mean(avg_cycles):.1f} 天")
        print(f"   最快期刊: {sorted_results[0]['journal_name']} ({sorted_results[0]['avg_review_days']:.1f}天)")
        print(f"   最慢期刊: {sorted_results[-1]['journal_name']} ({sorted_results[-1]['avg_review_days']:.1f}天)")
        print(f"   线程分布: {dict(sorted(thread_distribution.items()))}")

    def get_text(self, element) -> str:
        """安全获取元素文本"""
        if element is None:
            return ''
        return ''.join(element.itertext()).strip()

def main():
    """主函数"""
    # 创建并发分析器
    analyzer = ConcurrentJournalAnalyzer(
        max_workers=8,  # 8个并发线程
        rate_limit_delay=0.3  # 每个请求间隔0.3秒
    )
    
    # 执行并发分析
    results = analyzer.analyze_all_journals_concurrent()
    
    print(f"\n🎉 并发分析完成！共分析了 {len(results)} 个期刊")

if __name__ == "__main__":
    main()
