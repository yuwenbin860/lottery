"""
数据爬取模块 - 负责从官方网站获取彩票开奖数据
"""

import random
import time
import logging
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import json
import os
import re

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('crawler')


class LotteryCrawler:
    """彩票数据爬取类"""

    def __init__(self, config_path='config.json'):
        """初始化爬虫"""
        self.config = self._load_config(config_path)
        self.session = requests.Session()

    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def _get_random_user_agent(self):
        """获取随机User-Agent"""
        user_agents = self.config['crawler']['user_agents']
        return random.choice(user_agents)

    def _make_request(self, url, params=None, headers=None):
        """发送HTTP请求并处理重试逻辑"""
        default_headers = {
            'User-Agent': self._get_random_user_agent(),
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
        }

        # 使用自定义头覆盖默认头
        if headers:
            default_headers.update(headers)

        max_retries = self.config['crawler']['max_retries']
        retry_count = 0

        while retry_count < max_retries:
            try:
                response = self.session.get(url, headers=default_headers, params=params, timeout=15)
                response.raise_for_status()

                # 请求成功，返回响应
                return response
            except requests.exceptions.RequestException as e:
                retry_count += 1
                logger.warning(f"请求失败 ({retry_count}/{max_retries}): {str(e)}")

                if retry_count >= max_retries:
                    logger.error(f"达到最大重试次数，请求失败: {url}")
                    raise

                # 延迟重试
                time.sleep(self.config['crawler']['request_delay'])

    def _get_local_latest_issue(self, lottery_type):
        """获取本地最新期号"""
        data_dir = self.config['storage']['data_dir']
        filename = os.path.join(data_dir, f"{lottery_type}.csv")

        if not os.path.exists(filename):
            return None

        try:
            with open(filename, 'r', encoding='utf-8') as f:
                # 跳过标题行
                next(f, None)

                # 读取最后一行获取最新期号
                for line in f:
                    pass

                last_line = line.strip().split(',')
                if len(last_line) > 0:
                    return last_line[0]  # 第一列是期号
        except Exception as e:
            logger.error(f"读取本地数据失败: {str(e)}")

        return None

    def fetch_ssq_data(self, force_update=False):
        """获取双色球数据"""
        logger.info("开始获取双色球数据...")

        # 获取本地最新期号
        latest_local_issue = None if force_update else self._get_local_latest_issue('ssq')
        logger.info(f"本地最新期号: {latest_local_issue}")

        # 使用API直接获取数据
        api_url = "https://www.cwl.gov.cn/cwl_admin/front/cwlkj/search/kjxx/findDrawNotice"

        # 分页获取数据，初始值
        page_no = 1
        page_size = 30
        total_pages = 1  # 初始值，会在第一次请求后更新

        all_results = []

        try:
            while page_no <= total_pages:
                params = {
                    "name": "ssq",
                    "issueCount": "",
                    "issueStart": "",
                    "issueEnd": "",
                    "dayStart": "",
                    "dayEnd": "",
                    "pageNo": page_no,
                    "pageSize": page_size,
                    "week": "",
                    "systemType": "PC"
                }

                headers = {
                    'User-Agent': self._get_random_user_agent(),
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Accept-Language': 'zh,zh-CN;q=0.9,en-US;q=0.8,en;q=0.7',
                    'Referer': 'https://www.cwl.gov.cn/ygkj/wqkjgg/ssq/',
                    'X-Requested-With': 'XMLHttpRequest'
                }

                logger.info(f"请求双色球数据: 第{page_no}页")
                response = self._make_request(api_url, params=params, headers=headers)

                data = response.json()

                # 更新总页数
                if page_no == 1:
                    total_pages = data.get('pageNum', 1)
                    logger.info(f"双色球数据总页数: {total_pages}")

                # 检查请求是否成功
                if data.get('state') != 0:
                    logger.error(f"API请求失败: {data.get('message', '未知错误')}")
                    break

                # 处理当前页的结果
                page_results = data.get('result', [])

                for item in page_results:
                    code = item.get('code', '')

                    # 如果已经有此期数据且不是强制更新，跳过剩余处理
                    if latest_local_issue and code <= latest_local_issue and not force_update:
                        logger.info(f"已达到本地最新期号 {latest_local_issue}，停止获取")
                        return all_results

                    # 处理数据
                    draw_date = item.get('date', '').split('(')[0]  # 移除星期信息
                    red_balls = item.get('red', '').split(',')
                    blue_ball = item.get('blue', '')
                    prize_pool = item.get('poolmoney', '')

                    all_results.append({
                        'issue': code,
                        'date': draw_date,
                        'red_balls': red_balls,
                        'blue_ball': blue_ball,
                        'prize_pool': prize_pool
                    })

                # 增加请求间隔，避免被封
                time.sleep(self.config['crawler']['request_delay'])

                # 前往下一页
                page_no += 1

            logger.info(f"成功获取 {len(all_results)} 条双色球数据")
            return all_results

        except Exception as e:
            logger.error(f"获取双色球数据失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def fetch_dlt_data(self, force_update=False):
        """获取大乐透数据"""
        logger.info("开始获取大乐透数据...")

        # 获取本地最新期号
        latest_local_issue = None if force_update else self._get_local_latest_issue('dlt')
        logger.info(f"本地最新期号: {latest_local_issue}")

        # 体彩中心API
        api_url = "https://webapi.sporttery.cn/gateway/lottery/getHistoryPageListV1.qry"

        # 分页设置
        page_no = 1
        page_size = 30
        total_pages = 1  # 会在首次请求后更新

        all_results = []

        try:
            while page_no <= total_pages:
                params = {
                    "gameNo": "85",  # 大乐透代码
                    "provinceId": "0",
                    "pageSize": str(page_size),
                    "isVerify": "1",
                    "pageNo": str(page_no)
                }

                headers = {
                    'User-Agent': self._get_random_user_agent(),
                    'Accept': 'application/json, text/javascript, */*; q=0.01',
                    'Accept-Language': 'zh,zh-CN;q=0.9,en-US;q=0.8,en;q=0.7',
                    'Referer': 'https://static.sporttery.cn/',
                    'Origin': 'https://static.sporttery.cn',
                    'sec-ch-ua': '"Not(A:Brand";v="99", "Google Chrome";v="133", "Chromium";v="133"',
                    'sec-ch-ua-mobile': '?0',
                    'sec-ch-ua-platform': '"Windows"',
                    'sec-fetch-dest': 'empty',
                    'sec-fetch-mode': 'cors',
                    'sec-fetch-site': 'same-site'
                }

                logger.info(f"请求大乐透数据: 第{page_no}页")
                response = self._make_request(api_url, params=params, headers=headers)

                data = response.json()

                # 检查请求是否成功
                if not data.get('success'):
                    logger.error(f"API请求失败: {data.get('errorMessage', '未知错误')}")
                    break

                # 更新总页数
                if page_no == 1:
                    total_pages = data.get('value', {}).get('pages', 1)
                    logger.info(f"大乐透数据总页数: {total_pages}")

                # 处理当前页的结果
                items = data.get('value', {}).get('list', [])

                for item in items:
                    issue = item.get('lotteryDrawNum', '')

                    # 如果已经有此期数据且不是强制更新，则跳过剩余处理
                    if latest_local_issue and issue <= latest_local_issue and not force_update:
                        logger.info(f"已达到本地最新期号 {latest_local_issue}，停止获取")
                        return all_results

                    date = item.get('lotteryDrawTime', '')

                    # 提取号码
                    draw_result = item.get('lotteryDrawResult', '')
                    if draw_result:
                        numbers = draw_result.split(' ')
                        if len(numbers) >= 7:  # 确保有足够的号码
                            front_balls = numbers[:5]  # 前区5个号码
                            back_balls = numbers[5:7]  # 后区2个号码

                            # 处理奖池金额，去除逗号
                            prize_pool = item.get('poolBalanceAfterdraw', '').replace(',', '')

                            all_results.append({
                                'issue': issue,
                                'date': date,
                                'front_balls': front_balls,
                                'back_balls': back_balls,
                                'prize_pool': prize_pool
                            })

                # 增加请求间隔，避免被封
                time.sleep(self.config['crawler']['request_delay'])

                # 前往下一页
                page_no += 1

            logger.info(f"成功获取 {len(all_results)} 条大乐透数据")
            return all_results

        except Exception as e:
            logger.error(f"获取大乐透数据失败: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            return []

    def update_lottery_data(self, lottery_type, force_update=False):
        """更新指定彩票类型的数据"""
        if lottery_type == 'ssq':
            return self.fetch_ssq_data(force_update)
        elif lottery_type == 'dlt':
            return self.fetch_dlt_data(force_update)
        else:
            raise ValueError(f"不支持的彩票类型: {lottery_type}")
