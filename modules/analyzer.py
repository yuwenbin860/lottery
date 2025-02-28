"""
数据分析模块 - 负责彩票数据的统计分析
"""

import logging
import numpy as np
import pandas as pd
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('analyzer')


class LotteryAnalyzer:
    """彩票数据分析类"""

    def __init__(self, data=None):
        """初始化分析器"""
        self.data = data  # pandas DataFrame

    def set_data(self, data):
        """设置数据集"""
        self.data = data
        logger.info(f"已设置数据集，共 {len(data)} 条记录")

    def _get_ball_columns(self, lottery_type):
        """获取指定彩票类型的球号列名"""
        if lottery_type == 'ssq':
            red_columns = ['red1', 'red2', 'red3', 'red4', 'red5', 'red6']
            blue_columns = ['blue']
            return red_columns, blue_columns
        elif lottery_type == 'dlt':
            front_columns = ['front1', 'front2', 'front3', 'front4', 'front5']
            back_columns = ['back1', 'back2']
            return front_columns, back_columns
        else:
            raise ValueError(f"不支持的彩票类型: {lottery_type}")

    def frequency_analysis(self, lottery_type, period='all'):
        """号码频率分析"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据可供分析")
            return None

        # 获取数据子集
        if period != 'all':
            try:
                period = int(period)
                if period > len(self.data):
                    period = len(self.data)
                df = self.data.tail(period)
            except ValueError:
                logger.warning(f"无效的周期值: {period}，使用全部数据")
                df = self.data
        else:
            df = self.data

        # 获取球号列
        primary_columns, secondary_columns = self._get_ball_columns(lottery_type)

        # 分析主区号码频率
        primary_freq = {}
        for col in primary_columns:
            for num in df[col]:
                if pd.notna(num):  # 排除NaN值
                    num = int(num)
                    primary_freq[num] = primary_freq.get(num, 0) + 1

        # 分析副区号码频率
        secondary_freq = {}
        for col in secondary_columns:
            for num in df[col]:
                if pd.notna(num):  # 排除NaN值
                    num = int(num)
                    secondary_freq[num] = secondary_freq.get(num, 0) + 1

        # 计算出现概率
        total_primary = len(df) * len(primary_columns)
        total_secondary = len(df) * len(secondary_columns)

        primary_prob = {k: v / total_primary for k, v in primary_freq.items()}
        secondary_prob = {k: v / total_secondary for k, v in secondary_freq.items()}

        result = {
            'lottery_type': lottery_type,
            'period': period,
            'total_draws': len(df),
            'primary_frequency': primary_freq,
            'primary_probability': primary_prob,
            'secondary_frequency': secondary_freq,
            'secondary_probability': secondary_prob
        }

        logger.info(f"完成频率分析: {lottery_type}, 周期: {period}")
        return result

    def hot_cold_analysis(self, lottery_type, period=30):
        """热冷号分析"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据可供分析")
            return None

        # 获取最近的数据
        recent_data = self.data.tail(period)

        # 获取球号列
        primary_columns, secondary_columns = self._get_ball_columns(lottery_type)

        # 分析主区号码
        primary_numbers = []
        for col in primary_columns:
            primary_numbers.extend([int(x) for x in recent_data[col] if pd.notna(x)])

        primary_counter = Counter(primary_numbers)

        # 分析副区号码
        secondary_numbers = []
        for col in secondary_columns:
            secondary_numbers.extend([int(x) for x in recent_data[col] if pd.notna(x)])

        secondary_counter = Counter(secondary_numbers)

        # 确定热号和冷号的阈值（示例：出现次数排名前25%为热号，后25%为冷号）
        def get_hot_cold(counter, total_possible_numbers):
            # 确保所有可能的号码都在计数器中
            all_numbers = list(range(1, total_possible_numbers + 1))
            for num in all_numbers:
                if num not in counter:
                    counter[num] = 0

            sorted_items = sorted(counter.items(), key=lambda x: x[1], reverse=True)

            hot_threshold = len(sorted_items) // 4
            cold_threshold = len(sorted_items) - len(sorted_items) // 4

            hot_numbers = [item[0] for item in sorted_items[:hot_threshold]]
            cold_numbers = [item[0] for item in sorted_items[cold_threshold:]]

            return hot_numbers, cold_numbers

        # 获取热号和冷号
        primary_hot, primary_cold = get_hot_cold(primary_counter, 33 if lottery_type == 'ssq' else 35)
        secondary_hot, secondary_cold = get_hot_cold(secondary_counter, 16 if lottery_type == 'ssq' else 12)

        result = {
            'lottery_type': lottery_type,
            'period': period,
            'primary_hot': primary_hot,
            'primary_cold': primary_cold,
            'secondary_hot': secondary_hot,
            'secondary_cold': secondary_cold,
            'primary_counter': dict(primary_counter),
            'secondary_counter': dict(secondary_counter)
        }

        logger.info(f"完成热冷号分析: {lottery_type}, 周期: {period}")
        return result

    def odd_even_analysis(self, lottery_type, period='all'):
        """奇偶比分析"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据可供分析")
            return None

        # 获取数据子集
        if period != 'all':
            try:
                period = int(period)
                if period > len(self.data):
                    period = len(self.data)
                df = self.data.tail(period)
            except ValueError:
                logger.warning(f"无效的周期值: {period}，使用全部数据")
                df = self.data
        else:
            df = self.data

        # 获取球号列
        primary_columns, secondary_columns = self._get_ball_columns(lottery_type)

        # 计算奇偶分布
        def calculate_odd_even(row, columns):
            odd_count = 0
            even_count = 0

            for col in columns:
                if pd.notna(row[col]):
                    num = int(row[col])
                    if num % 2 == 0:
                        even_count += 1
                    else:
                        odd_count += 1

            return odd_count, even_count

        # 对每一期计算奇偶分布
        primary_distributions = []
        secondary_distributions = []

        for _, row in df.iterrows():
            primary_odd, primary_even = calculate_odd_even(row, primary_columns)
            secondary_odd, secondary_even = calculate_odd_even(row, secondary_columns)

            primary_distributions.append((primary_odd, primary_even))
            secondary_distributions.append((secondary_odd, secondary_even))

        # 统计各种奇偶比例的次数
        primary_odd_even_counts = Counter(primary_distributions)
        secondary_odd_even_counts = Counter(secondary_distributions)

        result = {
            'lottery_type': lottery_type,
            'period': period,
            'total_draws': len(df),
            'primary_odd_even_distribution': {f"{odd}:{even}": count
                                              for (odd, even), count in primary_odd_even_counts.items()},
            'secondary_odd_even_distribution': {f"{odd}:{even}": count
                                                for (odd, even), count in secondary_odd_even_counts.items()}
        }

        logger.info(f"完成奇偶比分析: {lottery_type}, 周期: {period}")
        return result

    def consecutive_numbers_analysis(self, lottery_type, period='all'):
        """连号分析"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据可供分析")
            return None

        # 获取数据子集
        if period != 'all':
            try:
                period = int(period)
                if period > len(self.data):
                    period = len(self.data)
                df = self.data.tail(period)
            except ValueError:
                logger.warning(f"无效的周期值: {period}，使用全部数据")
                df = self.data
        else:
            df = self.data

        # 获取球号列
        primary_columns, _ = self._get_ball_columns(lottery_type)

        # 检测每一期是否有连号
        consecutive_count = 0
        consecutive_details = []

        for _, row in df.iterrows():
            # 提取并排序主区号码
            numbers = sorted([int(row[col]) for col in primary_columns if pd.notna(row[col])])

            # 检查是否有连号
            has_consecutive = False
            consecutive_pairs = []

            for i in range(len(numbers) - 1):
                if numbers[i + 1] - numbers[i] == 1:
                    has_consecutive = True
                    consecutive_pairs.append((numbers[i], numbers[i + 1]))

            if has_consecutive:
                consecutive_count += 1
                consecutive_details.append({
                    'issue': row.get('issue', ''),
                    'date': row.get('date', ''),
                    'numbers': numbers,
                    'consecutive_pairs': consecutive_pairs
                })

        result = {
            'lottery_type': lottery_type,
            'period': period,
            'total_draws': len(df),
            'consecutive_count': consecutive_count,
            'consecutive_percentage': consecutive_count / len(df) * 100 if len(df) > 0 else 0,
            'consecutive_details': consecutive_details
        }

        logger.info(f"完成连号分析: {lottery_type}, 周期: {period}")
        return result

    def sum_analysis(self, lottery_type, period='all'):
        """和值分析"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据可供分析")
            return None

        # 获取数据子集
        if period != 'all':
            try:
                period = int(period)
                if period > len(self.data):
                    period = len(self.data)
                df = self.data.tail(period)
            except ValueError:
                logger.warning(f"无效的周期值: {period}，使用全部数据")
                df = self.data
        else:
            df = self.data

        # 获取球号列
        primary_columns, secondary_columns = self._get_ball_columns(lottery_type)

        # 计算每期的和值
        primary_sums = []
        secondary_sums = []

        for _, row in df.iterrows():
            # 计算主区和值
            primary_sum = sum(int(row[col]) for col in primary_columns if pd.notna(row[col]))
            primary_sums.append(primary_sum)

            # 计算副区和值
            secondary_sum = sum(int(row[col]) for col in secondary_columns if pd.notna(row[col]))
            secondary_sums.append(secondary_sum)

        # 统计和值分布
        primary_sum_counter = Counter(primary_sums)
        secondary_sum_counter = Counter(secondary_sums)

        # 计算和值统计信息
        primary_sum_stats = {
            'min': min(primary_sums) if primary_sums else None,
            'max': max(primary_sums) if primary_sums else None,
            'mean': sum(primary_sums) / len(primary_sums) if primary_sums else None,
            'median': sorted(primary_sums)[len(primary_sums) // 2] if primary_sums else None
        }

        secondary_sum_stats = {
            'min': min(secondary_sums) if secondary_sums else None,
            'max': max(secondary_sums) if secondary_sums else None,
            'mean': sum(secondary_sums) / len(secondary_sums) if secondary_sums else None,
            'median': sorted(secondary_sums)[len(secondary_sums) // 2] if secondary_sums else None
        }

        result = {
            'lottery_type': lottery_type,
            'period': period,
            'total_draws': len(df),
            'primary_sum_distribution': dict(primary_sum_counter),
            'primary_sum_stats': primary_sum_stats,
            'secondary_sum_distribution': dict(secondary_sum_counter),
            'secondary_sum_stats': secondary_sum_stats
        }

        logger.info(f"完成和值分析: {lottery_type}, 周期: {period}")
        return result
