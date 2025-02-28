"""
预测算法模块 - 实现各种彩票预测算法
"""

import logging
import numpy as np
import random
from datetime import datetime
import json
from collections import Counter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('predictor')


class LotteryPredictor:
    """彩票预测类"""

    def __init__(self, config_path='config.json', data=None):
        """初始化预测器"""
        self.config = self._load_config(config_path)
        self.data = data  # pandas DataFrame

    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise

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

    def _get_ball_ranges(self, lottery_type):
        """获取指定彩票类型的号码范围"""
        if lottery_type == 'ssq':
            return (1, 33), (1, 16)  # (红球范围, 蓝球范围)
        elif lottery_type == 'dlt':
            return (1, 35), (1, 12)  # (前区范围, 后区范围)
        else:
            raise ValueError(f"不支持的彩票类型: {lottery_type}")

    def _get_next_issue(self, lottery_type):
        """获取下一期期号"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据，无法确定下一期期号")
            return None

        # 获取最新一期的期号
        latest_issue = self.data['issue'].iloc[-1]

        # 简单示例：期号加1
        try:
            next_issue = str(int(latest_issue) + 1)
            return next_issue
        except:
            logger.warning(f"无法从当前期号生成下一期期号: {latest_issue}")
            return None

    def hot_number_prediction(self, lottery_type):
        """热号加权随机算法"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据可供预测")
            return None

        # 获取算法参数
        algo_params = self.config['prediction']['algorithms']['hot']['params']
        period = algo_params.get('period', 50)
        weight_factor = algo_params.get('weight_factor', 1.5)

        # 获取最近的数据
        recent_data = self.data.tail(period)

        # 获取球号列和范围
        primary_columns, secondary_columns = self._get_ball_columns(lottery_type)
        primary_range, secondary_range = self._get_ball_ranges(lottery_type)

        # 统计主区号码频率
        primary_counter = Counter()
        for col in primary_columns:
            for num in recent_data[col]:
                if pd.notna(num):
                    primary_counter[int(num)] += 1

        # 统计副区号码频率
        secondary_counter = Counter()
        for col in secondary_columns:
            for num in recent_data[col]:
                if pd.notna(num):
                    secondary_counter[int(num)] += 1

        # 计算加权概率
        primary_weights = {}
        for i in range(primary_range[0], primary_range[1] + 1):
            # 出现次数越多，权重越大
            count = primary_counter.get(i, 0)
            primary_weights[i] = count * weight_factor if count > 0 else 1

        secondary_weights = {}
        for i in range(secondary_range[0], secondary_range[1] + 1):
            count = secondary_counter.get(i, 0)
            secondary_weights[i] = count * weight_factor if count > 0 else 1

        # 根据权重随机选择
        primary_balls = []
        primary_numbers = list(range(primary_range[0], primary_range[1] + 1))
        primary_weights_list = [primary_weights[i] for i in primary_numbers]

        # 选择主区号码
        while len(primary_balls) < len(primary_columns):
            choice = random.choices(primary_numbers, weights=primary_weights_list, k=1)[0]
            if choice not in primary_balls:
                primary_balls.append(choice)

        # 选择副区号码
        secondary_balls = []
        secondary_numbers = list(range(secondary_range[0], secondary_range[1] + 1))
        secondary_weights_list = [secondary_weights[i] for i in secondary_numbers]

        while len(secondary_balls) < len(secondary_columns):
            choice = random.choices(secondary_numbers, weights=secondary_weights_list, k=1)[0]
            if lottery_type == 'dlt':
                # 大乐透的后区不能重复
                if choice not in secondary_balls:
                    secondary_balls.append(choice)
            else:
                # 双色球的蓝球是单个的
                secondary_balls.append(choice)
                break

        # 排序
        primary_balls.sort()
        secondary_balls.sort()

        # 生成预测结果
        next_issue = self._get_next_issue(lottery_type)

        result = {
            'lottery_type': lottery_type,
            'algorithm': 'hot',
            'target_issue': next_issue,
            'primary_balls': primary_balls,
            'secondary_balls': secondary_balls,
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        logger.info(f"完成热号预测: {lottery_type}, 目标期号: {next_issue}")
        return result

    def cold_number_prediction(self, lottery_type):
        """冷号补偿算法"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据可供预测")
            return None

        # 获取算法参数
        algo_params = self.config['prediction']['algorithms']['cold']['params']
        threshold = algo_params.get('threshold', 20)
        max_cold_numbers = algo_params.get('max_cold_numbers', 2)

        # 获取球号列和范围
        primary_columns, secondary_columns = self._get_ball_columns(lottery_type)
        primary_range, secondary_range = self._get_ball_ranges(lottery_type)

        # 计算每个号码的最近出现期数
        def calculate_last_appearance(ball_range, columns):
            last_appearance = {}

            # 初始化所有可能的号码
            for i in range(ball_range[0], ball_range[1] + 1):
                last_appearance[i] = len(self.data)  # 默认为最大期数

            # 从最近一期往前遍历
            for idx, row in self.data.iloc[::-1].iterrows():
                position = len(self.data) - idx - 1  # 距离最近一期的期数

                # 检查当前行的号码
                for col in columns:
                    if pd.notna(row[col]):
                        num = int(row[col])
                        # 如果是第一次遇到该号码，记录其位置
                        if last_appearance[num] == len(self.data):
                            last_appearance[num] = position

            return last_appearance

        # 计算主区和副区号码的最近出现期数
        primary_last_appearance = calculate_last_appearance(primary_range, primary_columns)
        secondary_last_appearance = calculate_last_appearance(secondary_range, secondary_columns)

        # 找出冷号（超过阈值未出现的号码）
        primary_cold = [num for num, pos in primary_last_appearance.items() if pos >= threshold]
        secondary_cold = [num for num, pos in secondary_last_appearance.items() if pos >= threshold]

        # 预测主区号码
        primary_balls = []

        # 从冷号中选择，但不超过max_cold_numbers个
        random.shuffle(primary_cold)
        cold_count = min(len(primary_cold), max_cold_numbers)
        primary_balls.extend(primary_cold[:cold_count])

        # 剩余的从非冷号中随机选择
        primary_warm = [i for i in range(primary_range[0], primary_range[1] + 1) if i not in primary_cold]
        remaining_count = len(primary_columns) - len(primary_balls)

        while len(primary_balls) < len(primary_columns):
            choice = random.choice(primary_warm)
            if choice not in primary_balls:
                primary_balls.append(choice)

        # 预测副区号码
        secondary_balls = []

        # 从冷号中选择
        random.shuffle(secondary_cold)
        cold_count = min(len(secondary_cold), 1)  # 副区通常只选一个冷号
        if cold_count > 0:
            secondary_balls.extend(secondary_cold[:cold_count])

        # 剩余的从非冷号中随机选择
        secondary_warm = [i for i in range(secondary_range[0], secondary_range[1] + 1) if i not in secondary_cold]

        while len(secondary_balls) < len(secondary_columns):
            choice = random.choice(secondary_warm)
            if lottery_type == 'dlt':
                # 大乐透的后区不能重复
                if choice not in secondary_balls:
                    secondary_balls.append(choice)
            else:
                # 双色球的蓝球是单个的
                secondary_balls.append(choice)
                break

        # 排序
        primary_balls.sort()
        secondary_balls.sort()

        # 生成预测结果
        next_issue = self._get_next_issue(lottery_type)

        result = {
            'lottery_type': lottery_type,
            'algorithm': 'cold',
            'target_issue': next_issue,
            'primary_balls': primary_balls,
            'secondary_balls': secondary_balls,
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        logger.info(f"完成冷号预测: {lottery_type}, 目标期号: {next_issue}")
        return result

    def frequency_prediction(self, lottery_type):
        """频率分析预测算法"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据可供预测")
            return None

        # 获取算法参数
        algo_params = self.config['prediction']['algorithms']['freq']['params']
        period = algo_params.get('period', 'all')

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

        # 获取球号列和范围
        primary_columns, secondary_columns = self._get_ball_columns(lottery_type)
        primary_range, secondary_range = self._get_ball_ranges(lottery_type)

        # 统计号码频率
        primary_counter = Counter()
        for col in primary_columns:
            for num in df[col]:
                if pd.notna(num):
                    primary_counter[int(num)] += 1

        secondary_counter = Counter()
        for col in secondary_columns:
            for num in df[col]:
                if pd.notna(num):
                    secondary_counter[int(num)] += 1

        # 计算每个号码的出现概率
        total_primary_draws = len(df) * len(primary_columns)
        total_secondary_draws = len(df) * len(secondary_columns)

        primary_probs = {}
        for i in range(primary_range[0], primary_range[1] + 1):
            primary_probs[i] = primary_counter.get(i, 0) / total_primary_draws

        secondary_probs = {}
        for i in range(secondary_range[0], secondary_range[1] + 1):
            secondary_probs[i] = secondary_counter.get(i, 0) / total_secondary_draws

        # 根据概率选择
        primary_numbers = list(range(primary_range[0], primary_range[1] + 1))
        primary_weights = [primary_probs[i] for i in primary_numbers]

        secondary_numbers = list(range(secondary_range[0], secondary_range[1] + 1))
        secondary_weights = [secondary_probs[i] for i in secondary_numbers]

        # 选择主区号码
        primary_balls = []
        while len(primary_balls) < len(primary_columns):
            choice = random.choices(primary_numbers, weights=primary_weights, k=1)[0]
            if choice not in primary_balls:
                primary_balls.append(choice)

        # 选择副区号码
        secondary_balls = []
        while len(secondary_balls) < len(secondary_columns):
            choice = random.choices(secondary_numbers, weights=secondary_weights, k=1)[0]
            if lottery_type == 'dlt':
                # 大乐透的后区不能重复
                if choice not in secondary_balls:
                    secondary_balls.append(choice)
            else:
                # 双色球的蓝球是单个的
                secondary_balls.append(choice)
                break

        # 排序
        primary_balls.sort()
        secondary_balls.sort()

        # 生成预测结果
        next_issue = self._get_next_issue(lottery_type)

        result = {
            'lottery_type': lottery_type,
            'algorithm': 'freq',
            'target_issue': next_issue,
            'primary_balls': primary_balls,
            'secondary_balls': secondary_balls,
            'prediction_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        logger.info(f"完成频率预测: {lottery_type}, 目标期号: {next_issue}")
        return result

    def predict(self, lottery_type, algorithm=None):
        """执行预测"""
        if algorithm is None:
            algorithm = self.config['prediction']['default_algorithm']

        if algorithm == 'hot':
            return self.hot_number_prediction(lottery_type)
        elif algorithm == 'cold':
            return self.cold_number_prediction(lottery_type)
        elif algorithm == 'freq':
            return self.frequency_prediction(lottery_type)
        else:
            raise ValueError(f"不支持的预测算法: {algorithm}")
