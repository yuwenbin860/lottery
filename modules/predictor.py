"""
预测算法模块 - 实现各种彩票预测算法
"""

import logging
import numpy as np
import random
from datetime import datetime
import json
from collections import Counter

import pandas as pd

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

    def predict_with_backtest(self, lottery_type, test_periods=10):
        """基于回测结果的预测"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据可供预测")
            return None

        # 回测各算法表现
        algorithms = ['hot', 'cold', 'freq']

        # 创建回测器
        from modules.backtest import LotteryBacktester
        backtester = LotteryBacktester()

        logger.info(f"正在回测算法表现以选择最佳算法...")

        # 执行回测
        comparison = backtester.compare_algorithms(
            self, self.data, lottery_type, algorithms, test_periods
        )

        # 分析回测结果，选择最佳算法
        best_algo = None
        best_score = -1
        algo_stats = {}

        if comparison and 'comparison' in comparison:
            for algo, stats in comparison['comparison'].items():
                # 计算算法分数 (使用中奖率和平均奖金综合评分)
                winning_rate = stats.get('winning_rate', 0)
                avg_prize = stats.get('avg_prize', 0)

                # 记录每个算法的表现
                algo_stats[algo] = {
                    '中奖率': f"{winning_rate:.2f}%",
                    '平均奖金': f"¥{avg_prize:.2f}"
                }

                # 简单加权计算分数
                score = winning_rate * 0.7 + (avg_prize / 100) * 0.3

                if score > best_score:
                    best_score = score
                    best_algo = algo

        # 如果找到最佳算法，使用它预测
        if best_algo:
            logger.info(f"基于回测结果选择算法: {best_algo}")
            result = self.predict(lottery_type, best_algo)

            # 添加回测信息到结果
            if result:
                result['backtest_info'] = algo_stats
                result['best_algorithm'] = best_algo

            return result
        else:
            # 回退到默认算法
            logger.warning("回测结果不明确，使用默认算法")
            return self.predict(lottery_type)

    def _get_next_issue(self, lottery_type):
        """获取下一期期号"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据，无法确定下一期期号")
            return None

        # 确保数据按期号排序（降序，最新的在最前面）
        sorted_data = self.data.sort_values(by='issue', ascending=False)

        # 获取最新一期的期号并确保是字符串类型
        latest_issue = str(sorted_data['issue'].iloc[0])

        logger.info(f"当前最新期号: {latest_issue}")

        try:
            if lottery_type == 'ssq':
                # 双色球格式: YYYYNNN
                if len(latest_issue) == 7:
                    year = int(latest_issue[:4])
                    issue_num = int(latest_issue[4:])

                    # 计算下一期
                    issue_num += 1

                    # 处理跨年情况
                    if issue_num > 150:  # 双色球通常一年150期左右
                        year += 1
                        issue_num = 1

                    next_issue = f"{year}{issue_num:03d}"
                    logger.info(f"生成的下一期双色球期号: {next_issue}")
                    return next_issue

            # 如果以上格式都不匹配，尝试直接加1
            next_issue = str(int(latest_issue) + 1)
            logger.info(f"使用简单加1方式生成下一期期号: {next_issue}")
            return next_issue
        except Exception as e:
            logger.error(f"生成下一期期号失败: {str(e)}")

            # 尝试备用方案
            import datetime
            current_year = datetime.datetime.now().year

            # 估算当前期号：按当年天数比例估算
            day_of_year = datetime.datetime.now().timetuple().tm_yday
            estimated_issue = int(day_of_year / 365 * 30) + 22  # 2025年初已到21期，加上一些调整

            next_issue = f"{current_year}{estimated_issue:03d}"
            logger.info(f"使用当前日期估算的下一期期号: {next_issue}")
            return next_issue

    def predict_for_issue(self, lottery_type, target_issue, algorithm=None):
        """对指定期号进行预测（使用该期之前的数据）"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据可供预测")
            return None

        # 排序数据
        sorted_data = self.data.sort_values(by='issue')

        # 找到目标期号在数据中的位置
        target_data = sorted_data[sorted_data['issue'] == target_issue]
        if len(target_data) == 0:
            logger.warning(f"未找到目标期号: {target_issue}")
            return None

        target_idx = target_data.index[0]

        # 获取该期之前的所有数据
        historical_data = sorted_data.loc[:target_idx - 1]

        if len(historical_data) == 0:
            logger.warning(f"期号{target_issue}之前没有历史数据可用于预测")
            return None

        # 保存原始数据
        original_data = self.data

        try:
            # 设置历史数据用于预测
            self.data = historical_data

            # 执行预测，但目标期号为指定期号而非自动生成的下一期
            result = self.predict(lottery_type, algorithm)
            if result:
                result['target_issue'] = target_issue

            return result
        finally:
            # 恢复原始数据
            self.data = original_data

    def backtest(self, lottery_type, algorithm=None, periods=10, start_issue=None, end_issue=None):
        """对历史数据进行回测"""
        if self.data is None or len(self.data) == 0:
            logger.warning("没有数据可供回测")
            return None

        logger.info(f"执行回测: 彩票类型={lottery_type}, 算法={algorithm}, 期数={periods}")

        # 排序数据
        sorted_data = self.data.sort_values(by='issue')

        # 确定回测范围
        if start_issue and end_issue:
            test_data = sorted_data[(sorted_data['issue'] >= start_issue) &
                                    (sorted_data['issue'] <= end_issue)]
        elif periods:
            # 使用最近的N期进行回测
            test_data = sorted_data.tail(periods)
        else:
            logger.warning("未指定回测范围或期数")
            return None

        if len(test_data) == 0:
            logger.warning("回测范围内没有数据")
            return None

        logger.info(f"开始回测 {lottery_type} 的 {len(test_data)} 期数据，算法: {algorithm or '默认'}")

        # 回测代码实现...

    def compare_algorithms(self, lottery_type, periods=10):
        """比较不同算法的表现"""
        algorithms = ['hot', 'cold', 'freq']
        results = {}

        for algo in algorithms:
            logger.info(f"评估算法: {algo}")
            backtest_result = self.backtest(lottery_type, algorithm=algo, periods=periods)
            if backtest_result:
                results[algo] = backtest_result

        # 找出表现最好的算法
        if results:
            # 根据综合指标评分
            scores = {}
            for algo, result in results.items():
                # 创建加权评分
                score = (
                        result['winning_rate'] * 0.4 +  # 中奖率权重40%
                        result['avg_prize'] / 1000 * 0.3 +  # 平均奖金权重30%
                        result['avg_primary_match'] / len(
                    result['detailed_results'][0]['actual_primary']) * 0.2 +  # 主区匹配率20%
                        result['avg_secondary_match'] / len(result['detailed_results'][0]['actual_secondary']) * 0.1
                # 副区匹配率10%
                )
                scores[algo] = score

            best_algo = max(scores.items(), key=lambda x: x[1])[0]

            return {
                'comparison': results,
                'scores': scores,
                'best_algorithm': best_algo
            }

        return None

    def predict_best(self, lottery_type, periods=10):
        """使用历史表现最好的算法进行预测"""
        logger.info(f"正在分析 {periods} 期历史数据，确定最佳预测算法...")
        comparison = self.compare_algorithms(lottery_type, periods)

        if comparison and 'best_algorithm' in comparison:
            best_algo = comparison['best_algorithm']
            logger.info(f"根据历史表现，选择最佳算法: {best_algo}")

            # 使用最佳算法预测
            prediction = self.predict(lottery_type, best_algo)

            # 添加算法选择信息
            if prediction:
                prediction['algorithm_comparison'] = {
                    'evaluated_periods': periods,
                    'best_algorithm': best_algo,
                    'algorithm_scores': comparison['scores']
                }

            return prediction
        else:
            logger.warning("无法确定最佳算法，使用默认算法")
            return self.predict(lottery_type)

    def _get_prize_level(self, lottery_type, primary_match, secondary_match):
        """获取奖级"""
        if lottery_type == 'ssq':
            if primary_match == 6 and secondary_match == 1:
                return "一等奖"
            elif primary_match == 6 and secondary_match == 0:
                return "二等奖"
            elif primary_match == 5 and secondary_match == 1:
                return "三等奖"
            elif (primary_match == 5 and secondary_match == 0) or (primary_match == 4 and secondary_match == 1):
                return "四等奖"
            elif (primary_match == 4 and secondary_match == 0) or (primary_match == 3 and secondary_match == 1):
                return "五等奖"
            elif primary_match == 2 and secondary_match == 1 or primary_match == 1 and secondary_match == 1 or primary_match == 0 and secondary_match == 1:
                return "六等奖"
            else:
                return "未中奖"
        elif lottery_type == 'dlt':
            if primary_match == 5 and secondary_match == 2:
                return "一等奖"
            elif primary_match == 5 and secondary_match == 1:
                return "二等奖"
            elif primary_match == 5 and secondary_match == 0:
                return "三等奖"
            elif primary_match == 4 and secondary_match == 2:
                return "四等奖"
            elif (primary_match == 4 and secondary_match == 1):
                return "五等奖"
            elif (primary_match == 3 and secondary_match == 2):
                return "六等奖"
            elif (primary_match == 4 and secondary_match == 0):
                return "七等奖"
            elif (primary_match == 3 and secondary_match == 1) or (primary_match == 2 and secondary_match == 2):
                return "八等奖"
            elif (primary_match == 3 and secondary_match == 0) or (primary_match == 2 and secondary_match == 1) or (
                    primary_match == 1 and secondary_match == 2) or (primary_match == 0 and secondary_match == 2):
                return "九等奖"
            else:
                return "未中奖"
        return "未知"

    def _calculate_prize(self, lottery_type, primary_match, secondary_match):
        """估算奖金（使用典型奖金）"""
        prize_level = self._get_prize_level(lottery_type, primary_match, secondary_match)

        # 使用典型奖金估算，实际奖金会根据奖池和中奖人数变化
        if lottery_type == 'ssq':
            prizes = {
                "一等奖": 5000000,  # 500万
                "二等奖": 200000,  # 20万
                "三等奖": 3000,  # 3000元
                "四等奖": 200,  # 200元
                "五等奖": 10,  # 10元
                "六等奖": 5,  # 5元
                "未中奖": 0
            }
        elif lottery_type == 'dlt':
            prizes = {
                "一等奖": 5000000,  # 500万
                "二等奖": 100000,  # 10万
                "三等奖": 10000,  # 1万
                "四等奖": 3000,  # 3000元
                "五等奖": 300,  # 300元
                "六等奖": 200,  # 200元
                "七等奖": 100,  # 100元
                "八等奖": 15,  # 15元
                "九等奖": 5,  # 5元
                "未中奖": 0
            }
        else:
            return 0

        return prizes.get(prize_level, 0)

    def _calculate_prize(self, lottery_type, primary_match, secondary_match):
        """估算奖金（使用典型奖金）"""
        prize_level = self._get_prize_level(lottery_type, primary_match, secondary_match)

        # 使用典型奖金估算，实际奖金会根据奖池和中奖人数变化
        if lottery_type == 'ssq':
            prizes = {
                "一等奖": 5000000,  # 500万
                "二等奖": 200000,  # 20万
                "三等奖": 3000,  # 3000元
                "四等奖": 200,  # 200元
                "五等奖": 10,  # 10元
                "六等奖": 5,  # 5元
                "未中奖": 0
            }
        elif lottery_type == 'dlt':
            prizes = {
                "一等奖": 5000000,  # 500万
                "二等奖": 100000,  # 10万
                "三等奖": 10000,  # 1万
                "四等奖": 3000,  # 3000元
                "五等奖": 300,  # 300元
                "六等奖": 200,  # 200元
                "七等奖": 100,  # 100元
                "八等奖": 15,  # 15元
                "九等奖": 5,  # 5元
                "未中奖": 0
            }
        else:
            return 0

        return prizes.get(prize_level, 0)

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
