"""
回测验证模块 - 用于验证预测算法的准确性
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('backtest')


class LotteryBacktester:
    """彩票预测回测类"""

    def __init__(self, config_path='config.json'):
        """初始化回测器"""
        self.config = self._load_config(config_path)

    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise

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

    def _calculate_match(self, prediction, actual, lottery_type):
        """计算预测与实际结果的匹配程度"""
        primary_columns, secondary_columns = self._get_ball_columns(lottery_type)

        # 获取预测号码
        pred_primary = prediction.get('primary_balls', [])
        pred_secondary = prediction.get('secondary_balls', [])

        # 获取实际号码
        actual_primary = [int(actual[col]) for col in primary_columns if pd.notna(actual[col])]
        actual_secondary = [int(actual[col]) for col in secondary_columns if pd.notna(actual[col])]

        # 计算匹配数
        primary_match = len(set(pred_primary) & set(actual_primary))
        secondary_match = len(set(pred_secondary) & set(actual_secondary))

        # 计算模拟奖金（简化版）
        prize = self._calculate_prize(primary_match, secondary_match, lottery_type)

        return {
            'primary_match': primary_match,
            'secondary_match': secondary_match,
            'total_match': primary_match + secondary_match,
            'prize': prize
        }

    def _calculate_prize(self, primary_match, secondary_match, lottery_type):
        """计算中奖金额（简化版）"""
        if lottery_type == 'ssq':
            # 双色球简化奖金规则
            if primary_match == 6 and secondary_match == 1:
                return 5000000  # 一等奖（简化）
            elif primary_match == 6 and secondary_match == 0:
                return 100000  # 二等奖（简化）
            elif primary_match == 5 and secondary_match == 1:
                return 3000  # 三等奖（简化）
            elif primary_match == 5 and secondary_match == 0 or primary_match == 4 and secondary_match == 1:
                return 200  # 四等奖（简化）
            elif primary_match == 4 and secondary_match == 0 or primary_match == 3 and secondary_match == 1:
                return 10  # 五等奖（简化）
            elif secondary_match == 1:
                return 5  # 六等奖（简化）
            else:
                return 0

        elif lottery_type == 'dlt':
            # 大乐透简化奖金规则
            if primary_match == 5 and secondary_match == 2:
                return 10000000  # 一等奖（简化）
            elif primary_match == 5 and secondary_match == 1:
                return 500000  # 二等奖（简化）
            elif primary_match == 5 and secondary_match == 0:
                return 10000  # 三等奖（简化）
            elif primary_match == 4 and secondary_match == 2:
                return 3000  # 四等奖（简化）
            elif primary_match == 4 and secondary_match == 1:
                return 300  # 五等奖（简化）
            elif primary_match == 3 and secondary_match == 2:
                return 200  # 六等奖（简化）
            elif primary_match == 4 and secondary_match == 0:
                return 100  # 七等奖（简化）
            elif primary_match == 3 and secondary_match == 1 or primary_match == 2 and secondary_match == 2:
                return 15  # 八等奖（简化）
            elif primary_match == 3 and secondary_match == 0 or primary_match == 1 and secondary_match == 2 or primary_match == 2 and secondary_match == 1:
                return 5  # 九等奖（简化）
            else:
                return 0

        else:
            raise ValueError(f"不支持的彩票类型: {lottery_type}")

    def backtest_algorithm(self, predictor, data, lottery_type, algorithm, test_periods=10):
        """回测指定算法的预测效果"""
        if len(data) <= test_periods:
            logger.warning(f"数据量不足，无法进行回测: {len(data)} <= {test_periods}")
            return None

        # 模拟历史时点进行预测
        results = []

        for i in range(test_periods):
            # 计算测试位置
            test_idx = len(data) - 1 - i
            train_data = data.iloc[:test_idx]
            test_data = data.iloc[test_idx:test_idx + 1]

            # 设置训练数据
            predictor.set_data(train_data)

            # 进行预测
            prediction = predictor.predict(lottery_type, algorithm)

            # 获取实际结果
            actual = test_data.iloc[0]

            # 计算匹配度
            match_result = self._calculate_match(prediction, actual, lottery_type)

            # 记录结果
            results.append({
                'issue': actual.get('issue', ''),
                'date': actual.get('date', ''),
                'prediction': prediction,
                'match': match_result
            })

        # 计算回测统计信息
        stats = self._calculate_backtest_stats(results)

        logger.info(f"完成回测: {lottery_type}, 算法: {algorithm}, 测试期数: {test_periods}")

        return {
            'lottery_type': lottery_type,
            'algorithm': algorithm,
            'test_periods': test_periods,
            'results': results,
            'stats': stats
        }

    def _calculate_backtest_stats(self, results):
        """计算回测统计信息"""
        if not results:
            return {}

        # 提取匹配结果
        primary_matches = [r['match']['primary_match'] for r in results]
        secondary_matches = [r['match']['secondary_match'] for r in results]
        prizes = [r['match']['prize'] for r in results]

        # 计算统计信息
        stats = {
            'total_tests': len(results),
            'avg_primary_match': sum(primary_matches) / len(results),
            'avg_secondary_match': sum(secondary_matches) / len(results),
            'total_prize': sum(prizes),
            'avg_prize': sum(prizes) / len(results),
            'winning_times': sum(1 for p in prizes if p > 0),
            'winning_rate': sum(1 for p in prizes if p > 0) / len(results) * 100
        }

        return stats

    def compare_algorithms(self, predictor, data, lottery_type, algorithms, test_periods=10):
        """比较多个算法的回测效果"""
        results = {}

        for algo in algorithms:
            backtest_result = self.backtest_algorithm(predictor, data, lottery_type, algo, test_periods)
            results[algo] = backtest_result

        # 比较算法效果
        comparison = {}
        for algo, result in results.items():
            if result:
                comparison[algo] = {
                    'avg_prize': result['stats']['avg_prize'],
                    'winning_rate': result['stats']['winning_rate'],
                    'avg_primary_match': result['stats']['avg_primary_match'],
                    'avg_secondary_match': result['stats']['avg_secondary_match']
                }

        logger.info(f"完成算法比较: {lottery_type}, 算法: {algorithms}")

        return {
            'lottery_type': lottery_type,
            'test_periods': test_periods,
            'algorithm_results': results,
            'comparison': comparison
        }
