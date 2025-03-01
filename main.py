#!/usr/bin/env python3
"""
彩票数据分析与预测系统 - 主程序
"""

import argparse
import json
import os
import sys
import logging
import pandas as pd
from tabulate import tabulate
from datetime import datetime

# 导入自定义模块
from modules.crawler import LotteryCrawler
from modules.storage import LotteryStorage
from modules.analyzer import LotteryAnalyzer
from modules.predictor import LotteryPredictor
from modules.backtest import LotteryBacktester
from modules.visualizer import LotteryVisualizer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("lottery_prediction.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('main')


class LotteryPredictionSystem:
    """彩票数据分析与预测系统"""

    def __init__(self, config_path='config.json'):
        """初始化系统"""
        self.config = self._load_config(config_path)

        # 初始化各个模块
        self.crawler = LotteryCrawler(config_path)
        self.storage = LotteryStorage(config_path)
        self.analyzer = LotteryAnalyzer()
        self.predictor = LotteryPredictor(config_path)
        self.backtester = LotteryBacktester(config_path)
        self.visualizer = LotteryVisualizer(config_path)

        logger.info("彩票数据分析与预测系统初始化完成")

    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def update_data(self, lottery_type, force_update=False):
        """更新彩票数据"""
        logger.info(f"开始更新{lottery_type}数据...")

        try:
            data = self.crawler.update_lottery_data(lottery_type, force_update)
            if data:
                self.storage.save_lottery_data(lottery_type, data, append=True)
                logger.info(f"成功更新{lottery_type}数据，共{len(data)}条记录")
                return len(data)
            else:
                logger.info(f"没有新的{lottery_type}数据需要更新")
                return 0
        except Exception as e:
            logger.error(f"更新{lottery_type}数据失败: {str(e)}")
            raise

    def analyze(self, lottery_type, analysis_type, period='all'):
        """分析彩票数据"""
        logger.info(f"开始分析{lottery_type}数据，分析类型: {analysis_type}, 周期: {period}")

        try:
            # 加载数据
            data = self.storage.load_lottery_data(lottery_type)
            if data.empty:
                logger.warning(f"没有{lottery_type}数据可供分析")
                return None

            # 设置数据
            self.analyzer.set_data(data)

            # 执行分析
            if analysis_type == 'frequency':
                result = self.analyzer.frequency_analysis(lottery_type, period)
            elif analysis_type == 'hot_cold':
                result = self.analyzer.hot_cold_analysis(lottery_type, period)
            elif analysis_type == 'odd_even':
                result = self.analyzer.odd_even_analysis(lottery_type, period)
            elif analysis_type == 'consecutive':
                result = self.analyzer.consecutive_numbers_analysis(lottery_type, period)
            elif analysis_type == 'sum':
                result = self.analyzer.sum_analysis(lottery_type, period)
            else:
                logger.error(f"不支持的分析类型: {analysis_type}")
                return None

            logger.info(f"完成{lottery_type}数据分析，分析类型: {analysis_type}")
            return result
        except Exception as e:
            logger.error(f"分析{lottery_type}数据失败: {str(e)}")
            raise

    def predict(self, lottery_type, algorithm=None):
        """预测彩票号码"""
        logger.info(f"开始预测{lottery_type}号码，算法: {algorithm}")

        try:
            # 加载数据
            data = self.storage.load_lottery_data(lottery_type)
            if data.empty:
                logger.warning(f"没有{lottery_type}数据可供预测")
                return None

            # 设置数据
            self.predictor.set_data(data)

            # 执行预测
            result = self.predictor.predict(lottery_type, algorithm)

            # 保存预测结果
            if result:
                self.storage.save_prediction(lottery_type, result.get('algorithm', ''), result)

            logger.info(f"完成{lottery_type}号码预测，算法: {result.get('algorithm', '')}")
            return result
        except Exception as e:
            logger.error(f"预测{lottery_type}号码失败: {str(e)}")
            raise

    def backtest(self, lottery_type, algorithm=None, test_periods=10):
        """回测预测算法"""
        logger.info(f"开始回测{lottery_type}预测算法: {algorithm}, 测试期数: {test_periods}")

        try:
            # 加载数据
            data = self.storage.load_lottery_data(lottery_type)
            if data.empty or len(data) <= test_periods:
                logger.warning(f"数据量不足，无法进行回测: {len(data) if not data.empty else 0} <= {test_periods}")
                return None

            # 执行回测
            if algorithm:
                # 单算法回测
                result = self.backtester.backtest_algorithm(
                    self.predictor, data, lottery_type, algorithm, test_periods
                )
            else:
                # 比较所有算法
                algorithms = list(self.config['prediction']['algorithms'].keys())
                result = self.backtester.compare_algorithms(
                    self.predictor, data, lottery_type, algorithms, test_periods
                )

            logger.info(f"完成{lottery_type}预测算法回测")
            return result
        except Exception as e:
            logger.error(f"回测{lottery_type}预测算法失败: {str(e)}")
            raise

    def visualize(self, data_or_result, chart_type, lottery_type=None, period='all'):
        """可视化数据或结果"""
        logger.info(f"开始生成{chart_type}图表")

        try:
            if chart_type == 'frequency':
                filepath = self.visualizer.plot_frequency_heatmap(data_or_result, lottery_type, period)
            elif chart_type == 'trend':
                filepath = self.visualizer.plot_trend_chart(data_or_result, lottery_type,
                                                            int(period) if period != 'all' else 30)
            elif chart_type == 'hot_cold':
                filepath = self.visualizer.plot_hot_cold_distribution(data_or_result)
            else:
                logger.error(f"不支持的图表类型: {chart_type}")
                return None

            logger.info(f"完成{chart_type}图表生成: {filepath}")
            return filepath
        except Exception as e:
            logger.error(f"生成{chart_type}图表失败: {str(e)}")
            raise


def print_table(data, headers=None, title=None):
    """以表格形式打印数据"""
    if title:
        print(f"\n=== {title} ===")

    if isinstance(data, pd.DataFrame):
        print(tabulate(data, headers='keys', tablefmt='fancy_grid'))
    elif isinstance(data, dict):
        if headers:
            rows = [[data.get(key, '') for key in headers]]
            print(tabulate(rows, headers=headers, tablefmt='fancy_grid'))
        else:
            rows = [[k, v] for k, v in data.items()]
            print(tabulate(rows, headers=['Key', 'Value'], tablefmt='fancy_grid'))
    elif isinstance(data, list) and all(isinstance(item, dict) for item in data):
        if headers:
            rows = [[item.get(key, '') for key in headers] for item in data]
            print(tabulate(rows, headers=headers, tablefmt='fancy_grid'))
        else:
            # 尝试从第一个字典获取所有键作为标题
            if data:
                headers = list(data[0].keys())
                rows = [[item.get(key, '') for key in headers] for item in data]
                print(tabulate(rows, headers=headers, tablefmt='fancy_grid'))
            else:
                print("空数据列表")
    else:
        print(data)


def format_prediction_result(result):
    """格式化预测结果输出"""
    if not result:
        return "无预测结果"

    lottery_type = result.get('lottery_type', '')
    algorithm = result.get('algorithm', '')
    target_issue = result.get('target_issue', '')
    primary_balls = result.get('primary_balls', [])
    secondary_balls = result.get('secondary_balls', [])

    # 检查是否由回测选择了算法
    if 'best_algorithm' in result:
        algorithm = f"{algorithm} (由回测选择的最佳算法)"

    output = [
        f"彩票类型: {lottery_type.upper()}",
        f"预测算法: {algorithm}",
        f"目标期号: {target_issue}",
        f"预测号码: {' '.join(map(str, primary_balls))} + {' '.join(map(str, secondary_balls))}",
        f"预测时间: {result.get('prediction_time', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}",
        f"\n声明: 预测结果仅供娱乐参考，请理性购彩"
    ]

    return "\n".join(output)


def print_backtest_result(result):
    """格式化输出回测结果"""
    print(f"\n=== {result['lottery_type'].upper()} 回测结果 (算法: {result['algorithm']}) ===\n")
    print(f"回测期数: {result['total_issues']}")
    print(f"中奖率: {result['winning_rate']:.2f}%")
    print(f"总奖金: ¥{result['total_prize']:.2f}")
    print(f"平均奖金: ¥{result['avg_prize']:.2f}")
    print(f"平均主区匹配: {result['avg_primary_match']:.2f} 个")
    print(f"平均副区匹配: {result['avg_secondary_match']:.2f} 个")

    print("\n奖级统计:")
    for level, count in result['prize_levels'].items():
        percent = (count / result['total_issues']) * 100
        print(f"  {level}: {count} 期 ({percent:.2f}%)")

    print("\n详细回测记录:")
    for r in result['detailed_results']:
        prize_str = f"¥{r['prize']}" if r['prize'] > 0 else "未中奖"
        match_str = f"{r['primary_match_count']}+{r['secondary_match_count']}"
        print(f"期号: {r['issue']} | 预测: {r['predicted_primary']}+{r['predicted_secondary']} | "
              f"开奖: {r['actual_primary']}+{r['actual_secondary']} | 匹配: {match_str} | 奖金: {prize_str}")


def print_algorithm_comparison(result):
    """格式化输出算法比较结果"""
    print("\n=== 算法表现比较 ===\n")

    # 提取所有算法的综合得分
    print("综合评分:")
    for algo, score in result['scores'].items():
        best_mark = " (最佳)" if algo == result['best_algorithm'] else ""
        print(f"  {algo}: {score:.2f}{best_mark}")

    # 打印详细指标比较
    print("\n详细指标比较:")
    metrics = ["winning_rate", "avg_prize", "avg_primary_match", "avg_secondary_match"]
    metric_names = {"winning_rate": "中奖率 (%)", "avg_prize": "平均奖金 (¥)",
                    "avg_primary_match": "平均主区匹配", "avg_secondary_match": "平均副区匹配"}

    # 打印表头
    print(f"{'指标':<15} | ", end="")
    for algo in result['scores'].keys():
        print(f"{algo:<10} | ", end="")
    print()
    print("-" * 50)

    # 打印各指标
    for metric in metrics:
        print(f"{metric_names[metric]:<15} | ", end="")
        for algo in result['scores'].keys():
            value = result['comparison'][algo][metric]
            if metric == "winning_rate":
                print(f"{value:.2f}%      | ", end="")
            elif metric == "avg_prize":
                print(f"{value:.2f}      | ", end="")
            else:
                print(f"{value:.2f}      | ", end="")
        print()

    # 打印奖级分布
    print("\n各算法中奖分布:")
    all_levels = set()
    for algo in result['scores'].keys():
        all_levels.update(result['comparison'][algo]['prize_levels'].keys())

    sorted_levels = sorted(all_levels, key=lambda x: 0 if x == "未中奖" else int(x[0]))

    # 打印表头
    print(f"{'奖级':<10} | ", end="")
    for algo in result['scores'].keys():
        print(f"{algo:<10} | ", end="")
    print()
    print("-" * 50)

    # 打印各奖级数据
    for level in sorted_levels:
        print(f"{level:<10} | ", end="")
        for algo in result['scores'].keys():
            count = result['comparison'][algo]['prize_levels'].get(level, 0)
            percent = (count / result['comparison'][algo]['total_issues']) * 100
            print(f"{count} ({percent:.1f}%) | ", end="")
        print()


def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='彩票数据分析与预测系统')

    # 定义子命令
    subparsers = parser.add_subparsers(dest='command', help='命令')

    # update 命令
    update_parser = subparsers.add_parser('update', help='抓取并更新最新数据')
    update_parser.add_argument('-t', '--type', required=True, choices=['ssq', 'dlt', 'all'], help='彩票类型')
    update_parser.add_argument('-f', '--force', action='store_true', help='强制更新所有数据')

    # analyze 命令
    analyze_parser = subparsers.add_parser('analyze', help='运行数据分析')
    analyze_parser.add_argument('-t', '--type', required=True, choices=['ssq', 'dlt'], help='彩票类型')
    analyze_parser.add_argument('-a', '--analysis', required=True,
                                choices=['frequency', 'hot_cold', 'odd_even', 'consecutive', 'sum'],
                                help='分析类型')
    analyze_parser.add_argument('-p', '--period', default='all', help='分析周期 (all/100/50/30)')
    analyze_parser.add_argument('-v', '--visualize', action='store_true', help='同时生成可视化图表')

    # predict 命令
    predict_parser = subparsers.add_parser('predict', help='生成预测结果')
    predict_parser.add_argument('-t', '--type', required=True, choices=['ssq', 'dlt'], help='彩票类型')
    predict_parser.add_argument('-a', '--algorithm', choices=['hot', 'cold', 'freq'], help='预测算法')
    # 预测命令的选项
    predict_parser.add_argument('-b', '--backtest', action='store_true', help='执行回测以评估预测准确率')
    predict_parser.add_argument('-B', '--best', action='store_true', help='使用历史表现最好的算法')
    predict_parser.add_argument('-p', '--periods', type=int, default=10, help='回测/评估的历史期数')
    predict_parser.add_argument('-i', '--issue', type=str, help='为指定期号预测（历史预测）')
    predict_parser.add_argument('-c', '--compare', action='store_true', help='比较所有算法的表现')

    # backtest 命令
    backtest_parser = subparsers.add_parser('backtest', help='回测算法有效性')
    backtest_parser.add_argument('-t', '--type', required=True, choices=['ssq', 'dlt'], help='彩票类型')
    backtest_parser.add_argument('-a', '--algorithm', choices=['hot', 'cold', 'freq'],
                                 help='预测算法（不指定则比较所有算法）')
    backtest_parser.add_argument('-p', '--periods', type=int, default=10, help='回测期数')
    backtest_parser.add_argument('-v', '--visualize', action='store_true', help='同时生成可视化图表')

    # visualize 命令
    visualize_parser = subparsers.add_parser('visualize', help='生成分析图表')
    visualize_parser.add_argument('-t', '--type', required=True, choices=['ssq', 'dlt'], help='彩票类型')
    visualize_parser.add_argument('-c', '--chart', required=True,
                                  choices=['frequency', 'trend', 'hot_cold'],
                                  help='图表类型')
    visualize_parser.add_argument('-p', '--period', default='all', help='分析周期 (all/100/50/30)')

    # 解析命令行
    args = parser.parse_args()

    # 如果没有命令，显示帮助
    if not args.command:
        parser.print_help()
        return

    # 初始化系统
    system = LotteryPredictionSystem()

    # 处理命令
    if args.command == 'update':
        if args.type == 'all':
            # 更新所有彩票类型
            updated_ssq = system.update_data('ssq', args.force)
            updated_dlt = system.update_data('dlt', args.force)
            print(f"更新完成: 双色球 {updated_ssq} 条记录, 大乐透 {updated_dlt} 条记录")
        else:
            # 更新指定彩票类型
            updated = system.update_data(args.type, args.force)
            print(f"更新完成: {args.type} {updated} 条记录")

    elif args.command == 'analyze':
        # 运行数据分析
        result = system.analyze(args.type, args.analysis, args.period)

        if result:
            # 格式化输出分析结果
            if args.analysis == 'frequency':
                print_table(result['primary_frequency'],
                            title=f"{args.type.upper()} 主区号码频率分析 (周期: {args.period})")
                print_table(result['secondary_frequency'],
                            title=f"{args.type.upper()} 副区号码频率分析 (周期: {args.period})")

            elif args.analysis == 'hot_cold':
                print(f"\n=== {args.type.upper()} 冷热号分析 (周期: {args.period}) ===")
                print(f"主区热号: {', '.join(map(str, result['primary_hot']))}")
                print(f"主区冷号: {', '.join(map(str, result['primary_cold']))}")
                print(f"副区热号: {', '.join(map(str, result['secondary_hot']))}")
                print(f"副区冷号: {', '.join(map(str, result['secondary_cold']))}")

            elif args.analysis == 'odd_even':
                print_table(result['primary_odd_even_distribution'],
                            title=f"{args.type.upper()} 主区奇偶比分析 (周期: {args.period})")
                print_table(result['secondary_odd_even_distribution'],
                            title=f"{args.type.upper()} 副区奇偶比分析 (周期: {args.period})")

            elif args.analysis == 'consecutive':
                print(f"\n=== {args.type.upper()} 连号分析 (周期: {args.period}) ===")
                print(f"总期数: {result['total_draws']}")
                print(f"出现连号期数: {result['consecutive_count']}")
                print(f"连号出现概率: {result['consecutive_percentage']:.2f}%")

            elif args.analysis == 'sum':
                print_table(result['primary_sum_stats'],
                            title=f"{args.type.upper()} 主区和值统计 (周期: {args.period})")
                print_table(result['secondary_sum_stats'],
                            title=f"{args.type.upper()} 副区和值统计 (周期: {args.period})")

            # 如果需要可视化，生成相应图表
            if args.visualize:
                if args.analysis == 'frequency':
                    chart_path = system.visualize(system.storage.load_lottery_data(args.type), 'frequency', args.type,
                                                  args.period)
                    print(f"频率热力图已保存: {chart_path}")

                elif args.analysis == 'hot_cold':
                    chart_path = system.visualize(result, 'hot_cold')
                    print(f"冷热号分布图已保存: {chart_path}")
        else:
            print(f"无法获取{args.type}的{args.analysis}分析结果")


    # 在处理predict命令的部分修改

    # 处理命令

    elif args.command == 'predict':
        if args.backtest:
            # 首先加载数据
            data = system.storage.load_lottery_data(args.type)
            if data.empty:
                print(f"没有{args.type}数据可供回测")
            else:
                # 设置数据到预测器
                system.predictor.set_data(data)
                # 执行回测
                result = system.predictor.backtest(args.type, algorithm=args.algorithm, periods=args.periods)
                if result:
                    print_backtest_result(result)
                else:
                    print(f"无法执行{args.type}的回测")
        elif args.compare:
            # 比较算法
            # 首先加载数据
            data = system.storage.load_lottery_data(args.type)
            if data.empty:
                print(f"没有{args.type}数据可供比较算法")
            else:
                system.predictor.set_data(data)
                result = system.predictor.compare_algorithms(args.type, periods=args.periods)
                if result:
                    print_algorithm_comparison(result)
                else:
                    print(f"无法比较{args.type}的算法表现")
        elif args.issue:
            # 针对特定期号的历史预测
            # 首先加载数据
            data = system.storage.load_lottery_data(args.type)
            if data.empty:
                print(f"没有{args.type}数据可供预测")
            else:
                system.predictor.set_data(data)
                result = system.predictor.predict_for_issue(args.type, args.issue, args.algorithm)
                if result:
                    print("\n" + format_prediction_result(result))
                else:
                    print(f"无法为{args.type}的{args.issue}期生成预测")
        elif args.best:
            # 使用历史最佳算法预测
            # 首先加载数据
            data = system.storage.load_lottery_data(args.type)
            if data.empty:
                print(f"没有{args.type}数据可供预测")
            else:
                system.predictor.set_data(data)
                result = system.predictor.predict_best(args.type, args.periods)
                if result:
                    print("\n" + format_prediction_result(result))
                    print("\n算法评估结果:")
                    for algo, score in result['algorithm_comparison']['algorithm_scores'].items():
                        mark = " (已选择)" if algo == result['algorithm_comparison']['best_algorithm'] else ""
                        print(f"  {algo}: {score:.2f}{mark}")
                else:
                    print(f"无法生成{args.type}的预测结果")
        else:
            # 常规预测
            # 加载数据
            data = system.storage.load_lottery_data(args.type)
            if data.empty:
                print(f"没有{args.type}数据可供预测")
            else:
                # 设置数据
                system.predictor.set_data(data)
                # 执行预测
                result = system.predict(args.type, args.algorithm)
                if result:
                    print("\n" + format_prediction_result(result))
                else:
                    print(f"无法生成{args.type}的预测结果")



    elif args.command == 'backtest':
        # 回测算法
        if args.algorithm:
            # 回测单个算法
            result = system.backtest(args.type, args.algorithm, args.periods)

            if result:
                print(f"\n=== {args.type.upper()} {args.algorithm} 算法回测结果 ===")
                print(f"测试期数: {result['test_periods']}")
                print(f"平均主区匹配数: {result['stats']['avg_primary_match']:.2f}")
                print(f"平均副区匹配数: {result['stats']['avg_secondary_match']:.2f}")
                print(f"总模拟奖金: ¥{result['stats']['total_prize']:.2f}")
                print(f"平均模拟奖金: ¥{result['stats']['avg_prize']:.2f}")
                print(f"中奖次数: {result['stats']['winning_times']}")
                print(f"中奖率: {result['stats']['winning_rate']:.2f}%")

                # 显示详细回测结果
                print("\n=== 回测详情 ===")
                for i, r in enumerate(result['results']):
                    print(f"\n[{i + 1}/{len(result['results'])}] 期号: {r['issue']}, 日期: {r['date']}")
                    print(
                        f"预测号码: {' '.join(map(str, r['prediction']['primary_balls']))} + {' '.join(map(str, r['prediction']['secondary_balls']))}")
                    print(
                        f"主区匹配: {r['match']['primary_match']}, 副区匹配: {r['match']['secondary_match']}, 奖金: ¥{r['match']['prize']}")
            else:
                print(f"无法进行{args.type}的{args.algorithm}算法回测")

        else:
            # 比较所有算法
            result = system.backtest(args.type, None, args.periods)

            if result:
                print(f"\n=== {args.type.upper()} 算法比较结果 (测试期数: {args.periods}) ===")

                # 显示比较结果
                comparison = result['comparison']
                for algo, stats in comparison.items():
                    print(f"\n[算法: {algo}]")
                    print(f"平均匹配数: 主区 {stats['avg_primary_match']:.2f}, 副区 {stats['avg_secondary_match']:.2f}")
                    print(f"平均奖金: ¥{stats['avg_prize']:.2f}, 中奖率: {stats['winning_rate']:.2f}%")
            else:
                print(f"无法进行{args.type}的算法比较")

    elif args.command == 'visualize':
        # 加载数据
        data = system.storage.load_lottery_data(args.type)

        if data.empty:
            print(f"没有{args.type}数据可供可视化")
            return

        # 生成图表
        if args.chart == 'frequency':
            chart_path = system.visualize(data, 'frequency', args.type, args.period)
        elif args.chart == 'trend':
            period = 30 if args.period == 'all' else int(args.period)
            chart_path = system.visualize(data, 'trend', args.type, period)
        elif args.chart == 'hot_cold':
            # 先进行热冷号分析
            result = system.analyze(args.type, 'hot_cold', args.period)
            if result:
                chart_path = system.visualize(result, 'hot_cold')
            else:
                print(f"无法获取{args.type}的热冷号分析结果")
                return

        if chart_path:
            print(f"图表已保存: {chart_path}")
        else:
            print(f"生成{args.type}的{args.chart}图表失败")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n程序已被用户中断")
        sys.exit(0)
    except Exception as e:
        logger.error(f"程序执行出错: {str(e)}", exc_info=True)
        print(f"程序执行出错: {str(e)}")
        sys.exit(1)
