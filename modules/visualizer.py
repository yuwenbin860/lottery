"""
数据可视化模块 - 实现彩票数据的图表可视化
"""

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import os
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('visualizer')


class LotteryVisualizer:
    """彩票数据可视化类"""

    def __init__(self, config_path='config.json'):
        """初始化可视化工具"""
        self.config = self._load_config(config_path)
        self.data_dir = self.config['storage']['data_dir']

        # 设置图表风格
        sns.set_style("whitegrid")

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

    def _get_ball_ranges(self, lottery_type):
        """获取指定彩票类型的号码范围"""
        if lottery_type == 'ssq':
            return (1, 33), (1, 16)  # (红球范围, 蓝球范围)
        elif lottery_type == 'dlt':
            return (1, 35), (1, 12)  # (前区范围, 后区范围)
        else:
            raise ValueError(f"不支持的彩票类型: {lottery_type}")

    def _ensure_output_dir(self, output_dir=None):
        """确保输出目录存在"""
        if output_dir is None:
            output_dir = os.path.join(self.data_dir, 'visualizations')

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"创建可视化输出目录: {output_dir}")

        return output_dir

    def plot_frequency_heatmap(self, data, lottery_type, period='all', output_dir=None):
        """绘制号码频率热力图"""
        if data is None or len(data) == 0:
            logger.warning("没有数据可供可视化")
            return None

        # 获取数据子集
        if period != 'all':
            try:
                period = int(period)
                if period > len(data):
                    period = len(data)
                df = data.tail(period)
            except ValueError:
                logger.warning(f"无效的周期值: {period}，使用全部数据")
                df = data
        else:
            df = data

        # 获取球号列和范围
        primary_columns, secondary_columns = self._get_ball_columns(lottery_type)
        primary_range, secondary_range = self._get_ball_ranges(lottery_type)

        # 计算主区号码频率
        primary_freq = np.zeros(primary_range[1] + 1)
        for col in primary_columns:
            for num in df[col]:
                if pd.notna(num):
                    primary_freq[int(num)] += 1

        # 计算副区号码频率
        secondary_freq = np.zeros(secondary_range[1] + 1)
        for col in secondary_columns:
            for num in df[col]:
                if pd.notna(num):
                    secondary_freq[int(num)] += 1

        # 创建热力图数据
        primary_freq = primary_freq[primary_range[0]:]
        secondary_freq = secondary_freq[secondary_range[0]:]

        # 创建子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 主区热力图
        primary_title = f"主区号码频率分布 ({lottery_type.upper()}, 期数: {period if period != 'all' else len(df)})"
        sns.heatmap(
            primary_freq.reshape(1, -1),
            annot=True,
            fmt='g',
            cmap='YlOrRd',
            ax=ax1,
            xticklabels=range(primary_range[0], primary_range[1] + 1),
            yticklabels=False,
            cbar_kws={'label': '出现次数'}
        )
        ax1.set_title(primary_title)
        ax1.set_xlabel('号码')

        # 副区热力图
        secondary_title = f"副区号码频率分布 ({lottery_type.upper()}, 期数: {period if period != 'all' else len(df)})"
        sns.heatmap(
            secondary_freq.reshape(1, -1),
            annot=True,
            fmt='g',
            cmap='YlOrRd',
            ax=ax2,
            xticklabels=range(secondary_range[0], secondary_range[1] + 1),
            yticklabels=False,
            cbar_kws={'label': '出现次数'}
        )
        ax2.set_title(secondary_title)
        ax2.set_xlabel('号码')

        # 调整布局
        plt.tight_layout()

        # 保存图表
        output_dir = self._ensure_output_dir(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{lottery_type}_frequency_heatmap_{period}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=self.config['visualization']['default_dpi'])
        plt.close()

        logger.info(f"频率热力图已保存: {filepath}")

        return filepath

    def plot_trend_chart(self, data, lottery_type, num_periods=30, output_dir=None):
        """绘制近期走势图"""
        if data is None or len(data) == 0:
            logger.warning("没有数据可供可视化")
            return None

        # 获取最近的数据
        if num_periods > len(data):
            num_periods = len(data)

        df = data.tail(num_periods).copy()

        # 反转数据使得最早的期数在左侧
        df = df.iloc[::-1].reset_index(drop=True)

        # 获取球号列和范围
        primary_columns, secondary_columns = self._get_ball_columns(lottery_type)
        primary_range, secondary_range = self._get_ball_ranges(lottery_type)

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # 主区走势图
        primary_title = f"主区号码走势图 ({lottery_type.upper()}, 最近{num_periods}期)"
        ax1.set_title(primary_title)

        # 绘制网格线
        for i in range(primary_range[0], primary_range[1] + 1):
            ax1.axhline(y=i, color='gray', linestyle='-', alpha=0.2)

        # 绘制每期开奖号码
        for idx, row in df.iterrows():
            primary_numbers = [int(row[col]) for col in primary_columns if pd.notna(row[col])]

            # 绘制点
            ax1.scatter([idx] * len(primary_numbers), primary_numbers, color='red', s=50)

            # 标注期号
            if idx % 5 == 0:  # 每5期标注一次期号
                ax1.text(idx, primary_range[0] - 1, str(row['issue']),
                         rotation=90, ha='center', va='top', fontsize=8)

        # 设置坐标轴
        ax1.set_xlim(-1, len(df))
        ax1.set_ylim(primary_range[0] - 2, primary_range[1] + 2)
        ax1.set_ylabel('主区号码')
        ax1.set_xticks([])

        # 副区走势图
        secondary_title = f"副区号码走势图 ({lottery_type.upper()}, 最近{num_periods}期)"
        ax2.set_title(secondary_title)

        # 绘制网格线
        for i in range(secondary_range[0], secondary_range[1] + 1):
            ax2.axhline(y=i, color='gray', linestyle='-', alpha=0.2)

        # 绘制每期开奖号码
        for idx, row in df.iterrows():
            secondary_numbers = [int(row[col]) for col in secondary_columns if pd.notna(row[col])]

            # 绘制点
            ax2.scatter([idx] * len(secondary_numbers), secondary_numbers, color='blue', s=50)

            # 标注期号
            if idx % 5 == 0:  # 每5期标注一次期号
                ax2.text(idx, secondary_range[0] - 0.5, str(row['issue']),
                         rotation=90, ha='center', va='top', fontsize=8)

        # 设置坐标轴
        ax2.set_xlim(-1, len(df))
        ax2.set_ylim(secondary_range[0] - 1, secondary_range[1] + 1)
        ax2.set_ylabel('副区号码')
        ax2.set_xticks(range(0, len(df), 5))
        ax2.set_xticklabels([f"{i + 1}" for i in range(0, len(df), 5)])
        ax2.set_xlabel('期数索引')

        # 调整布局
        plt.tight_layout()

        # 保存图表
        output_dir = self._ensure_output_dir(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{lottery_type}_trend_chart_{num_periods}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=self.config['visualization']['default_dpi'])
        plt.close()

        logger.info(f"走势图已保存: {filepath}")

        return filepath

    def plot_hot_cold_distribution(self, analysis_result, output_dir=None):
        """绘制冷热号分布图"""
        if not analysis_result:
            logger.warning("没有分析结果可供可视化")
            return None

        lottery_type = analysis_result.get('lottery_type')
        period = analysis_result.get('period')

        # 获取热号和冷号
        primary_hot = analysis_result.get('primary_hot', [])
        primary_cold = analysis_result.get('primary_cold', [])
        secondary_hot = analysis_result.get('secondary_hot', [])
        secondary_cold = analysis_result.get('secondary_cold', [])

        # 获取号码计数
        primary_counter = analysis_result.get('primary_counter', {})
        secondary_counter = analysis_result.get('secondary_counter', {})

        # 获取号码范围
        primary_range, secondary_range = self._get_ball_ranges(lottery_type)

        # 创建图表
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

        # 主区冷热号分布
        primary_title = f"主区冷热号分布 ({lottery_type.upper()}, 周期: {period}期)"
        ax1.set_title(primary_title)

        # 生成所有可能的号码
        primary_numbers = list(range(primary_range[0], primary_range[1] + 1))
        primary_counts = [primary_counter.get(num, 0) for num in primary_numbers]

        # 设置颜色
        primary_colors = ['#FF9999' if num in primary_hot else
                          '#9999FF' if num in primary_cold else
                          '#CCCCCC' for num in primary_numbers]

        # 绘制柱状图
        bars1 = ax1.bar(primary_numbers, primary_counts, color=primary_colors)

        # 添加数值标签
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f"{height:.0f}", ha='center', va='bottom')

        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FF9999', label='热号'),
            Patch(facecolor='#9999FF', label='冷号'),
            Patch(facecolor='#CCCCCC', label='平均')
        ]
        ax1.legend(handles=legend_elements)

        # 设置坐标轴
        ax1.set_xlabel('主区号码')
        ax1.set_ylabel('出现次数')
        ax1.set_xticks(primary_numbers)

        # 副区冷热号分布
        secondary_title = f"副区冷热号分布 ({lottery_type.upper()}, 周期: {period}期)"
        ax2.set_title(secondary_title)

        # 生成所有可能的号码
        secondary_numbers = list(range(secondary_range[0], secondary_range[1] + 1))
        secondary_counts = [secondary_counter.get(num, 0) for num in secondary_numbers]

        # 设置颜色
        secondary_colors = ['#FF9999' if num in secondary_hot else
                            '#9999FF' if num in secondary_cold else
                            '#CCCCCC' for num in secondary_numbers]

        # 绘制柱状图
        bars2 = ax2.bar(secondary_numbers, secondary_counts, color=secondary_colors)

        # 添加数值标签
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                     f"{height:.0f}", ha='center', va='bottom')

        # 添加图例
        ax2.legend(handles=legend_elements)

        # 设置坐标轴
        ax2.set_xlabel('副区号码')
        ax2.set_ylabel('出现次数')
        ax2.set_xticks(secondary_numbers)

        # 调整布局
        plt.tight_layout()

        # 保存图表
        output_dir = self._ensure_output_dir(output_dir)
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = f"{lottery_type}_hot_cold_{period}_{timestamp}.png"
        filepath = os.path.join(output_dir, filename)

        plt.savefig(filepath, dpi=self.config['visualization']['default_dpi'])
        plt.close()

        logger.info(f"冷热号分布图已保存: {filepath}")

        return filepath
