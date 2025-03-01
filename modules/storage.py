"""
数据存储模块 - 负责管理彩票数据的存储和读取
"""

import os
import csv
import json
import logging
import pandas as pd
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('storage')


class LotteryStorage:
    """彩票数据存储类"""

    def __init__(self, config_path='config.json'):
        """初始化存储管理器"""
        self.config = self._load_config(config_path)
        self.data_dir = self.config['storage']['data_dir']

        # 确保数据目录存在
        self._ensure_data_dir_exists()

    def _load_config(self, config_path):
        """加载配置文件"""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"加载配置文件失败: {str(e)}")
            raise

    def _ensure_data_dir_exists(self):
        """确保数据目录存在"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            logger.info(f"创建数据目录: {self.data_dir}")

    def save_lottery_data(self, lottery_type, data, append=True):
        """保存彩票数据到CSV文件"""
        if not data:
            logger.info(f"没有新数据需要保存: {lottery_type}")
            return

        file_path = os.path.join(self.data_dir, f"{lottery_type}.csv")
        file_exists = os.path.exists(file_path)

        mode = 'a' if append and file_exists else 'w'
        logger.info(f"保存{lottery_type}数据到 {file_path}, 模式: {mode}")

        try:
            with open(file_path, mode, newline='', encoding='utf-8') as f:
                if lottery_type == 'ssq':
                    fieldnames = ['issue', 'date', 'red1', 'red2', 'red3', 'red4', 'red5', 'red6', 'blue', 'prize_pool']
                elif lottery_type == 'dlt':
                    fieldnames = ['issue', 'date', 'front1', 'front2', 'front3', 'front4', 'front5', 'back1', 'back2',
                                  'prize_pool']
                else:
                    raise ValueError(f"不支持的彩票类型: {lottery_type}")

                writer = csv.DictWriter(f, fieldnames=fieldnames)

                # 如果是新文件或不追加，写入表头
                if mode == 'w':
                    writer.writeheader()

                # 写入数据
                for item in data:
                    writer.writerow(self._format_data_for_csv(lottery_type, item))

            logger.info(f"成功保存 {len(data)} 条记录到 {file_path}")

        except Exception as e:
            logger.error(f"保存数据失败: {str(e)}")
            raise

    def _format_data_for_csv(self, lottery_type, item):
        """格式化数据用于CSV存储"""
        if lottery_type == 'ssq':
            # 格式化双色球数据
            return {
                'issue': item.get('issue', ''),
                'date': item.get('date', ''),
                'red1': item.get('red_balls', ['', '', '', '', '', ''])[0],
                'red2': item.get('red_balls', ['', '', '', '', '', ''])[1],
                'red3': item.get('red_balls', ['', '', '', '', '', ''])[2],
                'red4': item.get('red_balls', ['', '', '', '', '', ''])[3],
                'red5': item.get('red_balls', ['', '', '', '', '', ''])[4],
                'red6': item.get('red_balls', ['', '', '', '', '', ''])[5],
                'blue': item.get('blue_ball', ''),
                'prize_pool': item.get('prize_pool', '')
            }
        elif lottery_type == 'dlt':
            # 格式化大乐透数据
            return {
                'issue': item.get('issue', ''),
                'date': item.get('date', ''),
                'front1': item.get('front_balls', ['', '', '', '', ''])[0],
                'front2': item.get('front_balls', ['', '', '', '', ''])[1],
                'front3': item.get('front_balls', ['', '', '', '', ''])[2],
                'front4': item.get('front_balls', ['', '', '', '', ''])[3],
                'front5': item.get('front_balls', ['', '', '', '', ''])[4],
                'back1': item.get('back_balls', ['', ''])[0],
                'back2': item.get('back_balls', ['', ''])[1],
                'prize_pool': item.get('prize_pool', '')
            }
        else:
            raise ValueError(f"不支持的彩票类型: {lottery_type}")

    def load_lottery_data(self, lottery_type):
        """从CSV文件加载彩票数据"""
        file_path = os.path.join(self.data_dir, f"{lottery_type}.csv")

        print(f"尝试加载数据文件: {file_path}")

        if not os.path.exists(file_path):
            print(f"错误：数据文件不存在: {file_path}")
            logger.warning(f"数据文件不存在: {file_path}")
            return pd.DataFrame()

        try:
            # 检查文件是否为空
            if os.path.getsize(file_path) == 0:
                print(f"错误：数据文件为空: {file_path}")
                logger.warning(f"数据文件为空: {file_path}")
                return pd.DataFrame()

            # 使用pandas读取CSV文件
            df = pd.read_csv(file_path)

            # 检查记录数
            if len(df) == 0:
                print(f"错误：数据文件不包含任何记录: {file_path}")
                logger.warning(f"数据文件不包含任何记录: {file_path}")
                return pd.DataFrame()

            print(f"成功加载 {len(df)} 条{lottery_type}记录")
            logger.info(f"成功加载 {len(df)} 条{lottery_type}记录")

            return df
        except Exception as e:
            print(f"错误：加载数据失败: {str(e)}")
            logger.error(f"加载数据失败: {str(e)}")
            import traceback
            traceback_str = traceback.format_exc()
            print(traceback_str)
            logger.error(traceback_str)
            return pd.DataFrame()

    def save_prediction(self, lottery_type, algorithm, prediction_data):
        """保存预测结果"""
        file_path = os.path.join(self.data_dir, "predictions.csv")
        file_exists = os.path.exists(file_path)

        try:
            with open(file_path, 'a' if file_exists else 'w', newline='', encoding='utf-8') as f:
                fieldnames = ['prediction_time', 'lottery_type', 'algorithm', 'target_issue', 'numbers', 'status']
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if not file_exists:
                    writer.writeheader()

                current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                # 格式化号码
                primary_balls = prediction_data.get('primary_balls', [])
                secondary_balls = prediction_data.get('secondary_balls', [])
                numbers = ','.join(map(str, primary_balls)) + '+' + ','.join(map(str, secondary_balls))

                writer.writerow({
                    'prediction_time': current_time,
                    'lottery_type': lottery_type,
                    'algorithm': algorithm,
                    'target_issue': prediction_data.get('target_issue', ''),
                    'numbers': numbers,
                    'status': 'pending'  # 初始状态为待验证
                })

            logger.info(f"成功保存预测结果: {lottery_type}, 算法: {algorithm}")

        except Exception as e:
            logger.error(f"保存预测结果失败: {str(e)}")
            raise
