基本命令
更新数据

python main.py update -t ssq        # 更新双色球数据
python main.py update -t dlt        # 更新大乐透数据
python main.py update -t all        # 更新所有彩票数据
python main.py update -t ssq -f     # 强制更新所有双色球数据
数据分析

python main.py analyze -t ssq -a frequency      # 双色球频率分析
python main.py analyze -t dlt -a hot_cold -p 50 # 大乐透最近50期热冷号分析
python main.py analyze -t ssq -a odd_even -v    # 双色球奇偶比分析并生成图表
预测号码

python main.py predict -t ssq               # 使用默认算法预测双色球
python main.py predict -t dlt -a cold       # 使用冷号补偿算法预测大乐透
算法回测

python main.py backtest -t ssq -a hot -p 20  # 回测热号算法最近20期的表现
python main.py backtest -t dlt               # 比较所有算法对大乐透的预测效果
数据可视化

python main.py visualize -t ssq -c frequency  # 生成双色球频率热力图
python main.py visualize -t dlt -c trend -p 30 # 生成大乐透最近30期走势图

python main.py predict -t ssq -b  # 基于回测的预测

python main.py predict -t ssq -b -p 20  # 回测最近20期再预测