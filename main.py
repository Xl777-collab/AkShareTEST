import akshare as ak
import pandas as pd

def get_market_data():
    print("正在从 AkShare 获取 A 股最新行情，请稍候...")
    # 获取东财接口的 A 股最新行情数据
    df = ak.stock_zh_a_spot_em()
    
    # 筛选今天涨跌幅大于 9.0% 的股票（近似涨停）
    filtered_df = df[df['涨跌幅'] >= 9.0].copy()
    
    # 只保留核心字段，减少传递给 AI 的数据量，提高 AI 分析准确度
    core_columns = ['代码', '名称', '最新价', '涨跌幅', '换手率', '成交额']
    # 如果筛选出来的股票太多，我们可以只取前 20 只
    core_data = filtered_df[core_columns].head(20)
    
    # 将 Pandas 表格转换为 Markdown 格式文本
    data_md = core_data.to_markdown(index=False)
    return data_md

if __name__ == "__main__":
    # 1. 抓取并处理数据
    md_data = get_market_data()
    
    # 2. 打印看看结果（这也就是我们未来要发给大模型的数据）
    print("\n====== 准备发送给 AI 的数据如下 ======\n")
    print(md_data)
    print("\n======================================\n")
    print("数据准备就绪！下一步就可以接入大模型 API 了。")
