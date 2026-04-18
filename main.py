import akshare as ak
import pandas as pd
import os
from openai import OpenAI

import time # 需要在顶部引入 time 模块

def get_market_data():
    max_retries = 3  # 设置最大重试次数为 3 次
    df = None
    
    for attempt in range(max_retries):
        try:
            print(f"正在从 AkShare 获取 A 股最新行情 (尝试第 {attempt + 1} 次)...")
            df = ak.stock_zh_a_spot_em()
            break  # 如果成功拿到数据，就跳出循环
        except Exception as e:
            print(f"获取失败，原因: {e}")
            if attempt < max_retries - 1:
                print("网络可能存在波动，等待 5 秒后重试...\n")
                time.sleep(5)
            else:
                print("已达到最大重试次数，放弃获取。")
                raise e  # 如果 3 次都失败了，再报错
                
    filtered_df = df[df['涨跌幅'] >= 9.0].copy()
    core_columns = ['代码', '名称', '最新价', '涨跌幅', '换手率', '成交额']
    
    core_data = filtered_df[core_columns]
    data_md = core_data.to_markdown(index=False)
    return data_md

def generate_ai_report(market_data_md):
    print("\n正在呼叫 DeepSeek 大模型撰写复盘报告...\n")
    
    # 从系统环境变量中读取刚才存在 GitHub Secrets 里的钥匙
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：找不到 API Key，请检查 GitHub Secrets 配置！")
        return

    # DeepSeek 的接口完全兼容 OpenAI 的 Python 库
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    # 编写给 AI 的指令（Prompt）
    prompt = f"""
    你是一个资深的 A 股市场分析师。以下是今天（收盘）涨幅大于 9% 的股票核心数据：
    
    {market_data_md}
    
    请根据以上数据，写一份专业的今日 A 股异动深度复盘报告。
    要求：
    1. 统计今天的涨停/大涨家数，并简要评估市场情绪。
    2. 尝试分析这些股票主要集中在哪些行业或概念板块。
    3. 语气专业客观，排版清晰美观。
    """
    
    # 发送请求给 DeepSeek 模型
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # 打印 AI 返回的报告
    print("================ AI 复盘报告 =================\n")
    print(response.choices[0].message.content)
    print("\n==============================================")

if __name__ == "__main__":
    md_data = get_market_data()
    generate_ai_report(md_data)
