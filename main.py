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
                
    # ================= 核心量化筛选逻辑 =================
    # 强制将需要用到的列转换为浮点数，防止格式报错
    for col in ['涨跌幅', '量比', '换手率']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # 条件 1：涨幅在 5% 到 9% 之间（有拉升动作，未涨停）
    # 条件 2：量比 >= 2.0（成交量是过去5天均量的2倍以上，有增量资金扫货）
    # 条件 3：换手率在 5.0% 到 20.0% 之间（交投活跃且筹码健康）
    filtered_df = df[
        (df['涨跌幅'] >= 5.0) & (df['涨跌幅'] <= 9.0) & 
        (df['量比'] >= 2.0) & 
        (df['换手率'] >= 5.0) & (df['换手率'] <= 20.0)
    ].copy()
    
    # 增加“量比”字段，传给大模型
    core_columns = ['代码', '名称', '最新价', '涨跌幅', '换手率', '量比', '成交额']
    
    # 按量比从大到小排序，取前 30 只最有代表性的异动股，防止数据过多超出 AI 处理范围
    core_data = filtered_df.sort_values(by='量比', ascending=False).head(30)
    
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
    你是一个资深的 A 股市场量化分析师。
    
    今天收盘后，我通过特定的量化策略筛选出了一批【底部放量异动潜力股】。
    我的核心筛选条件为：
    1. 涨幅在 5% 到 9% 之间（资金有强拉升意愿，但尚未触及涨停，有后续博弈空间）。
    2. 量比大于 2.0（意味着今日成交量是过去5日均量的2倍以上，增量资金明显流入）。
    3. 换手率在 5% 到 20% 之间（交投活跃，且筹码未过度发散）。
    
    以下是按“量比”降序排列的前 30 只核心异动股票数据：
    
    {market_data_md}
    
    请根据以上数据，写一份专业的今日 A 股量化异动深度复盘报告。
    要求：
    1. 【主力资金攻击方向】：分析这些增量资金主要集中在买入哪些行业或概念板块？提炼出 2-3 个核心主线。
    2. 【个股异动点评】：结合表格中的“量比”和“换手率”数据，挑选 3-4 只最具代表性的个股进行重点点评。
    3. 【情绪与策略】：简要评估当前的市场投机情绪，并给出明天的操作风向提示。
    4. 【排版美观】：语气专业客观，使用 Markdown 格式排版，多用加粗、列表、分割线等视觉元素增强易读性。
    """
    
# 发送请求给 DeepSeek 模型
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    
    # 获取 AI 生成的纯文本内容
    ai_report_content = response.choices[0].message.content
    
    # 打印到控制台（保持原样）
    print("================ AI 复盘报告 =================\n")
    print(ai_report_content)
    print("\n==============================================")

    # ====== 下面是你刚才提问的：新增的保存文件代码 ======
    import datetime
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    file_name = f"{today_str}-A股量化复盘.md"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(ai_report_content)
    print(f"\n成功！报告已自动保存为文件: {file_name}")

if __name__ == "__main__":
    md_data = get_market_data()
    generate_ai_report(md_data)
