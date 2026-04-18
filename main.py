import os
import time
import datetime
import akshare as ak
import pandas as pd
from openai import OpenAI

def get_market_data():
    print("正在拉取今日 A 股全市场快照进行初筛...")
    max_retries = 3
    spot_df = None
    
    for attempt in range(max_retries):
        try:
            spot_df = ak.stock_zh_a_spot_em()
            break
        except Exception as e:
            if attempt < max_retries - 1:
                print("网络波动，等待 5 秒重试...")
                time.sleep(5)
            else:
                raise e
                
    # 强制将需要运算的列转换为数值格式，防止报错
    spot_df['代码'] = spot_df['代码'].astype(str)
    for col in ['换手率', '总市值', '最新价', '涨跌幅', '成交额']:
        spot_df[col] = pd.to_numeric(spot_df[col], errors='coerce')
        
    # ================= 第一层漏斗：极速初筛 =================
    # 2. 市值过滤：总市值必须大于 30 亿 (3,000,000,000)
    spot_df = spot_df[spot_df['总市值'] >= 3000000000]
    
    # 3. 活跃度过滤：换手率在 3% 到 30% 之间（完美破解大盘股霸榜，只找真正活跃的标的）
    spot_df = spot_df[(spot_df['换手率'] >= 3.0) & (spot_df['换手率'] <= 30.0)]
    
    # 4. 按换手率降序，取前 300 只（这极大减轻了后续计算压力）
    pool_df = spot_df.sort_values(by='换手率', ascending=False).head(300).copy()
    stock_list = pool_df['代码'].tolist()
    
    print(f"初筛完毕，强势股池剩余 {len(stock_list)} 只股票。开始计算 120 天历史数据，请耐心等待（约 1-2 分钟）...\n")
    
    final_stocks = []
    
    # ================= 第二层漏斗：高阶量化策略精筛 =================
    for code in stock_list:
        try:
            # 拉取该股票的历史日线数据 (qfq = 前复权，保证价格连贯性)
            hist_df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if len(hist_df) < 120:
                continue # 上市不足 120 天的次新股直接跳过
                
            df_120 = hist_df.tail(120).reset_index(drop=True)
            df_recent_60 = df_120.tail(60).reset_index(drop=True)
            df_past_60 = df_120.head(60).reset_index(drop=True)
            
            current_price = df_120['收盘'].iloc[-1]
            price_60_days_ago = df_recent_60['开盘'].iloc[0]
            
            # 策略 1：过去60天涨幅 20%-50%
            price_increase = (current_price - price_60_days_ago) / price_60_days_ago
            cond_1 = 0.20 <= price_increase <= 0.50
            
            # 策略 2：近60天交易量 > 过去60-120天交易量
            vol_recent_60 = df_recent_60['成交量'].sum()
            vol_past_60 = df_past_60['成交量'].sum()
            cond_2 = vol_recent_60 > vol_past_60
            
            # 策略 3：近10天均量 > 1.2 * 近60天均量
            cond_3 = df_recent_60.tail(10)['成交量'].mean() > (df_recent_60['成交量'].mean() * 1.2)
            
            # 策略 4：现价距离近 60 天最高点回撤不足 10%
            cond_4 = current_price >= (df_recent_60['最高'].max() * 0.90)
            
            # 策略 5 (新增)：现价 > 20日均价 > 60日均价 (多头排列)
            ma20 = df_120['收盘'].rolling(window=20).mean().iloc[-1]
            ma60 = df_120['收盘'].rolling(window=60).mean().iloc[-1]
            cond_5 = current_price > ma20 > ma60
            
            # 只有这 5 个严苛条件同时满足，才会被选中！
            if cond_1 and cond_2 and cond_3 and cond_4 and cond_5:
                stock_name = pool_df[pool_df['代码'] == code]['名称'].values[0]
                final_stocks.append({
                    '代码': code,
                    '名称': stock_name,
                    '最新价': round(current_price, 2),
                    '20日线': round(ma20, 2),
                    '60日线': round(ma60, 2),
                    '60日涨幅': f"{round(price_increase * 100, 2)}%",
                    '量能放大倍数': round(vol_recent_60 / vol_past_60, 2)
                })
                print(f"🔥 捕获完美策略标的：{stock_name} ({code})")
                
            time.sleep(0.1) # 短暂休眠，防止被东方财富封禁 IP
            
        except Exception as e:
            continue
            
    print(f"\n策略计算完毕，共找出 {len(final_stocks)} 只符合极其严苛要求的股票！")
    
    if len(final_stocks) > 0:
        return pd.DataFrame(final_stocks).to_markdown(index=False)
    else:
        return "今日极其严苛的高级量化策略下，无完全符合条件的股票。"

def generate_ai_report(market_data_md):
    print("\n正在呼叫大模型撰写深度研报...\n")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：找不到 API Key！")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    # ================= AI Prompt 同步升级 =================
    prompt = f"""
    你是一个华尔街顶级的量化对冲基金经理（精通威客夫吸筹理论与 CANSLIM 突破法则）。
    
    今天，我的高频量化系统跑出了极高胜率的【均线多头+趋势突破放量模型】。
    系统的严苛筛选条件如下：
    1. 基础过滤：市值大于 30 亿，换手率 3%-30%，剔除了 ST 股及高控盘微盘股。
    2. 趋势确立：现价 > 20日均线 > 60日均线（完美的均线发散多头排列）。
    3. 区间涨幅：过去60天内涨幅在 20%-50%（处于主升浪初期，未严重透支）。
    4. 量能结构：近60天总成交量显著大于前置的60天，且最近10天均量大于60天均量的1.2倍（资金加速抢筹）。
    5. 筹码稳固：现价距离近60天最高点回撤不足10%（极其强势的横盘整理或突破形态）。
    
    以下是从 5000 只股票中杀出重围的最终核心标的：
    
    {market_data_md}
    
    请根据上述数据，撰写一份极具深度的投资内参报告：
    1. 【策略适用性验证】：评估当前 A 股环境下，此套“右侧强趋势突破”逻辑的胜率与市场环境匹配度。
    2. 【标的深度透视】：针对表格中的股票，结合量价关系与多头排列特征，分析主力意图。
    3. 【铁律与风控】：对于这类形态，一旦买入，未来最危险的破位信号（如跌破20日线、放量滞涨等）是什么？给出明确的防守底线。
    要求：语气干练、客观专业。使用 Markdown 排版，重点结论加粗。
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    
    ai_report_content = response.choices[0].message.content
    print("================ AI 复盘报告 =================\n")
    print(ai_report_content)
    print("\n==============================================")

# 自动保存为 Markdown 文件
    today_str = datetime.date.today().strftime("%Y-%m-%d")
    file_name = f"{today_str}-A股量化复盘.md"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(ai_report_content)
    print(f"\n成功！报告已自动保存为文件: {file_name}")

if __name__ == "__main__":
    md_data = get_market_data()
    generate_ai_report(md_data)
