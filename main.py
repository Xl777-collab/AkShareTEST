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
                
    spot_df['代码'] = spot_df['代码'].astype(str)
    for col in ['换手率', '总市值', '最新价', '涨跌幅', '成交额']:
        spot_df[col] = pd.to_numeric(spot_df[col], errors='coerce')
        
    # ================= 第一层漏斗：极速初筛 =================
    # 基础过滤：市值大于 30 亿，换手率 3%-30%
    spot_df = spot_df[spot_df['总市值'] >= 3000000000]
    spot_df = spot_df[(spot_df['换手率'] >= 3.0) & (spot_df['换手率'] <= 30.0)]
    
    # 提速优化：取换手率最高的前300，大幅缩短云端运算时间
    pool_df = spot_df.sort_values(by='换手率', ascending=False).head(300).copy()
    stock_list = pool_df['代码'].tolist()
    
    print(f"初筛完毕，强势股池剩余 {len(stock_list)} 只股票。开始计算 120 天历史数据...\n")
    
    final_stocks = []
    
    # ================= 第二层漏斗：新版高阶策略精筛 =================
    for code in stock_list:
        try:
            hist_df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if len(hist_df) < 120:
                continue 
                
            df_120 = hist_df.tail(120).reset_index(drop=True)
            df_recent_60 = df_120.tail(60).reset_index(drop=True)
            df_past_60 = df_120.head(60).reset_index(drop=True)
            
            current_price = df_120['收盘'].iloc[-1]
            price_60_days_ago = df_recent_60['开盘'].iloc[0]
            
            # 条件 1：趋势确立，现价 > 30日均线 > 60日均线
            ma30 = df_120['收盘'].rolling(window=30).mean().iloc[-1]
            ma60 = df_120['收盘'].rolling(window=60).mean().iloc[-1]
            cond_1 = current_price > ma30 > ma60
            
            # 条件 2：区间涨幅，过去60天内涨幅在 20%-50%
            price_increase = (current_price - price_60_days_ago) / price_60_days_ago
            cond_2 = 0.20 <= price_increase <= 0.50
            
            # 条件 3：量能结构，近60天总成交量 > 前置60天，且近10天均量 > 60天均量的 1.1 倍
            vol_recent_60 = df_recent_60['成交量'].sum()
            vol_past_60 = df_past_60['成交量'].sum()
            avg_vol_10 = df_recent_60.tail(10)['成交量'].mean()
            avg_vol_60 = df_recent_60['成交量'].mean()
            
            cond_3 = (vol_recent_60 > vol_past_60) and (avg_vol_10 > avg_vol_60 * 1.1)
            
            if cond_1 and cond_2 and cond_3:
                stock_name = pool_df[pool_df['代码'] == code]['名称'].values[0]
                final_stocks.append({
                    '代码': code,
                    '名称': stock_name,
                    '最新价': round(current_price, 2),
                    '30日线': round(ma30, 2),
                    '60日线': round(ma60, 2),
                    '60日涨幅': f"{round(price_increase * 100, 2)}%",
                    '近期量能放大': round(vol_recent_60 / vol_past_60, 2)
                })
                print(f"🔥 捕获新策略标的：{stock_name} ({code})")
                
            time.sleep(0.1) 
            
        except Exception as e:
            continue
            
    print(f"\n策略计算完毕，共找出 {len(final_stocks)} 只符合要求的股票！")
    
    if len(final_stocks) > 0:
        return pd.DataFrame(final_stocks).to_markdown(index=False)
    else:
        return "今日新版高级量化策略下，无完全符合条件的股票。"

def generate_ai_report(market_data_md):
    print("\n正在呼叫大模型撰写深度研报...\n")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        print("错误：找不到 API Key！")
        return

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
    prompt = f"""
    你是一个华尔街顶级的量化对冲基金经理（精通威客夫吸筹理论与 CANSLIM 突破法则）。
    
    今天，我的高频量化系统跑出了极高胜率的【均线多头+趋势突破放量模型】。
    调整系统的严苛筛选条件如下：
    1. 基础过滤：市值大于 30 亿，换手率 3%-30%。
    2. 趋势确立：现价 > 30日均线 > 60日均线（完美的均线发散多头排列）。
    3. 区间涨幅：过去60天内涨幅在 20%-50%（处于主升浪初期，未严重透支）。
    4. 量能结构：近60天总成交量显著大于前置的60天，且最近10天均量大于60天均量的1.1倍（资金加速抢筹）。
    
    以下是从 A 股市场中杀出重围的最终核心标的：
    
    {market_data_md}
    
    请严格根据上述数据，并调用你对 A 股上市公司的深度产业知识，撰写一份极具深度的投资内参报告。报告必须包含以下四个模块：
    
    1. 【量化策略当前环境匹配度】：评估当前 A 股宏观情绪下，此套“右侧强趋势突破”逻辑的胜率如何？
    2. 【行业与基本面共振分析（核心）】：逐一分析表格中股票的所属核心板块、主营业务壁垒。深度剖析：为什么资金会在这个阶段抢筹这些行业？是否存在政策催化、产业周期反转或业绩超预期的潜在逻辑？（如标的较多，可分类归纳为2-3条主线）
    3. 【技术与主力意图透视】：结合表格中的量能放大倍数与涨幅，分析主力资金在上述基本面背景下的介入深度与后续做多意图。
    4. 【防守铁律】：对于这类沿着 30 日生命线推升的强势形态，给出明确的防守底线（如破位特征）和止盈/止损纪律。
    
    要求：语气干练、客观专业，绝不含糊其辞。使用 Markdown 排版，重点结论加粗。
    """
    
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}]
    )
    
    ai_report_content = response.choices[0].message.content
    print("================ AI 复盘报告 =================\n")
    print(ai_report_content)
    print("\n==============================================")

    today_str = datetime.date.today().strftime("%Y-%m-%d")
    file_name = f"{today_str}-A股量化复盘.md"
    with open(file_name, "w", encoding="utf-8") as f:
        f.write(ai_report_content)
    print(f"\n成功！报告已自动保存为文件: {file_name}")

if __name__ == "__main__":
    md_data = get_market_data()
    generate_ai_report(md_data)
