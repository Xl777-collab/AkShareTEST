import os
import time
import datetime
import akshare as ak
import pandas as pd
from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header
import markdown
from xhtml2pdf import pisa

# ================= 第一部分：量化选股逻辑 =================

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
                print(f"网络波动 (第{attempt+1}次)，等待 5 秒重试...")
                time.sleep(5)
            else:
                raise e
                
    spot_df['代码'] = spot_df['代码'].astype(str)
    for col in ['换手率', '总市值', '最新价', '涨跌幅', '成交额']:
        spot_df[col] = pd.to_numeric(spot_df[col], errors='coerce')
        
    # 第一层漏斗：极速初筛
    # 剔除市值 < 30亿、换手率不在 3%-30% 的股票
    spot_df = spot_df[spot_df['总市值'] >= 3000000000]
    spot_df = spot_df[(spot_df['换手率'] >= 3.0) & (spot_df['换手率'] <= 30.0)]
    
    # 取换手率最高的前 150 只股票进行深度计算（兼顾效率与质量）
    pool_df = spot_df.sort_values(by='换手率', ascending=False).head(300).copy()
    stock_list = pool_df['代码'].tolist()
    
    print(f"初筛完毕，进入精筛池标的共 {len(stock_list)} 只。开始回溯历史数据...")
    
    final_stocks = []
    
    for code in stock_list:
        try:
            hist_df = ak.stock_zh_a_hist(symbol=code, period="daily", adjust="qfq")
            if len(hist_df) < 120: continue 
                
            df_120 = hist_df.tail(120).reset_index(drop=True)
            df_recent_60 = df_120.tail(60).reset_index(drop=True)
            df_past_60 = df_120.head(60).reset_index(drop=True)
            
            current_price = df_120['收盘'].iloc[-1]
            price_60_days_ago = df_recent_60['开盘'].iloc[0]
            
            # 策略条件计算
            ma30 = df_120['收盘'].rolling(window=30).mean().iloc[-1]
            ma60 = df_120['收盘'].rolling(window=60).mean().iloc[-1]
            price_increase = (current_price - price_60_days_ago) / price_60_days_ago
            
            vol_recent_60 = df_recent_60['成交量'].sum()
            vol_past_60 = df_past_60['成交量'].sum()
            avg_vol_10 = df_recent_60.tail(10)['成交量'].mean()
            avg_vol_60 = df_recent_60['成交量'].mean()
            
            # 五大核心策略条件
            cond_1 = current_price > ma30 > ma60 # 多头排列
            cond_2 = 0.20 <= price_increase <= 0.50 # 60日涨幅适中
            cond_3 = vol_recent_60 > vol_past_60 # 中期放量
            cond_4 = avg_vol_10 > (avg_vol_60 * 1.1) # 近期放量
            cond_5 = current_price >= (df_recent_60['最高'].max() * 0.90) # 处于高位附近
            
            if cond_1 and cond_2 and cond_3 and cond_4 and cond_5:
                stock_name = pool_df[pool_df['代码'] == code]['名称'].values[0]
                final_stocks.append({
                    '代码': code, '名称': stock_name, '最新价': round(current_price, 2),
                    '30日线': round(ma30, 2), '60日线': round(ma60, 2),
                    '60日涨幅': f"{round(price_increase * 100, 2)}%",
                    '量能放大': round(vol_recent_60 / vol_past_60, 2)
                })
                print(f"🔥 捕获标的：{stock_name} ({code})")
            time.sleep(0.1) 
        except: continue
            
    if len(final_stocks) > 0:
        return pd.DataFrame(final_stocks).to_markdown(index=False)
    return "今日量化模型未筛选出符合条件的标的。"

# ================= 第二部分：PDF 生成逻辑 =================

def generate_pdf(text, filename):
    html_content = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            body {{ font-family: Helvetica, Arial, sans-serif; line-height: 1.6; padding: 20px; }}
            h1, h2, h3 {{ color: #1a5276; border-bottom: 1px solid #eee; padding-bottom: 5px; }}
            table {{ border-collapse: collapse; width: 100%; margin: 15px 0; font-size: 12px; }}
            th, td {{ border: 1px solid #ddd; padding: 6px; text-align: left; }}
            th {{ background-color: #f8f9f9; }}
            .footer {{ font-size: 10px; color: #999; text-align: center; margin-top: 30px; }}
        </style>
    </head>
    <body>
        {markdown.markdown(text, extensions=['tables'])}
        <div class="footer">报告生成时间：{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </body>
    </html>
    """
    with open(filename, "wb") as f:
        pisa.CreatePDF(html_content, dest=f)

# ================= 第三部分：邮件发送逻辑 =================

def send_email(content, pdf_path, date_str):
    my_email = "18989565457@163.com"  # FIXME: 请替换为你的QQ邮箱
    password = os.getenv("EMAIL_AUTH_CODE")
    if not password: return

    msg = MIMEMultipart()
    msg['From'] = Header("AI 量化管家", 'utf-8')
    msg['To'] = Header("策略持有人", 'utf-8')
    msg['Subject'] = Header(f"📈 深度研报：A股量化复盘 ({date_str})", 'utf-8')

    msg.attach(MIMEText("您好！今日量化筛选结果及深度产业分析已生成，请查阅附件 PDF。", 'plain', 'utf-8'))

    with open(pdf_path, "rb") as f:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f'attachment; filename={pdf_path}')
        msg.attach(part)

    try:
        # 改成网易 163 的服务器，使用最稳定的 SSL 465 端口
        server = smtplib.SMTP_SSL('smtp.163.com', 465)
        server.login(my_email, password)
        server.sendmail(my_email, [my_email], msg.as_string())
        server.quit()
        print("🎉 研报邮件已成功送达！")
    except Exception as e:
        print(f"❌ 邮件发送失败: {e}")

# ================= 第四部分：主控制流 =================

def main():
    # 1. 选股
    md_table = get_market_data()
    
    # 2. 调用 AI 分析
    api_key = os.getenv("DEEPSEEK_API_KEY")
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    
# 提前获取一下当天的真实日期
    local_now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    date_tag = local_now.strftime("%Y年%m月%d日")

    prompt = f"""
    你是一个华尔街顶级的量化对冲基金经理。今天是 {date_tag}，请基于以下量化筛选出的异动标的：
    {md_table}
    
    撰写深度内参，包含：
    1. 【行业与业务深度透视】：调用你的知识库，严格分析这些股票所属的行业板块、核心业务壁垒及当前产业逻辑。
    2. 【主力意图剖析】：结合量价数据分析资金介入深度。
    3. 【风控底线】：明确给出沿着30日线推升形态的防守位。
    要求：专业客观，Markdown排版，文章标题必须包含今天的真实日期 {date_tag}。
    """
    
    response = client.chat.completions.create(model="deepseek-chat", messages=[{"role": "user", "content": prompt}])
    ai_report = response.choices[0].message.content
    
    # 3. 命名与保存 (修正时区)
    local_now = datetime.datetime.utcnow() + datetime.timedelta(hours=8)
    time_tag = local_now.strftime("%Y-%m-%d_%H-%M-%S")
    date_tag = local_now.strftime("%Y-%m-%d")
    
    md_file = f"{time_tag}-Report.md"
    pdf_file = f"{time_tag}-Report.pdf"
    
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(ai_report)
    
    generate_pdf(ai_report, pdf_file)
    
    # 4. 发送邮件
    send_email(ai_report, pdf_file, date_tag)

if __name__ == "__main__":
    main()
