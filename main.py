import os
import time
import requests
from datetime import datetime, timedelta, timezone
import pandas as pd
import akshare as ak
import mplfinance as mpf
from openai import OpenAI
import numpy as np
import markdown
from xhtml2pdf import pisa

# ==========================================
# 1. 数据获取模块
# ==========================================

def fetch_stock_data(symbol: str, period: str) -> pd.DataFrame:
    symbol_code = ''.join(filter(str.isdigit, symbol))
    print(f"   -> 正在获取 {symbol_code} 的 {period} 分钟数据...")

    try:
        df = ak.stock_zh_a_hist_min_em(
            symbol=symbol_code, 
            period=period, 
            adjust="qfq"
        )
    except Exception as e:
        print(f"   [Error] 接口报错: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    rename_map = {
        "时间": "date", "开盘": "open", "最高": "high",
        "最低": "low", "收盘": "close", "成交量": "volume"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    df["date"] = pd.to_datetime(df["date"])
    cols = ["open", "high", "low", "close", "volume"]
    df[cols] = df[cols].astype(float)
    
    if (df["open"] == 0).any():
        df["open"] = df["open"].replace(0, np.nan)
        df["open"] = df["open"].fillna(df["close"].shift(1))
        df["open"] = df["open"].fillna(df["close"])

    # 保留最近 100 根
    bars_count = int(os.getenv("BARS_COUNT", 100)) 
    df = df.sort_values("date").tail(bars_count).reset_index(drop=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()
    return df

# ==========================================
# 2. 本地绘图模块
# ==========================================

def generate_local_chart(symbol: str, df: pd.DataFrame, save_path: str, period: str):
    if df.empty: return

    plot_df = df.copy()
    plot_df.set_index("date", inplace=True)

    mc = mpf.make_marketcolors(
        up='#ff3333', down='#00b060', 
        edge='inherit', wick='inherit', 
        volume={'up': '#ff3333', 'down': '#00b060'},
        inherit=True
    )
    s = mpf.make_mpf_style(
        base_mpf_style='yahoo', 
        marketcolors=mc, 
        gridstyle=':', 
        y_on_right=True
    )

    apds = []
    if 'ma20' in plot_df.columns:
        apds.append(mpf.make_addplot(plot_df['ma20'], color='#ff9900', width=1.0))

    try:
        mpf.plot(
            plot_df, type='candle', style=s, addplot=apds, volume=True,
            title=f"SCOB Setup: {symbol} ({period}m)",
            savefig=dict(fname=save_path, dpi=100, bbox_inches='tight'),
            warn_too_much_data=2000
        )
    except Exception as e:
        print(f"   [Error] 绘图失败: {e}")

# ==========================================
# 3. AI 分析模块 (只看多头版)
# ==========================================

def get_scob_prompt(symbol, df, period):
    csv_data = df.tail(40).to_csv(index=False) 
    latest = df.iloc[-1]
    
    timeframe_context = ""
    if period == '60':
        timeframe_context = "这是一个 **60分钟** 大级别图表，请重点关注趋势反转信号。"
    else:
        timeframe_context = f"这是一个 **{period}分钟** 日内图表，请重点关注回调结束的切入点。"

    prompt = f"""
**Role**: 你是一位精通 SMC (Smart Money Concepts) 的 A 股交易员。
**Task**: 分析这张 {symbol} 的 **{period}分钟** K线数据，寻找【Bullish SCOB (看涨订单块)】形态。

**Constraint (重要)**:
1. A股市场只能做多 (Long Only)。
2. **请直接忽略所有 BEARISH (看跌) 信号。**
3. 如果是看跌形态，或者形态不标准，请直接回答 SCOB Signal: NO。

**Context**:
{timeframe_context}
当前最新价格: {latest['close']}
当前最新时间: {latest['date']}

**Analysis Logic (Bullish SCOB Criteria)**:
1. **Liquidity Sweep**: 下影线是否刺破了左侧的前低 (Swing Low) 也就是扫了止损？
2. **Displacement**: 价格是否在扫止损后迅速向上反弹，并收出阳线？
3. **Volume**: 关键K线是否伴随异常放量？

**Data**:
{csv_data}

**Output Format (Strictly follow this)**:
- **Timeframe**: {period} min
- **SCOB Signal**: [YES / NO] (仅当发现标准的 **看涨 (BULLISH)** 信号时才回答 YES)
- **Direction**: BULLISH
- **Confidence**: [1-10]
- **Analysis**: (简述理由)
- **Suggestion**: (给出建议入场位)
"""
    return prompt

def call_ai_api(prompt: str) -> str:
    # --- 1. 优先尝试：通义千问 (Qwen) ---
    qwen_key = os.getenv("DASHSCOPE_API_KEY")
    if qwen_key:
        try:
            client = OpenAI(
                api_key=qwen_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            resp = client.chat.completions.create(
                model="qwen-plus", 
                messages=[
                    {"role": "system", "content": "你是专业的A股SMC交易员，只关注做多机会。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.2
            )
            return resp.choices[0].message.content
        except Exception as e:
            print(f"   [Warn] Qwen (通义千问) 调用失败: {e}")

    # --- 2. 备用：Google Gemini ---
    gemini_key = os.getenv("GEMINI_API_KEY")
    if gemini_key:
        try:
            model_name = os.getenv("GEMINI_MODEL", "gemini-2.0-flash-exp")
            url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={gemini_key}"
            data = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"temperature": 0.2}
            }
            resp = requests.post(url, headers={'Content-Type': 'application/json'}, json=data)
            if resp.status_code == 200:
                return resp.json()['candidates'][0]['content']['parts'][0]['text']
        except Exception as e:
            print(f"   [Warn] Gemini 失败: {e}")
            
    return "Error: 所有 AI 接口均调用失败"

# ==========================================
# 4. PDF 生成模块
# ==========================================

def generate_pdf_report(symbol, chart_path, report_text, pdf_path, period):
    html_content = markdown.markdown(report_text)
    abs_chart_path = os.path.abspath(chart_path)
    font_path = "msyh.ttc" 
    if not os.path.exists(font_path): font_path = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"
    
    full_html = f"""
    <html>
    <head>
        <meta charset="utf-8">
        <style>
            @font-face {{ font-family: "MyChineseFont"; src: url("{font_path}"); }}
            @page {{ size: A4; margin: 1cm; }}
            body {{ font-family: "MyChineseFont", sans-serif; font-size: 12px; }}
            img {{ width: 16cm; }}
            .period-tag {{ background: #d35400; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; }}
        </style>
    </head>
    <body>
        <div style="margin-bottom:10px;">
            <span class="period-tag">BULLISH SCOB ONLY</span>
            <span class="period-tag" style="background:#2980b9;">{period} MIN</span>
        </div>
        <img src="{abs_chart_path}" />
        <hr/>
        {html_content}
        <div style="text-align:right; color:#bdc3c7; font-size:8px;">
            Symbol: {symbol} | Time: {datetime.now().strftime('%H:%M:%S')}
        </div>
    </body>
    </html>
    """
    try:
        with open(pdf_path, "wb") as pdf_file:
            pisa.CreatePDF(full_html, dest=pdf_file)
        return True
    except Exception as e:
        print(f"   [Error] PDF生成失败: {e}")
        return False

# ==========================================
# 5. 主程序
# ==========================================

def process_one_stock(symbol: str, generated_files: list):
    print(f"\n{'='*40}")
    print(f"🚀 分析标的: {symbol}")
    print(f"{'='*40}")

    target_periods = ['15', '30', '60']
    
    for period in target_periods:
        # 1. 获取数据
        df = fetch_stock_data(symbol, period)
        if df.empty: continue
        df = add_indicators(df)

        beijing_tz = timezone(timedelta(hours=8))
        ts = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M")
        chart_path = f"reports/{symbol}_{period}m_chart_{ts}.png"
        pdf_path = f"reports/{symbol}_{period}m_report_{ts}.pdf"

        # 2. 绘图
        generate_local_chart(symbol, df, chart_path, period)

        # 3. AI 分析
        prompt = get_scob_prompt(symbol, df, period)
        report_text = call_ai_api(prompt)

        # 4. === 关键过滤逻辑：只推送 BULLISH 信号 ===
        # 检查 AI 是否输出了 "SCOB Signal: YES"
        if "SCOB Signal: YES" in report_text:
            print(f"   🔥 发现【看涨】信号 ({period}m)，正在生成报告...")
            if generate_pdf_report(symbol, chart_path, report_text, pdf_path, period):
                generated_files.append(pdf_path)
        else:
            print(f"   💤 ({period}m) 无看涨机会，跳过推送。")
        
        time.sleep(1)

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    symbols = []
    if os.path.exists("stock_list.txt"):
        with open("stock_list.txt", "r", encoding="utf-8") as f:
            symbols = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]
    
    if not symbols:
        symbols = ["600519"]

    generated_pdfs = []

    for symbol in symbols:
        try:
            process_one_stock(symbol, generated_pdfs)
        except Exception as e:
            print(f"❌ {symbol} 全局错误: {e}")

    if generated_pdfs:
        with open("push_list.txt", "w", encoding="utf-8") as f:
            for pdf in generated_pdfs:
                f.write(f"{pdf}\n")
        print(f"\n📝 推送清单已更新: 包含 {len(generated_pdfs)} 份看涨研报")
    else:
        print("\n😴 本次扫描未发现看涨机会，不发送推送。")

if __name__ == "__main__":
    main()
