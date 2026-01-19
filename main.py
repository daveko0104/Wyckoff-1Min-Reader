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
# 1. 数据获取模块 (支持多周期)
# ==========================================

def fetch_stock_data(symbol: str, period: str) -> pd.DataFrame:
    """
    获取A股K线数据
    :param symbol: 股票代码
    :param period: 周期 ('15', '30', '60')
    """
    symbol_code = ''.join(filter(str.isdigit, symbol))
    print(f"   -> 正在获取 {symbol_code} 的 {period} 分钟数据...")

    try:
        # 东方财富接口支持: "1", "5", "15", "30", "60"
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
    
    # === Open=0 修复逻辑 ===
    if (df["open"] == 0).any():
        df["open"] = df["open"].replace(0, np.nan)
        df["open"] = df["open"].fillna(df["close"].shift(1))
        df["open"] = df["open"].fillna(df["close"])

    # 保留最近 100 根足够看 SCOB，减少 Token 消耗
    bars_count = int(os.getenv("BARS_COUNT", 100)) 
    df = df.sort_values("date").tail(bars_count).reset_index(drop=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # 均线辅助判断趋势
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
        print(f"   [OK] {period}m 图表已保存")
    except Exception as e:
        print(f"   [Error] 绘图失败: {e}")

# ==========================================
# 3. AI 分析模块 (通义千问版)
# ==========================================

def get_scob_prompt(symbol, df, period):
    """生成 SCOB 专用提示词"""
    csv_data = df.tail(40).to_csv(index=False) 
    latest = df.iloc[-1]
    
    timeframe_context = ""
    if period == '60':
        timeframe_context = "这是一个 **60分钟** 大级别图表，请重点关注趋势反转信号。"
    else:
        timeframe_context = f"这是一个 **{period}分钟** 日内图表，请重点关注回调结束的切入点。"

    prompt = f"""
**Role**: 你是一位精通 SMC (Smart Money Concepts) 的 A 股交易员。
**Task**: 分析这张 {symbol} 的 **{period}分钟** K线数据，寻找【Single Candle Order Block (SCOB)】形态。

**Context**:
{timeframe_context}
当前最新价格: {latest['close']}
当前最新时间: {latest['date']}

**Analysis Logic (SCOB Criteria)**:
1. **Liquidity Sweep (流动性掠夺)**: 
   - 观察最近的K线（特别是影线）是否刺破了左侧明显的短期高点或低点？
2. **Displacement (动能反转)**:
   - 扫掉止损后，价格是否迅速收回并向反方向运动？
3. **Volume**: 
   - 关键K线是否伴随异常成交量？

**Data**:
{csv_data}

**Output Format (Strictly follow this)**:
- **Timeframe**: {period} min
- **SCOB Signal**: [YES / NO] (仅当形态非常标准时回答 YES)
- **Direction**: [BULLISH (看涨) / BEARISH (看跌) / NONE]
- **Confidence**: [1-10]
- **Analysis**: (简述 50 字以内，指出哪一根K线是 Order Block)
- **Suggestion**: (如果 YES，给出激进买点；如果 NO，建议观望)
"""
    return prompt

def call_ai_api(prompt: str) -> str:
    """优先使用通义千问 (Qwen)，Gemini/GPT 作为备用"""
    
    # --- 1. 优先尝试：通义千问 (Qwen) ---
    qwen_key = os.getenv("DASHSCOPE_API_KEY")
    if qwen_key:
        try:
            # 使用 OpenAI SDK 兼容模式调用千问
            client = OpenAI(
                api_key=qwen_key,
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
            )
            
            # 使用 qwen-plus (性价比高)
            resp = client.chat.completions.create(
                model="qwen-plus", 
                messages=[
                    {"role": "system", "content": "你是专业的A股SMC交易员。"},
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
            
    return "Error: 所有 AI 接口均调用失败，请检查 Secret 设置。"

# ==========================================
# 4. PDF 生成模块
# ==========================================

def generate_pdf_report(symbol, chart_path, report_text, pdf_path, period):
    html_content = markdown.markdown(report_text)
    abs_chart_path = os.path.abspath(chart_path)
    # 简单字体回退逻辑
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
            .period-tag {{ background: #2c3e50; color: white; padding: 2px 8px; border-radius: 4px; font-size: 10px; }}
        </style>
    </head>
    <body>
        <div style="margin-bottom:10px;">
            <span class="period-tag">SCOB Strategy (Qwen AI)</span>
            <span class="period-tag" style="background:#e67e22;">{period} MIN Timeframe</span>
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
        # 1. 获取对应周期数据
        df = fetch_stock_data(symbol, period)
        if df.empty: continue
        df = add_indicators(df)

        # 2. 生成文件名
        beijing_tz = timezone(timedelta(hours=8))
        ts = datetime.now(beijing_tz).strftime("%Y%m%d_%H%M")
        
        chart_path = f"reports/{symbol}_{period}m_chart_{ts}.png"
        pdf_path = f"reports/{symbol}_{period}m_report_{ts}.pdf"

        # 3. 绘图
        generate_local_chart(symbol, df, chart_path, period)

        # 4. AI 分析 (Qwen)
        prompt = get_scob_prompt(symbol, df, period)
        report_text = call_ai_api(prompt)

        # 5. 生成 PDF
        if generate_pdf_report(symbol, chart_path, report_text, pdf_path, period):
            print(f"   ✅ {period}m 研报已生成")
            generated_files.append(pdf_path)
        
        time.sleep(1)

def main():
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    symbols = []
    if os.path.exists("stock_list.txt"):
        with open("stock_list.txt", "r", encoding="utf-8") as f:
            symbols = [line.strip() for line in f.readlines() if line.strip() and not line.startswith("#")]
    
    if not symbols:
        symbols = ["600519"] # 默认测试

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
        print(f"\n📝 推送清单已更新: 包含 {len(generated_pdfs)} 份报告")

if __name__ == "__main__":
    main()
