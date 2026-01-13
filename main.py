import os
import time
from datetime import datetime
import pandas as pd
import akshare as ak
import mplfinance as mpf
from openai import OpenAI

# ==========================================
# 1. 数据获取模块 (修改为1分钟K线, 600根)
# ==========================================

def get_symbol_with_prefix(symbol: str) -> str:
    """Akshare的分钟接口通常需要 sh/sz 前缀"""
    if symbol.startswith("6"):
        return f"sh{symbol}"
    elif symbol.startswith("0") or symbol.startswith("3"):
        return f"sz{symbol}"
    elif symbol.startswith("4") or symbol.startswith("8"):
        return f"bj{symbol}"
    return symbol

def fetch_a_share_minute(symbol: str) -> pd.DataFrame:
    """获取A股最近600根1分钟K线"""
    print(f"正在获取 {symbol} 的1分钟数据...")
    formatted_symbol = get_symbol_with_prefix(symbol)
    
    try:
        # period='1' 代表1分钟，adjust='qfq' 前复权
        df = ak.stock_zh_a_minute(
            symbol=formatted_symbol, 
            period="1", 
            adjust="qfq"
        )
    except Exception as e:
        print(f"获取失败，请检查股票代码格式或网络: {e}")
        return pd.DataFrame()

    # 统一列名
    rename_map = {
        "day": "date",
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    # 格式转换
    df["date"] = pd.to_datetime(df["date"])
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    
    # 截取最近600根
    df = df.sort_values("date").tail(600).reset_index(drop=True)
    return df

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """添加威科夫分析所需的背景均线 (MA50/200)"""
    df = df.copy()
    # 在分钟级别，MA50和MA200代表短周期内的长期趋势线
    df["ma50"] = df["close"].rolling(50).mean()
    df["ma200"] = df["close"].rolling(200).mean()
    return df

# ==========================================
# 2. 本地绘图模块 (解决API无法直接画图的问题)
# ==========================================

def generate_local_chart(symbol: str, df: pd.DataFrame, save_path: str):
    """
    使用 mplfinance 在本地生成威科夫风格图表
    因为 OpenAI API 只能返回文本，我们需要自己在本地生成图表以供参考
    """
    plot_df = df.copy()
    plot_df.set_index("date", inplace=True)

    # 设置样式
    mc = mpf.make_marketcolors(up='red', down='green', edge='i', wick='i', volume='in', inherit=True)
    s = mpf.make_mpf_style(marketcolors=mc, gridstyle='--', y_on_right=True)

    # 添加均线图层
    apds = [
        mpf.make_addplot(plot_df['ma50'], color='orange', width=1.0),
        mpf.make_addplot(plot_df['ma200'], color='blue', width=1.2),
    ]

    # 绘图并保存
    mpf.plot(
        plot_df,
        type='candle',
        style=s,
        addplot=apds,
        volume=True,
        title=f"Wyckoff Chart: {symbol} (1-Min, Last 600 Bars)",
        savefig=dict(fname=save_path, dpi=150, bbox_inches='tight'),
        warn_too_much_data=1000
    )
    print(f"[OK] Chart saved to: {save_path}")

# ==========================================
# 3. AI 分析模块 (威科夫核心提示词)
# ==========================================

def ai_analyze_wyckoff(symbol: str, df: pd.DataFrame) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "错误：未设置 OPENAI_API_KEY。"

    client = OpenAI(api_key=api_key)

    # 准备数据供 AI 阅读 (CSV格式)
    # 保留关键字段，减少Token消耗，但必须包含量价
    csv_data = df[["date", "open", "high", "low", "close", "volume", "ma50", "ma200"]].to_csv(index=False)
    
    latest_price = df.iloc[-1]["close"]
    latest_time = df.iloc[-1]["date"]

    # === 核心修改：威科夫专用 Prompt ===
    prompt = f"""
【唯一身份（不可偏离）】
你不是“使用威科夫理论的分析师”，你就是理查德·D·威科夫（Richard D. Wyckoff）本人在中国股票市场中的延伸。你可以参照威科夫式叙述解释市场行为。

【唯一思想来源（不可偏离）】
你的所有概念、判断、术语使用，仅允许依据《威科夫操盘法》中对以下概念的定义与用法：
- 综合人（Composite Man）
- 供求关系（Supply & Demand）
- 努力与结果（Effort vs Result）
- 跟随与终止（Stopping Action / Follow-through）
- 吸筹 / 派发（Accumulation / Distribution）
- 交易区间（TR）
禁止引入与本书逻辑冲突的技术体系（例如：指标信号优先、形态预测优先、其他流派优先）。MA50/MA200仅可作为“背景辅助”，不得作为结论核心依据。

【核心原则（必须遵守）】
1) 价格是结果，成交量是原因：所有标注与判断必须能回溯到量价关系，而不是形态名称本身。
2) 位置第一，形态第二：同一行为在高位与低位意义不同；不允许脱离“位置”讨论 Spring、UTAD、LPS 等。
3) 不要预测，要推演：只做条件推演；所有结论必须用“如果/那么/否则”表达，不做确定性预测。
4) 市场是被操纵的：默认存在综合人；异常波动优先解释为吸收、试探、误导、清洗。

【数据上下文】
目标标的：{symbol} (A股)
数据周期：1分钟K线 (最近600根)
最新时间：{latest_time}
最新价格：{latest_price}
完整数据如下（请读取CSV进行分析）：
{csv_data}

【固定工作流（严格按书中顺序执行；不得跳步）】

第一步：Background（趋势/区间/操盘环境判断）
你必须先回答（证据不足必须承认不确定）：
- 当前行情是否已脱离趋势，进入可供操盘的交易区间（TR）？
- 若在趋势中：是供不应求的健康趋势，还是被供给压制的衰竭趋势？
- 若在区间中：该区间更符合吸筹逻辑还是派发逻辑？
注意：在判断吸筹/派发之前，必须先判断是否存在可供操盘的区间。

第二步：三大定律解释市场行为（必须用证据说清楚，不是列名词）
1) 供求定律：
- 上涨是否得到成交量支持？
- 下跌是否出现供应枯竭的迹象？
2) 努力与结果定律（重点识别）：
- 放量但价格不再前进 → 可能有对手盘在吸收
- 缩量但价格仍能维持 → 一方力量已占据优势
- 巨大努力但结果有限 → 行情临近转折/测试区
3) 因果定律（区间=未来行情的原因）：
- Phase B 的横向宽度与时间决定后续潜在能量

第三步：结构识别（TR与Phase A–E）
1) TR边界识别：
- 区间真实上下边界以“收盘密集区”为主
- 指出哪些极端影线属于操盘行为，不应作为区间边界
2) Phase A–E（必须符合书中原意）：
- Phase A：停止原趋势（Stopping Action）
- Phase B：建仓/出货的主要工作区（操盘核心）
- Phase C：最终测试（Spring 或 UT/UTAD）
- Phase D：方向展开（SOS / SOW）
- Phase E：趋势离开区间
要求：
- 不允许为了好看强行补齐 所有Phase
- 若证据不足，只能标注“至 Phase B / C”，并说明缺失证据是什么

第四步：关键事件与行为（满足书中条件才允许使用术语）
你只能在满足书中条件时使用以下术语；每个术语必须附一句解释“为什么符合书中定义”，理由必须来自：供求/努力-结果/位置。
吸筹侧：SC, ST, Spring, LPS, SOS/JAC
派发侧：BC, UT/UTAD, SOW, LPSY

【输出结构（严格，必须按顺序输出）】

注意：由于这是纯文本对话，请详细描述图表应包含的关键点，以便读者对照本地生成的图表阅读。

B) 第二部分：完整分析（Markdown格式）
请按以下小标题输出完整的威科夫式分析：

1. Background（操盘环境与位置）
- 是否TR？吸筹/派发倾向与证据。
- 用“综合人”视角解释其可能的操盘目标。

2. 三大定律证据链（关键区段的量价行为）
- 供求分析
- 努力与结果（列出至少3处关键片段）
- 因果推演

3. 结构与阶段（TR边界 + Phase A–E）
- 详细描述TR边界的价格位置。
- 逐段说明Phase划分依据。

4. 关键事件核验
- 对每个事件（如Spring, UTAD）用一句话解释“为什么符合定义”。

5. 交易策略（符合中国股票市场 T+1 规则）
需要参考《威科夫操盘法》并且考虑以下问题：
- 怎么买（最少2个交易策略，明确买入价/止损价/目标价）
- 综合人可能动作（吸收/试探/误导/清洗/推进）
并明确指出：
- 哪条路径最符合当前结构
- 哪条路径代表操盘失败或你先前判断的误判信号

【开始】
严格按以上顺序完成分析。证据不足宁可少标注也不要滥用术语。
    """.strip()

    print("正在请求 OpenAI (Wyckoff Mode)...")
    resp = client.chat.completions.create(
        model="gpt-4o", # 建议使用 GPT-4o 以获得更好的逻辑推演能力
        messages=[
            {"role": "system", "content": "You are Richard D. Wyckoff."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3 # 降低随机性，提高分析严谨度
    )
    return resp.choices[0].message.content

# ==========================================
# 4. 主程序
# ==========================================

def main():
    # 默认股票代码 (可通过环境变量覆盖)
    symbol = os.getenv("SYMBOL", "600970") 
    
    # 1. 获取数据 (1分钟线, 600根)
    df = fetch_a_share_minute(symbol)
    if df.empty:
        print("未获取到数据，程序终止。")
        return
        
    df = add_indicators(df)

    # 建立输出目录
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data", exist_ok=True)
    os.makedirs("reports", exist_ok=True)

    # 2. 保存CSV
    csv_path = f"data/{symbol}_1min_{ts}.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"[OK] CSV Saved: {csv_path} ({len(df)} rows)")

    # 3. 本地生成图表 (因为API回传不了图片)
    chart_path = f"reports/{symbol}_chart_{ts}.png"
    generate_local_chart(symbol, df, chart_path)

    # 4. 生成威科夫分析报告
    report_text = ai_analyze_wyckoff(symbol, df)

    # 5. 保存报告
    report_path = f"reports/{symbol}_report_{ts}.md"
    
    # 将图片链接插入 Markdown 报告顶部，方便查看
    final_report = f"![Wyckoff Chart](./{os.path.basename(chart_path)})\n\n{report_text}"
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(final_report)

    print(f"[OK] Report Saved: {report_path}")

if __name__ == "__main__":
    main()
