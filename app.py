# 匯入所需套件
import streamlit as st
import pandas as pd
import feedparser
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
from statsmodels.tsa.statespace.sarimax import SARIMAX
import plotly.graph_objects as go
from FinMind.data import DataLoader
import datetime
import matplotlib.pyplot as plt

# 設定 matplotlib 字體為微軟正黑體，以避免中文亂碼
plt.rcParams['font.family'] = 'Microsoft JhengHei'

# 載入 FinBERT 模型
@st.cache_resource
def load_finbert():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return tokenizer, model

# 載入 HuggingFace 中文情感分析模型
@st.cache_resource
def load_chinese_sentiment():
    return pipeline("sentiment-analysis", model="uer/roberta-base-finetuned-jd-binary-chinese")

# 取得英文 RSS 新聞
def fetch_rss_news_en():
    rss_url = "https://news.google.com/rss/search?q=TSMC&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(rss_url)
    return [entry.title for entry in feed.entries if entry.title.strip()]

# 取得中文 RSS 新聞
def fetch_rss_news_zh():
    rss_url = "https://news.google.com/rss/search?q=台積電&hl=zh-TW&gl=TW&ceid=TW:zh-Hant"
    feed = feedparser.parse(rss_url)
    return [entry.title for entry in feed.entries if entry.title.strip()]

# 英文新聞情感分析
def finbert_sentiment(texts):
    tokenizer, model = load_finbert()
    sentiments = []
    valid_texts = []
    for text in texts:
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)[0].cpu().numpy()
            raw_score = probs[2] - probs[0]  # 正向機率 - 負向機率
            sentiments.append(float(raw_score))
            valid_texts.append(text)
        except:
            continue
    return valid_texts, sentiments

# 中文新聞情感分析
def chinese_sentiment(texts):
    classifier = load_chinese_sentiment()
    sentiments = []
    valid_texts = []
    for text in texts:
        try:
            result = classifier(text[:512])[0]
            score = result['score'] if result['label'] == 'positive' else -result['score']
            sentiments.append(score)
            valid_texts.append(text)
        except:
            continue
    return valid_texts, sentiments

# 準備股價資料，並在最新一天加入情感分數作為外生變數
def prepare_data(sentiment_score):
    today = datetime.date.today()
    start_date = today - datetime.timedelta(days=30)
    dl = DataLoader()
    try:
        data = dl.taiwan_stock_daily(stock_id="2330", start_date=str(start_date), end_date=str(today))
        if data.empty:
            st.error("⚠ 無法從 FinMind 取得股價資料")
            return pd.DataFrame()

        df = data[['date', 'close']].rename(columns={'date': 'date', 'close': 'price'})
        df['date'] = pd.to_datetime(df['date'])
        df = df.tail(15).reset_index(drop=True)  # 取近15天
        df['score'] = 0.0
        df.at[df.index[-1], 'score'] = float(sentiment_score)  # 將情感分數放在最新一天
        return df
    except Exception as e:
        st.error(f"⚠ 讀取 FinMind 資料時發生錯誤：{e}")
        return pd.DataFrame()

# 利用 SARIMAX 模型進行預測，使用情感分數作為外生變數
def forecast_arimax(df):
    df = df.dropna(subset=['price', 'score'])
    model = SARIMAX(df['price'], exog=df[['score']], order=(5,1,0))
    model_fit = model.fit(disp=False)
    future_sentiment = [[df['score'].iloc[-1]]] * 3  # 預測未來3天，外生變數同值
    forecast = model_fit.forecast(steps=3, exog=future_sentiment)
    return forecast

# 根據情感平均值與近5日股價變化給出投資建議
def investment_advice(avg_sentiment, price_series):
    if len(price_series) < 5:
        return "資料不足，無法提供投資建議"
    recent_change = (price_series.iloc[-1] - price_series.iloc[-5]) / price_series.iloc[-5]
    if avg_sentiment > 0.05 and recent_change > 0:
        return "💡 AI 投資建議：情感正向且股價上升，建議買入"
    elif avg_sentiment > 0.05 and recent_change <= 0:
        return "💡 AI 投資建議：情感正向但股價下跌，建議觀察後再決定"
    elif avg_sentiment < -0.05 and recent_change < 0:
        return "💡 AI 投資建議：情感負向且股價下跌，建議賣出"
    elif avg_sentiment < -0.05 and recent_change >= 0:
        return "💡 AI 投資建議：情感負向但股價上升，建議謹慎觀察"
    else:
        return "💡 AI 投資建議：情感中立，建議持續觀察"

# 主程式入口
def main():
    st.title("📈 透過雙語新聞情感分析進行台積電股價預測")

    # 抓取新聞與進行情感分析
    headlines_en = fetch_rss_news_en()
    headlines_en, sentiments_en = finbert_sentiment(headlines_en)

    headlines_zh = fetch_rss_news_zh()
    headlines_zh, sentiments_zh = chinese_sentiment(headlines_zh)

    # 美化：滾動表格樣式
    scroll_style = """
    <style>
    .scrollable-table {
        max-height: 300px;
        overflow-y: auto;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        background-color: #fafafa;
        margin-bottom: 20px;
    }
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th, td {
        border: 1px solid #ccc;
        padding: 6px;
        text-align: left;
        vertical-align: top;
    }
    </style>
    """
    st.markdown(scroll_style, unsafe_allow_html=True)

    # 顯示英文新聞與分析結果
    st.subheader("📰 英文新聞與情感分析")
    if headlines_en:
        df_en = pd.DataFrame({"Headline": headlines_en, "Sentiment": sentiments_en})
        table_html = df_en.to_html(classes="scrollable-table", index=False)
        st.markdown(f'<div class="scrollable-table">{table_html}</div>', unsafe_allow_html=True)
        avg_sent_en = sum(sentiments_en) / len(sentiments_en)
        st.markdown(f"**Average Sentiment：** {avg_sent_en:.4f}")
    else:
        st.warning("無法取得英文新聞")

    # 顯示中文新聞與分析結果
    st.subheader("📰 中文新聞與情感分析")
    if headlines_zh:
        df_zh = pd.DataFrame({"新聞標題": headlines_zh, "情感分數": sentiments_zh})
        table_html = df_zh.to_html(classes="scrollable-table", index=False)
        st.markdown(f'<div class="scrollable-table">{table_html}</div>', unsafe_allow_html=True)
        avg_sent_zh = sum(sentiments_zh) / len(sentiments_zh)
        st.markdown(f"**平均情感分數：** {avg_sent_zh:.4f}")
    else:
        st.warning("無法取得中文新聞")

    # 計算總體平均情感分數
    all_sentiments = sentiments_en + sentiments_zh
    avg_sentiment = sum(all_sentiments) / len(all_sentiments) if all_sentiments else 0

    # 顯示股價預測
    show_forecast = st.checkbox("顯示股價預測")
    if show_forecast:
        st.subheader("📊 台積電股價預測")
        df_price = prepare_data(avg_sentiment)
        if df_price.empty:
            st.error("⚠ 無法取得股價資料")
            return

        # 預測未來三日
        pred = forecast_arimax(df_price)
        future_dates = pd.date_range(start=df_price['date'].iloc[-1] + pd.Timedelta(days=1), periods=10)
        future_dates = future_dates[future_dates.weekday < 5][:3]

        # 繪製圖表
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_price['date'], y=df_price['price'],
                                mode='lines+markers', name='實際價格'))
        fig.add_trace(go.Scatter(x=future_dates, y=pred,
                                mode='lines+markers', name='預測價格',
                                line=dict(dash='dash', color='red')))
        fig.update_layout(
            title="台積電股價預測（含新聞情感）",
            xaxis_title="日期",
            yaxis_title="股價 (新台幣)",
            hovermode='x unified',
            template='plotly_white',
            width=900,
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # 投資建議
        advice = investment_advice(avg_sentiment, df_price['price'])
        st.info(advice)

# 程式執行入口
if __name__ == "__main__":
    main()
