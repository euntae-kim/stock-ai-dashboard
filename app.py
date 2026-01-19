import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
import google.generativeai as genai
from datetime import datetime, timedelta

# ==========================================
# [설정] 클라우드 배포용 보안 설정 (수정됨)
# ==========================================
try:
    # 1. 클라우드 금고(Secrets)에서 키를 꺼내 시도합니다.
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    # 2. 실패하면(로컬이면) 그냥 빈 문자열을 넣거나 내 키를 넣습니다.
    # (오류가 났던 이유는 이 'except' 줄이 없었기 때문입니다!)
    GEMINI_API_KEY = "여기에_진짜_키를_넣어도_됩니다"

# 모델 설정 (3.0 Pro Preview)
MODEL_NAME = "gemini-3.0-pro-preview" 

# AI 연결
try:
    genai.configure(api_key=GEMINI_API_KEY)
    model = genai.GenerativeModel(MODEL_NAME)
except Exception as e:
    st.error(f"API 키 설정 오류: {e}")

# ==========================================
# [기능 1] 주가 데이터 가져오기 (캐시 사용)
# ==========================================
@st.cache_data
def get_stock_data():
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    # 한국(TIGER 200), 미국(QQQ, SPY) 비교
    tickers = {
        'TIGER 200 (KR)': '102110.KS',
        'QQQ (US Nasdaq)': 'QQQ',
        'SPY (US S&P500)': 'SPY'
    }
    
    df_all = pd.DataFrame()
    for name, code in tickers.items():
        data = yf.download(code, start=start_date, end=end_date, progress=False)
        if not data.empty:
            # 종가(Close)만 가져오기
            if isinstance(data.columns, pd.MultiIndex):
                df_all[name] = data['Close'][code]
            else:
                df_all[name] = data['Close']
            
    return df_all.ffill().dropna()

# ==========================================
# [기능 2] AI 뉴스 요약 (RSS + Gemini)
# ==========================================
def get_ai_summary():
    # 야후 파이낸스 QQQ 관련 뉴스 RSS
    rss_url = "https://finance.yahoo.com/rss/headline?s=QQQ"
    feed = feedparser.parse(rss_url)
    
    summaries = []
    
    # 최신 뉴스 3개만 분석
    for i, entry in enumerate(feed.entries[:3]):
        try:
            prompt = f"""
            당신은 월스트리트의 전문 투자자입니다.
            아래 뉴스 제목을 보고, 한국 투자자가 알아야 할 핵심 내용을 
            '한글로', '한 줄로', '해요체로' 명확하게 요약해주세요.
            
            [뉴스 제목]
            {entry.title}
            """
            response = model.generate_content(prompt)
            summaries.append({
                "title": entry.title,
                "link": entry.link,
                "ai_comment": response.text.strip()
            })
        except Exception:
            summaries.append({"title": entry.title, "link": entry.link, "ai_comment": "AI 분석 중 오류가 발생했어요."})
            
    return summaries

# ==========================================
# [화면 구성] 웹사이트 디자인
# ==========================================
st.set_page_config(page_title="My Stock Dashboard", layout="wide", page_icon="📈")

st.title("📈 AI 기반 주식 인사이트 대시보드")
st.markdown("---")

# 레이아웃: 왼쪽(차트) 2 : 오른쪽(뉴스) 1 비율
col1, col2 = st.columns([2, 1]) 

# [왼쪽] 차트 영역
with col1:
    st.subheader("📊 한-미 증시 수익률 비교 (1년)")
    
    with st.spinner('데이터를 분석 중입니다...'):
        df = get_stock_data()
        
        if not df.empty:
            # 정규화 (시작점을 0%로 맞춤)
            df_norm = (df / df.iloc[0] - 1) * 100
            
            # 스트림릿 내장 라인 차트 (인터랙티브함)
            st.line_chart(df_norm, color=["#FF0000", "#0000FF", "#00FF00"])
            st.caption("※ TIGER 200(빨강) vs QQQ(파랑) vs SPY(초록) 수익률(%) 비교")
        else:
            st.warning("데이터를 불러오지 못했습니다.")

# [오른쪽] 뉴스 영역
with col2:
    st.subheader("🤖 AI 오늘의 시장 요약")
    
    if st.button("🔄 최신 뉴스 분석 시작", type="primary"):
        with st.spinner('Gemini가 뉴스를 읽고 있습니다...'):
            news_list = get_ai_summary()
            
            for news in news_list:
                with st.expander(f"📰 {news['ai_comment'][:25]}..."):
                    st.write(f"**AI 요약:** {news['ai_comment']}")
                    st.markdown(f"[원문 기사 보러가기]({news['link']})")
    else:
        st.info("버튼을 누르면 AI가 실시간으로 뉴스를 분석합니다.")

st.markdown("---")
st.markdown("Created with Python & Streamlit | Model: Google Gemini 3.0 pro")
