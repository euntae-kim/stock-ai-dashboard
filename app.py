import streamlit as st
import yfinance as yf
import pandas as pd
import feedparser
import google.generativeai as genai
from datetime import datetime, timedelta
import concurrent.futures
import os

# ==========================================
# [1. 기본 설정]
# ==========================================
st.set_page_config(page_title="Quant Dashboard (Ver 3.1)", layout="wide", page_icon="⚡")

# ==========================================
# [2. 보안] API 키 로드
# ==========================================
def load_api_key_from_file():
    key_file_path = "APP_key.txt"
    found_key = ""
    try:
        if os.path.exists(key_file_path):
            with open(key_file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip() or line.strip().startswith("#"):
                        continue
                    if "GEMINI_API_KEY" in line and "=" in line:
                        parts = line.split("=", 1)
                        if len(parts) == 2:
                            found_key = parts[1].strip().strip('"').strip("'")
                            break
        return found_key
    except Exception:
        return ""

GEMINI_API_KEY = load_api_key_from_file()

if not GEMINI_API_KEY and "GEMINI_API_KEY" in st.secrets:
    GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"].strip()

if not GEMINI_API_KEY:
    with st.sidebar:
        st.markdown("### 🔐 API 키 필요")
        st.info("설정 파일에서 키를 찾을 수 없습니다.")
        input_key = st.text_input("Gemini API 키 입력:", type="password")
        if input_key:
            GEMINI_API_KEY = input_key.strip()

# ==========================================
# [3. 모델 선택]
# ==========================================
with st.sidebar:
    st.header("⚙️ 엔진 설정 (2026 Ver)")
    
    model_options = {
        "⚡ Gemini 2.5 Flash (속도/가성비)": "models/gemini-2.5-flash",
        "🧠 Gemini 2.5 Pro (고지능/심층추론)": "models/gemini-2.5-pro",
    }
    
    selected_label = st.selectbox(
        "사용할 AI 모델을 선택하세요:",
        list(model_options.keys()),
        index=0
    )
    MODEL_NAME = model_options[selected_label]
    
    if "pro" in MODEL_NAME:
        st.info("🧠 **Pro 모델:** 복잡한 시장 상황을 깊이 있게 분석합니다.")
    else:
        st.success("⚡ **Flash 모델:** 빠르고 효율적으로 뉴스를 요약합니다.")
        
    st.caption(f"ID: `{MODEL_NAME}`")

# ==========================================
# [4. AI 연결]
# ==========================================
try:
    if GEMINI_API_KEY and len(GEMINI_API_KEY) > 10:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(MODEL_NAME)
    else:
        model = None
except Exception as e:
    st.error(f"⚠️ API 연결 오류: {e}")
    model = None

# ==========================================
# [5. 기능 함수] 뉴스 소스 4분할 및 분석 (안전 필터 해제)
# ==========================================
from google.generativeai.types import HarmCategory, HarmBlockThreshold # 필터 설정을 위한 도구 가져오기

class MockEntry:
    """뉴스가 없을 때 사용할 빈 객체"""
    def __init__(self, title, link):
        self.title = title
        self.link = link

@st.cache_data
def get_stock_data():
    """ 안정적인 일간(Daily) 데이터 수집 """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    tickers = {'TIGER 200 (KR)': '102110.KS', 'QQQ (US Nasdaq)': 'QQQ', 'SPY (US S&P500)': 'SPY', 'USD/KRW': 'KRW=X'}
    
    data_list = []
    for name, code in tickers.items():
        try:
            data = yf.download(code, start=start_date, end=end_date, progress=False)
            if not data.empty:
                series = data['Close'][code] if isinstance(data.columns, pd.MultiIndex) else data['Close']
                series.index = series.index.tz_localize(None)
                series.name = name
                data_list.append(series)
        except Exception:
            pass
    return pd.concat(data_list, axis=1).ffill() if data_list else pd.DataFrame()

def analyze_single_news(item):
    """ item: (entry, category) 형태의 튜플 """
    entry, category = item
    
    # 1. 링크가 없는 경우 (검색 실패)
    if entry.link == "":
        return {
            "title": entry.title, 
            "link": "#", 
            "category": category, 
            "ai_comment": "현재 관련 뉴스가 검색되지 않았습니다."
        }

    # 2. API 키 확인
    if not model:
        return {"title": entry.title, "link": entry.link, "category": category, "ai_comment": "API 키를 확인해주세요."}

    try:
        detail_level = "심층적으로" if "pro" in MODEL_NAME else "명확하게"
        
        prompt = f"""
        당신은 30대 퀀트 투자자입니다. 아래 뉴스 제목을 보고 투자 관점에서 분석해주세요.
        
        [뉴스 정보]
        분야: {category}
        제목: "{entry.title}"
        
        [요청사항]
        1. 내용 요약 (한 줄)
        2. 호재/악재/중립 판단
        3. 투자자 대응 ({detail_level})
        
        *반드시 '친근한 해요체'로 작성하고, 절대 빈칸으로 두지 마세요.*
        """
        
        # [핵심 수정] 안전 필터 강제 해제 (모든 내용을 차단하지 않고 표시)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        # 설정 적용하여 생성 요청
        response = model.generate_content(prompt, safety_settings=safety_settings)
        result_text = response.text.strip()
        
        if not result_text:
            result_text = "AI 분석 결과가 비어있습니다. (원문을 참고해주세요)"

        return {
            "title": entry.title,
            "link": entry.link,
            "category": category,
            "ai_comment": result_text
        }
    except Exception as e:
        # 에러가 나더라도 '오류 메시지'를 띄워서 링크만 나오는 현상 방지
        return {
            "title": entry.title, 
            "link": entry.link, 
            "category": category, 
            "ai_comment": f"⚠️ 분석 중 오류가 발생했습니다: {str(e)}"
        }

def get_ai_summary():
    search_map = {
        "🇺🇸 미국 실물경제": ["미국 경제 뉴스", "미국 연준"],
        "🇺🇸 미국 증시": ["미국 증시", "나스닥", "뉴욕 증시"],
        "🇰🇷 한국 실물경제": ["한국 경제", "한국 수출", "한국 금리"],
        "🇰🇷 한국 증시": ["한국 증시", "코스피", "국내 주식"]
    }
    
    tasks = []
    
    for category, queries in search_map.items():
        found_entry = None
        for query in queries:
            try:
                encoded_query = query.replace(" ", "+")
                rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
                feed = feedparser.parse(rss_url)
                if feed.entries:
                    found_entry = feed.entries[0]
                    break 
            except Exception:
                continue
        
        if found_entry:
            tasks.append((found_entry, category))
        else:
            dummy = MockEntry(title=f"'{category}' 관련 최신 기사가 없습니다.", link="")
            tasks.append((dummy, category))
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(analyze_single_news, tasks))
        
    return results

# ==========================================
# [6. UI 화면] 카테고리별 분리 표시
# ==========================================
display_model_name = MODEL_NAME.replace("models/", "")

st.title(f"⚡ AI 퀀트 대시보드 ({display_model_name})")
st.markdown("---")

col1, col2 = st.columns([2, 1], gap="medium") 

with col1:
    st.subheader("📊 자산 가치 변동 ($1,000 투자 시)")
    
    c1, c2 = st.columns([2, 1])
    with c1:
        period_options = ["12개월", "6개월", "3개월", "1개월", "1주"]
        selected_period = st.radio(
            "조회 기간:", period_options, index=2, horizontal=True, label_visibility="collapsed"
        )
    with c2:
        use_usd_base = st.toggle("💵 달러 환산 평가", value=True)

    with st.spinner('데이터 처리 중...'):
        df = get_stock_data()
        
        if not df.empty:
            days_map = {"12개월": 365, "6개월": 180, "3개월": 90, "1개월": 30, "1주": 7}
            start_date = df.index[-1] - timedelta(days=days_map[selected_period])
            df = df[df.index >= start_date]

            if use_usd_base and 'USD/KRW' in df.columns and 'TIGER 200 (KR)' in df.columns:
                df['TIGER 200 (KR)'] = df['TIGER 200 (KR)'] / df['USD/KRW']
            
            cols_to_plot = ['TIGER 200 (KR)', 'QQQ (US Nasdaq)', 'SPY (US S&P500)']
            cols_to_plot = [c for c in cols_to_plot if c in df.columns]
            df_view = df[cols_to_plot]

            if not df_view.empty:
                initial_investment = 1000 
                first_row = df_view.iloc[0].replace(0, 1)
                df_value = (df_view / first_row) * initial_investment
                
                st.line_chart(df_value, color=["#FF4B4B", "#1C83E1", "#00C805"], width="stretch")
                
                last_row = df_value.iloc[-1]
                st.markdown("#### 💰 $1,000 투자 시 현재 평가액")
                m1, m2, m3 = st.columns(3)
                
                def show_metric(col, label, value):
                    delta = value - 1000
                    col.metric(label, f"${value:,.2f}", f"{delta:+.2f} USD")

                if 'TIGER 200 (KR)' in last_row: show_metric(m1, "TIGER 200", last_row['TIGER 200 (KR)'])
                if 'QQQ (US Nasdaq)' in last_row: show_metric(m2, "QQQ (나스닥)", last_row['QQQ (US Nasdaq)'])
                if 'SPY (US S&P500)' in last_row: show_metric(m3, "SPY (S&P500)", last_row['SPY (US S&P500)'])
            else:
                st.warning("표시할 데이터가 부족합니다.")
        else:
            st.error("데이터 수집 실패")

with col2:
    st.subheader("🤖 AI 경제/증시 4분할 분석")
    st.write(f"**{display_model_name}** 모델이 4대 분야 주요 뉴스를 분석합니다.")
    
    if st.button(f"🚀 4대 뉴스 분석 실행 ({display_model_name})", type="primary", width="stretch"):
        with st.spinner('미국과 한국의 경제 뉴스를 읽고 있습니다...'):
            news_list = get_ai_summary()
            
            if news_list:
                for news in news_list:
                    # [변경] 카테고리를 제목에 함께 표시하여 구분
                    with st.expander(f"{news['category']} | {news['title'][:20]}...", expanded=True):
                        st.caption(f"🔗 소스: {news['title']}")
                        st.success(news['ai_comment'])
                        st.markdown(f"[기사 원문 보기]({news['link']})")
            else:
                st.warning("분석할 뉴스가 없습니다.")
    else:
        st.info("👆 버튼을 누르면 [미국 실물/증시, 한국 실물/증시] 뉴스를 각각 분석합니다.")

st.markdown("---")
st.markdown(f"<div style='text-align: center; color: gray;'>Created with Python & Streamlit | Engine: {display_model_name}</div>", unsafe_allow_html=True)
