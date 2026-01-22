import streamlit as st
import yfinance as yf
import pandas as pd
import feedparser
import google.generativeai as genai
from datetime import datetime, timedelta
import os
import time
import requests
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# ==========================================
# [1. 기본 설정]
# ==========================================
st.set_page_config(page_title="Quant Dashboard", layout="wide", page_icon="⚡")

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
    st.header("⚙️ 엔진 설정")
    
    model_options = {
        "⚡ Gemini 2.5 Flash": "models/gemini-2.5-flash",
        "🧠 Gemini 2.5 Pro": "models/gemini-2.5-pro",
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
# [5. 기능 함수] 안정성 최우선 로직
# ==========================================
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
    """ AI 분석 수행 함수 (실패 시에도 무조건 텍스트 반환) """
    entry, category = item
    
    # 1. 안전장치: 데이터 누락
    if entry is None:
         return {"title": "데이터 없음", "link": "#", "category": category, "ai_comment": "데이터 로드에 실패했습니다."}

    # 2. 링크 누락 (가짜 뉴스 객체)
    if not hasattr(entry, 'link') or entry.link in ["", "#", None]:
        return {
            "title": getattr(entry, 'title', '제목 없음'), 
            "link": "#", 
            "category": category, 
            "ai_comment": "관련된 최신 뉴스를 찾을 수 없습니다."
        }

    # 3. 모델 확인
    if not model:
        return {"title": entry.title, "link": entry.link, "category": category, "ai_comment": "API 키가 연결되지 않았습니다."}

    try:
        detail_level = "심층적으로" if "pro" in MODEL_NAME else "핵심만 명확하게"
        
        prompt = f"""
        당신은 30대 퀀트 투자자입니다. 
        분야: {category}
        기사 제목: "{entry.title}"
        
        1. 내용 요약 (한 줄)
        2. 시장 영향 (호재/악재/중립)
        3. 투자자 대응 ({detail_level})
        
        '친근한 해요체'로 3줄 이내 답변.
        """
        
        # 안전 필터 해제 (필수)
        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(prompt, safety_settings=safety_settings)
        result_text = response.text.strip()
        
        if not result_text:
            result_text = "AI 분석 결과를 생성하지 못했습니다. 원문을 참고해주세요."

        return {
            "title": entry.title,
            "link": entry.link,
            "category": category,
            "ai_comment": result_text
        }
    except Exception as e:
        # 에러 발생 시에도 빈 박스 대신 에러 메시지 출력
        return {
            "title": entry.title, 
            "link": entry.link, 
            "category": category, 
            "ai_comment": f"분석 중 오류 발생: {str(e)}"
        }

def get_ai_summary(status_container):
    """ 
    [박스 4개 보장 로직]
    중간에 에러가 나도 절대 멈추지 않고, 빈 박스라도 채워 넣습니다.
    """
    
    # 4개 분야 고정 (가장 확실한 검색어 1개씩만 사용)
    categories = [
        ("🇺🇸 미국 실물경제", "미국 경제 뉴스"),
        ("🇺🇸 미국 증시", "미국 증시"),
        ("🇰🇷 한국 실물경제", "한국 경제"),
        ("🇰🇷 한국 증시", "한국 증시")
    ]
    
    final_results = []
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    # 순차 처리 (Progress Bar 연동)
    total_steps = len(categories)
    
    for i, (category, query) in enumerate(categories):
        # 상태 업데이트 (UI)
        status_container.progress((i + 1) / total_steps, text=f"🔍 {category} 분석 중...")
        
        found_entry = None
        
        try:
            # 1. 뉴스 검색
            encoded_query = query.replace(" ", "+")
            rss_url = f"https://news.google.com/rss/search?q={encoded_query}&hl=ko&gl=KR&ceid=KR:ko"
            
            # 타임아웃 5초 설정 (무한 대기 방지)
            response = requests.get(rss_url, headers=headers, timeout=5)
            
            if response.status_code == 200:
                feed = feedparser.parse(response.content)
                if feed.entries:
                    found_entry = feed.entries[0]
            
        except Exception:
            # 검색 에러나면 그냥 넘어감 (found_entry는 None 상태)
            pass
        
        # 2. 검색 실패 시 Mock 객체 강제 주입
        if not found_entry:
            found_entry = MockEntry(title=f"'{query}' 관련 기사를 찾지 못했습니다.", link="#")
            
        # 3. AI 분석 실행 (이 결과는 무조건 append 됨)
        try:
            # 약간의 텀을 줘서 API 과부하 방지
            time.sleep(0.5)
            result = analyze_single_news((found_entry, category))
            final_results.append(result)
        except Exception as e:
            # 최악의 경우에도 에러 박스 추가
            final_results.append({
                "title": "시스템 오류",
                "link": "#",
                "category": category,
                "ai_comment": f"처리 실패: {e}"
            })
            
    return final_results

# ==========================================
# [6. UI 화면]
# ==========================================
display_model_name = MODEL_NAME.replace("models/", "")

st.title(f"⚡ AI 퀀트 대시보드")
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
    st.write(f"현재 **{display_model_name}** 모델이 시장을 분석합니다.")
    
    if st.button(f"🚀 분석 실행 ({display_model_name})", type="primary", width="stretch"):
        # 진행바를 표시할 빈 공간 생성
        status_container = st.empty()
        
        # 함수 실행 시 status_container를 넘겨줌
        news_list = get_ai_summary(status_container)
        
        # 완료 후 진행바 제거
        status_container.empty()
        
        if news_list:
            for news in news_list:
                category_text = news.get('category', '분야 미상')
                title_text = news.get('title', '제목 없음')
                ai_text = news.get('ai_comment', '')
                link_text = news.get('link', '#')
                
                # AI 내용이 비어있으면 강제 문구 삽입
                if not ai_text.strip():
                    ai_text = "분석 내용을 생성하지 못했습니다."

                with st.expander(f"{category_text} | {title_text[:20]}...", expanded=True):
                    st.caption(f"🔗 소스: {title_text}")
                    st.success(ai_text)
                    st.markdown(f"[기사 원문 보기]({link_text})")
        else:
            st.warning("분석할 뉴스가 없습니다.")
    else:
        st.info("👆 버튼을 누르면 [미국 실물/증시, 한국 실물/증시] 뉴스를 각각 분석합니다.")

st.markdown("---")
st.markdown(f"<div style='text-align: center; color: gray;'>Created with Python & Streamlit | Engine: {display_model_name}</div>", unsafe_allow_html=True)
