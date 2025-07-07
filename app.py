import asyncio
from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel, RunConfig
from dotenv import load_dotenv
import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

if not gemini_api_key:
    if "GEMINI_API_KEY" in st.secrets:
        gemini_api_key = st.secrets["GEMINI_API_KEY"]
    else:
        raise ValueError("GEMINI_API_KEY is not set. Please ensure it is defined in your .env file or Streamlit secrets.")

external_client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=external_client
)

config = RunConfig(
    model=model,
    model_provider=external_client,
    tracing_disabled=True
)

crypto_agent = Agent(
    name="Crypto Agent",
    instructions="""You are a knowledgeable and helpful cryptocurrency expert.
    Your task is to provide accurate and concise information about cryptocurrencies.
    You can answer questions about:
    - Current prices (though real-time data is not available, provide general knowledge or trends).
    - Market capitalization.
    - Historical data (general trends, not specific charts).
    - News and recent developments (general knowledge, not live news feeds).
    - Definitions of cryptocurrencies, blockchain, NFTs, DeFi, etc.
    - Explanations of how certain cryptocurrencies or blockchain technologies work.
    - General advice on crypto (e.g., "do your own research," "volatile market").

    Always aim to provide balanced and informative responses. Do not give financial advice.
    """,
)

async def get_crypto_info(user_query):
    return await Runner.run(
        crypto_agent,
        input=user_query,
        run_config=config
    )

def get_simulated_crypto_data(symbol: str):
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    if symbol.upper() == "BTC":
        base_price = 40000
        volatility = 500
    elif symbol.upper() == "ETH":
        base_price = 2500
        volatility = 100
    elif symbol.upper() == "XRP":
        base_price = 0.5
        volatility = 0.05
    else:
        base_price = 100
        volatility = 10

    prices = base_price + np.cumsum(np.random.normal(0, volatility, 100))
    prices = np.maximum(prices, base_price * 0.1) 
    
    df = pd.DataFrame({
        'Date': dates,
        'Price': prices
    })
    return df

st.set_page_config(layout="wide", page_title="AI Crypto Agent", initial_sidebar_state="collapsed")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="st-"] {
        font-family: 'Inter', sans-serif;
    }

    .main {
        background-color: #121212;
        color: #e0e0e0;
        padding: 20px 40px;
    }
    .stButton>button {
        background-color: #007bff;
        color: white;
        font-size: 18px;
        font-weight: 600;
        padding: 12px 25px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
        transition: all 0.3s ease;
        cursor: pointer;
        background-image: linear-gradient(45deg, #007bff, #00c6ff);
    }
    .stButton>button:hover {
        background-color: #0056b3;
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0,0,0,0.4);
        background-image: linear-gradient(45deg, #0056b3, #0099cc);
    }
    .stTextArea textarea, .stTextInput input {
        border-radius: 10px;
        padding: 12px;
        font-size: 16px;
        border: 1px solid #007bff;
        background-color: #212121;
        color: #e0e0e0;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    }
    .stSelectbox div[data-baseweb="select"] {
        border-radius: 10px;
        border: 1px solid #007bff;
        background-color: #212121;
        color: #e0e0e0;
        box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);
    }
    .stSelectbox div[data-baseweb="select"] > div {
        color: #e0e0e0;
    }
    h1, h2 {
        color: #00c6ff;
        text-align: center;
        margin-bottom: 25px;
        text-shadow: 0 2px 5px rgba(0,198,255,0.3);
        font-weight: 700;
    }
    .stMarkdown {
        text-align: center;
        color: #b0c4de;
        font-size: 1.15em;
        margin-bottom: 40px;
        line-height: 1.6;
    }
    .stAlert {
        border-radius: 10px;
        background-color: #2a2a2a !important;
        color: #e0e0e0 !important;
        border-left: 5px solid #007bff !important;
    }
    .stSpinner > div {
        color: #00c6ff !important;
    }
    .footer-heading {
        font-size: 1.3em;
        color: #00c6ff;
        text-align: center;
        margin-top: 60px;
        padding-top: 25px;
        border-top: 1px solid #007bff;
        font-weight: 600;
        text-shadow: 0 1px 3px rgba(0,198,255,0.2);
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("💰 AI Crypto Agent")
st.markdown("Ask me anything about cryptocurrencies, blockchain, or market trends!")

st.header("Ask for Information")
st.markdown("Get general information about any cryptocurrency.")

# Modified column layout to center the input and button
col_left_spacer, col_center_content, col_right_spacer = st.columns([1, 4, 1])
with col_center_content:
    user_input = st.text_area("Enter your crypto query here:", height=100, placeholder="e.g., What is Bitcoin? Explain Ethereum. What are NFTs?")
    if st.button("Get Info", use_container_width=True):
        if user_input.strip() == "":
            st.warning("Please enter a query.")
        else:
            with st.spinner("Fetching information..."):
                crypto_response = asyncio.run(get_crypto_info(user_input))
                
                if crypto_response and crypto_response.final_output:
                    st.success("Here is the information:")
                    st.write(crypto_response.final_output)
                else:
                    st.error("Could not retrieve information. Please try again.")

st.markdown("---")

st.header("View Crypto Trends")
st.markdown("View past trends for any cryptocurrency (simulated data).")

# Modified column layout to center the input and button for trends
col_left_spacer_trends, col_center_content_trends, col_right_spacer_trends = st.columns([1, 4, 1])
with col_center_content_trends:
    crypto_symbol = st.text_input("Enter cryptocurrency symbol (e.g., BTC, ETH, XRP):", placeholder="e.g., BTC")
    if st.button("Show Trends", use_container_width=True):
        if crypto_symbol.strip() == "":
            st.warning("Please enter a cryptocurrency symbol.")
        else:
            with st.spinner(f"Loading trends for {crypto_symbol.upper()}..."):
                df_trends = get_simulated_crypto_data(crypto_symbol)
                
                if not df_trends.empty:
                    st.subheader(f"{crypto_symbol.upper()} Price Trends")
                    
                    chart = alt.Chart(df_trends).mark_line(point=True).encode(
                        x=alt.X('Date:T', title='Date'),
                        y=alt.Y('Price:Q', title='Price (USD)'),
                        tooltip=['Date:T', 'Price:Q']
                    ).properties(
                        title=f'{crypto_symbol.upper()} Price Trends'
                    ).interactive()
                    
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning(f"No data found for {crypto_symbol.upper()}.")

st.markdown('<p class="footer-heading">Built by Hammad Mustafa</p>', unsafe_allow_html=True)
