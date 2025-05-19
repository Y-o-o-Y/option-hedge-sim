#streamlit run c:/Users/sofjk/Desktop/Hedge/option_hedge/123.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime

# 字體支援中文（可視需求切換）
import matplotlib
matplotlib.rcParams['axes.unicode_minus'] = False  # 避免負號亂碼
matplotlib.rcParams['font.family'] = ['Microsoft JhengHei']

# 語言包
LANGS = {
    "zh": {
        "title": "蒙地卡羅模擬器",
        "start_button": "開始模擬",
        "initial_price": "初始股價",
        "drift": "年化飄移 (mu)",
        "vol": "年化隱含波動率 (sigma)",
        "days": "模擬天數",
        "paths": "模擬條數",
        "chart_title": "模擬股價路徑",
        "xlabel": "天數",
        "ylabel": "價格",
        "sim_mode": "模擬模式",
        "mode_1": "純布朗運動",
        "mode_2": "飄移 + 波動狀態",
        "mode_3": "飄移 + 波動狀態 + 跳躍事件",
        "mode_4": "Heston 隨機波動率模型",
        "mode_5": "Heston 3層不確定模型",
        "regime_yearly": "預期每年高波動次數",
        "regime_prob": "高波動出現機率（每日）",
        "regime_mult": "高波動倍率",
        "jump_yearly": "預期每年跳躍次數",
        "jump_prob": "跳躍事件機率（每日）",
        "jump_size": "跳躍幅度強度",
        "kappa": "均值回歸速度 κ",
        "xi": "Vol of Vol ξ",
        "theta": "長期平均 θ",
        "v0": "初始波動率 v₀",
        "rho": "股價與波動率相關性 ρ",
        "param_block": "模擬參數設定",
        "trade_block": "開倉交易",
        "position_block": "💼當前持倉",
        "option_type": "選擇交易類型",
        "strike_input": "輸入 Strike 價",
        "days_to_expiry": "剩餘天數 (T)",
        "atm_iv": "輸入 ATM IV",
        "iv_change": "IV 行權價變化",
        "pnl": "💰當前總損益",
        "close_all": "全部平倉",
        "warning": "⚠️ 請先執行模擬，產生股價路徑後才能下單。",
        "warning2": "⛔ 已到達模擬終點 請重新開始模擬",
        "Long Call": "買看漲期權",
        "Long Put": "買看跌期權",
        'execute': "下單",
        "next_step": "📆 下一天",
        'total_pnl': "💰當前總損益",
        "nothing": "空空",
        'ticker':"股票代號(例如:AAPL)",
    },
    "en": {
        "title": "Monte Carlo Simulator",
        "start_button": "Start Simulation",
        "initial_price": "Initial Price",
        "drift": "Annual Drift (mu)",
        "vol": "Annual Implied Volatility (sigma)",
        "days": "Simulation Days",
        "paths": "Number of Paths",
        "chart_title": "Simulated Price Paths",
        "xlabel": "Days",
        "ylabel": "Price",
        "sim_mode": "Simulation Mode",
        "mode_1": "Pure Brownian Motion",
        "mode_2": "Drift + Volatility Regime",
        "mode_3": "Drift + Volatility Regime + Jump Events",
        "mode_4": "Heston Stochastic Volatility",
        "mode_5": "Heston 3-Layer Uncertainty Model",
        "regime_yearly": "Expected High Volatility Events per Year",
        "regime_prob": "Probability of High Vol Regime (Daily)",
        "regime_mult": "High Volatility Multiplier",
        "jump_yearly": "Expected Number of Jump Events per Year",
        "jump_prob": "Jump Event Probability",
        "jump_size": "Jump Size (Std Dev)",
        "kappa": "Mean Reversion Speed κ",
        "xi": "Vol of Vol ξ",
        "theta": "Long-Term Volatility Mean θ",
        "v0": "Initial Volatility v₀",
        "rho": "Correlation ρ",
        "param_block": "Simulation Parameters",
        "trade_block": "Trade Entry",
        "position_block": "💼Current Positions",
        "option_type": "Select Option Type",
        "strike_input": "Enter Strike Price",
        "days_to_expiry": "Days to Expiry (T)",
        "atm_iv": "Enter ATM IV",
        "iv_change": "IV Change per Strike",
        "pnl": "💰Current Total PnL",
        "close_all": "Close All Positions",
        "warning": "⚠️ Please run the simulation first before placing orders.",
        "warning2": "⛔ Simulation Ended. Please restart.",
        "Long Call": "Long Call",
        "Long Put": "Long Put",
        'execute': "Execute Trade",
        "next_step": "📆 Next Day",
        'total_pnl': "💰Total PnL",
        "nothing": "Nothing Here",
        'ticker':"Ticker(Ex:AAPL)",
    }
}

lang_map = {
    "中文": "zh",
    "English": "en"
}
lang_label = st.selectbox("語言 / Language", list(lang_map.keys()))
lang_choice = lang_map.get(lang_label, "zh")  # fallback to zh
T = LANGS.get(lang_choice, LANGS["zh"])   
# 模擬函數
def simulate_brownian(S0, sigma, T, steps):
    dt = T / steps
    prices = [S0]
    for _ in range(steps):
        dS = prices[-1] * (sigma * np.sqrt(dt) * np.random.normal())
        prices.append(prices[-1] + dS)
    return prices

def simulate_regime(S0, mu, sigma, T, steps, high_vol_prob, sigma_multiplier):
    dt = T / steps
    prices = [S0]
    for _ in range(steps):
        regime_sigma = sigma * sigma_multiplier if np.random.rand() < high_vol_prob else sigma
        dS = prices[-1] * (mu * dt + regime_sigma * np.sqrt(dt) * np.random.normal())
        prices.append(prices[-1] + dS)
    return prices

def simulate_jump_diffusion(S0, mu, sigma, T, steps, high_vol_prob, sigma_multiplier, jump_prob, jump_size):
    dt = T / steps
    prices = [S0]
    for _ in range(steps):
        regime_sigma = sigma * sigma_multiplier if np.random.rand() < high_vol_prob else sigma
        jump = 1 + np.random.normal(0, jump_size) if np.random.rand() < jump_prob else 1
        dS = prices[-1] * (mu * dt + regime_sigma * np.sqrt(dt) * np.random.normal())
        prices.append((prices[-1] + dS) * jump)
    return prices
def simulate_heston(S0, mu, T, steps, kappa, theta, xi, v0, rho):
    dt = T / steps
    S = [S0]
    v = [v0]  #方差  後續的程式有平方
    for _ in range(steps):
        z1, z2 = np.random.normal(), np.random.normal()
        dW1 = z1 * np.sqrt(dt)
        dW2 = (rho * z1 + np.sqrt(1 - rho**2) * z2) * np.sqrt(dt)

        vt = max(v[-1], 0)                     # xi is vol of vol
        vt_new = vt + kappa * (theta - vt) * dt + xi * np.sqrt(vt) * dW2
        vt_new = max(vt_new, 0)  # ✅ 防止變負
        v.append(vt_new)

        S_new = S[-1] * np.exp((mu - 0.5 * vt_new) * dt + np.sqrt(vt_new) * dW1)
        S.append(S_new)
    iv_path = [np.sqrt(max(v_, 0)) for v_ in v]
    return S, iv_path

# === Heston + Regime + Jump 模擬函數 ===
def simulate_heston_regime_jump(S0, mu, T, steps, kappa, theta, xi, v0, rho,
                                regime_prob, regime_mult, jump_prob, jump_size):
    dt = T / steps
    S = [S0]
    v = [v0]
    for _ in range(steps):
        z1, z2 = np.random.normal(), np.random.normal()
        dW1 = z1 * np.sqrt(dt)
        dW2 = (rho * z1 + np.sqrt(1 - rho**2) * z2) * np.sqrt(dt)

        vt = max(v[-1], 0)
        vt_new = vt + kappa * (theta - vt) * dt + xi * np.sqrt(vt) * dW2
        vt_new = max(vt_new, 0)
        v.append(vt_new)

        sigma_eff = np.sqrt(vt_new)
        if np.random.rand() < regime_prob:
            sigma_eff *= regime_mult

        jump = 1 + np.random.normal(0, jump_size) if np.random.rand() < jump_prob else 1
        dS = S[-1] * (mu * dt + sigma_eff * dW1)
        S_new = (S[-1] + dS) * jump
        S.append(S_new)
    iv_path = [np.sqrt(max(v_, 0)) for v_ in v]
    return S, iv_path

def skew_iv(base_iv, S, K, slope=-0.6):
    m = np.log(K / S)
    adjusted = base_iv * (1 + slope * m)
    
    return max(adjusted, 0.01)  # 避免 IV 為負


# 主介面
st.title(T["title"])
with st.expander("⚙️ " + T["param_block"], expanded=True):
    # === 使用者輸入 Ticker ===
    ticker_symbol = st.text_input(T["ticker"], value=T["ticker"]).upper()
    sim_mode = st.selectbox(T["sim_mode"], [T["mode_1"], T["mode_2"], 
                                            T["mode_3"], T["mode_4"], T["mode_5"]])

    # 自動抓取最新收盤價與整體IV
    try:
        yf_data = yf.Ticker(ticker_symbol)
        latest_price = yf_data.history(period="1d")["Close"][-1]
        default_iv = yf_data.info.get("impliedVolatility", 0.3)
    except:
        latest_price = 100.0
        default_iv = 0.3

    S0 = st.number_input(T["initial_price"], value=float(latest_price))
    mu = st.number_input(T["drift"], value=0.1)

    if sim_mode in [T["mode_4"], T["mode_5"]]:
        sigma = None
    else:
        sigma = st.number_input(T["vol"], value=float(default_iv))

    T_days = st.number_input(T["days"], value=30)
    paths = st.number_input(T["paths"], value=100, min_value=1, step=1)



    if sim_mode in [T["mode_4"], T["mode_5"]]:
        kappa = st.slider(T["kappa"], 0.1, 10.0, 5.0, step=0.1)
        xi = st.slider(T["xi"], 0.1, 1.0, 0.15, step=0.01)
        theta = st.slider(T["theta"], 0.01, 1.0, 0.4, step=0.01)
        v0 = st.slider(T["v0"], 0.01, 1.0, 0.4, step=0.01)
        rho = st.slider(T["rho"], -1.0, 1.0, -0.5, step=0.1)

    if sim_mode in [T["mode_2"], T["mode_3"], T["mode_5"]]:
        yearly_events = st.number_input(T["regime_yearly"], min_value=0, max_value=50, value=4)
        auto_regime_prob = round(yearly_events / 365, 4)
        regime_prob = st.slider(T["regime_prob"], 0.0, 0.2, auto_regime_prob, step=0.001, format="%.3f")
        regime_mult = st.slider(T["regime_mult"], 1.0, 3.0, 1.3, step=0.1)

    if sim_mode in [T["mode_3"], T["mode_5"]]:
        jump_events = st.number_input(T["jump_yearly"], min_value=0, max_value=50, value=4)
        auto_jump_prob = round(jump_events / 365, 4)
        jump_prob = st.slider(T["jump_prob"], 0.0, 0.1, auto_jump_prob, step=0.001, format="%.3f")
        jump_size = st.slider(T["jump_size"], 0.0, 0.5, 0.05, step=0.01)

   
# === Session State 初始化（建議放在 app.py 頂部） ===
if "force_resample" not in st.session_state:
    st.session_state.force_resample = False
if "price_path" not in st.session_state:
    st.session_state.price_path = []
if "v_path" not in st.session_state:
    st.session_state.v_path = []
if "day" not in st.session_state:
    st.session_state.day = 0
if "positions" not in st.session_state:
    st.session_state.positions = []
if "has_stepped" not in st.session_state:
    st.session_state.has_stepped = False
if "cash" not in st.session_state:
    st.session_state.cash = 100000
if "total_pnl" not in st.session_state:
    st.session_state.total_pnl = 0.0
if "last_closed_pnl" not in st.session_state:
    st.session_state.last_closed_pnl = 0.0
if "realized_pnl" not in st.session_state:
    st.session_state.realized_pnl = 0.0
if "all_paths" not in st.session_state:
    st.session_state.all_paths = []

    
# === 自動抓取最近到期日 & 對應 Strike IV ===
def find_nearest_expiry(desired_days, ticker, tolerance=3):
    t = yf.Ticker(ticker)
    today = datetime.today()
    closest_diff = float("inf")
    best_match = None

    for exp_str in t.options:
        exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
        days_to_expiry = (exp_date - today).days
        diff = abs(desired_days - days_to_expiry)
        if diff <= tolerance and diff < closest_diff:
            best_match = (exp_str, days_to_expiry)
            closest_diff = diff

    return best_match  # 回傳 (到期日字串, 實際天數)

def get_strike_iv(desired_days, strike, ticker):
    expiry_data = find_nearest_expiry(desired_days, ticker)
    if not expiry_data:
        return None, None, None

    expiry_str, actual_days = expiry_data
    t = yf.Ticker(ticker)
    spot = t.history(period="1d")["Close"][-1]
    calls = t.option_chain(expiry_str).calls.copy()
    calls["strike_diff"] = abs(calls["strike"] - strike)
    closest = calls.sort_values("strike_diff").iloc[0]
    strike_iv = closest["impliedVolatility"]

    return strike_iv, expiry_str, actual_days

# === Black-Scholes 定價函數 ===

def bs_price(S, K, T, r, sigma, option_type='call', return_d=False):
    if T <= 0:
        return (max(S - K, 0) if option_type == 'call' else max(K - S, 0)) if not return_d else (0, 0, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price
# === 模擬啟動按鈕與價格路徑選擇 ===
if st.button(T["start_button"]) or st.session_state.force_resample:
    st.session_state.force_resample = False
    st.session_state.has_stepped = False
    st.session_state.total_pnl = 0.0
    st.session_state.day = 0
    st.session_state.positions = []
    st.session_state.cash = 100000
    st.session_state.day = 0
    st.session_state.all_paths = []
    if sim_mode == T["mode_4"]:
        v0_adj = v0 ** 2
        theta_adj = theta ** 2
        fig, ax = plt.subplots(figsize=(8, 4))
        for _ in range(int(paths)):
            s_path, v_path = simulate_heston(S0, mu, T_days / 365, int(T_days), kappa, theta_adj, xi, v0_adj, rho)
            ax.plot(s_path, alpha=0.6)
            st.session_state.all_paths.append((s_path, v_path))
        st.pyplot(fig)
        chosen = np.random.choice(len(st.session_state.all_paths))
        st.session_state.price_path = st.session_state.all_paths[chosen][0]
        st.session_state.v_path = st.session_state.all_paths[chosen][1]
        fig_iv, ax_iv = plt.subplots(figsize=(8, 3))
        for i, (_, v_path) in enumerate(st.session_state.all_paths):
            ax_iv.plot(v_path, alpha=0.3, label="IV" if i == 0 else "")
    
        # 畫長期均值 θ 的水平線
        ax_iv.axhline(theta, color='gray', linestyle='--', label='θ Long Term Var Mean' if lang_choice == "en" else 'θ 長期方差平均')
        ax_iv.set_title("IV Path" if lang_choice == "en" else "模擬波動率變化 (IV 路徑)")
        ax_iv.set_xlabel(T["xlabel"])
        ax_iv.set_ylabel("IV"if lang_choice == "en" else "隱含波動率")
        ax_iv.legend()
        st.pyplot(fig_iv)

    elif sim_mode == T["mode_5"]:
        v0_adj = v0 ** 2
        theta_adj = theta ** 2
        fig, ax = plt.subplots(figsize=(8, 4))
        for _ in range(int(paths)):
            s_path, v_path = simulate_heston_regime_jump(S0, mu, T_days / 365, int(T_days),
                kappa, theta_adj, xi, v0_adj, rho, regime_prob, regime_mult, jump_prob, jump_size)
            ax.plot(s_path, alpha=0.6)
            st.session_state.all_paths.append((s_path, v_path))
        st.pyplot(fig)
        chosen = np.random.choice(len(st.session_state.all_paths))
        st.session_state.price_path = st.session_state.all_paths[chosen][0]
        st.session_state.v_path = st.session_state.all_paths[chosen][1]
        fig_iv, ax_iv = plt.subplots(figsize=(8, 3))
        for i, (_, v_path) in enumerate(st.session_state.all_paths):
            ax_iv.plot(v_path, alpha=0.3, label="IV" if i == 0 else "")
    
        # 畫長期均值 θ 的水平線
        ax_iv.axhline(theta, color='gray', linestyle='--', label='θ Long Term Var Mean' if lang_choice == "en" else 'θ 長期方差平均')
        ax_iv.set_title("IV Path" if lang_choice == "en" else "模擬波動率變化 (IV 路徑)")
        ax_iv.set_xlabel(T["xlabel"])
        ax_iv.set_ylabel("IV"if lang_choice == "en" else "隱含波動率")
        ax_iv.legend()
        st.pyplot(fig_iv)
    else:
        fig, ax = plt.subplots()
        for _ in range(int(paths)):
            if sim_mode == T["mode_1"]:
                path = simulate_brownian(S0, sigma, T_days / 365, int(T_days))
            elif sim_mode == T["mode_2"]:
                path = simulate_regime(S0, mu, sigma, T_days / 365, int(T_days), regime_prob, regime_mult)
            else:
                path = simulate_jump_diffusion(S0, mu, sigma, T_days / 365, int(T_days), regime_prob, regime_mult, jump_prob, jump_size)
            ax.plot(path, alpha=0.6)
            st.session_state.all_paths.append(path)
        st.pyplot(fig)
        st.session_state.price_path = st.session_state.all_paths[np.random.choice(len(st.session_state.all_paths))]

# 🛒 交易區塊（含 Heston IV 切換及真實到期日記錄）
with st.expander("🛒 " + T["trade_block"], expanded=True):
    price_path = st.session_state.get("price_path", [])
    current_day = st.session_state.get("day", 0)

    if not isinstance(price_path, list) or len(price_path) == 0:
        st.warning(T["warning"])
    else:
        option_label = st.selectbox(T["option_type"], [T["Long Call"], T["Long Put"]])
        option_type = "call" if "Call" in option_label or "看漲" in option_label else "put"

        strike = st.number_input(T["strike_input"], value=100.0)
        days_to_expiry = st.number_input(T["days_to_expiry"], value=30)
        strike_iv, _, true_days = get_strike_iv(days_to_expiry, strike, ticker_symbol)

        if strike_iv is None:
            st.error("❌ 無法取得該行權價的 IV，請調整 Strike 或到期日。")
        elif st.button(T['execute']):
            if current_day < len(price_path):
                S = price_path[current_day]

                # 確認實際到期天數
                actual_days = true_days if true_days else days_to_expiry

                # 1) 選 base_iv：
                #    - 第0天開倉用 yf 的 strike_iv
                #    - 之後若為 Heston 模型 (mode_4/5) 且已有模擬 v_path，就用模擬 IV
                if sim_mode in [T['mode_4'], T['mode_5']] \
                   and st.session_state.get('v_path') \
                   and current_day > 0:
                    base_iv = st.session_state['v_path'][current_day]
                else:
                    base_iv = strike_iv

                # 2) 建倉時不做微笑調整
                iv_used = base_iv

                # 3) 定價
                T_days = actual_days
                r = 0.045
                premium = bs_price(
                    S, strike, T_days / 365,
                    r, iv_used,
                    option_type=option_type
                )

                # 💰 扣款
                st.session_state.cash -= premium * 100

                # 📦 記錄持倉
                st.session_state.positions.append({
                    'type': option_type,
                    'strike': strike,
                    'premium': premium,
                    'T_days_input': actual_days,
                    'T': T_days / 365,
                    'day_entered': current_day,
                    'iv': iv_used
                })

                # 成功訊息：顯示開倉時使用的 IV
                st.success(
                    f"✅ 已建立 {option_type.upper()} 倉位，IV={iv_used:.2%}，T={T_days} 天，權利金={premium:.2f}"
                )
            else:
                st.error("⚠️ 模擬尚未開始或價格資料不足")


# === 模擬前進區塊 ===
col_left, col_right = st.columns([1, 4])
with col_left:
    st.markdown("### ") 
    st.markdown("### ")
    st.markdown("### ")
    st.markdown("### ")
    step_clicked = st.button(T["next_step"])
    st.markdown(f"**{T['total_pnl']}: ${st.session_state.total_pnl:.2f}**")
    
with col_right:
    price_path = st.session_state.get("price_path", [])

    if isinstance(price_path, list) and len(price_path) > 0:
        current_day = st.session_state.get("day", 0)

        if step_clicked:
            if current_day + 1 < len(price_path):
                st.session_state.day = current_day + 1
                st.session_state.has_stepped = True
                current_day = st.session_state.day

                # 自動結算到期倉位
                settled = []
                S = price_path[current_day]
                r = 0.045
                for i, pos in enumerate(st.session_state.positions):
                    days_held = current_day - pos["day_entered"]
                    remaining_days = pos["T_days_input"] - days_held
                    if remaining_days <= 0:
                        if pos["type"] == "call":
                            price = max(S - pos["strike"], 0)
                        else:
                            price = max(pos["strike"] - S, 0)
                        pnl = (price - pos["premium"]) * 100
                        st.session_state.total_pnl += pnl
                        settled.append(i)
                for i in reversed(settled):
                    del st.session_state.positions[i]

        price_path = st.session_state.get("price_path", [])
        if current_day < len(price_path):
            current_price = price_path[current_day]
            prev_price = price_path[current_day - 1] if current_day > 0 else current_price
            price_change_pct = ((current_price - prev_price) / prev_price) * 100 if current_day > 0 else 0

            color = "green" if price_change_pct > 0 else "red" if price_change_pct < 0 else "gray"

            st.markdown(
                f"""
                <div style='background-color:#eaf3fc; padding:10px; border-radius:8px'>
                    <span style='font-size:16px;'>📈 Day {current_day} Price = <b>{current_price:.2f}</b> 
                    <span style='color:{color};'>({price_change_pct:+.2f}%)</span></span>
                </div>
                """,
                unsafe_allow_html=True
            )

            fig, ax = plt.subplots()
            ax.plot(price_path[:current_day + 1], marker="o", alpha=0.7)
            ax.set_title(f"Price Path ({current_day} Day)" if lang_choice == "en" else f"價格路徑 (第 {current_day} 天)")
            ax.set_xlim(0, current_day + 5)
            ax.set_ylim(current_price * 0.9, current_price * 1.1)
            st.pyplot(fig)

           
            if st.button("🔁 重新模擬路徑"):
                st.session_state.force_resample = True
                
                if "all_paths" in st.session_state and st.session_state.all_paths:
                    chosen = np.random.choice(len(st.session_state.all_paths))
                    st.session_state.price_path = st.session_state.all_paths[chosen][0]
                    st.session_state.v_path = st.session_state.all_paths[chosen][1]
                    st.session_state.day = 0
                    st.session_state.positions = []
                    st.session_state.total_pnl = 0.0
                    st.session_state.cash = 100000
                    st.session_state.has_stepped = False
                    st.rerun()
        else:
            st.warning(T["warning2"])
    else:
        st.warning(T["warning2"])

# 🗂 持倉與浮動損益區塊（含 Heston IV 路徑與微笑調整）
# 📦 初始設置（只需寫一次）
if "trigger_close" not in st.session_state:
    st.session_state.trigger_close = False

with st.expander(T["position_block"], expanded=True):
    price_path = st.session_state.get("price_path", [])
    current_day = st.session_state.get("day", 0)

    if not isinstance(price_path, list) or len(price_path) == 0:
        st.warning("尚未模擬股價路徑")
    elif current_day >= len(price_path):
        st.warning(T["nothing"])
    else:
        S = price_path[current_day]
        r = 0.045
        floating_pnl = 0

        for i, pos in enumerate(st.session_state.positions):
            days_held = current_day - pos["day_entered"]
            remaining_days = max(int(pos["T_days_input"] - days_held), 0)
            remaining_T = max(pos["T"] - days_held / 365, 1e-4)

            # 到期 payoff
            if remaining_days <= 0:
                if pos["type"] == "call":
                    price = max(S - pos["strike"], 0)
                else:
                    price = max(pos["strike"] - S, 0)
                iv_used = pos["iv"]

            # 持倉當天不做第二次微笑調整
            elif current_day == pos["day_entered"]:
                iv_used = pos["iv"]
                price = bs_price(
                    S, pos["strike"], remaining_T,
                    r, iv_used,
                    option_type=pos["type"]
                )

            # 後續日子：Heston 路徑 + 微笑
            else:
                if sim_mode in [T['mode_4'], T['mode_5']] and st.session_state.get('v_path'):
                    base_iv = st.session_state['v_path'][current_day]
                else:
                    base_iv = pos['iv']
                iv_used = skew_iv(base_iv, S, pos["strike"])
                price = bs_price(
                    S, pos["strike"], remaining_T,
                    r, iv_used,
                    option_type=pos["type"]
                )

            pnl = (price - pos["premium"]) * 100
            floating_pnl += pnl

            st.write(
                f"{i+1}. {pos['type']} | Strike: {pos['strike']} | IV_used: {iv_used:.2%} | "
                f"剩餘T: {remaining_days} 天 | PnL: {pnl:.2f}"
            )

        if st.session_state.positions:
            st.write(f"💰 當前浮動損益：${floating_pnl:,.2f}")
        else:
            st.write(f"💰 累積總損益：${st.session_state.total_pnl:,.2f}")

        # 全部平倉同理
        if st.button("全部平倉"):
            for pos in st.session_state.positions:
                days_held = current_day - pos["day_entered"]
                remaining_days = max(int(pos["T_days_input"] - days_held), 0)
                remaining_T = max(pos["T"] - days_held / 365, 1e-4)

                if remaining_days <= 0:
                    price = max((S - pos["strike"]) if pos["type"] == "call" else (pos["strike"] - S), 0)
                elif current_day == pos["day_entered"]:
                    iv_used = pos["iv"]
                    price = bs_price(
                        S, pos["strike"], remaining_T,
                        r, iv_used,
                        option_type=pos["type"]
                    )
                else:
                    if sim_mode in [T['mode_4'], T['mode_5']] and st.session_state.get('v_path'):
                        base_iv = st.session_state['v_path'][current_day]
                    else:
                        base_iv = pos['iv']
                    iv_used = skew_iv(base_iv, S, pos["strike"])
                    price = bs_price(
                        S, pos["strike"], remaining_T,
                        r, iv_used,
                        option_type=pos["type"]
                    )

                pnl = (price - pos["premium"]) * 100
                st.session_state.total_pnl += pnl

            st.session_state.positions.clear()
            st.rerun()


