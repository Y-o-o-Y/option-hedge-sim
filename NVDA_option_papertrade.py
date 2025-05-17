import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import os
from pathlib import Path
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

# Black-Scholes期權定價（Call/Put）
def bs_price(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)
        else:
            return max(K - S, 0)
    d1 = (math.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*math.sqrt(T))
    d2 = d1 - sigma*math.sqrt(T)
    if option_type == 'call':
        return S*norm.cdf(d1) - K*math.exp(-r*T)*norm.cdf(d2)
    else:
        return K*math.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
 
# IV 微笑調整函數
def smile_adjusted_iv(base_iv, strike, spot):
    diff_pct = abs(strike - spot) / spot
    control=(0.001*spot)/base_iv  #baseiv*(1/135)*? = 0.1%
    return  base_iv+(diff_pct*(control)) # 微笑型結構
def term_structure_adjusted_iv(base_iv, strike, spot, T_days, long_term_boost=0.55):
    # """long_term_boost: x天期的增幅比例，預設為50%
    diff_pct = abs(strike - spot) / spot
    control = (0.001 * spot) / base_iv
    iv_smile = base_iv + (diff_pct * control)
    if T_days < 11 and T_days > 4:
        T_days = 7
    factor = 1 + np.abs(long_term_boost * ((T_days-7) / 30) ** 0.5)
    return iv_smile * factor
def update_iv_heston(sigma_daily, dt=1/365, kappa=5.0, theta=0.5793, xi=0.15,rho=-0.55):
    v_t = sigma_daily**2 # ← 換成 daily 方差
    theta = (theta ** 2) * dt   # ← 換成 daily 方差
    xi = xi / np.sqrt(365)  #Daily方差表準差
    z1 = np.random.normal()
    z2 = np.random.normal()
    dW_2 = rho * z1 + np.sqrt(1 - rho**2) * z2

    v_t = max(v_t, 1e-6)
    dv = kappa * (theta - v_t) * dt + xi * np.sqrt(v_t) * np.sqrt(dt) * dW_2
    v_next = max(v_t + dv, 1e-6)
    return np.sqrt(v_next) #傳出方差開更號(標準差) daily
# def call_delta(S, K, T, r, sigma):
#     if T <= 0 or sigma <= 0:
#         return 0
#     d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
#     return norm.cdf(d1)

# def adjusted_iv_tqqq(base_iv, strike_diff):
#     return base_iv + strike_diff * -0.02

# def adjusted_iv_sqqq(base_iv, strike_diff):
#     return base_iv + strike_diff * 0.02

# 持倉類別
class Position:
    def __init__(self, underlying, option_type, direction, K, premium, T):
        self.underlying = underlying
        self.option_type = option_type
        self.direction = direction
        self.K = K
        self.premium = premium
        self.T = T

    def update_T(self):
        self.T -= 1/365

    def value(self, S, r, base_iv):
        iv_used = term_structure_adjusted_iv(base_iv, self.K, S, self.T * 365)
        price = bs_price(S, self.K, max(self.T, 0), r, iv_used, self.option_type)
        return price

# 模擬器類別
class OptionSimulator:
    def __init__(self, initial_cash, S0_TQQQ, S0_SQQQ, sigma_annual,r,base_iv=0.85):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.S_TQQQ = S0_TQQQ
        self.S_SQQQ = S0_SQQQ
        self.sigma_annual = sigma_annual
        self.sigma_daily = sigma_annual / np.sqrt(365)
        self.r = r
        self.positions = []
        self.day = 0
        self.log = []
        self.prev_S_TQQQ = S0_TQQQ
        self.prev_S_SQQQ = S0_SQQQ
        self.base_iv = base_iv
        self.trades = []
        self.price_path = []
        self.iv_path = []

    def buy_option(self, underlying, K, option_type, days_to_expire, iv):
        S = self.S_TQQQ if underlying == 'TQQQ' else self.S_SQQQ
        T = days_to_expire / 365
        premium = bs_price(S, K, T, self.r, iv, option_type)
        self.cash -= premium * 100
        self.positions.append(Position(underlying, option_type, 'long', K, premium, T))


    def sell_option(self, underlying, K, option_type, days_to_expire, iv):
        S = self.S_TQQQ if underlying == 'TQQQ' else self.S_SQQQ
        T = days_to_expire / 365
        premium = bs_price(S, K, T, self.r, iv, option_type)
        self.cash += premium * 100
        self.positions.append(Position(underlying, option_type, 'long', K, premium, T))

    def auto_build_long_call(sim, iv1, iv2, days_to_expire, range_pct, lot1=1, lot2=1):
        S = sim.S_TQQQ
        low = int(S * (1 - range_pct))
        high = int(S * (1 + range_pct))
        print(f"可選 TQQQ Long Call 行權價區間: {low} ~ {high}")
        K1 = int(input("請輸入 TQQQ Long Call Strike："))
        for _ in range(lot1):
            sim.buy_option('TQQQ', K1, 'call', days_to_expire, iv1)

        S = sim.S_SQQQ
        low = int(S * (1 - range_pct))
        high = int(S * (1 + range_pct))
        print(f"可選 SQQQ Long Call 行權價區間: {low} ~ {high}")
        K2 = int(input("請輸入 SQQQ Long Call Strike："))
        for _ in range(lot2):
            sim.buy_option('SQQQ', K2, 'call', days_to_expire, iv2)


    def auto_build_sell_call(sim, iv1, iv2, days_to_expire, range_pct, lot1=1, lot2=1):
        S = sim.S_TQQQ
        low = int(S * (1 - range_pct))
        high = int(S * (1 + range_pct))
        print(f"可選 TQQQ Sell Call 行權價區間: {low} ~ {high}")
        K1 = int(input("請輸入 TQQQ Sell Call Strike："))
        for _ in range(lot1):
            sim.sell_option('TQQQ', K1, 'call', days_to_expire, iv1)

        S = sim.S_SQQQ
        low = int(S * (1 - range_pct))
        high = int(S * (1 + range_pct))
        print(f"可選 SQQQ Sell Call 行權價區間: {low} ~ {high}")
        K2 = int(input("請輸入 SQQQ Sell Call Strike："))
        for _ in range(lot2):
            sim.sell_option('SQQQ', K2, 'call', days_to_expire, iv2)


    def auto_build_long_put(sim, iv1, iv2, days_to_expire, range_pct, lot1=1, lot2=1):
        S = sim.S_TQQQ
        low = int(S * (1 - range_pct))
        high = int(S * (1 + range_pct))
        print(f"可選 TQQQ Long Put 行權價區間: {low} ~ {high}")
        K1 = int(input("請輸入 TQQQ Long Put Strike："))
        for _ in range(lot1):
            sim.buy_option('TQQQ', K1, 'put', days_to_expire, iv1)

        S = sim.S_SQQQ
        low = int(S * (1 - range_pct))
        high = int(S * (1 + range_pct))
        print(f"可選 SQQQ Long Put 行權價區間: {low} ~ {high}")
        K2 = int(input("請輸入 SQQQ Long Put Strike："))
        for _ in range(lot2):
            sim.buy_option('SQQQ', K2, 'put', days_to_expire, iv2)


    def auto_build_sell_put(sim, iv1, iv2, days_to_expire, range_pct, lot1=1, lot2=1):
        S = sim.S_TQQQ
        low = int(S * (1 - range_pct))
        high = int(S * (1 + range_pct))
        print(f"可選 TQQQ Sell Put 行權價區間: {low} ~ {high}")
        K1 = int(input("請輸入 TQQQ Sell Put Strike："))
        for _ in range(lot1):
            sim.sell_option('TQQQ', K1, 'put', days_to_expire, iv1)

        S = sim.S_SQQQ
        low = int(S * (1 - range_pct))
        high = int(S * (1 + range_pct))
        print(f"可選 SQQQ Sell Put 行權價區間: {low} ~ {high}")
        K2 = int(input("請輸入 SQQQ Sell Put Strike："))
        for _ in range(lot2):
            sim.sell_option('SQQQ', K2, 'put', days_to_expire, iv2)

    def build_diagonal_call(self, long_days, long_strike, short_days, short_strike, underlying='TQQQ'):
        S = self.S_TQQQ if underlying == 'TQQQ' else self.S_SQQQ
        T_long = long_days / 365
        T_short = short_days / 365
        iv_long = term_structure_adjusted_iv(self.base_iv, long_strike, S, long_days)
        iv_short = term_structure_adjusted_iv(self.base_iv, short_strike, S, short_days)

        premium_long = bs_price(S, long_strike, T_long, self.r, iv_long, 'call')
        premium_short = bs_price(S, short_strike, T_short, self.r, iv_short, 'call')

        self.cash -= premium_long * 100
        self.cash += premium_short * 100

        self.positions.append(Position(underlying, 'call', 'long', long_strike, premium_long, T_long))
        self.positions.append(Position(underlying, 'call', 'short', short_strike, premium_short, T_short))



    def build_diagonal_put(self, long_days, long_strike, short_days, short_strike, underlying='TQQQ'):
        S = self.S_TQQQ if underlying == 'TQQQ' else self.S_SQQQ
        T_long = long_days / 365
        T_short = short_days / 365
        iv_long = term_structure_adjusted_iv(self.base_iv, long_strike, S, long_days)
        iv_short = term_structure_adjusted_iv(self.base_iv, short_strike, S, short_days)

        premium_long = bs_price(S, long_strike, T_long, self.r, iv_long, 'put')
        premium_short = bs_price(S, short_strike, T_short, self.r, iv_short, 'put')

        self.cash -= premium_long * 100
        self.cash += premium_short * 100

        self.positions.append(Position(underlying, 'put', 'long', long_strike, premium_long, T_long))
        self.positions.append(Position(underlying, 'put', 'short', short_strike, premium_short, T_short))


    def build_diagonal_combo(sim, u, long_d_call, long_k_call, long_iv_call,
                          short_d_call, short_k_call, short_iv_call,
                          long_d_put, long_k_put,long_iv_put,
                          short_d_put, short_k_put, short_iv_put):
        sim.build_diagonal_call(long_d_call, long_k_call, long_iv_call, short_d_call, short_k_call, short_iv_call, u)
        sim.build_diagonal_put(long_d_put, long_k_put, short_d_put, short_k_put,u, long_iv_put, short_iv_put)
    
    def long_vertical_call(self, underlying, K_buy, K_sell, days_to_expire):
        S = self.S_TQQQ if underlying == 'TQQQ' else self.S_SQQQ
        T = days_to_expire / 365
        iv_buy = term_structure_adjusted_iv(self.base_iv, K_buy, S, days_to_expire)
        iv_sell = term_structure_adjusted_iv(self.base_iv, K_sell, S, days_to_expire)

        premium_buy = bs_price(S, K_buy, T, self.r, iv_buy, 'call')
        premium_sell = bs_price(S, K_sell, T, self.r, iv_sell, 'call')
        self.cash -= premium_buy * 100
        self.cash += premium_sell * 100
       
        self.positions.append(Position(underlying, 'call', 'long', K_buy, premium_buy, T))
        for _ in range(1):
            self.positions.append(Position(underlying, 'call', 'short', K_sell, premium_sell, T))

    def long_vertical_put(self, underlying, K_buy, K_sell, days_to_expire):
        S = self.S_TQQQ if underlying == 'TQQQ' else self.S_SQQQ
        T = days_to_expire / 365
        iv_buy = term_structure_adjusted_iv(self.base_iv, K_buy, S, days_to_expire)
        iv_sell = term_structure_adjusted_iv(self.base_iv, K_sell, S, days_to_expire)

        premium_buy = bs_price(S, K_buy, T, self.r, iv_buy, 'put')
        premium_sell = bs_price(S, K_sell, T, self.r,iv_sell, 'put')
        self.cash -= premium_buy * 100
        self.cash += premium_sell * 100
        
        self.positions.append(Position(underlying, 'put', 'long', K_buy, premium_buy, T))
        for _ in range(1):
            self.positions.append(Position(underlying, 'put', 'short', K_sell, premium_sell, T))

    
        

    def close_position(self, idx):
        if idx < 0 or idx >= len(self.positions):
            print("無效的持倉編號！")
            return
        pos = self.positions[idx]
        S = self.S_TQQQ if pos.underlying == 'TQQQ' else self.S_SQQQ
        price = pos.value(S, self.r,self.base_iv) 
        pnl = 0
        if pos.direction == 'long':
            self.cash += price * 100
            pnl = (price - pos.premium) * 100
        else:
            self.cash -= price * 100
            pnl = (pos.premium - price) * 100
        result = 'Win' if pnl > 0 else 'Loss'
        self.trades.append({
            'Day': self.day,
            'Underlying': pos.underlying,
            'Option Type': pos.option_type,
            'Strike': pos.K,
            'Direction': pos.direction,
            'Premium': round(pos.premium, 2),
            'Close Price': round(price, 2),
            'PnL': round(pnl, 2),
            'Result': result
        })
        print(f"已平倉: {pos.direction.upper()} {pos.option_type.upper()} {pos.underlying} K={pos.K} 以價格 {price:.2f} PnL={pnl:.2f}")
        self.positions.pop(idx)
    
    def next_day(self):
        if self.day == 0:
            self.iv_tqqq = 0.85
            self.iv_sqqq = 0.85
            self.v_for_price = self.sigma_daily 
            self.prev_mu = np.log(1 + 0.55) / 365 

        if np.random.rand() < 0.03:  #theta=0.5793, xi=0.15
            regime = 'HIGH'
            theta = 0.5793*1.3
            xi = 0.15*1.3
        else:
            regime = 'NORMAL'
            theta = 0.5793
            xi = 0.15
            
        self.v_for_price = update_iv_heston(self.v_for_price,theta=theta, xi=xi)  # 這裡的 rho 是 Heston 模型的參數
        market_sigma = self.v_for_price
        mu_daily = 0.8 * self.prev_mu + 0.2 * np.random.normal(scale=0.002)
        self.prev_mu = mu_daily  # AR(1)
        market_move = mu_daily + np.random.standard_t(df=20) * market_sigma  #T分布
        self.prev_S_TQQQ = self.S_TQQQ
        self.prev_S_SQQQ = self.S_SQQQ
     

        self.S_TQQQ *=  np.exp(market_move)
        self.S_SQQQ *=  np.exp(-market_move)
        
        for_iv_path  = self.v_for_price*np.sqrt(365)  # 轉回年化iv
        self.iv_path.append(for_iv_path)
        for pos in self.positions:
            pos.update_T()
        
        self.day += 1
        self.settle_expired()
        self.record_log()


       
    def settle_expired(self):
        to_remove = []
        for pos in self.positions:
            if pos.T <= 0:
                S = self.S_TQQQ if pos.underlying == 'TQQQ' else self.S_SQQQ
                intrinsic = max(0, S - pos.K) if pos.option_type == 'call' else max(0, pos.K - S)
                if pos.direction == 'long':
                    self.cash += intrinsic * 100
                else:
                    self.cash -= intrinsic * 100
                to_remove.append(pos)
        for pos in to_remove:
            self.positions.remove(pos)

    def portfolio_value(self):
        value = self.cash
        for pos in self.positions:
            S = self.S_TQQQ if pos.underlying == 'TQQQ' else self.S_SQQQ
            market_price = pos.value(S, self.r, self.base_iv)  # 用倉位建立時的 IV
            entry_price = pos.premium
            if pos.direction == 'long':
                value += market_price * 100  # long 倉現值
            else:
                value -= market_price * 100   # short 倉為已收保費 + 浮損益
        return value

    def show_status(self):
        if self.day == 0:
            chg_T = chg_S = 0
            iv_annual = self.sigma_annual
        else:
            chg_T = (self.S_TQQQ - self.prev_S_TQQQ) / self.prev_S_TQQQ * 100
            chg_S = (self.S_SQQQ - self.prev_S_SQQQ) / self.prev_S_SQQQ * 100
            iv_annual = self.v_for_price * np.sqrt(365)
        print(f"Day {self.day}, TQQQ: {self.S_TQQQ:.2f} ({chg_T:+.2f}%),IV: {iv_annual:.2%} ,Portfolio Value: {self.portfolio_value():.2f}")
    
        for idx, pos in enumerate(self.positions):
            S = self.S_TQQQ if pos.underlying == 'TQQQ' else self.S_SQQQ
            current_price = pos.value(S, self.r,self.base_iv)
            T_days = pos.T * 365
            current_iv = term_structure_adjusted_iv(self.base_iv, pos.K, S, T_days)
            if pos.direction == 'long':
                pnl = (current_price - pos.premium) * 100
            else:
                pnl = (pos.premium - current_price) * 100
            print(f"{idx}: {pos.direction.upper()} {pos.option_type.upper()} {pos.underlying} K={pos.K} T={max(pos.T*365,0):.1f}d Premium={pos.premium:.2f} IV={current_iv:.2f} PnL={pnl:.2f}")



    def record_log(self):
        self.log.append({
            'Day': self.day,
            'TQQQ': round(self.S_TQQQ, 2),
            'SQQQ': round(self.S_SQQQ, 2),
            'Cash': round(self.cash, 2),
            'Portfolio Value': round(self.portfolio_value(), 2),
            'Positions': len(self.positions)
        })

    def export_log(self):
        return pd.DataFrame(self.log)
    
    def grouped_trade_summary(self):
        if len(self.trades) < 2:
            print("交易紀錄不足，無法配對。")
            return

        paired = []
        for i in range(0, len(self.trades) - 1, 2):
            t1 = self.trades[i]
            t2 = self.trades[i + 1]

            pnl1 = t1['PnL']
            pnl2 = t2['PnL']
            total = pnl1 + pnl2

            paired.append({
                "Pair": f"{i}/{i+1}",
                "Day": f"{t1['Day']}/{t2['Day']}",
                "Underlying": f"{t1['Underlying']}/{t2['Underlying']}",
                "Type": f"{t1['Option Type']}/{t2['Option Type']}",
                "Strike": f"{t1['Strike']}/{t2['Strike']}",
                "Direction": f"{t1['Direction']}/{t2['Direction']}",
                "Premium": f"{t1['Premium']}/{t2['Premium']}",
                "Close Price": f"{t1['Close Price']}/{t2['Close Price']}",
                "PnL1": pnl1,
                "PnL2": pnl2,
                "Total PnL": total
            })

        df = pd.DataFrame(paired)
        desktop_path = r"C:\Users\sofjk\Desktop\Hedge\option_hedge"
        out_path = os.path.join(desktop_path, "option_pair_summary.csv")
        df.to_csv(out_path, index=False)
        print(f"✅ 已輸出交易配對損益至桌面: {out_path}")
        
        

    def summary(self):
        total_return = self.portfolio_value() - 10000
        return_rate = total_return / 10000 * 100
        pnl_list = [t['PnL'] for t in self.trades]
        wins = [p for p in pnl_list if p > 0]
        losses = [p for p in pnl_list if p < 0]
        win_rate = len(wins) / len(pnl_list) * 100 if pnl_list else 0
        avg_win = np.mean(wins) if wins else 0
        avg_loss = np.mean(losses) if losses else 0
        max_win = np.max(pnl_list) if pnl_list else 0
        max_loss = np.min(pnl_list) if pnl_list else 0
        duration_year = self.day / 365
        cagr = ((self.portfolio_value() / 10000) ** (1 / duration_year) - 1) * 100 if duration_year > 0 else 0

        print("\n===== 總結分析 =====")
        print(f"初始資金: 10000.00")
        print(f"結束資產: {self.portfolio_value():.2f}")
        print(f"總損益: {total_return:.2f} ({return_rate:.2f}%)")
        print(f"年化報酬率 (CAGR): {cagr:.2f}%")
        print(f"勝率: {win_rate:.2f}%")
        print(f"平均獲利: {avg_win:.2f}")
        print(f"平均虧損: {avg_loss:.2f}")
        print(f"最大獲利: {max_win:.2f}")
        print(f"最大虧損: {max_loss:.2f}")
        print("==================\n")
        
        self.grouped_trade_summary()  # 自動輸出兩倉配對損益
        # df = pd.DataFrame(self.trades)
        # desktop_path = r"C:\Users\sofjk\Desktop\Hedge\option_hedge"
        # out_path = os.path.join(desktop_path, "option_trades_summary.csv")
        # df.to_csv(out_path, index=False)
        # print(f"✅ 已輸出交易記錄至桌面: {out_path}")
    

    def interactive_mode(self):
        while True:
            print(f"\n=== 第 {self.day} 天 ===")
            print("1. 買 Call")
            print("2. 買 Put")
            print("3. 賣 Call")
            print("4. 賣 Put")
            print("5. 下一天")
            print("6. 顯示狀態")
            print("7. 平倉")
            print("8. 結束模擬")
            print("9. 一鍵平倉")
            print("10. 自建 T/S Long Call")
            print("11. 自建 T/S Sell Call")
            print("12. 自建 T/S Long Put")
            print("13. 自建 T/S Sell Put")
            print("14. 自建 long vertical call")
            print("15. 自建 long vertical put")
            print("16. 自建 Diagonal Call")
            print("17. 自建 Diagonal Put")
            choice = input("請輸入選項 (1-17)：")
            if choice in ['1', '2', '3', '4']:
                u = input("標的 (TQQQ/SQQQ)：").upper()
                if u == 'TQQQ':
                    print(f"目前 TQQQ 現價約 {self.S_TQQQ:.2f}")
                elif u == 'SQQQ':
                    print(f"目前 SQQQ 現價約 {self.S_SQQQ:.2f}")
                else:
                    print("標的輸入錯誤，請輸入 TQQQ 或 SQQQ")
                    continue
                k = float(input("行權價："))
                d = int(input("天數："))
                iv = float(input("IV (例如0.85)："))
                if choice == '1':
                    self.buy_option(u, k, 'call', d, iv)
                elif choice == '2':
                    self.buy_option(u, k, 'put', d, iv)
                elif choice == '3':
                    self.sell_option(u, k, 'call', d, iv)
                elif choice == '4':
                    self.sell_option(u, k, 'put', d, iv)
            elif choice == '5':
                self.next_day()
                self.show_status()
            elif choice == '6':
                self.show_status()
            elif choice == '7':
                self.show_status()
                idx = int(input("請輸入要平倉的持倉編號："))
                self.close_position(idx)
            elif choice == '8':
                print("模擬結束！")
                self.summary()
                break
            elif choice == '9':
                for i in reversed(range(len(self.positions))):
                    self.close_position(i)
            elif choice == '10':
                iv1 = float(input("輸入TQQQ IV: "))
                iv2 = float(input("輸入SQQQ IV: "))
                days = int(input("輸入天數: "))
                rng = float(input("輸入範圍百分比(例如0.1): "))
                lot1 = int(input("TQQQ輸入張數: "))
                lot2 = int(input("SQQQ輸入張數: "))
                self.auto_build_long_call(iv1, iv2, days,rng,lot1,lot2)
            elif choice == '11':
                iv1 = float(input("輸入TQQQ IV: "))
                iv2 = float(input("輸入SQQQ IV: "))
                days = int(input("輸入天數: "))
                rng = float(input("輸入範圍百分比(例如0.1): "))
                lot1 = int(input("TQQQ輸入張數: "))
                lot2 = int(input("SQQQ輸入張數: "))
                self.auto_build_sell_call(iv1, iv2, days, rng,lot1,lot2)
            elif choice == '12':
                iv1 = float(input("輸入TQQQ IV: "))
                iv2 = float(input("輸入SQQQ IV: "))
                days = int(input("輸入天數: "))
                rng = float(input("輸入範圍百分比(例如0.1): "))
                lot1 = int(input("TQQQ輸入張數: "))
                lot2 = int(input("SQQQ輸入張數: "))
                self.auto_build_long_put(iv1, iv2, days, rng,lot1,lot2)
            elif choice == '13':
                iv1 = float(input("輸入TQQQ IV: "))
                iv2 = float(input("輸入SQQQ IV: "))
                days = int(input("輸入天數: "))
                rng = float(input("輸入範圍百分比(例如0.1): "))
                lot1 = int(input("TQQQ輸入張數: "))
                lot2 = int(input("SQQQ輸入張數: "))
                self.auto_build_sell_put(iv1, iv2, days, rng,lot1,lot2)
            elif choice == '14':
                u = "TQQQ"
                days = int(input("輸入天數: "))
                self.long_vertical_call(
                    underlying=u,
                    K_buy=int(self.S_TQQQ * (1 + 0.02)),
                    K_sell=int(self.S_TQQQ * (1 + 0.04)),
                    days_to_expire=days,
                )
            elif choice == '15':
                u = "TQQQ"
                days = int(input("輸入天數: "))
                self.long_vertical_put(
                    underlying=u,
                    K_buy=int(self.S_TQQQ * (1 - 0.02)),
                    K_sell=int(self.S_TQQQ * (1 - 0.04)),
                    days_to_expire=days,
                    ) 

            elif choice == '16':
                u = "TQQQ"
                print("--- Long Call ---")
                long_d = int(input("天數（到期日）："))
                # long_k = float(input("履約價："))
                # long_iv = float(input("IV："))
                print("--- Short Call ---")
                short_d = int(input("天數（到期日）："))
                # short_k = float(input("履約價："))
                # short_iv = float(input("IV："))
                self.build_diagonal_call(long_d, int(self.S_TQQQ * (1 + 0.08)), short_d, int(self.S_TQQQ * (1 + 0.04)), u)

                

            elif choice == '17':
                u = "TQQQ"
                print("--- Long Put ---")
                long_d = int(input("天數（到期日）："))
                # long_k = float(input("履約價："))
                #long_iv = float(input("IV："))
                print("--- Short Put ---")
                short_d = int(input("天數（到期日）："))
                # short_k = float(input("履約價："))
                #short_iv = float(input("IV："))
                self.build_diagonal_put(long_d,int(self.S_TQQQ*(1-0.08)) ,short_d ,int(self.S_TQQQ*(1-0.04)), u)
                
            elif choice == '18':
                u = input("標的 (TQQQ/SQQQ)：").upper()
                print("--- Call 部分 ---")
                long_d_call = int(input("Long Call 到期日："))
                long_k_call = float(input("Long Call 履約價："))
                long_iv_call = float(input("Long Call IV："))
                short_d_call = int(input("Short Call 到期日："))
                short_k_call = float(input("Short Call 履約價："))
                short_iv_call = float(input("Short Call IV："))
                print("--- Put 部分 ---")
                long_d_put = int(input("Long Put 到期日："))
                long_k_put = float(input("Long Put 履約價："))
                long_iv_put = float(input("Long Put IV："))
                short_d_put = int(input("Short Put 到期日："))
                short_k_put = float(input("Short Put 履約價："))
                short_iv_put = float(input("Short Put IV："))
                self.build_diagonal_combo(u, long_d_call, long_k_call, long_iv_call,
                         short_d_call, short_k_call, short_iv_call,
                         long_d_put, long_k_put, long_iv_put,
                         short_d_put, short_k_put, short_iv_put)
            else:
                print("無效的選項，請重新輸入。")

# 啟動模擬器（請勿刪除）
sim = OptionSimulator(initial_cash=10000, S0_TQQQ=135, 
                      S0_SQQQ=28.5, sigma_annual=0.57, r=0.045,base_iv=0.44)
sim.interactive_mode()
# price_path = [sim.S_TQQQ]
# plt.plot(sim.iv_path)
# plt.title("IV Path Over Time")
# plt.xlabel("Day")
# plt.ylabel("Base IV")
# plt.grid(True)
# plt.show()