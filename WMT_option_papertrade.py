import numpy as np
import pandas as pd
import math
from scipy.stats import norm
import os
from pathlib import Path

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
def update_iv(current_iv, daily_return):
    noise = np.random.normal(0, 0.01)

    if daily_return <= -0.029:
        d_iv = +0.10 + noise  # 恐慌急跌
    elif daily_return <= -0.01:
        d_iv = +0.03 + noise  # 下跌情境
    elif daily_return >= 0.029:
        d_iv = 0.05 + noise  # 利多爆漲
    elif daily_return >= 0.01:
        d_iv = -0.03 + noise  # 穩健上漲，IV 緩慢滑落
    else:
        d_iv = noise  # 側盤時隨機

    return max(current_iv + d_iv, 0.01)
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
    def __init__(self, underlying, option_type, direction, K, premium, T, iv):
        self.underlying = underlying
        self.option_type = option_type
        self.direction = direction
        self.K = K
        self.premium = premium
        self.T = T
        self.iv = iv
        

    def update_T(self):
        self.T -= 1/365

    def value(self, S, r, current_iv=None):
        iv_used = current_iv if current_iv is not None else self.iv
        price = bs_price(S, self.K, max(self.T, 0), r, iv_used, self.option_type)
        return price
    
    def update_iv(self, daily_return):
        self.iv = update_iv(self.iv, daily_return)
# 模擬器類別
class OptionSimulator:
    def __init__(self, initial_cash, S0_TQQQ, S0_SQQQ, sigma_annual, r):
        
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.S_TQQQ = S0_TQQQ
        self.S_SQQQ = S0_SQQQ
        self.sigma_annual = sigma_annual
        self.sigma_daily = sigma_annual / np.sqrt(252)
        self.r = r
        self.positions = []
        self.day = 0
        self.log = []
        self.prev_S_TQQQ = S0_TQQQ
        self.prev_S_SQQQ = S0_SQQQ
        self.iv_tqqq = 0.85
        self.iv_sqqq = 0.85
        self.trades = []

    def buy_option(self, underlying, K, option_type, days_to_expire, iv):
        S = self.S_TQQQ if underlying == 'TQQQ' else self.S_SQQQ
        T = days_to_expire / 365
        premium = bs_price(S, K, T, self.r, iv, option_type)
        self.cash -= premium * 100
        self.positions.append(Position(underlying, option_type, 'long', K, premium, T, iv))

    def sell_option(self, underlying, K, option_type, days_to_expire, iv):
        S = self.S_TQQQ if underlying == 'TQQQ' else self.S_SQQQ
        T = days_to_expire / 365
        premium = bs_price(S, K, T, self.r, iv, option_type)
        self.cash += premium * 100
        self.positions.append(Position(underlying, option_type, 'short', K, premium, T, iv))
    
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

    def auto_build_long_vertical_call(self, underlying, iv, days_to_expire):
        S = self.S_TQQQ if underlying == 'TQQQ' else self.S_SQQQ
        print(f"目前 {underlying} 現價約 {S:.2f}")
        K_sell = int(input(f"請輸入要賣出的 Call Strike："))
        K_buy = K_sell - 1
        if K_buy <= 0:
            print("錯誤：買入 Strike 不可為負")
            return
        print(f"→ 建立 {underlying} Long Vertical Call：BUY {K_buy} / SELL {K_sell}")
        self.buy_option(underlying, K_buy, 'call', days_to_expire, iv)
        self.sell_option(underlying, K_sell, 'call', days_to_expire, iv)

    def auto_build_long_vertical_put(self, underlying, iv, days_to_expire):
        S = self.S_TQQQ if underlying == 'TQQQ' else self.S_SQQQ
        print(f"目前 {underlying} 現價約 {S:.2f}")
        K_sell = int(input(f"請輸入要賣出的 Put Strike："))
        K_buy = K_sell + 1
        if K_buy <= 0:
            print("錯誤：買入 Strike 不可為負")
            return
        print(f"→ 建立 {underlying} Long Vertical Put：BUY {K_buy} / SELL {K_sell}")
        self.buy_option(underlying, K_buy, 'put', days_to_expire, iv)
        self.sell_option(underlying, K_sell, 'put', days_to_expire, iv)

    def build_diagonal_call(sim, long_days, long_strike, long_iv, short_days, short_strike, short_iv, underlying='TQQQ'):
        sim.buy_option(underlying, long_strike, 'call', long_days, long_iv)
        sim.sell_option(underlying, short_strike, 'call', short_days, short_iv)

    def build_diagonal_put(sim, long_days, long_strike, long_iv, short_days, short_strike, short_iv, underlying='TQQQ'):
        sim.buy_option(underlying, long_strike, 'put', long_days, long_iv)
        sim.sell_option(underlying, short_strike, 'put', short_days, short_iv)

    def build_diagonal_combo(sim, u, long_d_call, long_k_call, long_iv_call,
                          short_d_call, short_k_call, short_iv_call,
                          long_d_put, long_k_put,long_iv_put,
                          short_d_put, short_k_put, short_iv_put):
        sim.build_diagonal_call(long_d_call, long_k_call, long_iv_call, short_d_call, short_k_call, short_iv_call, u)
        sim.build_diagonal_put(long_d_put, long_k_put, short_d_put, short_k_put,u, long_iv_put, short_iv_put)


    def close_position(self, idx):
        if idx < 0 or idx >= len(self.positions):
            print("無效的持倉編號！")
            return
        pos = self.positions[idx]
        S = self.S_TQQQ if pos.underlying == 'TQQQ' else self.S_SQQQ
        price = pos.value(S, self.r)
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
        if np.random.rand() < 0.005: #"0.1、0.2、0.0003、0.6... 隨機跳，直到跳到小於 0.02 為止  綜合下來只有2%  0.02機率跳中"
            regime = 'HIGH'
            sigma = self.sigma_daily * 1.3
        else:
            regime = 'NORMAL'
            sigma = self.sigma_daily*0.9

        mu_daily = 0.19 / 252  # 假設市場平均年化報酬率為 X%
        market_move = mu_daily + np.random.standard_t(df=20) * sigma  #T分布
        self.prev_S_TQQQ = self.S_TQQQ
        self.prev_S_SQQQ = self.S_SQQQ

        self.S_TQQQ *= (1 + 1 * market_move)
        self.S_SQQQ *= (1 - 1 * market_move)

        # 模擬 IV 根據資產報酬調整
        daily_return_tqqq = (self.S_TQQQ - self.prev_S_TQQQ) / self.prev_S_TQQQ
        daily_return_sqqq = (self.S_SQQQ - self.prev_S_SQQQ) / self.prev_S_SQQQ
        self.iv_tqqq = update_iv(self.iv_tqqq, daily_return_tqqq)
        self.iv_sqqq = update_iv(self.iv_sqqq, daily_return_sqqq)

        for pos in self.positions:
            pos.update_T()
            if pos.underlying == 'TQQQ':
                pos.update_iv(daily_return_tqqq)
            else:
                pos.update_iv(daily_return_sqqq)
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
            market_price = pos.value(S, self.r, current_iv=pos.iv)  # 用倉位建立時的 IV
            entry_price = pos.premium
            if pos.direction == 'long':
                value += market_price * 100  # long 倉現值
            else:
                value -= market_price * 100   # short 倉為已收保費 + 浮損益
        return value

    def show_status(self):
        if self.day == 0:
            chg_T = chg_S = 0
        else:
            chg_T = (self.S_TQQQ - self.prev_S_TQQQ) / self.prev_S_TQQQ * 100
            chg_S = (self.S_SQQQ - self.prev_S_SQQQ) / self.prev_S_SQQQ * 100
        print(f"Day {self.day}, TQQQ: {self.S_TQQQ:.2f} ({chg_T:+.2f}%), SQQQ: {self.S_SQQQ:.2f} ({chg_S:+.2f}%), Portfolio Value: {self.portfolio_value():.2f}")
    
        for idx, pos in enumerate(self.positions):
            S = self.S_TQQQ if pos.underlying == 'TQQQ' else self.S_SQQQ
            current_price = pos.value(S, self.r)
            current_iv = pos.iv
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
        print(df)
        

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
        duration_year = self.day / 252
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
            print("14. 自建 Long Vertical Call")
            print("15. 自建 Long Vertical Put")
            print("16. 自建 Diagonal Call")
            print("17. 自建 Diagonal Put")
            print("18. 自建 Diagonal Combo")
            choice = input("請輸入選項 (1-18)：")
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
                u = input("標的 (TQQQ/SQQQ)：").upper()
                iv = float(input("輸入 IV: "))
                days = int(input("輸入天數: "))
                self.auto_build_long_vertical_call(u, iv, days)
            elif choice == '15':
                u = input("標的 (TQQQ/SQQQ)：").upper()
                iv = float(input("輸入 IV: "))
                days = int(input("輸入天數: "))
                self.auto_build_long_vertical_put(u, iv, days)   

            elif choice == '16':
                u = input("標的 (TQQQ/SQQQ)：").upper()
                print("--- Long Call ---")
                long_d = int(input("天數（到期日）："))
                # long_k = float(input("履約價："))
                # long_iv = float(input("IV："))
                print("--- Short Call ---")
                short_d = int(input("天數（到期日）："))
                # short_k = float(input("履約價："))
                # short_iv = float(input("IV："))
                self.build_diagonal_call(long_d, int(self.S_TQQQ*(1+0.04)), 0.36, short_d, int(self.S_TQQQ*(1+0.02)), 0.45, u)
                # self.build_diagonal_call(long_d, long_k, long_iv, short_d, short_k, short_iv, u)

            elif choice == '17':
                u = input("標的 (TQQQ/SQQQ)：").upper()
                print("--- Long Put ---")
                long_d = int(input("天數（到期日）："))
                # long_k = float(input("履約價："))
                #long_iv = float(input("IV："))
                print("--- Short Put ---")
                short_d = int(input("天數（到期日）："))
                # short_k = float(input("履約價："))
                #short_iv = float(input("IV："))
                self.build_diagonal_put(long_d,int(self.S_TQQQ*(1-0.04)) ,0.4 ,short_d ,int(self.S_TQQQ*(1-0.02)) ,0.48, u)
                # self.build_diagonal_put(long_d, long_k, long_iv, short_d, short_k, short_iv, u)
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
sim = OptionSimulator(initial_cash=10000, S0_TQQQ=98.5, S0_SQQQ=28.5, sigma_annual=0.4, r=0.045)
sim.interactive_mode()

