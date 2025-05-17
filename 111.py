import matplotlib.pyplot as plt
from IPython.display import clear_output
import time

price_path = [100]  # 初始價格
sim = OptionSimulator(...)  # 你的模擬器

for _ in range(100):  # 模擬 100 天
    sim.next_day()
    price_path.append(sim.S_TQQQ)

    # 動態畫圖
    clear_output(wait=True)
    plt.figure(figsize=(8, 4))
    plt.plot(price_path, marker='o')
    plt.title(f"Day {sim.day} - Price Path")
    plt.xlabel("Day")
    plt.ylabel("Price")
    plt.grid(True)
    plt.show()

    time.sleep(0.2)  # 每步延遲 0.2 秒（可調整速度）
