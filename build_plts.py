import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("rng_speed.csv")


plt.figure(figsize=(9, 5))

plt.plot(df["N"], df["LCG(ns)"],    marker="o", label="LcgPerm32 (mini-PCG)")
plt.plot(df["N"], df["XSW(ns)"],    marker="o", label="Xsw32 (XORShift64+Weyl)")
plt.plot(df["N"], df["Salted(ns)"], marker="o", label="SaltedCb32 (SplitMix-salted)")
plt.plot(df["N"], df["mt19937(ns)"],marker="o", label="std::mt19937")

plt.yscale("log")
plt.xlabel("Размер выборки N (лог. масштаб)")
plt.ylabel("Время генерации, ns")
plt.title("Скорость генерации случайных чисел")
plt.grid(True, which="both", ls="--", alpha=0.4)
plt.legend(title="Генераторы", loc="upper left")
sizes = df["N"].tolist()
plt.xticks(
    sizes,
    [f"{n:,}" for n in sizes],
    rotation=90
)
plt.tight_layout()


plt.savefig('rng_speed_plot.png', dpi=150)