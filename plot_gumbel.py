import numpy as np
from scipy.stats import gumbel_r
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style()

x = np.linspace(
    gumbel_r.ppf(0.001),
    gumbel_r.ppf(0.999),
    1024
)

fig = plt.figure()
ax = plt.gca()
plt.plot(x, gumbel_r.pdf(x), label="Gumbel distribution\n(σ=0, β=1)")
ax.set_ylabel("pdf(x)")
ax.set_xlabel("x")
ax.legend()
plt.savefig("gumbel_pdf.pdf")