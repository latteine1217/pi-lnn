"""Cross-Re sensor-budget figure (thesis Fig. 4.12).

What: KE and vorticity relative error against sensor count K at five Reynolds numbers.
Why:  drawn 1:1 at the thesis \\linewidth (426.79 pt = 5.91 in) so LaTeX does not rescale
      it and shrink the type below the 80%-of-body-text rule.

Numbers are the uniform-128^2 evaluation (EXP-320..328, 330..335; slurm job 4664),
recorded in docs/experiment_log_v2.md section 1.4 Stage 3.
"""
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter, NullLocator
OUT = Path(__file__).resolve().parents[1] / "thesis" / "figures" / "results"

K=np.array([10,50,100])
KE={100:[8.71,1.91,0.73],500:[42.54,5.07,1.26],1000:[61.81,6.46,2.59],
    10000:[72.37,11.29,4.56],1000000:[57.93,18.71,13.50]}
OM={100:[8.67,2.26,1.65],500:[54.54,11.23,4.71],1000:[74.11,18.95,11.13],
    10000:[92.82,55.42,43.45],1000000:[89.36,67.63,62.60]}
ORDER=[100,500,1000,10000,1000000]
sty={100:("#E69F00","^","-"),500:("#009E73","s","--"),1000:("#0072B2","o","-."),
     10000:("#CC79A7","D",(0,(4,1,1,1))),1000000:("#D55E00","v",(0,(5,1.4)))}
lb={100:r"$Re=10^{2}$",500:r"$Re=5\times10^{2}$",1000:r"$Re=10^{3}$",
    10000:r"$Re=10^{4}$",1000000:r"$Re=10^{6}$"}

plt.rcParams.update({
    "font.family":"sans-serif","font.size":10,
    "axes.linewidth":0.8,"axes.labelsize":10,
    "xtick.labelsize":9.5,"ytick.labelsize":9.5,
    "xtick.direction":"in","ytick.direction":"in",
    "xtick.major.size":3.5,"ytick.major.size":3.5,
    "xtick.major.width":0.8,"ytick.major.width":0.8,
    "legend.fontsize":9.5,
})

W=5.90                                    # = \linewidth, 1:1 (no LaTeX rescaling)
fig,axes=plt.subplots(1,2,figsize=(W,3.75),sharey=True)
fig.subplots_adjust(left=0.105,right=0.988,top=0.855,bottom=0.245,wspace=0.06)

for ax,data,tag,name in [(axes[0],KE,"a","Kinetic energy"),(axes[1],OM,"b","Vorticity")]:
    for re in ORDER:
        c,m,ls=sty[re]
        ax.plot(K,data[re],marker=m,color=c,lw=1.5,ms=5.6,ls=ls,
                markerfacecolor="white",markeredgewidth=1.4,
                label=lb[re],clip_on=False,zorder=3)
    ax.axhline(10,color="#555",ls=":",lw=1.0,zorder=1)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(8.6,118); ax.set_ylim(0.55,130)
    ax.set_xticks(K); ax.set_xticklabels(["10","50","100"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.grid(True,which="major",alpha=0.18,lw=0.5)
    for s in ("top","right"): ax.spines[s].set_visible(False)
    ax.set_title(f"({tag}) {name}",fontsize=10,pad=5)

axes[0].set_yticks([1,2,5,10,20,50,100])
axes[0].set_yticklabels(["1","2","5","10","20","50","100"])
axes[0].set_ylabel("Relative error (%)")

fig.text(0.548,0.115,"Sensor count  $K$ (dimensionless)",ha="center",fontsize=10)
fig.suptitle("Sensor budget versus Reynolds number",fontsize=10.5,y=0.982)
h,l=axes[0].get_legend_handles_labels()
fig.legend(h,l,loc="lower center",bbox_to_anchor=(0.548,-0.008),ncol=5,frameon=False,
           handlelength=2.3,columnspacing=1.15,handletextpad=0.45)
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "crossre_sensor_budget.png", dpi=400, bbox_inches="tight", pad_inches=0.015)
fig.savefig(OUT / "crossre_sensor_budget.pdf", bbox_inches="tight", pad_inches=0.015)
print(f"saved -> {OUT}/crossre_sensor_budget.{{pdf,png}}")
