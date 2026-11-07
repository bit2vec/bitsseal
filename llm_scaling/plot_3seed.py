import re, sys, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D='s3_1B/'
seeds=[42,43,44]; TPS=32768; LOG=10
def losses(f):
    t=open(f,'rb').read().decode('utf8','ignore')
    return np.array([float(x) for x in re.findall(r"\{'loss': '([0-9.]+)',", t)])
def ppl(seed,arm):
    f=D+f'eval_s{seed}_{arm}.log'; out={}
    for m in re.finditer(r"ckpt=\S*checkpoint-(\d+) .*?ppl=([0-9.]+)", open(f,'rb').read().decode('utf8','ignore')):
        out[int(m.group(1))]=float(m.group(2))
    return out
fig,ax=plt.subplots(1,2,figsize=(8.2,3.0))
cols={'bits':'#1b7f3b','std':'#b03030'}; names={'bits':'Bits','std':'Standard table'}
for arm in ['std','bits']:
    L=np.stack([losses(D+f's{s}_{arm}.log')[:3051] for s in seeds])
    k=10; ma=np.stack([np.convolve(l,np.ones(k)/k,mode='valid') for l in L])
    x=(np.arange(ma.shape[1])+k)*LOG*TPS/1e9
    m=ma.mean(0); lo=ma.min(0); hi=ma.max(0)
    ax[0].plot(x,m,color=cols[arm],lw=1.2,label=names[arm]); ax[0].fill_between(x,lo,hi,color=cols[arm],alpha=.25,lw=0)
ax[0].set_xlabel('Training tokens (B)'); ax[0].set_ylabel('Training loss (100-step mean)'); ax[0].set_ylim(3.4,5.5); ax[0].legend(frameon=False); ax[0].set_title('7B training loss, 3 seeds (band = min to max)',fontsize=9)
steps=[3051*i for i in range(1,10)]+[30517]
for arm,key in [('std','standard'),('bits','bits')]:
    P=np.array([[ppl(s,key)[st] for st in steps] for s in seeds])
    x=np.array(steps)*TPS/1e9
    ax[1].errorbar(x,P.mean(0),yerr=P.std(0,ddof=1),color=cols[arm],marker='o',ms=3,lw=1.2,capsize=2,label=names[arm])
ax[1].set_xlabel('Training tokens (B)'); ax[1].set_ylabel('Held-out perplexity'); ax[1].set_yscale('log'); ax[1].set_yticks([40,50,60,80,100]); ax[1].set_yticklabels(['40','50','60','80','100']); ax[1].legend(frameon=False); ax[1].set_title('Held-out perplexity, mean ± std over 3 seeds',fontsize=9)
for a in ax: a.spines['top'].set_visible(False); a.spines['right'].set_visible(False)
plt.tight_layout()
for ext in ['pdf','png']: plt.savefig(f'figs/loss_curve_7b_3seed.{ext}',dpi=200)
print('ok')
