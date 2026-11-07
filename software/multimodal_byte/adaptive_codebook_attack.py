"""Genuine codebook-aware (Kerckhoffs) adaptive attack on Bits vs Standard.
The attacker KNOWS the codebook C, the FFN, the body/head, and the seed (unkeyed). It runs a DLG
gradient-inversion: optimize a soft token distribution A (L x V) so the induced gradient matches the
observed private gradient. For Bits the dummy is routed THROUGH the real FFN: emb = FFN(A @ C) (exact
at one-hot), so it must match the SHARED-FFN gradient (dense/entangled) rather than per-token rows.
Also runs the KEYED control (attacker uses a WRONG codebook). Metric = token-SET recovery |pred∩true|/L.
Usage: python adaptive_codebook_attack.py --arm standard|bits|bits_keyed --seqlens 1,4,8,16 --trials 30"""
import argparse, json, torch, torch.nn as nn, torch.nn.functional as F, numpy as np
DEV='cuda' if torch.cuda.is_available() else 'cpu'

class Model(nn.Module):
    def __init__(s, arm, V, d, nbits, W, nc, seed):
        super().__init__(); s.arm=arm; s.V=V
        if arm=='standard':
            s.E=nn.Parameter(torch.randn(V,d)*0.05)
        else:
            g=torch.Generator().manual_seed(seed)
            s.register_buffer('C',(torch.rand(V,nbits,generator=g)<0.5).float()*2-1)
            s.ffn=nn.Sequential(nn.Linear(nbits,W),nn.GELU(),nn.Linear(W,d))
        s.head=nn.Linear(d,nc)
    def emb_ids(s,ids):
        return s.E[ids] if s.arm=='standard' else s.ffn(s.C[ids])
    def emb_soft(s,A,Cknown=None):
        if s.arm=='standard': return (A@s.E).unsqueeze(0)
        C = s.C if Cknown is None else Cknown
        return s.ffn(A@C).unsqueeze(0)
    def logits_from_emb(s,emb): return s.head(emb.mean(1))

def real_grad(m,ids,label):
    m.zero_grad()
    loss=F.cross_entropy(m.logits_from_emb(m.emb_ids(ids)),label)
    g=torch.autograd.grad(loss,[p for p in m.parameters()],create_graph=False)
    return [x.detach() for x in g]

def dlg_attack(m,g_star,label,L,Cknown=None,iters=600,lr=0.1):
    Z=torch.randn(L,m.V,device=DEV,requires_grad=True)
    opt=torch.optim.Adam([Z],lr=lr)
    params=[p for p in m.parameters()]
    for _ in range(iters):
        opt.zero_grad()
        A=F.softmax(Z,dim=-1)
        emb=m.emb_soft(A,Cknown)
        loss=F.cross_entropy(m.logits_from_emb(emb),label)
        g=torch.autograd.grad(loss,params,create_graph=True)
        match=sum(((a-b)**2).sum() for a,b in zip(g,g_star))
        match.backward(); opt.step()
    return F.softmax(Z.detach(),dim=-1)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--arm',required=True,choices=['standard','bits','bits_keyed'])
    ap.add_argument('--seqlens',default='1,4,8,16'); ap.add_argument('--trials',type=int,default=30)
    ap.add_argument('--V',type=int,default=1000); ap.add_argument('--d',type=int,default=64)
    ap.add_argument('--nbits',type=int,default=128); ap.add_argument('--W',type=int,default=256)
    ap.add_argument('--nc',type=int,default=10); ap.add_argument('--iters',type=int,default=600)
    ap.add_argument('--out',default='adaptive_codebook_results.jsonl'); a=ap.parse_args()
    torch.manual_seed(0); np.random.seed(0)
    base_arm='bits' if a.arm.startswith('bits') else 'standard'
    m=Model(base_arm,a.V,a.d,a.nbits,a.W,a.nc,seed=12345).to(DEV)
    # keyed control: attacker uses a DIFFERENT codebook than the model's
    Cknown=None
    if a.arm=='bits_keyed':
        g=torch.Generator().manual_seed(99999)   # wrong key
        Cknown=((torch.rand(a.V,a.nbits,generator=g)<0.5).float()*2-1).to(DEV)
    fout=open(a.out,'a')
    for L in [int(x) for x in a.seqlens.split(',')]:
        recs=[]; pos=[]
        for t in range(a.trials):
            ids=torch.randint(0,a.V,(1,L),device=DEV)
            label=torch.randint(0,a.nc,(1,),device=DEV)   # attacker granted the label (generous)
            gs=real_grad(m,ids,label)
            A=dlg_attack(m,gs,label,L,Cknown,iters=a.iters)
            predl=A.argmax(-1).tolist(); truel=ids.view(-1).tolist()
            pred=set(predl); true=set(truel)
            recs.append(len(pred&true)/len(true))
            pos.append(sum(int(p==t) for p,t in zip(predl,truel))/L)   # per-position exact match, chance 1/V
        r=float(np.mean(recs)); rp=float(np.mean(pos))
        rec=dict(arm=a.arm,seqlen=L,recovery=round(r,4),chance=round(L/a.V,5),pos_recovery=round(rp,4),pos_chance=round(1/a.V,5),trials=a.trials)
        fout.write(json.dumps(rec)+'\n'); print(f"[{a.arm}] L={L:3d}: set-recovery={r:.4f} (chance {L/a.V:.4f})  pos-recovery={rp:.4f} (chance {1/a.V:.4f})",flush=True)
    fout.close()

if __name__=='__main__': main()
