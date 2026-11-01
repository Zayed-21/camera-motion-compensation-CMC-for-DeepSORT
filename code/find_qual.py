import numpy as np
from scipy.optimize import linear_sum_assignment
S="MOT17-13"; W="work/trackers/MOT17-train/"
gt=np.loadtxt(f"data/MOT17/train/{S}/gt/gt.txt",delimiter=","); gt=gt[(gt[:,6]==1)&(gt[:,7]==1)]
def iou(a,b):
    x1=np.maximum(a[:,None,0],b[None,:,0]); y1=np.maximum(a[:,None,1],b[None,:,1])
    x2=np.minimum(a[:,None,0]+a[:,None,2],b[None,:,0]+b[None,:,2]); y2=np.minimum(a[:,None,1]+a[:,None,3],b[None,:,1]+b[None,:,3])
    i=np.clip(x2-x1,0,None)*np.clip(y2-y1,0,None); return i/(a[:,None,2]*a[:,None,3]+b[None,:,2]*b[None,:,3]-i)
def assign(t):
    tr=np.loadtxt(W+t+f"/data/{S}.txt",delimiter=","); m={}
    for f in np.unique(gt[:,0]):
        g=gt[gt[:,0]==f]; r=tr[tr[:,0]==f]
        if not len(r): continue
        c=iou(g[:,2:6],r[:,2:6]); a,b=linear_sum_assignment(-c)
        for i,j in zip(a,b):
            if c[i,j]>0.5: m[(int(g[i,1]),int(f))]=int(r[j,1])
    return m
B,C=assign("base"),assign("cmc_s0")
for gid in np.unique(gt[:,1]).astype(int):
    fr=sorted(f for (g,f) in B if g==gid)
    for k in range(1,len(fr)):
        f0,f1=fr[k-1],fr[k]
        if B[(gid,f0)]!=B[(gid,f1)] and f1-f0<=3:
            win=[f for f in range(f0-15,f1+16) if (gid,f) in C]
            if len(win)>25 and len({C[(gid,f)] for f in win})==1:
                h=gt[(gt[:,1]==gid)&(gt[:,0]==f1)][0,5]
                print(f"gt {gid} frame {f1} base {B[(gid,f0)]}->{B[(gid,f1)]} cmc {C[(gid,f1)]} h={h:.0f}")
