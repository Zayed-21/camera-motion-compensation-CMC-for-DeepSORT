"""Fig: MOT17-13 (real moving camera), GT person 17; box = track matched to that person (IoU-Hungarian, >0.5)."""
import numpy as np, cv2, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
exec(open("find_qual.py").read().split("for gid in")[0])   # B, C: (gt id, frame) -> track id
S="MOT17-13"; GID=17; FR=[92,118,135]; W,H=420,260
gtall=np.loadtxt(f"data/MOT17/train/{S}/gt/gt.txt",delimiter=",")
plt.rcParams.update({"font.size":8,"font.family":"serif"})
fig,ax=plt.subplots(2,3,figsize=(7.0,2.75))
for r,(t,lab,D) in enumerate((("base","Baseline",B),("cmc_s0","Estimated CMC",C))):
    T=np.loadtxt(f"work/trackers/MOT17-train/{t}/data/{S}.txt",delimiter=",")
    for c,f in enumerate(FR):
        g=gtall[(gtall[:,1]==GID)&(gtall[:,0]==f)][0,2:6]
        x0=int(np.clip(g[0]+g[2]/2-W/2,0,1920-W)); y0=int(np.clip(g[1]+g[3]/2-H/2,0,1080-H))
        img=cv2.cvtColor(cv2.imread(f"data/MOT17/train/{S}/img1/{f:06d}.jpg"),cv2.COLOR_BGR2RGB)[y0:y0+H,x0:x0+W].copy()
        tid=D[(GID,f)]; x,y,w,h=T[(T[:,0]==f)&(T[:,1]==tid)][0,2:6]
        col=(255,40,40) if tid!=36 else (255,215,0)
        cv2.rectangle(img,(int(x-x0),int(y-y0)),(int(x+w-x0),int(y+h-y0)),col,3)
        cv2.putText(img,f"ID {tid}",(int(x-x0)-10,int(y-y0)-8),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)
        ax[r,c].imshow(img); ax[r,c].set_xticks([]); ax[r,c].set_yticks([])
        if r==0: ax[r,c].set_title(f"frame {f}")
    ax[r,0].set_ylabel(lab)
fig.tight_layout(pad=0.2); fig.savefig("fig_real13.pdf",dpi=200); fig.savefig("fig_real13.png",dpi=110)
print([B[(GID,f)] for f in FR],[C[(GID,f)] for f in FR])
