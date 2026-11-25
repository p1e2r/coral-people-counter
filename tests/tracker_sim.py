# Simulated counting-accuracy test for the tracker. Run: PYTHONPATH=app python3 tests/tracker_sim.py
import random, sys, tracker as T
orig=T.Tracker._hit
def hit(self,t,box,score): t.det=box; orig(self,t,box,score)
T.Tracker._hit=hit
def run(seed, drop=0.0, jitter=0.0, occl=False, fps=15, verbose=False):
    random.seed(seed); ppl=[]; t=0
    for i in range(200):
        t+=random.expovariate(1/(fps*1.2)); ltr=random.random()<.5; sp=random.uniform(.15,.45)/fps
        ppl.append(dict(i=i,start=int(t),x=-.1 if ltr else 1.1,v=sp if ltr else -sp,y=random.uniform(.3,.7),h=random.uniform(.18,.4),hes=random.random()<.05,turned=False,done=False,counted=[]))
    tr=T.Tracker(max_missed=int(fps*1.5)); owners={}
    def ev(d,tk):
        o=owners.get(tk.id,{})
        g=max(o,key=o.get) if o else None
        if g is not None: ppl[g]["counted"].append(d)
    ct=T.Counter(ev); ct.configure({"line":[0.5,0,0.5,1]})
    for f in range(int(t)+150):
        dets=[];gid={}
        for p in ppl:
            if f<p["start"] or p["done"]: continue
            p["x"]+=p["v"]
            if p["hes"] and not p["turned"] and ((p["v"]>0 and p["x"]>.56) or (p["v"]<0 and p["x"]<.44)): p["turned"]=True; p["v0"]=p["v"]; p["k"]=0
            if p.get("v0") and p["k"]<12: p["k"]+=1; p["v"]=p["v0"]*(1-p["k"]/6)  # slow down, stop, walk back (~0.8 s)
            if not -.15<p["x"]<1.15: p["done"]=True; continue
            w=p["h"]*.35; box=(p["x"]-w/2+random.gauss(0,jitter),p["y"]-p["h"]/2+random.gauss(0,jitter),p["x"]+w/2+random.gauss(0,jitter),p["y"]+p["h"]/2+random.gauss(0,jitter))
            vis=(min(box[2],1)-max(box[0],0))/(box[2]-box[0])
            if vis<0.4 or random.random()<drop: continue
            box=(max(box[0],0),box[1],min(box[2],1),box[3])  # detector sees the visible part
            sc=random.uniform(.5,.9)
            if occl:
                for q in ppl:
                    if q is not p and not q["done"] and f>=q["start"] and abs(q["x"]-p["x"])<w*.6 and q["y"]>p["y"]: sc=random.uniform(.2,.4)
            dets.append((box,sc)); gid[box]=p["i"]
        for tk in tr.update(dets,.45):
            if tk.missed==0 and getattr(tk,'det',None) in gid:
                owners.setdefault(tk.id,{}); owners[tk.id][gid[tk.det]]=owners[tk.id].get(gid[tk.det],0)+1
        ct.process(tr.tracks, now=f/fps)
    # per-person outcome: net count (in/out minus undo) should be 1 (0 for hesitators)
    def net(p): return len([c for c in p["counted"] if c!="undo"]) - p["counted"].count("undo")
    ok=sum(1 for p in ppl if net(p)==(0 if p["hes"] else 1))
    missed=[p for p in ppl if not p["hes"] and net(p)<1]; extra=[p for p in ppl if net(p)>(0 if p["hes"] else 1)]
    multi=sum(1 for o in owners.values() if len(o)>1)
    total_truth=sum(0 if p["hes"] else 1 for p in ppl); total=sum(net(p) for p in ppl)
    if verbose:
        for p in missed[:8]: print("  missed",p["i"],"hes",p["hes"],"counted",p["counted"], "h=%.2f y=%.2f v=%.3f"%(p["h"],p["y"],p["v"]))
    return total_truth,total,len(missed),len(extra),multi
if '--check' not in sys.argv:
  for kw in [dict(),dict(jitter=.006,drop=.15),dict(jitter=.006,drop=.15,occl=True)]:
      r=[run(s,**kw) for s in range(1,6)]
      tt=sum(x[0] for x in r); tc=sum(x[1] for x in r)
      print(kw, f"truth={tt} counted={tc} err={100*(tc-tt)/tt:+.1f}% missed={sum(x[2] for x in r)} extra={sum(x[3] for x in r)} idmix={sum(x[4] for x in r)}")


if __name__ == "__main__" and "--check" in sys.argv:
    # CI gate: overall count error must stay within 5% in every condition
    worst = 0.0
    for kw in [dict(), dict(jitter=.006, drop=.15), dict(jitter=.006, drop=.15, occl=True)]:
        r = [run(s, **kw) for s in range(1, 6)]
        tt, tc = sum(x[0] for x in r), sum(x[1] for x in r)
        worst = max(worst, abs(100 * (tc - tt) / tt))
    print(f"worst total error: {worst:.1f}%")
    sys.exit(1 if worst > 5.0 else 0)
