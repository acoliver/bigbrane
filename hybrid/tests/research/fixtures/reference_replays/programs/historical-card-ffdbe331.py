import json, math
import numpy as np

def rho(e):
    return 2*math.atanh(math.sqrt(float(e@e)))
def dist(x,y):
    nx=float(x@x); ny=float(y@y); dd=float((x-y)@(x-y))
    return math.acosh(1+2*dd/((1-nx)*(1-ny)))
def G(x,y):
    return (rho(x)+rho(y)-dist(x,y))/2

OPS={'+','-','?'}

class Card:
    # card update[1..3] and readout[1], written as on the card
    def __init__(s,E,W,delta,lam):
        s.E=E; s.W=W; s.d=delta; s.l=lam; s.T={}
    def step(s,w,op,x):
        if w not in s.W or op not in OPS or x not in s.E:
            return ('none',None)
        E=s.E
        if op=='+':
            if w not in s.T:
                s.T[w]=[x,G(E[x],E[x]),0.0,1]
            else:
                st=s.T[w]; st[1]=min(st[1],G(E[x],E[st[0]])); st[3]+=1
            return ('write',None)
        if op=='-':
            if w in s.T:
                st=s.T[w]; st[2]=max(st[2],G(E[x],E[st[0]]))
            return ('write',None)
        if w not in s.T:
            return ('query',0.5)
        a,g,h,k=s.T[w]
        z=G(E[x],E[a]); c=s.d*k-s.l
        if g>h:
            m=max(h,min(z,g))
            if c!=0:
                P=(math.exp(c*m)-math.exp(c*h))/(math.exp(c*g)-math.exp(c*h))
            else:
                P=(m-h)/(g-h)
        else:
            P=1.0 if z>=g else 0.0
        return ('query',min(1.0,max(0.0,P)))

class Known:
    # Shepard/Tenenbaum-Griffiths size-principle learner over the continuous nested family
    # h_r = {y : s(y) >= r}, s(y) = G(y, first positive), |h_r| = exp(-delta r), prior lam*exp(-lam r).
    # Stores raw examples; posterior integrated piecewise over r from scratch (no g/h statistic, no closed form).
    def __init__(s,E,W,delta,lam):
        s.E=E; s.W=W; s.d=delta; s.l=lam; s.ex={}
    def step(s,w,op,x):
        if w not in s.W or op not in OPS or x not in s.E:
            return ('none',None)
        if op=='+':
            s.ex.setdefault(w,[]).append(('+',x)); return ('write',None)
        if op=='-':
            if w in s.ex:
                s.ex[w].append(('-',x))
            return ('write',None)
        if w not in s.ex:
            return ('query',0.5)
        return ('query',s.post(s.ex[w],x))
    def post(s,exs,y):
        E=s.E; a=exs[0][1]
        sim=lambda u: G(E[u],E[a])
        ps=[sim(u) for o,u in exs if o=='+']
        ns=[sim(u) for o,u in exs if o=='-']
        sy=sim(y)
        logprior_rate=-s.l
        loglik_rate=s.d*len(ps)  # each positive: 1[s(x)>=r]/|h_r| = exp(delta r)
        c=logprior_rate+loglik_rate
        pts=sorted(set([0.0,sy]+ps+ns))
        M=pts[-1] if c>0 else 0.0
        Z=0.0; num=0.0
        for u,v in zip(pts[:-1],pts[1:]):
            mid=0.5*(u+v)
            ok=all(mid<=p for p in ps) and all(mid>q for q in ns)
            if not ok:
                continue
            if c==0:
                wt=v-u
            else:
                wt=(math.exp(c*(v-M))-math.exp(c*(u-M)))/c
            Z+=wt
            if v<=sy:
                num+=wt
        if Z<=0:
            return None
        return num/Z

def run(E,W,toks,delta,lam,eps=1e-3):
    C=sorted(E.keys())
    V=sorted(W)+C+['+','-','?','yes','no','BOS']
    iy=V.index('yes'); ino=V.index('no')
    card=Card(E,W,delta,lam); kn=Known(E,W,delta,lam)
    buf=['BOS','BOS']+toks
    maxd=0.0; nq=0; ninc=0; Pc=[]
    for t in range(2,len(buf)):
        w,op,x=buf[t-2],buf[t-1],buf[t]
        r1=card.step(w,op,x); r2=kn.step(w,op,x)
        if r1[0]!=r2[0]:
            return None
        p1=np.full(len(V),1.0/len(V)); p2=p1.copy()
        if r1[0]=='query':
            if r2[1] is None:
                ninc+=1; Pc.append(r1[1]); continue
            nq+=1; Pc.append(r1[1])
            for p,P in ((p1,r1[1]),(p2,r2[1])):
                p[:]=eps/len(V); p[iy]+=(1-eps)*P; p[ino]+=(1-eps)*(1-P)
            maxd=max(maxd,abs(r1[1]-r2[1]))
        maxd=max(maxd,float(np.max(np.abs(p1-p2))))
    return maxd,nq,ninc,Pc

rng=np.random.default_rng(12345)
d=4; NC=30
E={}
for i in range(NC):
    v=rng.normal(size=d); v/=np.linalg.norm(v)
    E['c%d'%i]=v*rng.uniform(0.3,0.95)
Wlist=['w%d'%i for i in range(12)]
W=set(Wlist)|{'wq'}
def make_stream():
    queues={}
    for w in Wlist:
        pool=rng.choice(NC,3,replace=False)
        sch=[]
        if rng.random()<0.3:
            sch.append(('-',int(rng.integers(NC))))
        sch.append(('+',int(pool[0])))
        for _ in range(7):
            r=rng.random()
            if r<0.35: sch.append(('+',int(pool[rng.integers(3)])))
            elif r<0.5: sch.append(('-',int(rng.integers(NC))))
            else: sch.append(('?',int(rng.integers(NC))))
        queues[w]=sch
    queues['wq']=[('?',int(rng.integers(NC))) for _ in range(3)]
    toks=[]
    while any(queues.values()):
        ks=[k for k in queues if queues[k]]
        w=ks[rng.integers(len(ks))]
        op,x=queues[w].pop(0)
        toks+= [w,op,'c%d'%x]
        if op=='?': toks.append('yes')
    return toks

settings=[(float(rng.uniform(0.3,2.0)),float(rng.uniform(0.0,1.0))) for _ in range(4)]+[(1.0,0.5),(0.5,0.5)]
MAX=0.0; NQ=0; NI=0; ok=True
for (de,la) in settings:
    for rep in range(3):
        r=run(E,W,make_stream(),de,la)
        if r is None:
            ok=False; continue
        MAX=max(MAX,r[0]); NQ+=r[1]; NI+=r[2]

# card toy geometry
L3=math.log(3.0)
def pt(R,th):
    s=math.tanh(R/2); return np.array([s*math.cos(th),s*math.sin(th)])
def ang(R1,R2,q):
    dd=R1+R2-2*q
    return math.acos((math.cosh(R1)*math.cosh(R2)-math.cosh(dd))/(math.sinh(R1)*math.sinh(R2)))
ET={'dalmatian':pt(2*L3,0.0),'poodle':pt(2*L3,ang(2*L3,2*L3,L3)),'terrier':pt(2*L3,-ang(2*L3,2*L3,L3)),'beagle':pt(3*L3,ang(2*L3,3*L3,L3)),'cat':pt(2*L3,math.pi)}
stream='dax + dalmatian dax ? poodle dax ? cat dax + dalmatian dax + dalmatian dax ? poodle fep + dalmatian fep + poodle fep + terrier fep ? beagle fep ? cat wug + dalmatian wug - poodle wug ? dalmatian wug ? beagle'
rt=run(ET,{'dax','fep','wug'},stream.split(),1.0,0.5)
toy_ok=False
if rt is not None:
    MAX=max(MAX,rt[0]); NQ+=rt[1]; NI+=rt[2]
    exp=[0.366,0.0,0.06,1.0,0.0,1.0,0.0]
    toy_ok=len(rt[3])==7 and all(abs(round(p,3)-e)<=0.001 for p,e in zip(rt[3],exp))
else:
    ok=False

equiv=bool(ok and NQ>0 and MAX<1e-6)
notes='compared %d queries + all positions (eps-smoothed distributions); %d inconsistent-evidence queries (g<=h) excluded: Bayesian posterior empty there, card uses point-mass fallback; toy matches card values: %s' % (NQ,NI,toy_ok)
print(json.dumps({'equivalent':equiv,'max_abs_diff':float(MAX),'notes':notes}))
