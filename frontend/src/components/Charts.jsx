export function BarChart({data,title,max=1,format=v=>`${(v*100).toFixed(1)}%`}) {
  if(!data?.length)return <div className="empty">No measured data yet.</div>;
  return <div className="bar-chart" role="img" aria-label={title}>{data.map((d,i)=><div className="bar-row" key={i}><span>{d.label}</span><div className="bar-track"><div style={{width:`${Math.max(0,Math.min(100,d.value/max*100))}%`,background:d.color||'var(--teal)'}}/></div><b>{format(d.value)}</b></div>)}</div>;
}
export function LineChart({series,title,xLabel='',yLabel='',domain=[0,1],xDomain=[0,1]}){
  const W=640,H=220,L=48,R=20,T=18,B=36;
  const x=v=>L+(v-xDomain[0])/(xDomain[1]-xDomain[0]||1)*(W-L-R);
  const y=v=>H-B-(v-domain[0])/(domain[1]-domain[0]||1)*(H-T-B);
  return <figure className="linechart"><figcaption>{title}</figcaption><svg viewBox={`0 0 ${W} ${H}`} role="img" aria-label={title}>
    {[0,.25,.5,.75,1].map((v,i)=><g key={i}><line x1={L} y1={y(domain[0]+v*(domain[1]-domain[0]))} x2={W-R} y2={y(domain[0]+v*(domain[1]-domain[0]))} stroke="#dce5e7"/><text x={L-8} y={y(domain[0]+v*(domain[1]-domain[0]))+4} textAnchor="end">{(domain[0]+v*(domain[1]-domain[0])).toFixed(2)}</text><text x={x(xDomain[0]+v*(xDomain[1]-xDomain[0]))} y={H-16} textAnchor="middle">{(xDomain[0]+v*(xDomain[1]-xDomain[0])).toFixed(1)}</text></g>)}
    {series.map((s,i)=><polyline key={i} fill="none" stroke={s.color||'var(--teal)'} strokeWidth="2.5" points={s.points.filter(p=>p.every(Number.isFinite)).map(p=>`${x(p[0])},${y(p[1])}`).join(' ')}/>)}
    <text x={W/2} y={H-1} textAnchor="middle">{xLabel}</text><text x="8" y="12">{yLabel}</text>
  </svg><div className="legend">{series.map(s=><span key={s.label}><i style={{background:s.color||'var(--teal)'}}/>{s.label}</span>)}</div></figure>;
}
