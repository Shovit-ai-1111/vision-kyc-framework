import {useEffect,useRef,useState} from 'react';
import {api,auth,errorMessage} from '../api';
import {LineChart} from './Charts';
export function captureGeometry(width,height,ratio=width/height){const w=Math.min(width,height*ratio),h=Math.min(height,width/ratio);return [(width-w)/2,(height-h)/2,w,h];}
export function captureFrame(video,ratio){
 if(!video.videoWidth)throw new Error('Camera is not ready.');
 const c=document.createElement('canvas'),[x,y,w,h]=captureGeometry(video.videoWidth,video.videoHeight,ratio),scale=Math.min(1,800/Math.max(w,h));
 c.width=Math.round(w*scale);c.height=Math.round(h*scale);c.getContext('2d').drawImage(video,x,y,w,h,0,0,c.width,c.height);return c.toDataURL('image/jpeg',.88);
}
export function cameraError(error){
 if(error.name==='NotAllowedError')return 'Camera permission denied. Allow camera access in your browser, then choose Retry camera.';
 if(error.name==='NotFoundError')return 'No camera found. Connect a webcam and choose Retry camera.';
 if(error.name==='NotReadableError')return 'The camera is in use by another application. Close it there and retry.';
 return errorMessage(error);
}
const contours=[[33,160,158,133,153,144,33],[362,385,387,263,373,380,362],[61,40,37,0,267,270,291,321,314,17,84,91,61],[10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10],[168,6,197,195,5,4,1]];
const stepNames={NEUTRAL:'Look straight',CENTER:'Look straight',BLINK:'Close and reopen eyes',TURN_LEFT:'Turn left',TURN_RIGHT:'Turn right'};
function Check({name,value}){return <div className={`quality-check ${value==null?'pending':value?'good':'attention'}`}><b>{value==null?'○':value?'✓':'!'}</b><span>{name}</span><small>{value==null?'Waiting':value?'Good':'Adjust'}</small></div>;}
export default function Camera({session,onComplete}){
 const video=useRef(null),canvas=useRef(null),stream=useRef(null),abort=useRef(null),mounted=useRef(true),running=useRef(false);
 const [ready,setReady]=useState(false),[busy,setBusy]=useState(false),[state,setState]=useState(null),[snapshot,setSnapshot]=useState(null),[error,setError]=useState(''),[history,setHistory]=useState([]),[latency,setLatency]=useState(0),[attempt,setAttempt]=useState(0),[complete,setComplete]=useState(false),[stopped,setStopped]=useState(false),[portraitView,setPortraitView]=useState(()=>window.innerWidth<750);
 useEffect(()=>{const resize=()=>setPortraitView(window.innerWidth<750);window.addEventListener('resize',resize);return()=>window.removeEventListener('resize',resize);},[]);
 useEffect(()=>{
  let cancelled=false;mounted.current=true;setReady(false);setError('');
  (async()=>{try{
   if(!navigator.mediaDevices||!window.isSecureContext)throw new Error('Camera requires localhost or HTTPS. Open the secure application address.');
   const s=await navigator.mediaDevices.getUserMedia({video:{width:{ideal:1280},height:{ideal:window.innerWidth<750?1280:720},aspectRatio:{ideal:window.innerWidth<750?.75:16/9},facingMode:'user'},audio:false});
   if(cancelled){s.getTracks().forEach(t=>t.stop());return;}stream.current=s;video.current.srcObject=s;await video.current.play();if(!cancelled)setReady(true);
  }catch(e){if(!cancelled)setError(cameraError(e));}})();
  return()=>{cancelled=true;mounted.current=false;abort.current?.abort();stream.current?.getTracks().forEach(t=>t.stop());};
 },[attempt]);
 useEffect(()=>{const c=canvas.current;if(!c)return;const ctx=c.getContext('2d');ctx.clearRect(0,0,c.width,c.height);const points=state?.metrics?.landmarks;if(!points?.length)return;ctx.fillStyle='#49ffb5';points.forEach(([x,y])=>{ctx.beginPath();ctx.arc(x*c.width,y*c.height,1.3,0,Math.PI*2);ctx.fill();});ctx.strokeStyle='#2ae6ef';ctx.lineWidth=1.5;contours.forEach(ids=>{ctx.beginPath();ids.forEach((id,i)=>{const p=points[id];if(p)ctx[i?'lineTo':'moveTo'](p[0]*c.width,p[1]*c.height);});ctx.stroke();});},[state,snapshot]);
 async function start(){
  if(running.current)return;running.current=true;setBusy(true);setError('');setStopped(false);setHistory([]);setState(null);setSnapshot(null);setComplete(false);
  const controller=new AbortController();abort.current=controller;const config={...auth(session),signal:controller.signal,timeout:30000};
  try{
   let s=(await api.post(`/api/cases/${session.id}/challenge`,{},config)).data;if(!mounted.current)return;setState(s);const startAt=performance.now();
   while(s.status==='IN_PROGRESS'&&!controller.signal.aborted){
    if(document.hidden)throw new Error('Keep this tab visible during verification. Restart the challenge when ready.');
    const ratio=window.innerWidth<750?.75:video.current.videoWidth/video.current.videoHeight,frame=captureFrame(video.current,ratio),t=performance.now();
    s=(await api.post(`/api/cases/${session.id}/challenge/${s.challenge_id}/frames`,{sequence:s.next_sequence,image:frame},config)).data;
    if(!mounted.current||controller.signal.aborted)return;setSnapshot({url:frame,ratio});setState(s);setLatency(performance.now()-t);
    if(Number.isFinite(s.metrics?.ear))setHistory(h=>[...h,{t:(performance.now()-startAt)/1000,ear:s.metrics.ear,nose:s.metrics.nose_offset}].slice(-180));
    if(s.status==='IN_PROGRESS')await new Promise(r=>setTimeout(r,120));
   }
   if(controller.signal.aborted)return;
   if(s.status!=='PASSED')throw new Error(s.issue||'Liveness challenge failed. Try again in even lighting.');
   setComplete(true);const result=(await api.post(`/api/cases/${session.id}/finish`,{},config)).data;await new Promise(r=>setTimeout(r,800));
   if(mounted.current&&!controller.signal.aborted){stream.current?.getTracks().forEach(t=>t.stop());onComplete(result);}
  }catch(e){if(mounted.current&&e.code!=='ERR_CANCELED')setError(cameraError(e));}finally{running.current=false;if(mounted.current)setBusy(false);}
 }
 const guidance=state?.metrics?.guidance,color=complete?'green':error||stopped?'red':guidance?.color||'red';
 const prompt=error?'Verification interrupted':complete?'Verification captured!':stopped?'Session stopped':!busy?'Position your face, then start':!guidance?'Looking for your face…':!guidance.ready?guidance.prompt:state?.issue||state?.prompt||'Good, hold still…';
 const maxT=Math.max(5,history.at(-1)?.t||0),progress=Math.round((state?.progress||0)*100),cameraRatio=portraitView?.75:video.current?.videoWidth?video.current.videoWidth/video.current.videoHeight:16/9,analyzedRatio=snapshot?.ratio||cameraRatio,guideHeight=1000/cameraRatio,ovalWidth=Math.min(560,guideHeight*.76*.75);
 return <section className={`card live-workspace ${error?'has-error':''} ${complete?'has-success':''}`}>
  <div className="row spread"><div><span className="eyebrow">03 / LIVE VERIFICATION</span><h2>Your face, clearly in view.</h2></div><span className="badge">{busy?'LIVE CHECK':complete?'CAPTURED':'CAMERA'}</span></div>
  <p className="muted">Sit in even light. Keep your face inside the oval and follow each instruction. When asked, close both eyes for two seconds, then reopen.</p>
  <div className={`live-stage guide-${color}`} style={{aspectRatio:cameraRatio}}>
   <video ref={video} autoPlay muted playsInline aria-label="Live camera preview" className="live-video"/>
   <svg className="face-guide" viewBox={`0 0 1000 ${guideHeight}`} preserveAspectRatio="none" aria-label="Face positioning guide"><ellipse cx="500" cy={guideHeight*.48} rx={ovalWidth/2} ry={ovalWidth/1.5}/></svg>
   <div className="live-topline"><span className="live-badge"><i/>{ready?'Camera on':'Opening camera…'}</span><span>{busy?`${state?.remaining_seconds??120}s remaining`:'Mirrored preview'}</span></div>
   <div className="live-status">{guidance?.face_count>1?'Multiple faces detected':guidance?.face_detected?'Face detected':busy?'Searching for one face':'Ready when you are'}</div>
   {complete&&<div className="capture-success" aria-label="Capture successful">✓</div>}
  </div>
  <div className={`live-prompt ${color}`} role="status" aria-live="polite"><span>{prompt}</span><div className="progress-ring" style={{'--progress':`${progress}%`}} role="progressbar" aria-label="Challenge progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress}><b>{progress}%</b></div></div>
  <div className="quality-grid"><Check name="One face" value={guidance?guidance.face_count===1:null}/><Check name="Lighting" value={guidance?.lighting_ok}/><Check name="Face size" value={guidance?.size_ok}/><Check name="Position & focus" value={guidance?guidance.centered&&guidance.focus_ok:null}/></div>
  <div className="steps readable">{state?.steps.map((step,i)=><span key={i} className={i<state.completed_steps?'done':i===state.completed_steps?'current':''}>{i<state.completed_steps?'✓ ':''}{stepNames[step]||step}</span>)}</div>
  {error&&<p className="error" role="alert">{error}</p>}
  <div className="row camera-actions"><button className="primary" disabled={!ready||busy} onClick={start}>{busy?'Checking live frames…':'Start verification'}</button>{busy&&<button onClick={()=>{abort.current?.abort();setStopped(true);}}>Stop verification</button>}{!ready&&<button onClick={()=>setAttempt(v=>v+1)}>Retry camera</button>}</div>
  <div className="inference-panel"><div><h3>What the server sees</h3><p className="caption">Landmarks below belong to the exact analyzed frame. Guidance on the live preview updates after each server response.</p><div className="analyzed" style={{aspectRatio:analyzedRatio}}>{snapshot?<img src={snapshot.url} alt="Last server-analyzed frame"/>:<div className="empty">Start verification to see MediaPipe landmarks.</div>}<canvas ref={canvas} width="800" height={Math.round(800/analyzedRatio)}/><span className="camera-label">MediaPipe · {state?.metrics?.landmarks?.length||0} points · unmirrored</span></div></div><div className="stats compact"><div><b>{state?.metrics?.ear?.toFixed(3)||'—'}</b><span>Eye aspect ratio</span></div><div><b>{state?.metrics?.nose_offset?.toFixed(3)||'—'}</b><span>Head-turn signal</span></div><div><b>{Math.round(latency)} ms</b><span>Last server response</span></div></div></div>
  <details><summary>Live movement graphs</summary><div className="two-cols"><LineChart title="Eye closure over time" xLabel="Seconds" yLabel="EAR" domain={[0,.5]} xDomain={[0,maxT]} series={[{label:'Observed EAR',points:history.map(h=>[h.t,h.ear])}]}/><LineChart title="Head movement over time" xLabel="Seconds" yLabel="Offset" domain={[-.7,.7]} xDomain={[0,maxT]} series={[{label:'Nose / eye-width proxy',points:history.map(h=>[h.t,h.nose]),color:'#e6a358'}]}/></div></details>
  <p className="caption">Movement checks do not establish certified resistance to printed photos or replay attacks.</p>
 </section>;
}
