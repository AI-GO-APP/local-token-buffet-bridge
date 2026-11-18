// No request/response payload is written to Durable Object storage or logs.
// D-50: request ceiling = Claude Code's 32 MiB request limit (smallest upstream
// inference ceiling; Workers Free/Pro accept 100 MB bodies). Pending bodies live
// in this Durable Object's 128 MB isolate, so their total is budgeted instead.
const MAX_BODY = 32 * 1024 * 1024;
const PENDING_BYTES = 48 * 1024 * 1024;
const BODY_READ_MS = 60_000;
const INFERENCE_CAPACITY=64, CONTROL_CAPACITY=8;
const BATCH_LIMIT=8, BATCH_BYTES=6_100_000;
const MAX_WORK_BYTES=2*MAX_BODY+65536;
const RELAY_VERSION='0.9.6-relay';
const encoder=new TextEncoder();
const QUOTE=0x22, BACKSLASH=0x5c, SHORT_ESCAPES={0x08:0x62,0x09:0x74,0x0a:0x6e,0x0c:0x66,0x0d:0x72}, HEX=encoder.encode('0123456789abcdef');
// JSON-string escaping on UTF-8 bytes, chunk by chunk, so a large body is never
// held twice (raw string plus JSON.stringify copy). Same output as JSON.stringify.
const escapeChunk=src=>{
  let extra=0;
  for(const b of src){if(b===QUOTE||b===BACKSLASH)extra+=1;else if(b<0x20)extra+=SHORT_ESCAPES[b]?1:5;}
  if(!extra)return src;
  const out=new Uint8Array(src.byteLength+extra);let o=0;
  for(const b of src){
    if(b===QUOTE||b===BACKSLASH){out[o++]=BACKSLASH;out[o++]=b;}
    else if(b<0x20){out[o++]=BACKSLASH;if(SHORT_ESCAPES[b])out[o++]=SHORT_ESCAPES[b];else{out.set([0x75,0x30,0x30,HEX[b>>4],HEX[b&15]],o);o+=5;}}
    else out[o++]=b;
  }
  return out;
};
const inferencePath=(method,path)=>method==="POST"&&["/api/v1/chat/completions","/v1/chat/completions","/api/alpha/decisions","/bridge/v1/jobs","/bridge/v1/models/probe"].includes(path);
const COST_HEADERS=["X-Bridge-Cost-Source","X-Bridge-Pricing-Model","X-Bridge-Pricing-Fetched-At"];
const error = (code, type, message) => Response.json({error:{code,message,metadata:{error_type:type}}},
  {status:code,headers:{"Cache-Control":"no-store"}});
const hash = async text => [...new Uint8Array(await crypto.subtle.digest("SHA-256",new TextEncoder().encode(text)))].map(b=>b.toString(16).padStart(2,"0")).join("");
const envelopeBase=batch=>encoder.encode(batch?'{"requests":[]}':'{"request":null}').byteLength;
// Streams pre-serialized works; each work's bytes are released once handed to the stream.
const envelope=(batch,items,release)=>{
  if(!items.length)return new Response(batch?'{"requests":[]}':'{"request":null}',{headers:{'Content-Type':'application/json'}});
  const queue=[encoder.encode(batch?'{"requests":[':'{"request":')];
  items.forEach((item,i)=>{if(i)queue.push(encoder.encode(','));queue.push(item);});
  queue.push(encoder.encode(batch?']}':'}'));
  return new Response(new ReadableStream({
    pull(controller){
      const entry=queue.shift();
      if(entry===undefined){controller.close();return;}
      if(entry instanceof Uint8Array){controller.enqueue(entry);return;}
      for(const part of entry.parts)controller.enqueue(part);
      release(entry);
    },
    cancel(){for(const entry of queue)if(!(entry instanceof Uint8Array))release(entry);queue.length=0;}
  }),{headers:{'Content-Type':'application/json'}});
};

export default {
  async fetch(request, env) {
    if (!env.DEVICE_KEY) return error(503,"not_configured","Relay is not paired.");
    const id=env.DEVICE.idFromName("primary-device");
    const response=await env.DEVICE.get(id).fetch(request);
    const headers=new Headers(response.headers);
    headers.set('X-Bridge-Relay-Version',RELAY_VERSION);
    return new Response(response.body,{status:response.status,headers});
  }
};

export class DeviceRelay {
  constructor(ctx, env) {
    this.ctx=ctx; this.env=env;
    this.pending=new Map(); this.bytes=0; this.reading=0; this.readingInference=0; this.readingControl=0; this.waiter=null; this.lastSeen=0; this.keys=new Set();this.agentSession=null;
  }
  async body(request,max=MAX_BODY) {
    const length=Number(request.headers.get("content-length")||0);
    if(length>max) throw new Error("size");
    const reader=request.body?.getReader();
    if(!reader) return "";
    let size=0; const parts=[];
    let timer;
    const expired=new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('body timeout')),15_000);});
    try {
      for(;;){ const {done,value}=await Promise.race([reader.read(),expired]); if(done)break;
        size+=value.byteLength; if(size>max)throw new Error('size'); parts.push(value); }
    } catch(error) {reader.cancel().catch(()=>{});throw error;}
    finally {clearTimeout(timer);}
    const all=new Uint8Array(size); let pos=0;
    for(const part of parts){all.set(part,pos);pos+=part.byteLength;}
    return new TextDecoder().decode(all);
  }
  // Reads an API body straight into JSON-escaped UTF-8 chunks (see escapeChunk).
  async workBody(request,account) {
    const reader=request.body?.getReader();
    if(!reader) return {parts:[],size:0};
    const validator=new TextDecoder("utf-8",{fatal:true});
    let raw=0,size=0,first=true; const parts=[];
    let timer;
    const expired=new Promise((_,reject)=>{timer=setTimeout(()=>reject(new Error('body timeout')),BODY_READ_MS);});
    try {
      for(;;){ let {done,value}=await Promise.race([reader.read(),expired]); if(done)break;
        raw+=value.byteLength; if(raw>MAX_BODY)throw new Error('size');
        validator.decode(value,{stream:true});
        // TextDecoder dropped a leading BOM before D-50; keep that behaviour.
        if(first&&value[0]===0xef&&value[1]===0xbb&&value[2]===0xbf)value=value.subarray(3);
        first=false;
        const escaped=escapeChunk(value); size+=escaped.byteLength; parts.push(escaped);
        // Charge the shared budget as bytes arrive, so chunked uploads need no worst-case reservation.
        this.bytes+=escaped.byteLength; account.bytes+=escaped.byteLength;
        if(this.bytes>PENDING_BYTES)throw new Error('budget'); }
      validator.decode();
    } catch(error) {reader.cancel().catch(()=>{});throw error;}
    finally {clearTimeout(timer);}
    return {parts,size};
  }
  // Pending bytes are counted from admission until the work is streamed, answered or expired.
  release(item) {
    if(!item||!item.parts)return;
    this.bytes-=item.size; item.parts=null;
  }
  forget(id,item) {
    this.pending.delete(id); clearTimeout(item.timer); this.release(item); item.work=null;
  }
  async fetch(request) {
    const path=new URL(request.url).pathname;
    try {
      if(path.startsWith("/agent/")) {
        const supplied=request.headers.get("Authorization")||"";
        // Hash both secrets before comparison; never emit either value.
        if(await hash(supplied)!==await hash("Bearer "+this.env.DEVICE_KEY))return error(401,"authentication","Invalid device key.");
        if(request.method!=="POST")return error(405,"invalid_request","POST required.");
        const data=JSON.parse(await this.body(request,4_100_000));
        if(path==="/agent/poll") {
          const batch=Object.hasOwn(data,'batch_size');
          if(batch&&(!Number.isInteger(data.batch_size)||data.batch_size<1||data.batch_size>BATCH_LIMIT))
            return error(400,"invalid_request","Invalid batch size.");
          if(!Array.isArray(data.key_digests)||data.key_digests.length>1000||data.key_digests.some(k=>typeof k!=="string"||!/^[a-f0-9]{64}$/.test(k)))return error(400,"invalid_request","Invalid key metadata.");
          if(data.agent_session!==undefined&&(typeof data.agent_session!=="string"||!/^[a-f0-9]{32}$/.test(data.agent_session)))return error(400,"invalid_request","Invalid agent session.");
          // Agents before D-50 read at most BATCH_BYTES; newer ones declare a larger single-work limit.
          if(data.max_work_bytes!==undefined&&(!Number.isInteger(data.max_work_bytes)||data.max_work_bytes<1))return error(400,"invalid_request","Invalid work size.");
          const single=Math.max(BATCH_BYTES,Math.min(data.max_work_bytes??0,MAX_WORK_BYTES));
          // A closed local process can leave an HTTP poll alive at the edge. Never
          // send new work to that stale waiter, or replay a dispatched inference.
          if(this.waiter)this.waiter(null);
          const session=data.agent_session||"legacy";
          if(this.agentSession!==null&&this.agentSession!==session){
            for(const [id,item] of this.pending){
              if(!item.dispatched)continue;
              this.forget(id,item);
              item.resolve(error(503,"device_restarted","Local agent restarted; request was not replayed."));
            }
          }
          this.agentSession=session;
          this.keys=new Set(data.key_digests); this.lastSeen=Date.now();
          const selected=[];
          const limit=batch?data.batch_size:1;
          const base=envelopeBase(batch);let bytes=base;
          // A work this agent cannot read is answered 413 now instead of being dispatched.
          const fits=item=>base+item.size+1<=single;
          const oversized=item=>{this.forget(item.work.id,item);item.resolve(error(413,"request_too_large","The local Bridge cannot accept a request this large; update it first."));};
          for(const item of [...this.pending.values()]){
            if(item.dispatched)continue;
            if(!fits(item)){oversized(item);continue;}
            // Several works share one envelope only within BATCH_BYTES; a larger one travels alone.
            if(selected.length&&bytes+item.size+1>BATCH_BYTES)break;
            selected.push(item);bytes+=item.size+1;
            if(selected.length>=limit||bytes>BATCH_BYTES)break;
          }
          const dispatch=items=>{for(const item of items)item.dispatched=true;return envelope(batch,items,item=>this.release(item));};
          if(selected.length)return dispatch(selected);
          // One device, one long poll. Never use timers as durable storage.
          const response=await new Promise(resolve=>{
            let completed=false;
            const finish=item=>{
              if(completed)return;
              if(item&&!fits(item)){oversized(item);return;}
              completed=true;clearTimeout(timer);this.waiter=null;resolve(dispatch(item?[item]:[]));
            };
            const timer=setTimeout(()=>finish(null),20_000);
            this.waiter=finish;
          });
          this.lastSeen=Date.now();
          return response;
        }
        if(path==="/agent/result") {
          this.lastSeen=Date.now();
          const item=this.pending.get(data.id);
          if(!item)return Response.json({accepted:false});
          if(!Number.isInteger(data.status)||data.status<200||data.status>599||typeof data.body!=="string"||data.body.length>4_000_000)return error(400,"invalid_request","Invalid response.");
          this.forget(data.id,item);
          const headers={"Content-Type":"application/json; charset=utf-8","Cache-Control":"no-store"};
          for(const name of COST_HEADERS){const value=data.headers?.[name];if(typeof value==="string"&&/^[\x20-\x7e]{1,200}$/.test(value))headers[name]=value;}
          item.resolve(new Response(data.body,{status:data.status,headers}));
          return Response.json({accepted:true});
        }
        return error(404,"not_found","Unknown agent endpoint.");
      }
      if(path==="/health")return Response.json({ok:true,device_online:Date.now()-this.lastSeen<35_000,version:RELAY_VERSION,max_concurrent_requests:INFERENCE_CAPACITY,shared_provider_capacity:true,relay_protocol:'batch8-v1',max_batch_size:BATCH_LIMIT,max_batch_bytes:BATCH_BYTES,max_request_bytes:MAX_BODY,pending_bytes_budget:PENDING_BYTES},{headers:{"Cache-Control":"no-store"}});
      const allowed=(request.method==="POST"&&["/api/v1/chat/completions","/v1/chat/completions","/api/alpha/decisions","/bridge/v1/jobs","/bridge/v1/preflight","/bridge/v1/models/refresh","/bridge/v1/models/probe"].includes(path))
        ||(request.method==="GET"&&(["/api/v1/models","/v1/models","/bridge/v1/model-capabilities"].includes(path)||/^\/bridge\/v1\/jobs\/[a-f0-9]{32}$/.test(path)||/^\/bridge\/v1\/receipts\/[a-f0-9-]{36}$/i.test(path)));
      if(!allowed)return error(404,"not_found","Unknown API endpoint.");
      const authorization=request.headers.get("Authorization")||"";
      if(!authorization.startsWith("Bearer ")||authorization.length>263)return error(401,"authentication","Invalid API key.");
      if(Date.now()-this.lastSeen>=35_000)return error(503,"device_offline","Local computer is offline.");
      if(!this.keys.has(await hash(authorization.slice(7))))return error(401,"authentication","Invalid or revoked API key.");
      const inference=inferencePath(request.method,path);
      const active=[...this.pending.values()].filter(item=>item.inference===inference).length;
      const reading=inference?this.readingInference:this.readingControl;
      if(active+reading>=(inference?INFERENCE_CAPACITY:CONTROL_CAPACITY))return error(429,"queue_full","Device concurrent capacity reached.");
      const requestId=request.headers.get("X-AIEA-Request-Id");
      if(requestId&&!/^[a-f0-9-]{36}$/i.test(requestId))return error(400,"invalid_request","Invalid request ID.");
      const declared=request.method==="POST"?Number(request.headers.get("content-length")||0):0;
      if(!(declared>=0&&declared<=MAX_BODY))return error(413,"request_too_large","Request body exceeds 32 MiB.");
      if(this.bytes+declared>PENDING_BYTES)return error(429,"queue_full","Device relay memory budget reached.");
      // Reserve capacity synchronously before yielding to body I/O; bytes are charged while reading.
      this.reading++; if(inference)this.readingInference++;else this.readingControl++;
      const account={bytes:0};
      let body;
      try {body=request.method==="POST"?await this.workBody(request,account):{parts:[],size:0};}
      catch(failure){this.bytes-=account.bytes;throw failure;}
      finally {this.reading--;if(inference)this.readingInference--;else this.readingControl--;}
      const id=crypto.randomUUID();
      const deadline=Math.min(Date.now()+90_000,Number(request.headers.get("X-Bridge-Deadline"))||Date.now()+90_000);
      const work={id,path,method:request.method,authorization,deadline,request_id:requestId};
      const head=JSON.stringify(work);
      const parts=[encoder.encode(head.slice(0,-1)+',"body":"'),...body.parts,encoder.encode('"}')];
      const size=parts.reduce((total,part)=>total+part.byteLength,0);
      this.bytes+=size-account.bytes;
      if(size>MAX_WORK_BYTES||this.bytes>PENDING_BYTES){
        this.bytes-=size;
        return size>Math.min(MAX_WORK_BYTES,PENDING_BYTES)?error(413,"request_too_large","Escaped request exceeds the relay limit.")
          :error(429,"queue_full","Device relay memory budget reached.");
      }
      return await new Promise(resolve=>{
        const timer=setTimeout(()=>{
          const item=this.pending.get(id);
          if(item){this.pending.delete(id);this.release(item);item.work=null;}
          resolve(error(408,"timeout","Device response timed out."));
        },Math.max(1,deadline-Date.now()+2000));
        const item={work,parts,size,resolve,timer,dispatched:false,inference};this.pending.set(id,item);
        if(this.waiter)this.waiter(item);
      });
    } catch(failure) {
      if(failure?.message==="size")return error(413,"request_too_large","Request body exceeds 32 MiB.");
      if(failure?.message==="budget")return error(429,"queue_full","Device relay memory budget reached.");
      return error(400,"invalid_request","Invalid or oversized request.");
    }
  }
}
