#!/usr/bin/env python3
"""Durable shared-GPU matrix runner: one child process per config/context."""
import argparse, csv, json, os, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONTEXTS=(128,512,2048,4096,8192,16384,32768)
FIELDS=("case_id","context_tokens","status","evidence_class","weight_store","transfer_mode","slots","prefetch_depth","embedding_placement","lm_head_placement","ttft_ms","prefill_ms","tpot_ms","tokens_per_s","h2d_bytes","h2d_bytes_per_token","h2d_time_ms","peak_vram_bytes","output_correctness","reason","duration_seconds","bundle")

def stamp(): return datetime.now(timezone.utc).isoformat()
def args():
 p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,default=ROOT/'reports/llama70b_full_configuration_20260813/performance_matrix.csv'); p.add_argument('--checkpoint',type=Path,required=True); p.add_argument('--output-dir',type=Path,default=ROOT/'reports/llama70b_shared_matrix_20260813'); p.add_argument('--physical-gpu',type=int,default=4); p.add_argument('--python',default=sys.executable); p.add_argument('--timeout-seconds',type=float,default=1800); p.add_argument('--limit',type=int); p.add_argument('--resume',action='store_true'); return p.parse_args()
def write_json(p,v): p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def clean(s): return ''.join(c if c.isalnum() or c in '._-' else '_' for c in s)
def main():
 a=args(); rows=list(csv.DictReader(a.config.open())); jobs=[(r,c) for r in rows for c in CONTEXTS]; jobs=jobs[:a.limit] if a.limit else jobs; a.output_dir.mkdir(parents=True,exist_ok=True); out=a.output_dir/'results.csv'; js=a.output_dir/'results.json'; results=json.loads(js.read_text()) if a.resume and js.exists() else []; done={(x['case_id'],int(x['context_tokens'])) for x in results}
 write_json(a.output_dir/'manifest.json',{'mode':'shared_smoke','physical_gpu':a.physical_gpu,'logical_device':'cuda:0','contexts':CONTEXTS,'job_count':len(jobs),'started_at':stamp()})
 for n,(cfg,ctx) in enumerate(jobs,1):
  if (cfg['case_id'],ctx) in done: continue
  bundle=a.output_dir/'cases'/clean(f"{cfg['case_id']}-ctx{ctx}"); bundle.mkdir(parents=True,exist_ok=True); t0=time.perf_counter(); started=stamp()
  cmd=[a.python,str(ROOT/'tools/qualify_llama70b_single_request.py'),'--mode','smoke','--allow-shared-smoke','--checkpoint',str(a.checkpoint),'--device','cuda:0','--case',f'{ctx}:1','--repeat','1','--weight-store',cfg['weight_store'],'--granularity',cfg['transfer_mode'],'--slots',cfg['slots'],'--prefetch-depth',cfg['prefetch_depth'],'--embedding-placement',cfg['embedding_placement'],'--lm-head-placement',cfg['lm_head_placement'],'--kv-page-size','16','--kv-attention-backend','generic_cuda','--kv-prefill-backend','gather_sdpa_prefill','--ignore-memlock-limit','--timeout-seconds',str(a.timeout_seconds),'--output-dir',str(bundle),'--raw-dir',str(a.output_dir/'raw'),'--python',a.python]
  e=os.environ.copy(); e['CUDA_VISIBLE_DEVICES']=str(a.physical_gpu); p=subprocess.run(cmd,cwd=ROOT,env=e,capture_output=True,text=True); finished=stamp(); (bundle/'stdout.log').write_text(p.stdout); (bundle/'stderr.log').write_text(p.stderr)
  try: case=json.loads((bundle/'cases.json').read_text())['cases'][0]
  except Exception: case={'status':'FAILED','reason':f'child_exit_{p.returncode}'}
  m=case.get('metrics') or {}; perf=m.get('performance') or {}; mem=m.get('memory') or {}; ws=m.get('weight_streaming') or {}; val=lambda x: x.get('value') if isinstance(x,dict) and 'value' in x else x; tpot=val(perf.get('tpot_mean_ms')); item={'case_id':cfg['case_id'],'context_tokens':ctx,'status':case.get('status','FAILED'),'evidence_class':case.get('evidence_class','SMOKE_ONLY'),'weight_store':cfg['weight_store'],'transfer_mode':cfg['transfer_mode'],'slots':cfg['slots'],'prefetch_depth':cfg['prefetch_depth'],'embedding_placement':cfg['embedding_placement'],'lm_head_placement':cfg['lm_head_placement'],'ttft_ms':val(perf.get('ttft_ms')) or 'NOT_RUN','prefill_ms':val(perf.get('prefill_transformer_ms')) or 'NOT_RUN','tpot_ms':tpot or 'NOT_RUN','tokens_per_s':val(perf.get('token_per_second')) or (1000/tpot if tpot else 'NOT_RUN'),'h2d_bytes':val(ws.get('h2d_bytes')) or 'NOT_RUN','h2d_bytes_per_token':val(ws.get('h2d_bytes_per_generated_token')) or 'NOT_RUN','h2d_time_ms':val(ws.get('h2d_time_ms')) or 'NOT_RUN','peak_vram_bytes':val(mem.get('cuda_allocated_peak_bytes')) or 'NOT_RUN','output_correctness':'PASS' if case.get('status')=='PASS' else 'NOT_RUN','reason':case.get('reason',''),'duration_seconds':time.perf_counter()-t0,'bundle':str(bundle),'started_at':started,'finished_at':finished,'child_returncode':p.returncode}
  exists=out.exists()
  with out.open('a',newline='') as f:
   w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader() if not exists else None; w.writerow({k:item.get(k,'NOT_RUN') for k in FIELDS}); f.flush(); os.fsync(f.fileno())
  results.append(item); write_json(js,results); write_json(a.output_dir/'state.json',{'completed':len(results),'total':len(jobs),'last':item,'updated_at':finished}); print(f'[{n}/{len(jobs)}] {cfg["case_id"]} ctx={ctx} {item["status"]} rc={p.returncode}',flush=True)
 write_json(a.output_dir/'manifest.json',{'mode':'shared_smoke','physical_gpu':a.physical_gpu,'logical_device':'cuda:0','contexts':CONTEXTS,'job_count':len(jobs),'completed_count':len(results),'finished_at':stamp()})
if __name__=='__main__': main()
