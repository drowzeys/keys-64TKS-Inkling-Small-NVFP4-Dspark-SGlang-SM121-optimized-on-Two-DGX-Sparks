import json, time, urllib.request, concurrent.futures as cf, statistics, sys
URL="http://10.100.10.1:30000/generate"
PROMPTS={
 "list":"List 40 US states with one-line fun facts, numbered:",
 "essay":"Write a detailed essay about the spirit of American innovation:",
 "reading":"Read this and answer: The Constitution establishes three branches of government. The legislative branch makes laws, the executive enforces them, and the judicial interprets them. Checks and balances prevent any branch from dominating. Question: explain the system in detail:",
}
def one(prompt, ct):
    body=json.dumps({"text":prompt,"sampling_params":{"max_new_tokens":ct,"temperature":0.7,"top_p":0.95}}).encode()
    t0=time.time()
    r=urllib.request.urlopen(urllib.request.Request(URL,body,{"Content-Type":"application/json"}),timeout=600)
    d=json.load(r); dt=time.time()-t0
    return d["meta_info"]["completion_tokens"], dt, d["meta_info"].get("spec_accept_length")
# warmup
one("Warmup.",16)
print("task,conc,agg_toks,wall_s,agg_tok/s,per_stream_tok/s,accept")
for task,prompt in PROMPTS.items():
    for conc in (1,4,8):
        t0=time.time()
        with cf.ThreadPoolExecutor(conc) as ex:
            rs=list(ex.map(lambda i: one(prompt+f" (variation {i})",256), range(conc)))
        wall=time.time()-t0
        toks=sum(r[0] for r in rs)
        accs=[r[2] for r in rs if r[2]]
        print(f"{task},{conc},{toks},{wall:.1f},{toks/wall:.1f},{toks/wall/conc:.1f},{statistics.mean(accs):.2f}" if accs else f"{task},{conc},{toks},{wall:.1f},{toks/wall:.1f},{toks/wall/conc:.1f},n/a", flush=True)
