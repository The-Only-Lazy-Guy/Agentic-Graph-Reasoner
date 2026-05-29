from __future__ import annotations
"""Generate NGR-v1a goal-spec tasks, not one mandatory trajectory."""
import argparse, json, random
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from graph_core import MemoryGraph, canonical_relation
from ngr_v1_env import split_signal_spans

def clean(x,max_len=260): return ' '.join(str(x or '').split())[:max_len].rstrip()
def write_jsonl(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        for r in rows: f.write(json.dumps(r,ensure_ascii=False)+'\n')
def graph_files(d): return sorted(p for p in Path(d).glob('*.json') if p.is_file())
def neigh(g,nid):
    out=[]
    for e in g.edges:
        if e.src==nid and e.dst in g.nodes: out.append(e.dst)
        elif e.dst==nid and e.src in g.nodes: out.append(e.src)
    seen=set(); return [x for x in out if not (x in seen or seen.add(x))]
def out_neigh(g,nid):
    out=[]
    for e in g.edges:
        if e.src==nid and e.dst in g.nodes:
            out.append(e.dst)
        elif not getattr(e,'directed',True) and e.dst==nid and e.src in g.nodes:
            out.append(e.src)
    seen=set(); return [x for x in out if not (x in seen or seen.add(x))]
def rel_between(g,a,b):
    for e in g.edges:
        if e.src==a and e.dst==b: return canonical_relation(e.relation)
    return 'related'
def row(rid,gp,typ,signal,goal,mem,meta=None): return {'id':rid,'task_type':typ,'graph_path':str(gp),'signal':signal,'initial_memory_node_ids':list(mem),'spans':[s.__dict__ for s in split_signal_spans(signal)],'goal':dict(goal),'metadata':dict(meta or {})}
def covered(g,gp,rng,limit):
    rows=[]; ids=[nid for nid,n in g.nodes.items() if len(str(n.text))>=20]
    if len(ids) < 3: return []
    delimiters = ["; ", " | ", " --- ", ". ", " AND "]
    while len(rows) < limit:
        rng.shuffle(ids)
        for i in range(0,len(ids)-2,3):
            if len(rows)>=limit: break
        chosen=ids[i:i+3]; 
        texts = [clean(g.nodes[n].text,160) for n in chosen]
        if rng.random() < 0.5: rng.shuffle(texts)
        sig=rng.choice(delimiters).join(texts)
        if rng.random() < 0.2: sig = f"Cover these points: {sig}"
        goal={'covered_mappings':[{'span_text':clean(g.nodes[n].text,160),'memory_id':n} for n in chosen],'session_nodes':[],'session_edges':[],'memory_attachments':[],'final_commits':[{'action':'no_op'}]}
        rows.append(row(f'{gp.stem}::covered_long_signal::{len(rows):06d}',gp,'covered_long_signal',sig,goal,chosen,{'covered_node_ids':chosen}))
    return rows
def long_decompose(g,gp,rng,limit):
    rows=[]; edges=[e for e in g.edges if e.src in g.nodes and e.dst in g.nodes and e.src!=e.dst]
    if not edges: return []
    delimiters = ["; ", " | ", " --- ", ". ", "\n"]
    while len(rows) < limit:
        rng.shuffle(edges)
        for e in edges:
            if len(rows)>=limit: break
        nb=[x for x in out_neigh(g,e.dst) if x!=e.src]
        if not nb: continue
        c=rng.choice(nb); nodes=[e.src,e.dst,c]
        texts = [clean(g.nodes[n].text,140) for n in nodes]
        
        # 50% chance to shuffle the input signal so the model can't rely on linear order
        if rng.random() < 0.5:
            sig_texts = list(texts)
            rng.shuffle(sig_texts)
            sig = rng.choice(delimiters).join(sig_texts)
        else:
            sig = rng.choice(delimiters).join(texts)
            
        if rng.random() < 0.2: sig = f"Decompose: {sig}"
        elif rng.random() < 0.2: sig = f"Notes: {sig}"
        
        goal={'session_nodes':[{'name':f's{i}','span_text':texts[i],'node_type':'concept'} for i in range(3)],'session_edges':[{'src':'s0','dst':'s1','relation':canonical_relation(e.relation)},{'src':'s1','dst':'s2','relation':rel_between(g,e.dst,c)}],'covered_mappings':[],'memory_attachments':[],'final_commits':[{'action':'add_node','session':f's{i}'} for i in range(3)]}
        rows.append(row(f'{gp.stem}::long_decompose::{len(rows):06d}',gp,'long_decompose',sig,goal,[],{'source_nodes':nodes,'directed_chain':True}))
    return rows
def multi_attach(g,gp,rng,limit):
    rows=[]; ids=[nid for nid,n in g.nodes.items() if len(str(n.text))>=20]
    if len(ids) < 2: return []
    while len(rows) < limit:
        rng.shuffle(ids)
        for i in range(0,len(ids)-1,2):
            if len(rows)>=limit: break
        a,b=ids[i],ids[i+1]
        if a==b: continue
        t_a = clean(g.nodes[a].text,120)
        t_b = clean(g.nodes[b].text,120)
        
        sig_variants = [
            f"A new bridge concept connects these ideas: {t_a} and {t_b}.",
            f"Create a bridge linking {t_a} with {t_b}.",
            f"Synthesize {t_a} and {t_b} into a single bridge note.",
            f"Note how {t_b} relates to {t_a} through a new bridging concept.",
            f"Bridge: {t_a} <-> {t_b}."
        ]
        sig = rng.choice(sig_variants)
        support_text = t_a
        
        bridge_variants = [
            f"{clean(t_a,90)} and {clean(t_b,90)} are connected by a shared bridge concept.",
            f"Bridge integrating the concepts of {clean(t_a,90)} and {clean(t_b,90)}.",
            f"Synthesis of {clean(t_a,90)} alongside {clean(t_b,90)}.",
            f"A shared abstraction connecting {clean(t_a,90)} with {clean(t_b,90)}."
        ]
        bridge_text = clean(rng.choice(bridge_variants), 180)
        
        goal={
            'session_nodes':[
                {'name':'support_note','span_text':support_text,'node_type':'concept'},
                {'name':'bridge','span_text':bridge_text,'node_type':'bridge'},
            ],
            'session_edges':[{'src':'support_note','dst':'bridge','relation':'support'}],
            'covered_mappings':[],
            'memory_attachments':[{'session':'bridge','memory_id':a,'relation':'related'},{'session':'bridge','memory_id':b,'relation':'related'}],
            'final_commits':[
                {'action':'add_node','session':'support_note'},
                {'action':'add_node','session':'bridge'},
                {'action':'link_nodes','session':'bridge','memory_id':a,'relation':'related'},
                {'action':'link_nodes','session':'bridge','memory_id':b,'relation':'related'},
            ]
        }
        rows.append(row(f'{gp.stem}::multi_region_attach::{len(rows):06d}',gp,'multi_region_attach',sig,goal,[a,b],{'attach_to':[a,b]}))
    return rows
def mixed(g,gp,rng,limit):
    rows=[]; edges=[e for e in g.edges if e.src in g.nodes and e.dst in g.nodes and e.src!=e.dst]
    if not edges: return []
    while len(rows) < limit:
        rng.shuffle(edges)
        for e in edges:
            if len(rows)>=limit: break
        t_dst = clean(g.nodes[e.dst].text,120)
        t_src = clean(g.nodes[e.src].text,120)
        
        sig_variants = [
            f"Add a new note related to {t_dst}: {t_src}",
            f"New concept: {t_src}. Link it to {t_dst}.",
            f"Attach {t_src} as a new note under {t_dst}.",
            f"Expand on {t_dst} by adding: {t_src}",
            f"Regarding {t_dst}, note that {t_src}"
        ]
        sig = rng.choice(sig_variants)
        rel=canonical_relation(e.relation)
        source_text = clean(g.nodes[e.src].text,160)
        
        goal={
            'session_nodes':[
                {'name':'new_note','span_text':source_text,'node_type':'concept'},
            ],
            'session_edges':[],
            'covered_mappings':[],
            'memory_attachments':[{'session':'new_note','memory_id':e.dst,'relation':rel}],
            'final_commits':[
                {'action':'add_node','session':'new_note'},
                {'action':'link_nodes','session':'new_note','memory_id':e.dst,'relation':rel},
            ]
        }
        rows.append(row(f'{gp.stem}::mixed_add_link::{len(rows):06d}',gp,'mixed_add_link',sig,goal,[e.dst],{'source_node':e.src,'attach_to':e.dst}))
    return rows
def gen(gp,rng,per):
    g=MemoryGraph.load_json(gp); return covered(g,gp,rng,per)+long_decompose(g,gp,rng,per)+multi_attach(g,gp,rng,per)+mixed(g,gp,rng,per)
def split_by_graph(rows, graphs, rng, val):
    graphs = [str(g) for g in graphs]
    if not graphs:
        return rows, [], [], []
    shuffled = list(graphs)
    rng.shuffle(shuffled)
    n_val = min(max(1, int(round(len(shuffled) * val))), max(len(shuffled) - 1, 1))
    val_graphs = set(shuffled[:n_val])
    train_graphs = set(shuffled[n_val:])
    train = [r for r in rows if str(r.get('graph_path', '')) in train_graphs]
    val_rows = [r for r in rows if str(r.get('graph_path', '')) in val_graphs]
    return train, val_rows, sorted(train_graphs), sorted(val_graphs)
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--graphs-dir',default='graphs'); ap.add_argument('--out-dir',default='artifacts/tasks_v1a'); ap.add_argument('--max-tasks',type=int,default=2000); ap.add_argument('--per-type-per-graph',type=int,default=80); ap.add_argument('--val-ratio',type=float,default=.2); ap.add_argument('--seed',type=int,default=42); args=ap.parse_args()
    rng=random.Random(args.seed); per_graph_rows=[]; graphs=graph_files(args.graphs_dir)
    for gp in graphs:
        try:
            grew = gen(gp,rng,args.per_type_per_graph)
            per_graph_rows.extend(grew)
        except Exception as e: print('[warn] failed',gp,e)
    rng.shuffle(per_graph_rows)
    per_graph_rows=per_graph_rows[:args.max_tasks]
    train,val,train_graphs,val_graphs=split_by_graph(per_graph_rows,graphs,rng,args.val_ratio)
    out=Path(args.out_dir); write_jsonl(out/'ngr_v1_train.jsonl',train); write_jsonl(out/'ngr_v1_val.jsonl',val)
    summary={'graphs':[str(g) for g in graphs],'train_graphs':train_graphs,'val_graphs':val_graphs,'graph_heldout_split':True,'total':len(per_graph_rows),'train':len(train),'val':len(val),'task_counts':dict(Counter(r['task_type'] for r in per_graph_rows)),'train_task_counts':dict(Counter(r['task_type'] for r in train)),'val_task_counts':dict(Counter(r['task_type'] for r in val)),'goal_spec_not_single_trajectory':True}; (out/'ngr_v1_summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8'); print(json.dumps(summary,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
