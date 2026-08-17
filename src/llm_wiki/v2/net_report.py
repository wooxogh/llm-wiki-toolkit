"""Portable NET visualizations; no graph database or browser package required."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from llm_wiki.v2.net_store import NetStore
from llm_wiki.v2.schemas import EdgeType, NodeType


_TREE_EDGE_TYPES = {
    EdgeType.PARENT_OF.value,
    EdgeType.CONTAINS_DOCUMENT.value,
    EdgeType.DOCUMENT_HAS_CONCEPT.value,
}
_TYPE_ORDER = {
    NodeType.TOPIC.value: 0,
    NodeType.COLLECTION.value: 1,
    NodeType.DOCUMENT.value: 2,
    NodeType.CONCEPT.value: 3,
}


def export_mermaid(vault: Path | None = None, out: Path | None = None) -> Path:
    store = NetStore(vault)
    out = out or (store.root / "NET.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    labels = {node.id: node.label.replace('"', "'") for node in store.nodes()}
    lines = ["# NET", "", "```mermaid", "graph TD"]
    for edge in store.edges():
        left, right = _id(edge.source), _id(edge.target)
        label = edge.relation or edge.type
        lines.append(f'  {left}["{labels.get(edge.source, edge.source)}"] -->|{label}| {right}["{labels.get(edge.target, edge.target)}"]')
    lines.extend(["```", ""])
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def render_tree(vault: Path | None = None, *, show_concepts: bool = False,
                show_ids: bool = False, max_depth: int | None = None,
                active_only: bool = False, ascii_only: bool = False) -> str:
    """Render the strict Topic/Collection/Document/Concept backbone as a tree."""
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be zero or greater")

    store = NetStore(vault)
    all_nodes = store.nodes()
    all_edges = store.edges()
    nodes = {
        node.id: node for node in all_nodes
        if (show_concepts or node.type != NodeType.CONCEPT.value)
        and (not active_only or _node_state(node) == "ACTIVE")
    }
    children: dict[str, list[str]] = defaultdict(list)
    incoming: set[str] = set()
    hierarchy_edges = []
    for edge in all_edges:
        if edge.type not in _TREE_EDGE_TYPES:
            continue
        if edge.source not in nodes or edge.target not in nodes:
            continue
        children[edge.source].append(edge.target)
        incoming.add(edge.target)
        hierarchy_edges.append(edge)

    def sort_key(node_id: str) -> tuple[int, str, str]:
        node = nodes[node_id]
        return (_TYPE_ORDER.get(node.type, 99), node.label.casefold(), node.id)

    for values in children.values():
        values.sort(key=sort_key)
    roots = sorted((node_id for node_id in nodes if node_id not in incoming), key=sort_key)
    if "topic:knowledge" in roots:
        roots.remove("topic:knowledge")
        roots.insert(0, "topic:knowledge")

    if not nodes:
        return "NET is empty. Run `wiki-net build` first."

    concept_counts: dict[str, int] = defaultdict(int)
    for edge in all_edges:
        if edge.type == EdgeType.DOCUMENT_HAS_CONCEPT.value:
            concept_counts[edge.source] += 1

    branch, last_branch, vertical, space = (
        ("|-- ", "`-- ", "|   ", "    ") if ascii_only
        else ("\u251c\u2500\u2500 ", "\u2514\u2500\u2500 ", "\u2502   ", "    ")
    )
    lines = [f"NET ({len(nodes)} nodes, {len(hierarchy_edges)} tree edges)"]
    visited: set[str] = set()

    def label_for(node_id: str) -> str:
        node = nodes[node_id]
        label = f"{node.label} [{node.type}]"
        state = _node_state(node)
        if state != "ACTIVE":
            label += f" ({state})"
        if node.type == NodeType.DOCUMENT.value and not show_concepts:
            count = concept_counts.get(node.id, 0)
            label += f" ({count} concept{'s' if count != 1 else ''})"
        if show_ids:
            label += f" <{node.id}>"
        return label

    def visit(node_id: str, prefix: str, is_last: bool, depth: int,
              ancestors: frozenset[str], root: bool = False) -> None:
        connector = "" if root else (last_branch if is_last else branch)
        lines.append(prefix + connector + label_for(node_id))
        if node_id in ancestors:
            lines.append(prefix + space + "[cycle suppressed]")
            return
        visited.add(node_id)
        descendants = children.get(node_id, [])
        child_prefix = prefix if root else prefix + (space if is_last else vertical)
        if max_depth is not None and depth >= max_depth:
            if descendants:
                lines.append(child_prefix + last_branch + f"... ({len(descendants)} hidden)")
            return
        next_ancestors = ancestors | {node_id}
        for index, child_id in enumerate(descendants):
            visit(child_id, child_prefix, index == len(descendants) - 1,
                  depth + 1, next_ancestors)

    for index, root_id in enumerate(roots):
        if index:
            lines.append("")
        visit(root_id, "", True, 0, frozenset(), root=True)

    # A malformed store should still be inspectable instead of silently hiding nodes.
    for node_id in sorted(set(nodes) - visited, key=sort_key):
        lines.append("")
        visit(node_id, "", True, 0, frozenset(), root=True)

    graph_edges = sum(edge.type not in _TREE_EDGE_TYPES for edge in all_edges)
    if graph_edges:
        lines.extend(["", f"{graph_edges} semantic/membership edge(s) are omitted from the tree; use `wiki-net visualize` to inspect them."])
    return "\n".join(lines)


def export_html(vault: Path | None = None, out: Path | None = None,
                *, active_only: bool = False) -> Path:
    """Write a self-contained interactive HTML explorer for every NET edge."""
    store = NetStore(vault)
    out = out or (store.root / "NET.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    nodes = [node for node in store.nodes() if not active_only or _node_state(node) == "ACTIVE"]
    node_ids = {node.id for node in nodes}
    edges = [edge for edge in store.edges() if edge.source in node_ids and edge.target in node_ids]
    payload = {
        "nodes": [node.to_dict() for node in nodes],
        "edges": [edge.to_dict() for edge in edges],
    }
    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # JSON lives inside a script element, so escape HTML-significant characters.
    data = data.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    out.write_text(_HTML_TEMPLATE.replace("__NET_DATA__", data), encoding="utf-8")
    return out


def _node_state(node) -> str:
    if node.type == NodeType.CONCEPT.value:
        return str(node.attrs.get("concept_state", node.state)).upper()
    return str(node.state).upper()


def _id(value: str) -> str:
    return "n_" + "".join(char if char.isalnum() else "_" for char in value)


_HTML_TEMPLATE = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>LLM Wiki NET Explorer</title>
<style>
:root { color-scheme: light; --ink:#182026; --muted:#667078; --line:#d8dee3; --panel:#f7f8f9; }
* { box-sizing:border-box; }
html,body { width:100%; height:100%; margin:0; overflow:hidden; font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; color:var(--ink); background:#fff; }
#app { display:grid; grid-template-columns:280px 1fr; width:100%; height:100%; }
aside { z-index:2; display:flex; flex-direction:column; min-width:0; border-right:1px solid var(--line); background:var(--panel); }
header { padding:18px 16px 12px; border-bottom:1px solid var(--line); }
h1 { margin:0 0 3px; font-size:18px; letter-spacing:0; }
#stats { color:var(--muted); font-size:12px; }
.controls { padding:14px 16px; overflow:auto; }
label.title { display:block; margin:15px 0 7px; color:#454e54; font-size:11px; font-weight:700; text-transform:uppercase; }
input[type=search] { width:100%; height:34px; padding:0 9px; border:1px solid #b9c2c9; border-radius:5px; background:#fff; color:var(--ink); }
.check { display:flex; align-items:center; gap:8px; margin:7px 0; cursor:pointer; }
.swatch { width:10px; height:10px; border-radius:50%; flex:0 0 auto; }
.buttons { display:grid; grid-template-columns:1fr 1fr; gap:7px; margin-top:14px; }
button { height:32px; border:1px solid #aeb8bf; border-radius:5px; background:#fff; color:var(--ink); cursor:pointer; }
button:hover { background:#eef1f3; }
#details { margin-top:auto; padding:14px 16px; min-height:125px; border-top:1px solid var(--line); background:#fff; overflow:auto; }
#details strong { display:block; overflow-wrap:anywhere; }
#details .meta { margin-top:5px; color:var(--muted); font-size:12px; overflow-wrap:anywhere; }
main { position:relative; min-width:0; min-height:0; }
canvas { display:block; width:100%; height:100%; cursor:grab; }
canvas.dragging { cursor:grabbing; }
#tip { position:absolute; display:none; max-width:340px; padding:7px 9px; pointer-events:none; border:1px solid #bcc5cb; border-radius:5px; background:rgba(255,255,255,.96); box-shadow:0 4px 16px rgba(0,0,0,.12); font-size:12px; overflow-wrap:anywhere; }
#empty { position:absolute; inset:0; display:none; place-items:center; color:var(--muted); pointer-events:none; }
@media (max-width:720px) { #app { grid-template-columns:1fr; grid-template-rows:auto 1fr; } aside { max-height:245px; border-right:0; border-bottom:1px solid var(--line); } .controls { display:grid; grid-template-columns:1fr 1fr; gap:0 14px; } #details { display:none; } }
</style>
</head>
<body>
<div id="app">
  <aside>
    <header><h1>LLM Wiki NET</h1><div id="stats"></div></header>
    <div class="controls">
      <input id="search" type="search" placeholder="Search label or ID" autocomplete="off">
      <label class="title">Node types</label>
      <label class="check"><input type="checkbox" data-node="TOPIC" checked><span class="swatch" style="background:#267a68"></span>Topics</label>
      <label class="check"><input type="checkbox" data-node="COLLECTION" checked><span class="swatch" style="background:#a86b18"></span>Collections</label>
      <label class="check"><input type="checkbox" data-node="DOCUMENT" checked><span class="swatch" style="background:#316fa6"></span>Documents</label>
      <label class="check"><input type="checkbox" data-node="CONCEPT" checked><span class="swatch" style="background:#8055a5"></span>Concepts</label>
      <label class="title">Edges</label>
      <label class="check"><input id="hierarchy" type="checkbox" checked>Hierarchy and membership</label>
      <label class="check"><input id="relations" type="checkbox" checked>Semantic relations</label>
      <div class="buttons"><button id="fit" type="button">Fit graph</button><button id="labels" type="button">Toggle labels</button></div>
    </div>
    <div id="details"><strong>Select a node</strong><div class="meta">Click a node to inspect its ID, state, and attributes.</div></div>
  </aside>
  <main><canvas id="graph"></canvas><div id="tip"></div><div id="empty">No nodes match the current filters.</div></main>
</div>
<script>
const DATA=__NET_DATA__;
const COLORS={TOPIC:'#267a68',COLLECTION:'#a86b18',DOCUMENT:'#316fa6',CONCEPT:'#8055a5'};
const TYPE_ORDER=['TOPIC','COLLECTION','DOCUMENT','CONCEPT'];
const RELATION_EDGE='RELATES_TO';
const canvas=document.getElementById('graph'),ctx=canvas.getContext('2d'),main=canvas.parentElement;
const tip=document.getElementById('tip'),stats=document.getElementById('stats'),details=document.getElementById('details'),empty=document.getElementById('empty');
let width=0,height=0,dpr=1,scale=1,offsetX=0,offsetY=0,dragNode=null,panning=false,lastX=0,lastY=0,showLabels=true,selected=null;
const nodes=DATA.nodes.map(n=>({...n,x:0,y:0}));
const byId=new Map(nodes.map(n=>[n.id,n]));
const edges=DATA.edges.filter(e=>byId.has(e.source)&&byId.has(e.target));

function initialLayout(){
  const groups=new Map(TYPE_ORDER.map(t=>[t,[]]));
  nodes.forEach(n=>(groups.get(n.type)||groups.get('CONCEPT')).push(n));
  TYPE_ORDER.forEach((type,column)=>{
    const group=groups.get(type).sort((a,b)=>(a.label||a.id).localeCompare(b.label||b.id));
    group.forEach((node,row)=>{ node.x=column*360; node.y=(row-(group.length-1)/2)*62; });
  });
}
function enabledTypes(){ return new Set([...document.querySelectorAll('[data-node]:checked')].map(x=>x.dataset.node)); }
function visible(){
  const types=enabledTypes();
  const query=document.getElementById('search').value.trim().toLocaleLowerCase();
  const result=nodes.filter(n=>types.has(n.type));
  return {nodes:result,ids:new Set(result.map(n=>n.id)),query};
}
function edgeVisible(edge,ids){
  if(!ids.has(edge.source)||!ids.has(edge.target)) return false;
  return edge.type===RELATION_EDGE?document.getElementById('relations').checked:document.getElementById('hierarchy').checked;
}
function resize(){
  const rect=main.getBoundingClientRect(); dpr=Math.max(1,window.devicePixelRatio||1); width=rect.width; height=rect.height;
  canvas.width=Math.round(width*dpr); canvas.height=Math.round(height*dpr); canvas.style.width=width+'px'; canvas.style.height=height+'px'; draw();
}
function screen(node){ return {x:node.x*scale+offsetX,y:node.y*scale+offsetY}; }
function world(x,y){ return {x:(x-offsetX)/scale,y:(y-offsetY)/scale}; }
function radius(node){ return node.type==='TOPIC'?10:node.type==='COLLECTION'?9:node.type==='DOCUMENT'?8:7; }
function fit(){
  const v=visible().nodes;
  if(!v.length){ scale=1; offsetX=width/2; offsetY=height/2; draw(); return; }
  const xs=v.map(n=>n.x),ys=v.map(n=>n.y),minX=Math.min(...xs),maxX=Math.max(...xs),minY=Math.min(...ys),maxY=Math.max(...ys);
  const graphW=Math.max(120,maxX-minX+120),graphH=Math.max(120,maxY-minY+120);
  scale=Math.min(1.5,Math.max(.08,Math.min((width-80)/graphW,(height-80)/graphH)));
  offsetX=width/2-(minX+maxX)/2*scale; offsetY=height/2-(minY+maxY)/2*scale; draw();
}
function draw(){
  ctx.setTransform(dpr,0,0,dpr,0,0); ctx.clearRect(0,0,width,height);
  const v=visible(),shownEdges=edges.filter(e=>edgeVisible(e,v.ids));
  empty.style.display=v.nodes.length?'none':'grid'; stats.textContent=`${v.nodes.length} nodes / ${shownEdges.length} edges`;
  ctx.lineCap='round';
  shownEdges.forEach(edge=>{
    const a=screen(byId.get(edge.source)),b=screen(byId.get(edge.target)),semantic=edge.type===RELATION_EDGE;
    ctx.beginPath(); ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.strokeStyle=semantic?'rgba(182,68,72,.58)':'rgba(94,108,117,.30)';
    ctx.lineWidth=semantic?1.8:1; ctx.setLineDash(semantic?[5,4]:[]); ctx.stroke();
  });
  ctx.setLineDash([]);
  v.nodes.forEach(node=>{
    const p=screen(node),r=Math.max(4,radius(node)*Math.min(1.3,Math.max(.7,scale))),match=v.query&&(`${node.label} ${node.id}`.toLocaleLowerCase().includes(v.query));
    ctx.beginPath(); ctx.arc(p.x,p.y,r,0,Math.PI*2); ctx.fillStyle=COLORS[node.type]||'#667078'; ctx.fill();
    ctx.lineWidth=node===selected||match?3:1.5; ctx.strokeStyle=node===selected?'#111':match?'#d33a2c':'#fff'; ctx.stroke();
    if(showLabels&&(scale>.48||match||node===selected)){
      ctx.font='12px system-ui, sans-serif'; ctx.fillStyle='#20272c'; ctx.textBaseline='middle';
      const label=node.label||node.id,short=label.length>45?label.slice(0,42)+'...':label; ctx.fillText(short,p.x+r+5,p.y);
    }
  });
}
function hit(x,y){
  const v=visible().nodes;
  for(let i=v.length-1;i>=0;i--){ const p=screen(v[i]),r=Math.max(9,radius(v[i])*scale+4); if((p.x-x)**2+(p.y-y)**2<=r*r)return v[i]; }
  return null;
}
function showDetails(node){
  selected=node; details.textContent=''; const title=document.createElement('strong'); title.textContent=node.label||node.id; details.appendChild(title);
  const meta=document.createElement('div'); meta.className='meta'; const state=node.type==='CONCEPT'?(node.attrs?.concept_state||node.state):node.state;
  meta.textContent=`${node.type} | ${state} | ${node.id}`; details.appendChild(meta);
  if(node.attrs&&Object.keys(node.attrs).length){ const attrs=document.createElement('div'); attrs.className='meta'; attrs.textContent=JSON.stringify(node.attrs,null,2); attrs.style.whiteSpace='pre-wrap'; details.appendChild(attrs); }
  draw();
}
canvas.addEventListener('mousedown',event=>{ const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top; dragNode=hit(x,y); panning=!dragNode; lastX=x; lastY=y; canvas.classList.add('dragging'); if(dragNode)showDetails(dragNode); });
window.addEventListener('mousemove',event=>{
  const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top;
  if(dragNode){ const p=world(x,y); dragNode.x=p.x; dragNode.y=p.y; draw(); }
  else if(panning){ offsetX+=x-lastX; offsetY+=y-lastY; lastX=x; lastY=y; draw(); }
  else { const node=hit(x,y); tip.style.display=node?'block':'none'; if(node){ tip.textContent=`${node.label} (${node.type})`; tip.style.left=Math.min(width-350,x+14)+'px'; tip.style.top=Math.max(8,y-8)+'px'; } }
});
window.addEventListener('mouseup',()=>{ dragNode=null; panning=false; canvas.classList.remove('dragging'); });
canvas.addEventListener('wheel',event=>{ event.preventDefault(); const rect=canvas.getBoundingClientRect(),x=event.clientX-rect.left,y=event.clientY-rect.top,before=world(x,y),factor=event.deltaY<0?1.12:.89; scale=Math.min(4,Math.max(.04,scale*factor)); offsetX=x-before.x*scale; offsetY=y-before.y*scale; draw(); },{passive:false});
document.querySelectorAll('input').forEach(input=>input.addEventListener('input',draw));
document.getElementById('fit').addEventListener('click',fit);
document.getElementById('labels').addEventListener('click',()=>{ showLabels=!showLabels; draw(); });
window.addEventListener('resize',resize);
initialLayout(); resize(); fit();
</script>
</body>
</html>
'''
