#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import difflib
import html
import json
import re
from pathlib import Path


TOKEN_RE = re.compile(r"\s+|[\w]+|[^\w\s]", re.UNICODE)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a self-contained visual P0 pair-review page")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title")
    parser.add_argument("--mode", choices=("independent", "adjudication"), default="independent")
    args = parser.parse_args()
    rows = load_csv(args.input)
    if not rows:
        raise ValueError(f"annotation packet is empty: {args.input}")
    packet_id = f"{args.input.stem}:{len(rows)}:{rows[0]['pair_id']}:{rows[-1]['pair_id']}"
    title = args.title or (
        "双人分歧仲裁" if args.mode == "adjudication" else f"独立核查：{args.input.stem}"
    )
    payload = []
    for row in rows:
        item = dict(row)
        claim_left, claim_right = diff_html(row["original_claim"], row["changed_claim"])
        evidence_left, evidence_right = diff_html(row["original_evidence"], row["changed_evidence"])
        item["claim_diff_original"] = claim_left
        item["claim_diff_changed"] = claim_right
        item["evidence_diff_original"] = evidence_left
        item["evidence_diff_changed"] = evidence_right
        payload.append(item)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = PAGE_TEMPLATE.replace("__TITLE__", html.escape(title)).replace(
        "__PACKET_JSON__", safe_json({
            "title": title,
            "packetId": packet_id,
            "mode": args.mode,
            "source": str(args.input),
            "columns": list(rows[0]),
            "rows": payload,
        })
    )
    args.output.write_text(rendered, encoding="utf-8")
    print(f"wrote {len(rows)} review items to {args.output}")


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {
        "pair_id", "parent_id", "pool_role", "original_claim", "original_evidence",
        "changed_claim", "changed_evidence", "source_label", "proposed_target_label",
        "transform_type", "edit_category", "verified", "verification_method",
        "annotator_id", "notes",
    }
    missing = sorted(required - set(rows[0] if rows else {}))
    if missing:
        raise ValueError(f"{path} misses columns: {missing}")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise ValueError(f"duplicate pair_id in {path}")
    return rows


def diff_html(left: str, right: str) -> tuple[str, str]:
    left_tokens, right_tokens = TOKEN_RE.findall(left), TOKEN_RE.findall(right)
    matcher = difflib.SequenceMatcher(a=left_tokens, b=right_tokens, autojunk=False)
    left_parts, right_parts = [], []
    for operation, i1, i2, j1, j2 in matcher.get_opcodes():
        left_text = html.escape("".join(left_tokens[i1:i2]))
        right_text = html.escape("".join(right_tokens[j1:j2]))
        if operation == "equal":
            left_parts.append(left_text)
            right_parts.append(right_text)
        elif operation == "delete":
            left_parts.append(f'<mark class="removed">{left_text}</mark>')
        elif operation == "insert":
            right_parts.append(f'<mark class="added">{right_text}</mark>')
        else:
            left_parts.append(f'<mark class="removed">{left_text}</mark>')
            right_parts.append(f'<mark class="added">{right_text}</mark>')
    return "".join(left_parts), "".join(right_parts)


def safe_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False).replace("<", "\\u003c").replace(">", "\\u003e")


PAGE_TEMPLATE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --ink:#17202a; --muted:#64748b; --line:#d8e0ea; --panel:#fff; --bg:#f3f6fa;
  --yes:#087f5b; --yesbg:#dff7ed; --no:#c92a2a; --nobg:#ffe3e3; --blue:#2457c5; }
* { box-sizing:border-box; } body { margin:0; color:var(--ink); background:var(--bg);
  font:15px/1.55 system-ui,-apple-system,"Segoe UI","Noto Sans CJK SC",sans-serif; }
header { position:sticky; top:0; z-index:10; display:flex; gap:18px; align-items:center;
  padding:12px 22px; color:white; background:#18243a; box-shadow:0 2px 10px #0003; }
header h1 { font-size:18px; margin:0; white-space:nowrap; } .grow { flex:1; }
.progress { height:9px; background:#ffffff2b; border-radius:99px; overflow:hidden; margin-top:5px; }
.progress > div { height:100%; background:#63e6be; transition:width .2s; }
button, select, input, textarea { font:inherit; } button { border:1px solid transparent; border-radius:8px;
  padding:8px 13px; cursor:pointer; font-weight:650; } button:disabled { opacity:.45; cursor:not-allowed; }
.export { background:#ffd43b; color:#212529; } .secondary { background:white; border-color:var(--line); }
main { max-width:1500px; margin:18px auto; padding:0 18px 40px; display:grid;
  grid-template-columns:300px minmax(0,1fr); gap:18px; }
aside, .card { background:var(--panel); border:1px solid var(--line); border-radius:12px;
  box-shadow:0 3px 14px #1f29370d; }
aside { padding:18px; align-self:start; position:sticky; top:82px; max-height:calc(100vh - 100px); overflow:auto; }
aside h2 { font-size:15px; margin:0 0 8px; } aside ol { padding-left:20px; margin:8px 0 18px; }
.warning { padding:10px; background:#fff4e6; border-left:4px solid #f08c00; border-radius:5px; }
.field { display:grid; gap:5px; margin:12px 0; } .field label { font-size:12px; color:var(--muted); font-weight:700; }
.field input,.field select,.field textarea { width:100%; padding:9px; border:1px solid #bac5d3; border-radius:7px; background:white; }
.card { overflow:hidden; } .meta { padding:13px 18px; background:#f8fafc; border-bottom:1px solid var(--line);
  display:flex; gap:8px; flex-wrap:wrap; align-items:center; }
.badge { border-radius:99px; background:#e7eef9; padding:3px 9px; font-size:12px; font-weight:700; }
.badge.status { margin-left:auto; } .columns { display:grid; grid-template-columns:1fr 1fr; }
.column { padding:20px; min-width:0; } .column + .column { border-left:1px solid var(--line); }
.column h2 { margin:0 0 10px; font-size:16px; } .block { margin:0 0 20px; }
.block h3 { color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em; margin:0 0 5px; }
.text { white-space:pre-wrap; overflow-wrap:anywhere; padding:13px; background:#f8fafc; border-radius:8px; min-height:58px; }
mark { padding:1px 2px; border-radius:3px; } .removed { background:#ffc9c9; color:#9b1c1c; text-decoration:line-through; }
.added { background:#b2f2bb; color:#176b36; } .decision { border-top:1px solid var(--line); padding:18px 20px; }
.decision-grid { display:grid; grid-template-columns:1fr 1fr; gap:12px; }
.choice { padding:15px; border:2px solid var(--line); border-radius:10px; background:white; text-align:left; }
.choice.yes.selected { border-color:var(--yes); background:var(--yesbg); color:var(--yes); }
.choice.no.selected { border-color:var(--no); background:var(--nobg); color:var(--no); }
.reasons { display:flex; flex-wrap:wrap; gap:7px; margin:12px 0; } .reason { font-weight:500; font-size:13px;
  color:#7f1d1d; background:#fff5f5; border:1px solid #ffc9c9; }
.reason.selected { background:#c92a2a; color:white; } textarea { resize:vertical; min-height:70px; }
.nav { display:flex; align-items:center; gap:10px; margin-top:16px; } .nav button { background:#e8eef8; color:#243b64; }
.nav .next { margin-left:auto; background:var(--blue); color:white; } .counter { color:var(--muted); font-variant-numeric:tabular-nums; }
.agreement { display:none; margin:12px 0; padding:12px; border-radius:8px; background:#f1f5f9; }
.agreement.show { display:block; } .kbd { font:12px ui-monospace,monospace; padding:2px 5px; border:1px solid #bbc5d1; border-radius:4px; background:#fff; }
.locked { padding:10px 12px; margin-bottom:12px; color:#334155; background:#e2e8f0; border-radius:8px; }
@media (max-width:900px) { main { grid-template-columns:1fr; } aside { position:static; max-height:none; } .columns { grid-template-columns:1fr; }
  .column + .column { border-left:0; border-top:1px solid var(--line); } header { flex-wrap:wrap; } }
</style>
</head>
<body>
<header><h1 id="title"></h1><div class="grow"><div id="summary"></div><div class="progress"><div id="bar"></div></div></div>
<button class="export" id="export">导出 CSV</button></header>
<main>
<aside>
  <h2>唯一判断问题</h2>
  <p><strong>改写后的 claim–evidence 关系，是否仍严格对应页面给出的目标标签？</strong></p>
  <ol>
    <li><b>SUPPORTS</b>：证据必须能推出改写 claim；仅“不矛盾”不够。</li>
    <li><b>REFUTES</b>：证据必须明确否定改写 claim。</li>
    <li><b>NEI</b>：证据既不能支持，也不能否定。</li>
    <li>实体、时间、数量、否定范围、指代和决定性证据均不能被意外改变。</li>
    <li>句子必须自然且语义明确；双重否定、only/not 的范围含糊时判不通过。</li>
  </ol>
  <div class="warning">不要查看模型预测，不用外部世界知识补足证据。两位标注者在各自导出结果前不要讨论。</div>
  <div class="field"><label for="annotator">标注者 ID（必填且两人不同）</label><input id="annotator" placeholder="例如 reviewer_A"></div>
  <div class="field"><label for="filter">显示范围</label><select id="filter"><option value="all">全部</option><option value="pending">未完成</option><option value="yes">通过</option><option value="no">不通过</option><option value="disagreement">仅双人分歧</option></select></div>
  <p>快捷键：<span class="kbd">A</span> 通过，<span class="kbd">R</span> 不通过，<span class="kbd">←</span>/<span class="kbd">→</span> 切换。</p>
  <p id="storageNote" style="color:var(--muted)">结果自动保存在当前浏览器；正式交付请点击“导出 CSV”。</p>
</aside>
<section>
  <div class="card">
    <div class="meta"><span class="badge" id="role"></span><span class="badge" id="labels"></span><span class="badge" id="edit"></span><span class="badge" id="pair"></span><span class="badge status" id="status"></span></div>
    <div class="agreement" id="agreement"></div>
    <div class="columns">
      <div class="column"><h2>原始样本</h2><div class="block"><h3>Claim</h3><div class="text" id="originalClaim"></div></div><div class="block"><h3>Evidence</h3><div class="text" id="originalEvidence"></div></div></div>
      <div class="column"><h2>改写样本</h2><div class="block"><h3>Changed claim</h3><div class="text" id="changedClaim"></div></div><div class="block"><h3>Changed evidence</h3><div class="text" id="changedEvidence"></div></div></div>
    </div>
    <div class="decision">
      <div class="locked" id="locked" hidden>两位审核员结论一致，此条已锁定；仲裁员只需处理分歧项。</div>
      <div class="decision-grid"><button class="choice yes" id="yes">✓ 通过：目标标签仍成立</button><button class="choice no" id="no">✕ 不通过：关系或语义已改变</button></div>
      <div class="reasons" id="reasons"></div>
      <div class="field"><label for="notes">备注（不通过时建议写明原因）</label><textarea id="notes" placeholder="例如：American 不能严格推出 not British；新增 only 改变了量词范围。"></textarea></div>
      <div class="nav"><button id="prev">← 上一条</button><span class="counter" id="counter"></span><button id="next" class="next">下一条 →</button></div>
    </div>
  </div>
</section>
</main>
<script>
const packet = __PACKET_JSON__;
const reasons = ["目标标签不再成立","否定/only 范围错误","实体或指代改变","时间/数量/关系改变","决定性证据改变","语法不自然或含糊","changed field 不匹配","其他"];
const $ = id => document.getElementById(id);
let index = 0, visible = [];
const stateKey = id => `p0-review:${packet.packetId}:${id}`;
let decisions = {};
function annotator() { return $("annotator").value.trim(); }
function loadState() { decisions = JSON.parse(localStorage.getItem(stateKey(annotator())) || "{}"); render(); }
function saveState() { if (annotator()) localStorage.setItem(stateKey(annotator()), JSON.stringify(decisions)); }
function baseDecision(row) {
  const value = String(row.verified || "").toLowerCase();
  return value === "true" ? true : value === "false" ? false : null;
}
function decisionFor(row) { return decisions[row.pair_id] || { verified:baseDecision(row), notes:row.notes || "", reasons:[], annotator_id:row.annotator_id || "" }; }
function filteredRows() {
  const f = $("filter").value;
  return packet.rows.filter(row => { const d=decisionFor(row); if(f==="pending") return d.verified===null; if(f==="yes") return d.verified===true;
    if(f==="no") return d.verified===false; if(f==="disagreement") return row.review_status==="disagreement"; return true; });
}
function render() {
  visible = filteredRows(); if(index >= visible.length) index=Math.max(0,visible.length-1);
  const done = packet.rows.filter(r => decisionFor(r).verified !== null).length;
  $("title").textContent=packet.title; $("summary").textContent=`已完成 ${done}/${packet.rows.length} · 通过 ${packet.rows.filter(r=>decisionFor(r).verified===true).length} · 不通过 ${packet.rows.filter(r=>decisionFor(r).verified===false).length}`;
  $("bar").style.width=`${100*done/packet.rows.length}%`;
  if(!visible.length){ $("counter").textContent="当前筛选没有条目"; $("yes").disabled=$("no").disabled=true; return; }
  const row=visible[index], d=decisionFor(row), locked=packet.mode==="adjudication"&&row.review_status==="agreement";
  $("yes").disabled=$("no").disabled=locked; $("notes").disabled=locked; $("locked").hidden=!locked;
  $("role").textContent=row.pool_role; $("labels").textContent=`${row.source_label} → ${row.proposed_target_label}`; $("edit").textContent=`${row.edit_category} / ${row.transform_type}`; $("pair").textContent=row.pair_id;
  $("status").textContent=d.verified===true?"已通过":d.verified===false?"已拒绝":"未判断";
  $("originalClaim").innerHTML=row.claim_diff_original; $("changedClaim").innerHTML=row.claim_diff_changed; $("originalEvidence").innerHTML=row.evidence_diff_original; $("changedEvidence").innerHTML=row.evidence_diff_changed;
  $("yes").classList.toggle("selected",d.verified===true); $("no").classList.toggle("selected",d.verified===false); $("notes").value=d.notes||"";
  $("reasons").innerHTML=""; reasons.forEach(reason=>{const b=document.createElement("button"); b.className="reason"+(d.reasons.includes(reason)?" selected":""); b.textContent=reason; b.disabled=locked; b.onclick=()=>toggleReason(reason); $("reasons").appendChild(b);});
  const compare = row.verified_a !== undefined; $("agreement").classList.toggle("show",compare); $("agreement").innerHTML=compare?`<b>标注者 A：</b>${row.verified_a}　${escapeHtml(row.notes_a||"")}<br><b>标注者 B：</b>${row.verified_b}　${escapeHtml(row.notes_b||"")}`:"";
  $("counter").textContent=`${index+1}/${visible.length}（全包 ${packet.rows.indexOf(row)+1}/${packet.rows.length}）`; $("prev").disabled=index===0; $("next").disabled=index===visible.length-1;
}
function update(values) { if(!visible.length)return; const row=visible[index]; if(packet.mode==="adjudication"&&row.review_status==="agreement")return; const d=decisionFor(row); decisions[row.pair_id]={...d,...values,annotator_id:annotator()||d.annotator_id}; saveState(); render(); }
function choose(value){ if(!annotator()){alert("请先填写标注者 ID"); $("annotator").focus(); return;} update({verified:value}); }
function toggleReason(reason){ const d=decisionFor(visible[index]), r=d.reasons.includes(reason)?d.reasons.filter(x=>x!==reason):[...d.reasons,reason]; update({reasons:r,verified:false}); }
function move(delta){ if(!visible.length)return; index=Math.max(0,Math.min(visible.length-1,index+delta)); render(); window.scrollTo({top:0,behavior:"smooth"}); }
function escapeHtml(value){const div=document.createElement("div");div.textContent=value;return div.innerHTML;}
function csvCell(value){ const s=String(value??""); return /[",\r\n]/.test(s)?`"${s.replaceAll('"','""')}"`:s; }
function exportCsv(){
  if(!annotator()){alert("请填写标注者 ID");return;} const incomplete=packet.rows.filter(r=>decisionFor(r).verified===null).length;
  if(incomplete && !confirm(`还有 ${incomplete} 条未判断，仍然导出吗？`))return;
  const extra=["review_status","verified_a","annotator_a","notes_a","verified_b","annotator_b","notes_b"].filter(c=>packet.rows.some(r=>r[c]!==undefined));
  const columns=[...packet.columns.filter(c=>!extra.includes(c)),...extra,"rejection_reasons"];
  const lines=[columns.map(csvCell).join(",")]; packet.rows.forEach(row=>{const d=decisionFor(row); const out={...row,verified:d.verified===null?"":String(d.verified),verification_method:"human",annotator_id:d.annotator_id||annotator(),notes:d.notes||"",rejection_reasons:(d.reasons||[]).join(";")}; lines.push(columns.map(c=>csvCell(out[c])).join(","));});
  const blob=new Blob(["\ufeff"+lines.join("\r\n")],{type:"text/csv;charset=utf-8"}); const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`${packet.rows[0].pool_role}_${annotator()}_${packet.mode}.csv`;a.click();URL.revokeObjectURL(a.href);
}
$("yes").onclick=()=>choose(true); $("no").onclick=()=>choose(false); $("prev").onclick=()=>move(-1); $("next").onclick=()=>move(1); $("export").onclick=exportCsv;
$("notes").oninput=e=>update({notes:e.target.value}); $("filter").onchange=()=>{index=0;render();}; $("annotator").onchange=loadState;
document.addEventListener("keydown",e=>{if(e.target.matches("input,textarea,select"))return;if(e.key.toLowerCase()==="a")choose(true);if(e.key.toLowerCase()==="r")choose(false);if(e.key==="ArrowLeft")move(-1);if(e.key==="ArrowRight")move(1);});
render();
</script>
</body></html>
'''


if __name__ == "__main__":
    main()
