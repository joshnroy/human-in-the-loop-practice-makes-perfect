'use strict';

const $ = id => document.getElementById(id);
const name = a => typeof a === 'string' ? a : a?.name ?? JSON.stringify(a);
const number = v => typeof v === 'number' ? v.toFixed(6) : 'pending';
const key = (state, belief, cost, horizon) => JSON.stringify([state, belief, cost, horizon]);
let summaries = [],
  record = null,
  nodes = new Map(),
  keys = new Map(),
  milestones = [],
  position = 0,
  clock = null,
  requestId = 0,
  treeLayer = null,
  treeView = {x: 0, y: 0, scale: 1},
  dragPoint = null;
function stopPlayback() {
  if (clock !== null) clearInterval(clock);
  clock = null;
  $('play').textContent = 'Play search';
}
function busy(value) {
  for (const id of ['previous', 'next', 'decision', 'download', 'start', 'back', 'play', 'forward', 'finish', 'timeline']) $(id).disabled = value;
}
function message(text) {
  $('status').textContent = text;
}
async function get(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error(await response.text());
  return response.json();
}
async function getDecision(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error(await response.text());
  if (!response.body?.getReader) return response.json();
  const total = Number(response.headers.get('X-Uncompressed-Content-Length')) || 0,
    reader = response.body.getReader(), chunks = [];
  let loaded = 0;
  $('loading').hidden = false;
  $('loading-progress').value = 0;
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.length;
    const ratio = total ? Math.min(loaded / total, 1) : 0;
    $('loading-progress').value = ratio;
    $('loading-percent').textContent = total ? Math.round(ratio * 100) + '%' : (loaded / 1048576).toFixed(1) + ' MB';
    $('loading-label').textContent = 'Loading search JSON · ' + (loaded / 1048576).toFixed(1) + (total ? ' / ' + (total / 1048576).toFixed(1) + ' MB' : ' MB');
  }
  const bytes = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
  $('loading-progress').value = 1;
  $('loading-percent').textContent = '100%';
  $('loading-label').textContent = 'Parsing loaded search JSON…';
  return JSON.parse(new TextDecoder().decode(bytes));
}
function resolve(value) {
  if (!value || typeof value !== 'object' || value.$ref === undefined || !value.$kind) return value;
  return record?.search_interns?.[value.$kind]?.[String(value.$ref)] ?? value;
}
function field(event, fieldName) {
  return resolve(event?.[fieldName]);
}
function expanded(event) {
  return Object.fromEntries(Object.entries(event).map(([fieldName, value]) => [fieldName, resolve(value)]));
}
function bar(parent, label, value, max, isStop = false) {
  const row = document.createElement('div');
  row.className = 'label';
  row.textContent = label + '  ' + number(value);
  const line = document.createElement('div');
  line.className = 'bar' + (isStop ? ' stop' : '');
  line.style.width = 100 * Math.abs(value) / (max || 1) + '%';
  parent.append(row, line);
}
function indexTrace() {
  nodes = new Map();
  keys = new Map();
  milestones = [];
  record.search.forEach((event, i) => {
    event.traceIndex = i;
    if (event.event === 'node') {
      nodes.set(event.node, {
        info: event,
        events: [],
        actions: new Map()
      });
      keys.set(key(field(event, 'environment_state'), field(event, 'belief_state'), event.summed_cost, event.horizon), event.node);
    }
    const node = nodes.get(event.node);
    if (node) {
      node.events.push(event);
      const action = field(event, 'action');
      if (action && action !== 'STOP' && ['branch', 'action_value'].includes(event.event)) {
        const akey = JSON.stringify(action);
        if (!node.actions.has(akey)) node.actions.set(akey, {
          action,
          events: []
        });
        node.actions.get(akey).events.push(event);
      }
    }
    if (event.event !== 'sample') milestones.push(i);
  });
  if (!milestones.length) milestones = [0];
}
function drawSummary() {
  const root = nodes.get(0),
    values = root ? root.events.filter(e => e.event === 'stop_value' || e.event === 'action_value') : [];
  $('values').replaceChildren();
  const max = Math.max(...values.map(e => Math.abs(e.value)), .001);
  for (const e of values) bar($('values'), e.event === 'stop_value' ? 'STOP' : name(field(e, 'action')), e.value, max, e.event === 'stop_value');
  $('competences').replaceChildren();
  for (const [skill, v] of Object.entries(record.competences ?? {})) bar($('competences'), skill, v, 1);
  $('metadata').textContent = 'Cycle ' + record.cycle + ' · decision ' + record.decision + ' · chose ' + name(record.action) + ' · H=' + record.horizon + ' · C=' + (root?.info.summed_cost ?? '?') + ' · B=' + (record.model?.hard_budget ?? 'linear') + ' · ' + number(record.search_duration_seconds) + ' seconds · ' + nodes.size + ' unique states';
}
function svg(tag, attributes = {}) {
  const element = document.createElementNS('http://www.w3.org/2000/svg', tag);
  for (const [attribute, value] of Object.entries(attributes)) element.setAttribute(attribute, value);
  return element;
}
function treeData(id, cutoff, seen = new Set(), budget = {left: 120, truncated: false}) {
  const node = nodes.get(id);
  if (!node || node.info.traceIndex > cutoff) return null;
  if (budget.left-- <= 0) { budget.truncated = true; return null; }
  const visible = node.events.filter(event => event.traceIndex <= cutoff), choice = visible.find(event => event.event === 'choice');
  const state = {kind: 'state', id: 's' + id, label: 'S' + id + ' · V=' + number(choice?.value) + (choice ? ' → ' + name(field(choice, 'action')) : ' · evaluating'), data: expanded(node.info), children: []};
  if (seen.has(id)) { state.label += ' · cached'; return state; }
  seen.add(id);
  for (const action of node.actions.values()) {
    const events = action.events.filter(event => event.traceIndex <= cutoff);
    if (!events.length) continue;
    const value = events.find(event => event.event === 'action_value');
    const actionNode = {kind: 'action', label: name(action.action) + ' · Q=' + number(value?.value), data: {action: action.action, events}, children: []};
    for (const branch of events.filter(event => event.event === 'branch')) {
      const chance = {kind: 'chance', label: 'p=' + number(branch.probability) + ' · Δ=' + number(branch.contribution), data: expanded(branch), children: []};
      const child = keys.get(key(field(branch, 'successor'), field(branch, 'belief_state'), branch.summed_cost, branch.horizon));
      const childNode = child === undefined ? null : treeData(child, cutoff, seen, budget);
      if (childNode) chance.children.push(childNode);
      actionNode.children.push(chance);
    }
    state.children.push(actionNode);
  }
  return state;
}
function drawDecisionTree(cutoff) {
  const budget = {left: 120, truncated: false}, root = treeData(0, cutoff, new Set(), budget);
  $('tree').replaceChildren();
  if (!root) return;
  const flat = [], links = [];
  let nextLeaf = 0;
  function layout(node, depth = 0) {
    node.depth = depth;
    flat.push(node);
    for (const child of node.children) { links.push([node, child]); layout(child, depth + 1); }
    node.x = node.children.length ? node.children.reduce((total, child) => total + child.x, 0) / node.children.length : 80 + nextLeaf++ * 190;
    node.y = 55 + depth * 115;
  }
  layout(root);
  $('tree').setAttribute('viewBox', '0 0 1200 700');
  const layer = svg('g');
  treeLayer = layer;
  treeView = {x: 600 - root.x * .8, y: 20, scale: .8};
  $('tree').append(layer);
  transformTree();
  for (const [from, to] of links) layer.append(svg('line', {x1: from.x, y1: from.y, x2: to.x, y2: to.y, class: 'tree-edge'}));
  for (const node of flat) {
    const group = svg('g', {transform: 'translate(' + node.x + ' ' + node.y + ')', class: 'tree-node ' + node.kind});
    group.dataset.kind = node.kind;
    if (node.kind === 'state') group.append(svg('circle', {r: 13}));
    else if (node.kind === 'chance') group.append(svg('path', {d: 'M 0 -13 L 13 0 L 0 13 L -13 0 Z'}));
    else group.append(svg('rect', {x: -18, y: -12, width: 36, height: 24, rx: 6}));
    const label = svg('text', {x: 20, y: 5}); label.textContent = node.label; group.append(label);
    group.onclick = () => { $('node-info').textContent = JSON.stringify(node.data, null, 2); };
    layer.append(group);
  }
  $('node-info').textContent = 'Showing ' + flat.filter(node => node.kind === 'state').length + ' of ' + nodes.size + ' recorded states' + (budget.truncated ? '. Pan horizontally or replay earlier computations to inspect a smaller frontier.' : '.') + ' Click a node for its logged values.';
}
function transformTree() {
  if (treeLayer) treeLayer.setAttribute('transform', 'translate(' + treeView.x + ' ' + treeView.y + ') scale(' + treeView.scale + ')');
}
$('tree').addEventListener('wheel', event => {
  event.preventDefault();
  treeView.scale = Math.max(.15, Math.min(4, treeView.scale * (event.deltaY < 0 ? 1.12 : .89)));
  transformTree();
});
$('tree').addEventListener('pointerdown', event => {
  $('tree').setPointerCapture(event.pointerId);
  dragPoint = {x: event.clientX, y: event.clientY, pointerId: event.pointerId};
});
$('tree').addEventListener('pointermove', event => {
  if (!dragPoint) return;
  treeView.x += event.clientX - dragPoint.x;
  treeView.y += event.clientY - dragPoint.y;
  dragPoint = {x: event.clientX, y: event.clientY};
  transformTree();
});
$('tree').addEventListener('pointerup', event => {
  if (dragPoint) $('tree').releasePointerCapture(event.pointerId);
  dragPoint = null;
});
$('tree').addEventListener('pointerleave', () => { dragPoint = null; });
function drawReplay() {
  if (!record) return;
  const cutoff = milestones[position],
    event = record.search[cutoff];
  $('timeline').value = position;
  $('event-position').textContent = position + 1 + ' / ' + milestones.length;
  $('event-info').textContent = event ? 'Trace event ' + (cutoff + 1) + ' / ' + record.search.length + ' · ' + event.event + ' · node ' + event.node + (event.elapsed_seconds !== undefined ? ' · ' + number(event.elapsed_seconds) + ' s' : '') : '';
  drawDecisionTree(cutoff);
}
function seek(next) {
  position = Math.max(0, Math.min(milestones.length - 1, next));
  drawReplay();
}
async function selectDecision() {
  stopPlayback();
  const generation = ++requestId;
  busy(true);
  record = null;
  nodes.clear();
  keys.clear();
  $('tree').replaceChildren();
  $('loading').hidden = false;
  message('Loading one complete search trace…');
  try {
    const result = await getDecision('/api/decision?seed=' + encodeURIComponent($('seed').value) + '&index=' + encodeURIComponent($('decision').value));
    if (generation !== requestId) return;
    record = result;
    indexTrace();
    drawSummary();
    $('timeline').max = milestones.length - 1;
    seek(milestones.length - 1);
    busy(false);
    $('previous').disabled = +$('decision').value === 0;
    $('next').disabled = +$('decision').value === summaries.length - 1;
    message('Complete saved search loaded. Navigate decisions or replay its structural events.');
    $('loading').hidden = true;
  } catch (error) {
    if (generation === requestId) message('Could not load search: ' + error.message);
  }
}
async function selectSeed() {
  stopPlayback();
  const generation = ++requestId;
  busy(true);
  record = null;
  nodes.clear();
  keys.clear();
  $('tree').replaceChildren();
  message('Indexing this seed’s decision log. The first visit may take a moment…');
  try {
    const result = await get('/api/index?seed=' + encodeURIComponent($('seed').value));
    if (generation !== requestId) return;
    summaries = result.decisions;
    $('decision').replaceChildren();
    for (const entry of summaries) {
      const o = document.createElement('option');
      o.value = entry.index;
      o.textContent = 'Cycle ' + entry.cycle + ' · ' + entry.decision + ' · ' + name(entry.action);
      $('decision').append(o);
    }
    if (summaries.length) await selectDecision();else message('No completed decisions recorded.');
  } catch (error) {
    if (generation === requestId) message('Could not index seed: ' + error.message);
  }
}
$('seed').onchange = selectSeed;
$('decision').onchange = selectDecision;
$('previous').onclick = () => {
  $('decision').value = +$('decision').value - 1;
  selectDecision();
};
$('next').onclick = () => {
  $('decision').value = +$('decision').value + 1;
  selectDecision();
};
$('start').onclick = () => {
  stopPlayback();
  seek(0);
};
$('finish').onclick = () => {
  stopPlayback();
  seek(milestones.length - 1);
};
$('back').onclick = () => {
  stopPlayback();
  seek(position - 1);
};
$('forward').onclick = () => {
  stopPlayback();
  seek(position + 1);
};
$('timeline').oninput = () => {
  stopPlayback();
  seek(+$('timeline').value);
};
$('play').onclick = () => {
  if (clock !== null) {
    stopPlayback();
    return;
  }
  if (position === milestones.length - 1) seek(0);
  $('play').textContent = 'Pause';
  clock = setInterval(() => {
    seek(position + 1);
    if (position === milestones.length - 1) stopPlayback();
  }, 180);
};
$('download').onclick = () => {
  if (!record) return;
  const blob = new Blob([JSON.stringify(record, (k, v) => k === 'traceIndex' ? undefined : v)], {
      type: 'application/json'
    }),
    url = URL.createObjectURL(blob),
    a = document.createElement('a');
  a.href = url;
  a.download = 'seed_' + $('seed').value + '_cycle_' + record.cycle + '_decision_' + record.decision + '.json';
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};
busy(true);
get('/api/seeds').then(data => {
  for (const seed of data.seeds) {
    const o = document.createElement('option');
    o.value = seed;
    o.textContent = seed;
    $('seed').append(o);
  }
  if (data.seeds.length) selectSeed();else message('No logged seeds found.');
}).catch(error => message(error.message));
