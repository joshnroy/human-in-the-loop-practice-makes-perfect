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
  requestId = 0;
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
function raw(parent, label, value) {
  const d = document.createElement('details'),
    s = document.createElement('summary');
  s.textContent = label;
  d.append(s);
  d.addEventListener('toggle', () => {
    if (!d.open || d.dataset.built) return;
    d.dataset.built = '1';
    const pre = document.createElement('pre');
    pre.textContent = JSON.stringify(value, null, 2);
    d.append(pre);
  });
  parent.append(d);
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
      keys.set(key(event.environment_state, event.belief_state, event.summed_cost, event.horizon), event.node);
    }
    const node = nodes.get(event.node);
    if (node) {
      node.events.push(event);
      if (event.action && event.action !== 'STOP' && ['branch', 'action_value'].includes(event.event)) {
        const akey = JSON.stringify(event.action);
        if (!node.actions.has(akey)) node.actions.set(akey, {
          action: event.action,
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
  for (const e of values) bar($('values'), e.event === 'stop_value' ? 'STOP' : name(e.action), e.value, max, e.event === 'stop_value');
  $('competences').replaceChildren();
  for (const [skill, v] of Object.entries(record.competences ?? {})) bar($('competences'), skill, v, 1);
  $('metadata').textContent = 'Cycle ' + record.cycle + ' · decision ' + record.decision + ' · chose ' + name(record.action) + ' · H=' + record.horizon + ' · C=' + (root?.info.summed_cost ?? '?') + ' · B=' + (record.model?.hard_budget ?? 'linear') + ' · ' + number(record.search_duration_seconds) + ' seconds · ' + nodes.size + ' unique states';
}
function drawState(parent, id, cutoff, open = false) {
  const node = nodes.get(id);
  if (!node || node.info.traceIndex > cutoff) return;
  const visible = node.events.filter(e => e.traceIndex <= cutoff),
    choice = visible.find(e => e.event === 'choice'),
    stop = visible.find(e => e.event === 'stop_value');
  const d = document.createElement('details'),
    s = document.createElement('summary');
  const atoms = node.info.environment_state?.atoms ?? [];
  s.textContent = 'State ' + id + ' · ' + (atoms.length ? atoms.join(', ') : '∅') + ' · h=' + node.info.horizon + ' · C=' + node.info.summed_cost + ' · V=' + number(choice?.value) + (choice ? ' → ' + name(choice.action) : ' · evaluating');
  d.append(s);
  parent.append(d);
  d.addEventListener('toggle', () => {
    if (!d.open || d.dataset.built) return;
    d.dataset.built = '1';
    raw(d, 'Physical state and belief', node.info);
    raw(d, 'STOP = ' + number(stop?.value) + ' · θ samples (' + visible.filter(e => e.event === 'sample').length + ')', visible.filter(e => ['sample', 'stop_value'].includes(e.event)));
    for (const action of node.actions.values()) {
      const events = action.events.filter(e => e.traceIndex <= cutoff);
      if (!events.length) continue;
      const v = events.find(e => e.event === 'action_value'),
        ad = document.createElement('details'),
        as = document.createElement('summary');
      as.textContent = name(action.action) + ' · Q=' + number(v?.value);
      if (choice && JSON.stringify(choice.action) === JSON.stringify(action.action)) as.className = 'selected';
      ad.append(as);
      d.append(ad);
      ad.addEventListener('toggle', () => {
        if (!ad.open || ad.dataset.built) return;
        ad.dataset.built = '1';
        for (const b of events.filter(e => e.event === 'branch')) {
          const bd = document.createElement('details'),
            bs = document.createElement('summary');
          bs.textContent = 'p=' + number(b.probability) + ' × V=' + number(b.successor_value) + ' = ' + number(b.contribution) + ' · cost ' + b.sampled_cost;
          bd.append(bs);
          ad.append(bd);
          bd.addEventListener('toggle', () => {
            if (!bd.open || bd.dataset.built) return;
            bd.dataset.built = '1';
            raw(bd, 'Full chance outcome', b);
            const child = keys.get(key(b.successor, b.belief_state, b.summed_cost, b.horizon));
            if (child === undefined) {
              const p = document.createElement('p');
              p.textContent = 'Successor not found in the recorded trace';
              bd.append(p);
            } else drawState(bd, child, cutoff, true);
          });
        }
      });
    }
    if (choice) raw(d, 'Choice and reason', choice);else {
      const p = document.createElement('p');
      p.className = 'pending';
      p.textContent = 'This node has not returned at this replay position.';
      d.append(p);
    }
  });
  d.open = open;
}
function drawReplay() {
  if (!record) return;
  const cutoff = milestones[position],
    event = record.search[cutoff];
  $('timeline').value = position;
  $('event-position').textContent = position + 1 + ' / ' + milestones.length;
  $('event-info').textContent = event ? 'Trace event ' + (cutoff + 1) + ' / ' + record.search.length + ' · ' + event.event + ' · node ' + event.node + (event.elapsed_seconds !== undefined ? ' · ' + number(event.elapsed_seconds) + ' s' : '') : '';
  $('tree').replaceChildren();
  $('active').replaceChildren();
  drawState($('tree'), 0, cutoff, true);
  if (event && event.node !== 0) {
    const h = document.createElement('h2');
    h.textContent = 'Active recorded node';
    $('active').append(h);
    drawState($('active'), event.node, cutoff, true);
  }
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
  $('active').replaceChildren();
  message('Loading one complete search trace…');
  try {
    const result = await get('/api/decision?seed=' + encodeURIComponent($('seed').value) + '&index=' + encodeURIComponent($('decision').value));
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
  $('active').replaceChildren();
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
