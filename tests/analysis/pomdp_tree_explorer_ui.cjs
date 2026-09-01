const fs = require('fs'),
  vm = require('vm'),
  assert = require('assert');
const path = require('path');
const assets = path.resolve(__dirname, '../../analysis/pomdp_tree_explorer');
const html = fs.readFileSync(path.join(assets, 'index.html'), 'utf8');
class Element {
  constructor(tag = 'div') {
    this.tag = tag;
    this.children = [];
    this.dataset = {};
    this.style = {};
    this.listeners = {};
    this._value = '';
    this.valueHistory = [];
    this.textContent = '';
  }
  set value(v) { this._value = v; this.valueHistory.push(Number(v)); }
  get value() { return this._value; }
  setAttribute(name, value) { this[name] = String(value); }
  getAttribute(name) { return this[name]; }
  setPointerCapture(pointerId) { this.capturedPointer = pointerId; }
  releasePointerCapture(pointerId) { this.releasedPointer = pointerId; }
  append(...items) {
    this.children.push(...items);
    if (this.tag === 'select' && this.value === '' && items[0]) this.value = String(items[0].value);
  }
  replaceChildren(...items) {
    this.children = [];
    this.value = '';
    this.append(...items);
  }
  addEventListener(event, fn) {
    this.listeners[event] = fn;
  }
  set open(v) {
    this._open = v;
    if (this.listeners.toggle) this.listeners.toggle();
  }
  get open() {
    return this._open;
  }
  click() {
    if (this.onclick) this.onclick();
  }
}
const elements = {};
for (const match of html.matchAll(/<(\w+)[^>]*\bid="([^"]+)"/g)) elements[match[2]] = new Element(match[1]);
const child = {
    state: {
      environment_state: 'HOLDING'
    }
  },
  parent = {
    state: {
      environment_state: 'READY'
    }
  },
  action = {
    name: 'PickCube'
  };
const trace = [{
  event: 'node',
  node: 0,
  environment_state: parent,
  belief_state: {},
  summed_cost: 0,
  horizon: 1
}, {
  event: 'sample',
  node: 0,
  theta: {
    p: .5
  },
  pomdp_value: .4
}, {
  event: 'stop_value',
  node: 0,
  value: .4
}, {
  event: 'node',
  node: 3,
  environment_state: child,
  belief_state: {},
  summed_cost: 1,
  horizon: 0
}, {
  event: 'stop_value',
  node: 3,
  value: .6
}, {
  event: 'choice',
  node: 3,
  value: .6,
  action: 'STOP'
}, {
  event: 'branch',
  node: 0,
  action,
  successor: child,
  belief_state: {},
  summed_cost: 1,
  horizon: 0,
  sampled_cost: 1,
  probability: 1,
  successor_value: .6,
  contribution: .6
}, {
  event: 'action_value',
  node: 0,
  action,
  value: .6
}, {
  event: 'choice',
  node: 0,
  action,
  value: .6
}];
const record = {
  cycle: 0,
  decision: 1,
  action,
  horizon: 1,
  model: {
    hard_budget: 150
  },
  competences: {
    PickCube: .5
  },
  search_duration_seconds: 1,
  search: trace
};
const context = vm.createContext({
  console,
  setTimeout,
  clearTimeout,
  setInterval,
  clearInterval,
  Blob,
  URL,
  document: {
    getElementById: id => {
      assert(elements[id], id);
      return elements[id];
    },
    createElement: tag => new Element(tag)
    ,createElementNS: (_namespace, tag) => new Element(tag)
  },
  TextDecoder,
  fetch: async url => {
    const value = url === '/api/seeds' ? {
      seeds: [0]
    } : url.startsWith('/api/index') ? {
      decisions: [{
        index: 0,
        cycle: 0,
        decision: 1,
        action
      }, {
        index: 1,
        cycle: 0,
        decision: 2,
        action: 'STOP'
      }]
    } : JSON.parse(JSON.stringify(record));
    if (!url.startsWith('/api/decision')) return {ok: true, json: async () => value};
    const bytes = Buffer.from(JSON.stringify(value));
    let part = 0;
    return {
      ok: true,
      headers: {get: key => key.toLowerCase() === 'x-uncompressed-content-length' ? String(bytes.length) : null},
      body: {getReader: () => ({read: async () => part++ === 0 ? {done: false, value: bytes.subarray(0, Math.floor(bytes.length / 2))} : part === 2 ? {done: false, value: bytes.subarray(Math.floor(bytes.length / 2))} : {done: true}})}
    };
  }
});
vm.runInContext(fs.readFileSync(path.join(assets, 'explorer.js'), 'utf8'), context);
(async () => {
  for (let i = 0; i < 10; i++) await new Promise(setImmediate);
  assert.match(elements.metadata.textContent, /2 unique states/);
  assert(elements['loading-progress'].valueHistory.some(value => value > 0 && value < 1));
  assert.equal(elements['loading-progress'].value, 1);
  const descendants = element => [element, ...element.children.flatMap(descendants)];
  const rendered = descendants(elements.tree);
  assert(rendered.some(e => e.dataset.kind === 'state'));
  assert(rendered.some(e => e.dataset.kind === 'action'));
  assert(rendered.some(e => e.dataset.kind === 'chance'));
  assert(rendered.some(e => e.tag === 'line' || e.tag === 'path'));
  const y = element => Number(element.transform.match(/translate\([^ ]+ ([^)]+)\)/)[1]);
  const rootState = rendered.find(e => e.dataset.kind === 'state');
  const actionNode = rendered.find(e => e.dataset.kind === 'action');
  const chanceNode = rendered.find(e => e.dataset.kind === 'chance');
  assert(y(rootState) < y(actionNode));
  assert(y(actionNode) < y(chanceNode));
  const layer = elements.tree.children[0], originalTransform = layer.transform;
  const initialScale = Number(originalTransform.match(/scale\(([^)]+)\)/)[1]);
  assert(initialScale >= .55, 'initial tree must remain readable instead of fitting every leaf');
  elements.tree.listeners.pointerdown({pointerId: 7, clientX: 10, clientY: 10});
  elements.tree.listeners.pointermove({pointerId: 7, clientX: 35, clientY: 45});
  assert.notEqual(layer.transform, originalTransform);
  assert.equal(elements.tree.capturedPointer, 7);
  elements.start.click();
  assert(descendants(elements.tree).some(e => /evaluating/.test(e.textContent)));
  elements.forward.click();
  assert.match(elements['event-info'].textContent, /stop_value/);
  elements.finish.click();
  assert(descendants(elements.tree).some(e => /PickCube/.test(e.textContent)));
  elements.next.click();
  for (let i = 0; i < 10; i++) await new Promise(setImmediate);
  assert.equal(elements.decision.value, 1);
  assert.equal(elements.next.disabled, true);
  elements.previous.click();
  for (let i = 0; i < 10; i++) await new Promise(setImmediate);
  assert.equal(elements.previous.disabled, true);
  console.log('UI checks passed: streamed loading, branching SVG tree, replay, previous/next decision.');
})().catch(e => {
  console.error(e);
  process.exitCode = 1;
});
