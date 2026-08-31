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
    this.value = '';
    this.textContent = '';
  }
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
  },
  fetch: async url => ({
    ok: true,
    json: async () => url === '/api/seeds' ? {
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
    } : JSON.parse(JSON.stringify(record))
  })
});
vm.runInContext(fs.readFileSync(path.join(assets, 'explorer.js'), 'utf8'), context);
(async () => {
  for (let i = 0; i < 10; i++) await new Promise(setImmediate);
  assert.match(elements.metadata.textContent, /2 unique states/);
  assert.equal(elements.tree.children.length, 1);
  const root = elements.tree.children[0];
  assert.equal(root.open, true);
  const actionDetail = root.children.find(e => e.children?.[0]?.textContent.startsWith('PickCube · Q='));
  assert(actionDetail);
  actionDetail.open = true;
  const branch = actionDetail.children[1];
  branch.open = true;
  assert(branch.children.some(e => e.children?.[0]?.textContent.startsWith('State 3')));
  elements.start.click();
  assert.match(elements.tree.children[0].children[0].textContent, /evaluating/);
  elements.forward.click();
  assert.match(elements['event-info'].textContent, /stop_value/);
  elements.finish.click();
  assert.match(elements.tree.children[0].children[0].textContent, /PickCube/);
  elements.next.click();
  for (let i = 0; i < 10; i++) await new Promise(setImmediate);
  assert.equal(elements.decision.value, 1);
  assert.equal(elements.next.disabled, true);
  elements.previous.click();
  for (let i = 0; i < 10; i++) await new Promise(setImmediate);
  assert.equal(elements.previous.disabled, true);
  console.log('UI checks passed: initial load, root values, lazy chance/child expansion, replay, previous/next decision.');
})().catch(e => {
  console.error(e);
  process.exitCode = 1;
});
