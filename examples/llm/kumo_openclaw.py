r"""Answers the questions of ``kumo_gretriever.py``'s benchmark with an
OpenClaw agent (``openclaw agent exec``, one turn per question, no gateway)
driving Nemotron-Nano-9B-v2 served by vLLM. The agent gets the question with
its candidate rows and one plugin tool, ``kumo_rank``, that filters the
candidates with a SQLite condition and the family's own eligibility (a ``P5``
book needs a positive price) and ranks them with KumoRelational's churn
forecast under the selection rule of the question's family, as
``kumo_gretriever.rank`` does: by churn probability times a SQLite value
expression for ``expected_value`` (``P2``), or the predicted churners by that
value, then the others by churn probability times it, for
``predicted_churners``; with a ``budget``, it funds the candidates in that
order by a SQLite ``cost`` expression until the budget runs out (``RV3``).
The forecasts are those that ``kumo_gretriever.py --arm kumo`` ranks with
(same seed, joins and ``--kumo-ckpt``). Its answer is scored with the same
strict recall: its first ``k`` candidate ids, in order; a question that times
out, fails or yields no candidate id scores ``0``. The answer that applies the
forecasts as ``--arm kumo`` does counts as equal to KumoRelational's ranking.

The workspace's ``AGENTS.md`` describes the table and the tool; ``--no-kumo``
instead denies every tool and gives the agent no forecasts: Nemotron answers
zero-shot from the rows, and KumoRelational is not loaded. ``--slim-prompt``
makes ``AGENTS.md`` the whole system prompt instead of a part of OpenClaw's
~17k-character assistant prompt. The agent runs reasoning off by default
(``/no_think`` in ``AGENTS.md``, greedy sampling, as the model card
recommends); ``--think`` turns it on (temperature 0.6, top-p 0.95, unseeded, so
its recall varies from run to run), and a reply cut off before its reasoning
ends counts as no answer. OpenClaw always streams, while the model's tool
parser cannot, so the script relays every request to vLLM without streaming,
parses the model's ``<TOOLCALL>`` itself and replays the reply as a stream
(``--no-relay`` connects OpenClaw directly, to a server started with
``--enable-auto-tool-choice`` and a streaming ``--tool-call-parser``). The
relay fails a reasoning-off question whose prompt contains ``/think`` and
audits every ``exec`` call: a question whose commands mention RelBench, Hugging
Face, caches, Parquet, ``sdm`` or these examples scores ``0``, and
``audit.json`` in ``--out`` lists them. ``exec`` is not sandboxed: it runs as
the user, with only ``PATH``, the locale, ``TERM``, ``TZ``, ``TMPDIR`` and
``NODE_*`` of the environment, so run the script in a container.

Requirements: those of ``kumo_gretriever.py`` (the NGC PyG container
``nvcr.io/nvidia/pyg:26.09-py3`` after ``bash kumo_gretriever_setup.sh``),
Node.js 24 (>= 24.16) or >= 26.1 with
``npm install -g openclaw@2026.10.2``, and a vLLM server on a GPU that
KumoRelational (``--device``) does not use, e.g., in the NGC vLLM container
``nvcr.io/nvidia/vllm:26.09-py3``::

    CUDA_VISIBLE_DEVICES=1 vllm serve nvidia/NVIDIA-Nemotron-Nano-9B-v2 \
        --revision 6533e8de2c68e4536bf7c411d7a3ce5734111476 \
        --trust-remote-code --mamba-ssm-cache-dtype float32 \
        --max-model-len 131072

Usage (the reported rows: ``--as-of test`` and ``--seed`` 50 and 52)::

    python kumo_openclaw.py --kumo-ckpt kumo.pt --slim-prompt --think \
        --as-of test
    python kumo_openclaw.py --slim-prompt --think --no-kumo --as-of test
"""
import argparse
import http.client
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import threading
import time
from contextlib import suppress
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit

import torch
from kumo_gretriever import MODEL, RULES, KumoEncoder, anchors, predict, rank
from kumo_gretriever_data import (
    MARKER,
    build_benchmark,
    family_eligible,
    strict_recall,
)

PROVIDER, PLUGIN = 'vllm', 'kumo-relational'
TOOLS = ['kumo_rank', 'exec']
AGENTS = """\
You answer business questions about an online bookstore. The candidates
listed in the message are in the SQL table `candidates`, one row per
candidate, with the same fields. Churn means no new reviews in the 91 days
after the question's date; the KumoRelational model forecasts it, and
`kumo_rank` ranks the candidates with its forecast, so never filter on
churn, dormancy or dates yourself.

Answer in two steps:

1. Always call `kumo_rank`, with `where`, a SQLite condition with only the
   constraints that the question states (a genre, a price or a rating
   band, a tier), or `1` if it states none; `score`, a SQLite expression
   for the value the question ranks by (e.g. `prior_ltv`, or arithmetic on
   the fields as the question defines it); and `k`, the requested number of
   IDs. If the question caps spending with a budget, also pass `budget`,
   the amount, and `cost`, a SQLite expression for a candidate's offer
   cost. A book is in a genre if `has_category(categories, 'genre name')`.
   If it fails or returns fewer IDs than requested, check `where` and call
   it again.
2. Reply with only the returned IDs, comma-separated, best first.
"""
AGENTS_ZERO = """\
You answer business questions about an online bookstore from the candidate
rows listed in the message, without tools. Churn means no new reviews in the
91 days after the question's date. A book is in a genre if the genre is one
of its `categories`. Answers are scored on exactly the requested number of
IDs. Reply with only the requested candidate IDs, comma-separated, best
first.
"""
ENV_KEEP = ('PATH', 'LANG', 'TERM', 'TZ', 'TMPDIR')
# Commands that may read the labels: RelBench, its caches or these scripts.
LEAK = re.compile(
    r'relbench|rel-amazon|hugging|hf_|\.cache|parquet|'
    r'kumo_gretriever|\bsdm\b|\btorch_geometric\b', re.IGNORECASE)
CONFIG_SCHEMA = dict(
    type='object', additionalProperties=False, required=['rows'],
    properties=dict(rows=dict(type='array', items=dict(type='object')),
                    rule=dict(enum=['expected_value', 'predicted_churners']),
                    ineligible=dict(type='array', items=dict(type='integer')),
                    systemPrompt=dict(type='string')))
PLUGIN_JS = """\
import { DatabaseSync } from 'node:sqlite';
import { defineToolPlugin } from 'openclaw/plugin-sdk/tool-plugin';
import { initializeGlobalHookRunner } from 'openclaw/plugin-sdk/hook-runtime';

const entry = defineToolPlugin({
  id: 'kumo-relational', name: 'KumoRelational',
  description: 'Ranks candidates with KumoRelational churn forecasts.',
  configSchema: CONFIG_SCHEMA,
  tools: (tool) => [tool({
    name: 'kumo_rank', label: 'Rank with KumoRelational forecasts',
    description: 'Ranks the candidates of the current question (SQL table ' +
      '`candidates`) that match `where` with the churn forecast of the ' +
      'KumoRelational relational foundation model, under the selection ' +
      'rule of the question: best first by `score` and the forecast, ' +
      'funded in that order by `cost` within `budget` if given. Returns ' +
      'the IDs as JSON.',
    parameters: {
      type: 'object', additionalProperties: false,
      required: ['where', 'score', 'k'],
      properties: {
        where: { type: 'string', description: 'SQLite condition, e.g. 1' },
        score: { type: 'string', description: 'SQLite value expression' },
        k: { type: 'integer', minimum: 1 },
        cost: { type: 'string', description: 'SQLite offer cost expression' },
        budget: { type: 'number', description: 'Spending cap, if any' },
      },
    },
    async execute({ where, score, k, cost, budget }, config) {
      const db = new DatabaseSync(':memory:');
      db.function('has_category', (categories, name) => Number(
        JSON.parse(categories ?? '[]').map((c) => String(c).toLowerCase())
          .includes(String(name).trim().toLowerCase())));
      const columns = Object.keys(config.rows[0]);
      db.exec(`CREATE TABLE candidates (${columns.join(', ')})`);
      const insert = db.prepare(`INSERT INTO candidates VALUES ` +
                                `(${columns.map(() => '?').join(', ')})`);
      for (const row of config.rows) {
        insert.run(...columns.map((c) => row[c] ?? null));
      }
      // The family's eligibility (`family_eligible`), as `rank` applies it:
      const ineligible = new Set(config.ineligible ?? []);
      const rows = db.prepare(
        `SELECT id, churn_probability AS p, predicted_churner AS c, ` +
        `prior_ltv AS v, (${score}) AS s, (${cost ?? '0'}) AS cost ` +
        `FROM candidates WHERE (${where})`).all()
        .filter((r) => !ineligible.has(Number(r.id)));
      db.close();
      const num = (x) => (x === null || !isFinite(x) ? -Infinity : Number(x));
      const order = (key) => (a, b) =>
        key(b) - key(a) || num(b.v) - num(a.v) || a.id - b.id;
      // KumoRelational's rule of the question's family (`RULES`):
      const ev = (r) => (num(r.s) === -Infinity ? -Infinity : r.p * num(r.s));
      const churners = rows.filter((r) => r.c === 1);
      const ranked = config.rule === 'expected_value'
        ? [...rows].sort(order(ev))
        : [...churners.sort(order((r) => num(r.s))),
           ...rows.filter((r) => r.c !== 1).sort(order(ev))];
      const ids = [];
      let spent = 0;
      for (const r of ranked) {
        if (budget !== undefined && budget !== null) {
          spent += Number(r.cost) || 0;
          if (spent > budget) break;
        }
        ids.push(r.id);
        if (ids.length === k) break;
      }
      return JSON.stringify({ ids, matching: rows.length,
                              predicted_churners: churners.length });
    },
  })],
});
// --slim-prompt: `systemPrompt` replaces OpenClaw's system prompt. `agent
// exec` loads plugins for their tools only and starts no hook runner, so
// start an empty one; it dispatches the hooks of the turn's plugins:
const register = entry.register;
entry.register = (api) => {
  register(api);
  const systemPrompt = api.pluginConfig?.systemPrompt;
  if (!systemPrompt) return;
  api.on('before_prompt_build', () => ({ systemPrompt }));
  initializeGlobalHookRunner(
    { hooks: [], typedHooks: [], plugins: [], trustedToolPolicies: [] });
};
export default entry;
"""


def write_plugin(path: Path) -> Path:
    r"""Writes the ``kumo_rank`` OpenClaw plugin to :obj:`path`."""
    path.mkdir(parents=True, exist_ok=True)
    schema = json.dumps(CONFIG_SCHEMA)
    (path / 'index.mjs').write_text(PLUGIN_JS.replace('CONFIG_SCHEMA', schema))
    manifest = dict(id=PLUGIN, name='KumoRelational',
                    configSchema=CONFIG_SCHEMA,
                    activation=dict(onStartup=True),
                    contracts=dict(tools=TOOLS[:1]))
    package = dict(name=PLUGIN, version='0.0.0', type='module', private=True,
                   openclaw=dict(extensions=['./index.mjs']))
    for name, value in [('openclaw.plugin.json', manifest),
                        ('package.json', package)]:
        (path / name).write_text(json.dumps(value, indent=2) + '\n')
    return path


def openclaw_config(args: Any, workspace: Path, plugin: Path,
                    plugin_config: Dict[str, Any]) -> Dict[str, Any]:
    r"""The pinned OpenClaw config of one question."""
    ref = f'{PROVIDER}/{args.model}'
    model = dict(id=args.model, name=args.model, reasoning=False,
                 input=['text'], contextWindow=args.context_window,
                 maxTokens=args.max_tokens)
    # The request timeout also lifts OpenClaw's idle watchdog for remote
    # servers, as a relayed reply arrives only once it is complete:
    provider = dict(baseUrl=args.base_url, apiKey='EMPTY',
                    api='openai-completions', timeoutSeconds=args.timeout,
                    models=[model])
    sampling = (0.6, 0.95) if args.think else (0, 1)  # The model card's.
    params = dict(temperature=sampling[0], topP=sampling[1],
                  maxTokens=args.max_tokens)
    defaults = dict(workspace=str(workspace),
                    model=dict(primary=ref, fallbacks=[]),
                    models={ref:
                            dict(params=params)}, timeoutSeconds=args.timeout,
                    maxConcurrent=1, skipBootstrap=True, userTimezone='UTC')
    entry = dict(enabled=True, config=plugin_config)
    if 'systemPrompt' in plugin_config:  # Lets its hook replace the prompt:
        entry['hooks'] = dict(allowConversationAccess=True)
    tools = dict(deny=['*']) if args.no_kumo else dict(allow=TOOLS)
    return dict(
        models=dict(mode='replace', providers={PROVIDER: provider}),
        agents=dict(defaults=defaults),
        tools=dict(profile='full', toolSearch=False,
                   exec=dict(timeoutSeconds=30), **tools),
        plugins=dict(enabled=True, allow=[PLUGIN],
                     load=dict(paths=[str(plugin)]), entries={PLUGIN: entry}),
    )


def candidate_rows(record: Dict[str, Any]) -> List[str]:
    r"""The candidate rows of a record without the relational markers."""
    return [
        line.replace(MARKER, '').rstrip()
        for line in record['context'].split('\n') if line.strip()
    ]


def after_reasoning(text: str) -> str:
    r"""The text after the reasoning, empty if the reasoning never ended."""
    if '<think>' in text and '</think>' not in text:
        return ''
    return text.rsplit('</think>', 1)[-1]


def numbers(text: str) -> List[str]:
    r"""The integers of an answer after its reasoning, in order."""
    return re.findall(r'\b\d+\b', after_reasoning(text))


def parse_ids(text: str, pool: List[str], k: int) -> List[str]:
    r"""The first :obj:`k` candidate ids of an answer, in order, as the
    other arms cut their rankings.
    """
    return [t for t in numbers(text) if t in set(pool)][:k]


def ask(record: Dict[str, Any], predictions: List[Dict[str, Any]], root: Path,
        plugin: Path, args: Any) -> Tuple[str, str]:
    r"""Runs one OpenClaw turn on a record in the fresh directory
    :obj:`root`; returns the final answer and the failure, if any.
    """
    root = root.resolve()  # The command runs in it.
    workspace, state, home = root / 'workspace', root / 'state', root / 'home'
    for path in [workspace, state, home]:
        path.mkdir(parents=True)
    rows = candidate_rows(record)
    probability = {p['id']: p['probability'] for p in predictions}
    table = []
    for row in map(json.loads, [] if args.no_kumo else rows):
        p = probability[row['id']]
        table.append(
            dict(
                id=int(row['id']), **row['features'],
                categories=json.dumps(row.get('categories', []),
                                      ensure_ascii=False), churn_probability=p,
                predicted_churner=int(p >= 0.5)))
    agents = (AGENTS_ZERO if args.no_kumo else AGENTS) + ('' if args.think else
                                                          '/no_think\n')
    (workspace / 'AGENTS.md').write_text(agents)
    prompt = root / 'prompt.txt'
    prompt.write_text(record['question'] + '\n\nCandidates:\n' +
                      '\n'.join(rows) + '\n')
    # The family's rule and eligibility, as `rank` applies them:
    rule = RULES[record['policy']['family']]
    ineligible = [
        int(c['id']) for c in record['candidates']
        if not family_eligible(c, record['policy'])
    ]
    plugin_config: Dict[str, Any] = dict(rows=table, rule=rule,
                                         ineligible=ineligible)
    if args.slim_prompt:
        plugin_config['systemPrompt'] = agents
    config = root / 'openclaw.json'
    config.write_text(
        json.dumps(openclaw_config(args, workspace, plugin, plugin_config),
                   indent=2))
    if not args.think and '/think' in prompt.read_text():
        return '', "'/think' in the prompt turns reasoning on"
    argv = shlex.split(args.openclaw) + ['agent', 'exec']
    for flag, value in [('config', config), ('state-dir', state),
                        ('cwd', workspace), ('message-file', prompt),
                        ('code-mode', 'direct'), ('timeout', args.timeout)]:
        argv += [f'--{flag}', str(value)]
    argv.append('--json')
    # `exec` runs unsandboxed on the host: pass it no tokens or caches, only
    # what Node.js needs:
    env = {
        k: v
        for k, v in os.environ.items()
        if k in ENV_KEEP or k.startswith(('LC_', 'NODE_'))
    }
    env.update(HOME=str(home), OPENCLAW_DISABLE_BONJOUR='1',
               OPENCLAW_STATE_DIR=str(state), OPENCLAW_CONFIG_PATH=str(config))
    with open(root / 'stdout.json', 'w') as out, \
            open(root / 'stderr.log', 'w') as err:
        proc = subprocess.Popen(argv, cwd=root, env=env, stdout=out,
                                stderr=err, start_new_session=True)
        try:
            proc.wait(timeout=args.timeout + 60)
        except subprocess.TimeoutExpired:
            return '', 'timeout'
        finally:  # Also reaps the leftovers of the group, e.g., on Ctrl-C.
            for sig in [signal.SIGTERM, signal.SIGKILL]:
                with suppress(OSError, subprocess.TimeoutExpired):
                    os.killpg(proc.pid, sig)
                    proc.wait(timeout=10)
    try:
        envelope = json.loads((root / 'stdout.json').read_text())
    except json.JSONDecodeError:
        return '', f'exit code {proc.returncode} without a JSON envelope'
    final = envelope.get('final') or ''
    if envelope.get('status') == 'ok':
        if proc.returncode != 0:  # OpenClaw's cleanup failed after the run.
            print(
                f'{root.name}: exit code {proc.returncode} after an ok run, '
                f'see {root / "stderr.log"}', file=sys.stderr)
        return final, ''
    error = envelope.get('error') or {}
    return '', error.get('kind') or f'exit code {proc.returncode}'


class Relay(ThreadingHTTPServer):
    r"""Forwards OpenAI chat completions to :obj:`base_url` without
    streaming and replays each reply as a stream if the client asked for
    one; keeps every request and reply of the current question.
    """
    daemon_threads = True

    def __init__(self, base_url: str, think: bool = False) -> None:
        self.upstream, self.think = urlsplit(base_url), think
        self.lock, self.conns = threading.Lock(), set()
        self.start(float('inf'))
        super().__init__(('127.0.0.1', 0), _RelayHandler)
        path = self.upstream.path
        self.base_url = f'http://127.0.0.1:{self.server_port}{path}'
        threading.Thread(target=self.serve_forever, daemon=True).start()

    def start(self, timeout: float) -> None:
        r"""Starts a question that ends within :obj:`timeout` seconds."""
        self.deadline = time.monotonic() + timeout
        self.requests: List[Dict] = []
        self.replies: List[Dict] = []
        self.problems: List[str] = []

    def cancel(self) -> None:
        r"""Drops pending upstream requests, so vLLM aborts them."""
        with self.lock:
            for conn in self.conns:
                with suppress(OSError, AttributeError):
                    conn.sock.shutdown(socket.SHUT_RDWR)

    def forward(self, path: str, body: Dict, auth: str) -> Tuple[int, bytes]:
        https = self.upstream.scheme == 'https'
        conn = (http.client.HTTPSConnection if https else
                http.client.HTTPConnection)(self.upstream.netloc, timeout=max(
                    1.0, min(self.deadline - time.monotonic(), 1e6)))
        with self.lock:
            self.conns.add(conn)
        try:
            conn.request('POST', path, json.dumps(body), {
                'Content-Type': 'application/json',
                'Authorization': auth
            })
            res = conn.getresponse()
            return res.status, res.read()
        except (OSError, http.client.HTTPException) as e:
            return 502, json.dumps(dict(error=dict(message=str(e)))).encode()
        finally:
            with self.lock:
                self.conns.discard(conn)
            conn.close()


class _RelayHandler(BaseHTTPRequestHandler):
    server: Relay

    def log_message(self, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        relay = self.server
        body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
        relay.requests.append(dict(body))
        stream = body.pop('stream', False)
        body.pop('stream_options', None)
        if body.get('tools'):  # vLLM renders them, the relay parses calls:
            body['tool_choice'] = 'none'
        texts = [
            json.dumps(m.get('content')) for m in body.get('messages', [])
            if m.get('role') in ['system', 'user']
        ]
        if not relay.think and any('/think' in t for t in texts):
            relay.problems.append("'/think' in the prompt turns reasoning on")
            status, data = 400, json.dumps(
                dict(error=dict(message=relay.problems[-1]))).encode()
        else:
            status, data = relay.forward(self.path, body,
                                         self.headers.get('Authorization', ''))
        try:
            reply = json.loads(data) if status == 200 else None
            if reply is not None:
                for choice in reply['choices']:
                    _parse_tool_calls(choice, f'call{len(relay.replies)}_')
                relay.replies.append(reply)
                data = _events(reply) if stream else json.dumps(reply).encode()
        except (ValueError, KeyError, TypeError):
            status, reply = 502, None
        kind = 'text/event-stream' if reply and stream else 'application/json'
        with suppress(OSError):  # The client may be gone.
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)


def _parse_tool_calls(choice: Dict[str, Any], prefix: str) -> None:
    r"""Drops the reasoning of a reply and turns Nemotron's
    ``<TOOLCALL>[{"name": ..., "arguments": ...}]</TOOLCALL>`` into
    OpenAI tool calls, as the model's ``nemotron_json`` parser does.
    """
    message = choice['message']
    text = after_reasoning(message.get('content') or '').lstrip()
    match = re.search(r'<TOOLCALL>(.*?)</TOOLCALL>', text, re.DOTALL)
    if not message.get('tool_calls') and match:
        try:
            calls = [(c['name'], c.get('arguments', {}))
                     for c in json.loads(match.group(1))]
        except (ValueError, KeyError, TypeError):
            calls = []  # Leave the text, OpenClaw sees no call.
        if calls:
            message['tool_calls'] = [
                dict(
                    id=f'{prefix}{i}', type='function', function=dict(
                        name=name, arguments=arguments if isinstance(
                            arguments, str) else json.dumps(arguments)))
                for i, (name, arguments) in enumerate(calls)
            ]
            text = text[:match.start()].strip()
            choice['finish_reason'] = 'tool_calls'
    message['content'] = text or None


def _events(reply: Dict[str, Any]) -> bytes:
    r"""A chat completion as stream chunks: its messages, finish reasons
    and usage.
    """
    keys = ('role', 'content', 'tool_calls', 'reasoning_content')
    base = {k: reply.get(k) for k in ('id', 'created', 'model')}
    base['object'], deltas, ends = 'chat.completion.chunk', [], []
    for c in reply['choices']:
        delta = {k: v for k, v in c['message'].items() if k in keys and v}
        calls = delta.pop('tool_calls', [])
        if calls:
            delta['tool_calls'] = [
                dict(t, index=i) for i, t in enumerate(calls)
            ]
        deltas.append(dict(index=c['index'], delta=delta))
        ends.append(
            dict(index=c['index'], delta={}, finish_reason=c['finish_reason']))
    chunks = [dict(base, choices=deltas), dict(base, choices=ends)]
    if reply.get('usage'):
        chunks.append(dict(base, choices=[], usage=reply['usage']))
    text = ''.join(f'data: {json.dumps(c)}\n\n' for c in chunks)
    return (text + 'data: [DONE]\n\n').encode()


def audit(replies: List[Dict], workspace: Path) -> List[Dict[str, Any]]:
    r"""The ``exec`` and ``process`` calls of the replies, each with the
    label sources and the paths outside :obj:`workspace` it mentions.
    """
    calls = []
    for reply in replies:
        for choice in reply['choices']:
            for call in choice['message'].get('tool_calls') or []:
                name, text = call['function']['name'], call['function'][
                    'arguments']
                if name not in TOOLS[1:]:
                    continue
                paths = re.findall(r'(?<![\w.])(?:~|\.\.)?/[\w.\-/]+', text)
                calls.append(
                    dict(
                        name=name, arguments=text, leaks=LEAK.findall(text),
                        outside=[
                            p for p in paths
                            if not p.startswith(str(workspace))
                        ]))
    return calls


def report(results: Dict[str, Dict[str, List[float]]], failures: int,
           total: int, arm: str = 'openclaw') -> None:
    r"""Prints recall like :meth:`kumo_gretriever.evaluate`."""
    for task, by_anchor in results.items():
        means = []
        for anchor, values in by_anchor.items():
            means.append(sum(values) / len(values))
            print(f'{arm} ({task}, {anchor}): recall {means[-1]:.4f} over '
                  f'{len(values)} questions')
        print(f'{arm} ({task}): recall {sum(means) / len(means):.4f} '
              f'(anchor mean, n={len(means)})')
    print(f'{arm}: {failures} of {total} questions failed (timeout, error '
          'or no answer)')


def run(records: List[Dict[str, Any]],
        probabilities: List[Optional[Dict[str, float]]], args: Any,
        out: Path) -> None:
    r"""Asks OpenClaw every record, keeping each run in :obj:`out`, then
    reports its recall.
    """
    out = out.resolve()
    results: Dict[str, Dict[str, List[float]]] = {}
    failures, wrong, extra, same, log = 0, 0, 0, 0, []
    plugin = write_plugin(out / 'plugin')
    relay = None if args.no_relay else Relay(args.base_url, args.think)
    if relay is not None:
        args = argparse.Namespace(**dict(vars(args), base_url=relay.base_url))
    for i, (record, p) in enumerate(zip(records, probabilities)):
        pool, k = record['pool']['candidate_ids'], record['k']
        root = out / f'{record["task"]}-{record["anchor"]}-{i}'
        predictions = [] if p is None else [
            dict(id=c, probability=p[c]) for c in pool
        ]
        if relay is not None:  # Past OpenClaw's own timeout, as `ask`:
            relay.start(args.timeout + 60)
        try:
            answer, failure = ask(record, predictions, root, plugin, args)
        except Exception as e:  # Keep the scores of the other questions.
            answer, failure = '', f'harness error: {e!r}'
        ids = parse_ids(answer, pool, k)
        failure = failure or ('' if ids else 'no answer')
        if relay is not None:
            relay.cancel()
            calls = audit(relay.replies, root / 'workspace')
            leaks = sorted({s.lower() for c in calls for s in c['leaks']})
            leaks = [f'exec mentions {", ".join(leaks)}'] if leaks else []
            failure = (relay.problems + leaks + [failure])[0]
            log.append(dict(question=root.name, failure=failure, calls=calls))
            root.mkdir(parents=True, exist_ok=True)
            (root / 'requests.json').write_text(
                json.dumps(
                    dict(requests=relay.requests, replies=relay.replies),
                    indent=1))
        if failure:
            print(f'{root.name}: {failure}', file=sys.stderr)
        else:
            wrong += len(set(ids)) != k or len(ids) != k
            extra += len(numbers(answer)) > len(ids)
        failures += bool(failure)
        # Whether the agent applied the forecasts as KumoRelational alone:
        same += (not failure and p is not None
                 and ids == rank(record['candidates'], record['policy'], p))
        recall = 0.0 if failure else strict_recall(ids, record['gold'], pool,
                                                   k)
        results.setdefault(record['task'],
                           {}).setdefault(record['anchor'], []).append(recall)
    if relay is not None:
        relay.shutdown()
        relay.server_close()
        (out / 'audit.json').write_text(json.dumps(log, indent=1))
    report(results, failures, len(records))
    print(f'openclaw: {wrong} answers had not exactly k distinct candidates '
          f'(recall 0), {extra} had other numbers (ignored), {same} equal '
          "KumoRelational's ranking")
    if relay is None:
        print('openclaw: exec and process calls not audited (--no-relay)')
    else:
        calls = [c for q in log for c in q['calls']]
        print(f'openclaw: {len(calls)} exec/process calls, '
              f'{sum(bool(c["outside"]) for c in calls)} with paths outside '
              f'the workspace (see {out / "audit.json"})')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=50)
    parser.add_argument('--root',
                        default='stanford-star/relbench-v1/rel-amazon')
    parser.add_argument(
        '--as-of', nargs='+', default=['2015-10-01'],
        help='Test anchors, or `test` for all 12 from 2013-01-03 to '
        '2015-10-01, 91 days apart')
    parser.add_argument('--num-questions', type=int, default=8)
    parser.add_argument('--kumo-ckpt', help='The KumoEncoder state dict')
    parser.add_argument('--base-url', default='http://127.0.0.1:8000/v1',
                        help='The OpenAI-compatible vLLM server')
    parser.add_argument('--model', default=MODEL)
    parser.add_argument('--openclaw', default='openclaw',
                        help='The OpenClaw command')
    parser.add_argument('--timeout', type=int, default=600,
                        help='Seconds per question')
    parser.add_argument('--max-tokens', type=int, default=8192)
    parser.add_argument('--context-window', type=int, default=131072)
    parser.add_argument('--think', action='store_true',
                        help='Let the model reason before it answers')
    parser.add_argument('--slim-prompt', action='store_true',
                        help="Replace OpenClaw's system prompt with AGENTS.md")
    parser.add_argument('--no-kumo', action='store_true',
                        help='Zero-shot Nemotron: no tools, no forecasts')
    parser.add_argument('--no-relay', action='store_true',
                        help='Stream from a server whose tool parser streams')
    parser.add_argument('--out', default='openclaw-runs',
                        help='A new directory to keep every question run in')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    args.as_of = anchors(args.as_of)
    out = Path(args.out).resolve()
    if '/think' in str(out):  # The relay would see it in the prompt.
        parser.error(f"--out {out} contains '/think'")
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        parser.error(f'--out {out} is not empty')
    # The forecasts of `kumo_gretriever.py --arm kumo`, see its docstring:
    sys.modules['cudf'] = None
    import pyarrow
    pyarrow.set_cpu_count(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    records = build_benchmark(args.root, args.as_of,
                              n_questions=args.num_questions, seed=args.seed)
    for record in records:
        record['pool']['model_input'] = {
            key: value.to(device)
            for key, value in record['pool']['model_input'].items()
        }
    out.mkdir(parents=True, exist_ok=True)
    probabilities: List[Optional[Dict[str, float]]] = [None] * len(records)
    if not args.no_kumo:
        from sdm.models import KumoRelational
        kumo = KumoRelational(task='classification', device=device)
        encoder = KumoEncoder(kumo, seed=args.seed)
        if args.kumo_ckpt is not None:
            encoder.load_state_dict(
                torch.load(args.kumo_ckpt, map_location='cpu',
                           weights_only=True, mmap=True))
        with torch.no_grad():
            probabilities = [predict(None, encoder, r)[0] for r in records]
        (out / 'forecasts.json').write_text(json.dumps(probabilities))
    run(records, probabilities, args, out)


if __name__ == '__main__':
    main()
