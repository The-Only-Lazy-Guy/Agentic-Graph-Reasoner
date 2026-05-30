"""Build data/question_bank.json — a shardable bank for multi-machine data gen.

Each machine takes a disjoint slice (by index modulo num-shards) so the corpora
are UNIQUE, then merges. Bank = the curated 50 + 50 more across the same domains.
"""
import json, os

base = json.load(open('artifacts/phase15_test_50.json', encoding='utf-8'))['tasks']
steps = {'easy': 4, 'medium': 6, 'hard': 8, 'extreme': 10}

more = [
 ('physics','easy','What is the difference between mass and weight?'),
 ('physics','easy','Why do objects of different mass fall at the same rate in a vacuum?'),
 ('physics','medium','Explain the Doppler effect and give an everyday example.'),
 ('physics','medium','Why does a spinning figure skater speed up when pulling their arms in?'),
 ('physics','hard','Why does time dilation occur for objects moving near the speed of light?'),
 ('physics','medium','What is the difference between heat and temperature?'),
 ('physics','hard','How does a transformer change AC voltage without moving parts?'),
 ('chem','easy','What is the difference between an element and a compound?'),
 ('chem','medium','Why is water a polar molecule and why does that matter?'),
 ('chem','medium','What is the difference between an acid and a base in terms of pH?'),
 ('chem','hard','Why does Le Chateliers principle predict how equilibrium shifts under pressure?'),
 ('bio','easy','What is the difference between DNA and RNA?'),
 ('bio','medium','How does an enzyme speed up a biochemical reaction?'),
 ('bio','medium','Why do antibiotics work on bacteria but not viruses?'),
 ('bio','hard','How does the sodium-potassium pump maintain a cells resting potential?'),
 ('math','easy','What is the difference between a permutation and a combination?'),
 ('math','medium','Why is the sum of the first n odd numbers always a perfect square?'),
 ('math','medium','What does it mean for a function to be continuous but not differentiable?'),
 ('math','hard','Explain why the harmonic series diverges.'),
 ('math','medium','What is the geometric meaning of an eigenvector?'),
 ('math','hard','Why does Gaussian elimination solve a linear system, and when does it fail?'),
 ('cs','easy','What is the difference between compiled and interpreted languages?'),
 ('cs','medium','What problem does virtual memory solve in an operating system?'),
 ('cs','medium','Why can floating-point arithmetic give 0.1 + 0.2 != 0.3?'),
 ('cs','hard','How does a B-tree keep database index lookups logarithmic on disk?'),
 ('cs','medium','What is the difference between a process and a thread?'),
 ('cs','hard','How does TLS establish a secure channel over an untrusted network?'),
 ('cs','medium','What is the difference between authentication and authorization?'),
 ('algo','easy','What is the difference between O(n) and O(log n) growth?'),
 ('algo','medium','Why does merge sort guarantee O(n log n) while quicksort does not?'),
 ('algo','medium','How does a hash set detect duplicates in O(n)?'),
 ('algo','hard','How does Dijkstra use a priority queue to achieve its time bound?'),
 ('algo','medium','When is a stack the right data structure for an algorithm?'),
 ('algo','hard','Why does topological sort require a directed acyclic graph?'),
 ('algo','medium','What is memoization and how does it change recursion cost?'),
 ('algo','hard','How does union-find achieve near-constant time with path compression?'),
 ('logic','easy','What makes an argument valid versus sound?'),
 ('logic','medium','Explain the difference between correlation and causation.'),
 ('logic','medium','What is a counterexample and how does it disprove a universal claim?'),
 ('logic','hard','Why can you not prove a universal statement by checking finitely many cases?'),
 ('sysdesign','easy','What is the purpose of a cache in a web system?'),
 ('sysdesign','medium','When would you choose a message queue over a direct API call?'),
 ('sysdesign','medium','What is eventual consistency and when is it acceptable?'),
 ('sysdesign','hard','How does consistent hashing reduce reshuffling when a node is added?'),
 ('sysdesign','medium','Why use a read replica, and what staleness risk does it introduce?'),
 ('sysdesign','hard','How would you design an idempotent retry mechanism for payments?'),
 ('sysdesign','medium','What is backpressure and why does a streaming system need it?'),
 ('sysdesign','hard','How does a write-ahead log help a database recover after a crash?'),
 ('cs','medium','What is the difference between symmetric and asymmetric encryption?'),
 ('math','easy','Why is dividing by zero undefined?'),
]
tasks = list(base)
for i, (dom, diff, q) in enumerate(more, start=51):
    tasks.append({'id': f'q_{i:03d}_{diff}_{dom}', 'difficulty': diff, 'question': q,
                  'max_steps': steps[diff], 'expected': 'graph-grounded explanation'})
os.makedirs('data', exist_ok=True)
json.dump({'tasks': tasks}, open('data/question_bank.json', 'w', encoding='utf-8'),
          ensure_ascii=False, indent=2)
print(f"wrote {len(tasks)} questions -> data/question_bank.json")
