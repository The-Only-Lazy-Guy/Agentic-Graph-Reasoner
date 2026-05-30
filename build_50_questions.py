import json
base = json.load(open('artifacts/phase15_test_20.json', encoding='utf-8'))
tasks = list(base['tasks'])
steps = {'easy': 4, 'medium': 6, 'hard': 8, 'extreme': 10}
new = [
 ('physics','easy','Why does ice float on water?'),
 ('physics','medium','Explain why the sky appears blue during the day.'),
 ('physics','medium','State Newtons third law and give one everyday example.'),
 ('physics','hard','How does total internal reflection enable optical fibers to carry signals?'),
 ('chem','easy','What distinguishes an ionic bond from a covalent bond?'),
 ('chem','medium','Why does increasing temperature generally speed up a chemical reaction?'),
 ('bio','easy','What is the role of mitochondria in a cell?'),
 ('bio','medium','Explain how natural selection leads to adaptation over generations.'),
 ('math','easy','What does the derivative of a function represent geometrically?'),
 ('math','medium','Use the pigeonhole principle to explain why two people in a group of 13 share a birth month.'),
 ('math','hard','Prove there are infinitely many prime numbers.'),
 ('math','medium','What is Bayes theorem and what does it let you compute?'),
 ('cs','easy','What is the difference between a stack and a queue?'),
 ('cs','medium','Why can a hash table offer average O(1) lookup, and when does it degrade?'),
 ('cs','medium','What are the ACID properties of a database transaction?'),
 ('cs','hard','How does a deadlock arise and what are the four necessary conditions?'),
 ('cs','medium','What is the difference between TCP and UDP?'),
 ('cs','hard','Explain how copying garbage collection reclaims memory.'),
 ('algo','easy','What precondition must hold for binary search to be correct?'),
 ('algo','medium','What is the worst-case time of quicksort and what input causes it?'),
 ('algo','medium','When would you use BFS instead of DFS on a graph?'),
 ('algo','hard','What property must a problem have for dynamic programming to apply?'),
 ('algo','medium','Why does a greedy algorithm fail to give optimal change for some coin systems?'),
 ('algo','hard','How does Bellman-Ford handle negative edge weights where Dijkstra cannot?'),
 ('logic','easy','What is the contrapositive of "if it rains, the ground is wet"?'),
 ('logic','medium','Explain the difference between a necessary and a sufficient condition.'),
 ('sysdesign','medium','What problem does a load balancer solve and name one balancing strategy.'),
 ('sysdesign','hard','State the CAP theorem and what it forces you to trade off during a partition.'),
 ('sysdesign','medium','Why should a payment API endpoint be idempotent?'),
 ('sysdesign','hard','What is database sharding and what new problem does it introduce?'),
]
for i, (dom, diff, q) in enumerate(new, start=21):
    tasks.append({'id': f'q_{i:02d}_{diff}_{dom}', 'difficulty': diff, 'question': q,
                  'max_steps': steps[diff], 'expected': 'graph-grounded explanation'})
out = {'tasks': tasks[:50]}
json.dump(out, open('artifacts/phase15_test_50.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
print('wrote', len(out['tasks']), 'tasks -> artifacts/phase15_test_50.json')
