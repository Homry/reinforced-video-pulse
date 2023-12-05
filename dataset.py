with open('input.txt', 'r') as f:
    lines = f.read().split('\n')

lines = [i.split(' ') for i in lines]
animals = [i[1] for i in lines]
animals = list(set(animals))
res = {i: [] for i in animals}
for i in lines:
    res[i[1]].append(i[0])
for i in res.keys():
    res[i].sort()

animals.sort(key=len)

res_str = []
for i in animals:
    res_str.append(f'{i}: {", ".join(res[i])}')
print('\n'.join(res_str))
