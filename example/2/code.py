import sys
n,k = map(int, sys.stdin.readline().split())
arr = list(map(int, sys.stdin.readline().split()))
tot = sum(arr)
if tot % k != 0:
	print('No')
	return
tot //= k
idx,cur = 0,0
ans = []
for i in range(n):
	cur += arr[i]
	idx += 1
	if cur == tot:
		ans.append(idx)
		idx = 0
		cur = 0
	elif cur > tot:
		print('No')
		return
if sum(ans) != n:
	print('No')
	return
print('Yes')
for an in ans:
	print(an,end=' ')
