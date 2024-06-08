import sys
reader = (s.rstrip() for s in sys.stdin)
input = reader.__next__

def solve():
    s,c = input().split()
    n = len(s)
    for i in range(n-1):
        prev = s[i]
        pos = i
        for j in range(i+1, n):
            if s[j]<prev:
                prev = s[j]
                pos = j
            elif s[j] == prev:
                pos = j
        if prev == s[i]:
            continue
        t = list(s)
        t[i], t[pos] = prev, s[i]
        s = "".join(t)
        break
    if s<c:
        print(s)
    else:
        print("---")

t = int(input())
for i in range(t):
    solve()
