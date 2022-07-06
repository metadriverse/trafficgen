res = 0

x = [1,1,1,2,2,2,3,3,4,4]

for i in range(len(x)):
    res = res ^ x[i]

k = res&(-res)

a,b=0,0

for i in range(len(x)):
    if (x[i]^k)>0:a=a&x[i]
    else: b = b&x[i]

print(a)