import re

big = file('big.txt').read()
big = re.sub(' +', ' ', big)
N = len(big)

print "letter"
print "string"
print ""

for i in xrange(0, N):
    c = big[i]
    if ord(c) > 31 and ord(c) < 127:
        print c
