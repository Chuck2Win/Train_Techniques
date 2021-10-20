import itertools
# 반복자를 만드는 모듈
letters = ['a','b','c','d','e','f']
booleans = [True, False, True, False, False, True]
decimals = [0.1,0.7,0.4,0.4,0.5]

itertools.chain(letters, booleans, decimals) # <itertools.chain at 0x7f352af148e0>
list(itertools.chain(letters, booleans, decimals)) # ['a','b','c','d','e','f',True, False, True, False, False, True,0.1,0.7,0.4,0.4,0.5]
