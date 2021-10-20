import random
# random choices - 가중치 반영해 랜덤 요소 반환
code = ['HTML','CSS','JS']
print(random.choices(code, weights=[5,1,1],k=7)) # 각 feature 마다 weight( 높을수록 잘 나옴) k : return의 개수
