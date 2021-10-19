# jsonl 읽기
f = open('example.jsonl','r')
data = []
for i in f:
  data.append(i) # 이러면 json이 추가될 것임
  
