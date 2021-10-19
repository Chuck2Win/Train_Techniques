# jsonl 읽기
f = open('example.jsonl','r')
data = []
for i in f:
  data.append(i) # 이러면 json이 추가될 것임
  
import json 
row_data = [{"id":"a","title":"higuys","genre":"drama"}, {"id":"b","title":"make_me_happy","genre":"action"}] 
with open("make_jsonl.jsonl" , encoding= "utf-8",mode="w") as file: 
  for i in row_data: 
    file.write(json.dumps(i) + "\n")
