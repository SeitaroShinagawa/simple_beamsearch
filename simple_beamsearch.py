#!usr/bin/python

import sys
beam = int(sys.argv[1])

"""
"a": 0.3,  
"b": 0.7,  
"a-a":0.1, #0.3*0.1=0.03
"a-b":0.9, #0.3*0.9=0.27
"b-a":0.4, #0.7*0.4=0.28
"b-b":0.6, #0.7*0.6=0.42
"a-a-a":0.3,  #0.3*0.1*0.3=0.009
"a-a-b":0.7,  #0.3*0.1*0.7=0.021
"a-b-a":0.8,  #0.3*0.9*0.8=0.216
"a-b-b":0.2,  #0.3*0.9*0.2=0.054
"b-a-a":0.6,  #0.7*0.4*0.6=0.168
"b-a-b":0.4,  #0.7*0.4*0.4=0.112
"b-b-a":0.5,  #0.7*0.6*0.5=0.21
"b-b-b":0.5,  #0.7*0.6*0.5=0.21

a:0
b:1
"""

probs = [[[],[0.3,0.7]],  
[[0],[0.1,0.9]],
[[1],[0.4,0.6]],
[[0,0],[0.3,0.7]],
[[0,1],[0.8,0.2]],
[[1,0],[0.6,0.4]],
[[1,1],[0.5,0.5]]]

def prob_gen(lis):
  tmp = [x[1] for x in probs if x[0]==lis]
  return tmp[0]

def list_print(in_):
  for i in in_:
    print(i)

print("step 1")
out_list = []
prob=prob_gen(out_list)

print("prob a:{},b:{}".format(prob[0],prob[1]))
candidate_list = [[[i],j] for i,j in enumerate(prob)]
candidate_list = sorted(candidate_list,key=lambda x:x[1],reverse=True)
out_list = candidate_list[:beam] #[ [[0],0.3] , [[1],0.7] ]

list_print(out_list)

print("step 2")

candidate_list=[]
for lis in out_list:
    prob=prob_gen(lis[0])
    print("prob a:{},b:{}".format(prob[0],prob[1]),"conditioned by p(",lis[0],")=",lis[1])
    for i,j in enumerate(prob):
      A = lis[0]+[i]
      B = lis[1]*j
      candidate_list.append([A,B])
candidate_list = sorted(candidate_list,key=lambda x:x[1],reverse=True)
out_list = candidate_list[:beam] #[ [[0],0.3] , [[1],0.7] ]
list_print(out_list)

print("step 3")

candidate_list=[]
for lis in out_list:
    prob=prob_gen(lis[0])
    print("prob a:{},b:{}".format(prob[0],prob[1]),"conditioned by p(",lis[0],")=",lis[1])
    for i,j in enumerate(prob):
      A = lis[0]+[i]
      B = lis[1]*j
      candidate_list.append([A,B])

candidate_list = sorted(candidate_list,key=lambda x:x[1],reverse=True)
out_list = candidate_list[:beam] #[ [[0],0.3] , [[1],0.7] ]
list_print(out_list)





