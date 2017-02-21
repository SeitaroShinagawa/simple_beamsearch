#!/usr/bin/env python3

import numpy as np
import math

"""
beam search in chainer

reset
hset_list <- candidate_list
next_h,prob = NeuralNetwork(hset_list)
search_list <- accum_hidden_one(<s>,h,word_list,0)
candidate_list,result_list <- search(search_list)
while len(candidate_list==0:
  search_list <- accum_hidden_one(x_i,h_i,word_list,cur_logprob)
  candidate_list,result_list <- search(search_list)
"""

def pseudo_NN(hidden,word,word_list):
    hidden_next = hidden+[word]
    word_prob = NN_out_dict["-".join([word_list[i] for i in hidden_next])] 
    return hidden_next,word_prob

#def Nbest(output,num):
#    nbest = []
#    n=1
#    for i,v in sorted(enumerate(output),key=lambda x:x[-1], reverse=True):
#        nbest.append([i,v])
#        if n==num:
#            break
#        else:
#            n+=1
#    return nbest

def Nbest(output,num):
    nbest = [(i,v) for i,v in sorted(enumerate(output),key=lambda x:x[-1], reverse=True)[:num]]
    return nbest

class BeamSearch():
    """
    candidate_list: [(hidden_state, word_list, cur_logprob),...]  #word_list:["<s>","there","is",...]
    search_list:    [(next_hidden_state, word_list+next_word, cur_logprob+next_logprob),...]

    ***pseudo_code***
    Initialize candidate_list
    count = 0
    while len(candidate_list)>0 or count<limit:
        for candidate in candidate_list:
            cur_h, word_list, cur_logprob = candidate
            next_h, next_prob = NN(cur_h, word_list[-1])   
            accum_hidden_one(candidate,next_h,next_prob)
        search()
    *****************
    """
    def __init__(self,init_h,init_x,stop_word,beam_size=2):
        """
        init_h: initial hidden variable (often zero vector)
        init_x: initial word (often the number indicates <s>, e.g. init_x=1 if vocab["<s>"]=1)
        candidate_list: [(h,word_list,cur_logprob),...]
        """
        self.beam_size = beam_size
        self.init_h = init_h
        self.init_x = init_x
        self.stop_word = stop_word
        self.candidate_list = [(self.init_h,[self.init_x],0)] #beam_sizeの数だけここに入れる 
        self.search_list = []
        self.result_list = [] #</s>が出たらここに入れる

    def reset(self):
        self.candidate_list = [(self.init_h,[self.init_x],0)] #beam_sizeの数だけここに入れる
        self.search_list = []
        self.result_list = [] #</s>が出たらここに入れる

    def get(self,batch_size=1):
        """
        get beamsearch candidate pair(current hidden state,word_list,log probabilities)
        """
        num = 0
        while batch_size*(num) < len(self.candidate_list):
            yield self.candidate_list[batch_size*num:batch_size*(num+1)]
            num += 1

    def accum_hidden_one(self,hset,next_hidden_state,next_prob): 

        """
        accumrate candidate pair calclated by NN
        hset        : the element of candidate_list (hidden_state, word_list, cur_logprob)
        next_h      : next hidden state of Neural Network (1xH) H:hidden_size
        next_prob   : softmax applied output of Neural Network (1xV) V:vocab_size

        """
        cur_h, word_list, cur_logprob = hset 
        prob_list = [(i,prob) for i,prob in enumerate(next_prob)]
        self.search_list = self.search_list + [(next_hidden_state,word_list+[i],cur_logprob+math.log(prob)) for i,prob in sorted(prob_list,key=lambda x:x[-1],reverse=True)[:self.beam_size]]
           
    def accum_hidden(self,hset_list,next_hidden_state_mat,next_prob_mat): #h_mat:BxH, out_distribution_mat:BxV 
        """
        mini-batch version of accum_hidden_one
        hset_list               : [hset1,hset2,...], hset:(hidden_state, word_list, cur_logprob)
        next_hidden_state_mat   : numpy matrix (BxH) B:batch_size, H:hidden_size 
        next_prob_mat           : numpy matrix (BxV) B:batch_size, V:vocab_size
        """
        B,V = next_prob_mat.shape
        prob_list = [(int(i/V),i%V,prob) for i,prob in enumerate(next_prob_mat.flatten())] #[(h_index0,prob0),(h_index1,prob1),...]
        nbest = [(next_hidden_state_mat[h_index],hset_list[h_index][1]+[word],hset_list[h_index][-1]+np.log(prob)) for h_index,word,prob in sorted(prob_list,key=lambda x:x[-1],reverse=True)[:self.beam_size]] #[(h,logprob),...] 
        self.search_list = self.search_list + nbest

    def search(self): #select N-best probabilities up to beam_size
        self.candidate_list = sorted(self.search_list,key=lambda x:x[-1],reverse=True)[:self.beam_size]
        lis = [word_list[-1] for next_h,word_list,cur_logprob in self.candidate_list]
        index_list = []
        for i,index in enumerate(lis):
            if index == self.stop_word:
                index_list.append(i)
        
        for i in index_list[::-1]:
            self.result_list.append(self.candidate_list[i])
            self.candidate_list.pop(i) 
        
        self.search_list = []

if __name__ == '__main__':

    #test by handcraft settings as below
    word_list=["<s>","a","b","c","</s>"]  #vocab size

    NN_out_dict = {}
    #t=1
    NN_out_dict["<s>"] = [0.0,0.5,0.4,0.09,0.01]     #the output probabilities: [a,b,c,</s>]
    #t=2
    NN_out_dict["<s>-a"] = [0.0,0.1,0.5,0.3,0.1]
    NN_out_dict["<s>-b"] = [0.0,0.12,0.05,0.8,0.03]
    NN_out_dict["<s>-c"] = [0.0,0.5,0.3,0.05,0.15]
    #t=3 
    NN_out_dict["<s>-a-a"] = [0.0,0.1,0.5,0.3,0.1]
    NN_out_dict["<s>-a-b"] = [0.0,0.25,0.2,0.5,0.05]
    NN_out_dict["<s>-a-c"] = [0.0,0.1,0.5,0.3,0.1]
    NN_out_dict["<s>-b-a"] = [0.0,0.1,0.5,0.3,0.1]
    NN_out_dict["<s>-b-b"] = [0.0,0.1,0.5,0.3,0.1]
    NN_out_dict["<s>-b-c"] = [0.0,0.1,0.1,0.1,0.7]
    NN_out_dict["<s>-c-a"] = [0.0,0.1,0.5,0.3,0.1]
    NN_out_dict["<s>-c-b"] = [0.0,0.1,0.5,0.3,0.1]
    NN_out_dict["<s>-c-c"] = [0.0,0.1,0.5,0.3,0.1]
    #t=3
    NN_out_dict["<s>-a-a-a"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-a-a-b"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-a-a-c"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-a-b-a"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-a-b-b"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-a-b-c"] = [0.0,0.06,0.04,0.1,0.8]
    NN_out_dict["<s>-a-c-a"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-a-c-b"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-a-c-c"] = [0.0,0.06,0.04,0.1,0.8]
    NN_out_dict["<s>-b-a-a"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-b-a-b"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-b-a-c"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-b-b-a"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-b-b-b"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-b-b-c"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-b-c-a"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-b-c-b"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-b-c-c"] = [0.0,0.06,0.04,0.1,0.8]
    NN_out_dict["<s>-c-a-a"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-c-a-b"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-c-a-c"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-c-b-a"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-c-b-b"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-c-b-c"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-c-c-a"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-c-c-b"] = [0.0,0.1,0.3,0.1,0.6]
    NN_out_dict["<s>-c-c-c"] = [0.0,0.06,0.04,0.1,0.8]

    #Nbest result
    word_ini = 0 #initialize vocab["<s>"] = 0
    hidden_ini = []
    print("initial:",hidden_ini,word_ini)
    word = word_ini
    hidden = hidden_ini
    for i in range(4): #setting is four sequence decoding 
        hidden,word_prob = pseudo_NN(hidden,word,word_list)
        best_list = Nbest(word_prob,1)
        word_index,prob = best_list[0]
        word= word_index
        print("phase{}:{},{}".format(i,hidden,word))
    print("")

    #beam search result
    stop_word = 4
    BEAM = BeamSearch(hidden_ini,word_ini,stop_word,beam_size=4)
    for j in range(4):
        for candidate in BEAM.get(batch_size=1):
            print("candidate:",candidate)
            h,words,cur_logprob = candidate[0]
            next_h,next_prob = pseudo_NN(h,words[-1],word_list)  #words[-1] indicates input word
            print("next_h:{},next_prob:{}".format(next_h,next_prob))
            BEAM.accum_hidden_one(candidate[0],next_h,next_prob)
        BEAM.search()
        for i in BEAM.candidate_list:
            print("searched result phase{}:{}".format(j,i))
        print("")
    for i in BEAM.result_list:
        print("result",i)
    
    
    
     
