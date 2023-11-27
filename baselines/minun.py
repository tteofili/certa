import pandas as pd
import numpy as np
import random
import sys
import os
from collections import defaultdict
from itertools import product, combinations

from tqdm import tqdm
from torch.utils import data

class MinunExplainer(object):
    # The explainer class
    def __init__(self,
                 model,
                 ):
        # will be the model itself that can be directly used for prediction
        self.model = model


    def to_df(self, instance):
        df_d = dict()
        for i in range(len(instance[0])):
            df_d['ltable_' + instance[3][i]] = instance[0][i]
        for i in range(len(instance[1])):
            df_d['rtable_' + instance[3][i]] = instance[1][i]
        df_d['label'] = instance[2]
        return pd.DataFrame(pd.Series(df_d)).T

    def explain(self, instance, method="greedy" ,k=10):
        '''
            k: number of explaiantions to return, cannot exceed self.maximum_samples
        Returns:
            exp_result(DefaultDict): the explaination of given instance in the format of key-value pairs, where 
        '''
        # 1. For each pais of attribute in the instance, identify all possible edit ops
        # 2. Repeatly search for N samples with _generate_samples algorithms
        # 3. for each sample, evaluate the black box model by calling _eval_model function
        # 4. End above loop if finding k samples that can flip the results or finish enumerating the search space 
        # 5. refine the candidates and return explaination
        '''list(pd_instance.loc[:, pd_instance.columns.str.startswith('ltable_')].values[0])
        list(pd_instance.loc[:, pd_instance.columns.str.startswith('rtable_')].values[0])
        pd_instance['label']
        [c.replace('ltable_', '') for c in pd_instance.columns if c.startswith('ltable_')]'''

        attrs = self._entity2pair_attrs(instance)
        cands4attrs = [[]]
        for pair in attrs:
            if instance[2] == 0:
                res = self._generate_candidates_from_one_attribute_neg(pair[0], pair[1])
            else:
                res = self._generate_candidates_from_one_attribute_pos(pair[0], pair[1])
            cands4attrs.append(res)  
        cands4attrs = cands4attrs[1:]
        assert len(cands4attrs) > 0
        if method == "greedy":
            candidates,eval_cnt = self._explain_permutation(instance, cands4attrs, k)
        elif method == "min":
            candidates,eval_cnt = self._explain_min_attr(instance, cands4attrs, k)
        elif method == "binary":
            candidates,eval_cnt = self._explain_binary(instance, cands4attrs, k)
        else:
            raise Exception('Not Support this kind of explaination method!') 
        # corner case: if no flip happens:
        # original prediction is 0: just add all attributes of the right entity as explaination
        # original prediction is 1: return an empty dict denoting delete all attributes.
        exp_result = defaultdict()
        if len(candidates) == 0: 
            print("cannot flip!")
            return (defaultdict(), pd.DataFrame()), eval_cnt
        else:
            top_explaination = self._find_explaination(candidates)
            for idx,onum in enumerate(top_explaination[-1]):
                if onum != 0:
                    exp_result[instance[-1][idx]] = cands4attrs[idx][onum]
            res = (exp_result, top_explaination[0])
            return res, eval_cnt


    def _explain_permutation(self, instance, cands4attrs, k):
        '''
        Args:
            cands4attrs(Array): list of candidates in each attributes: array length equals to 
            the number of attributes, each item contain a list of possible transformation in that attribute
            k(int): number of explaiantions to return, cannot exceed self.maximum_samples
        Returns:
            candidates(Array): an array contains k explainations
        '''
        eval_cnt = 0 
        candidates = []
        perms = []  
        for cands in cands4attrs:
            perms.append(range(len(cands)))
        # print(perms)
        for indices in product(*perms):
            num = sum(indices)
            if num == 0:
                continue
            sample = self._format_instance(cands4attrs, indices, self.to_df(instance))
            predict, logit = self._eval_model(sample)
            eval_cnt = eval_cnt + 1
            if str(predict[0]) != str(instance[2]): # result is flipped
                # candidates.append((sample,num,logit[0],indices))
                # print(str(predict[0])+"#"+str(instance[2])+"#")
                candidates.append((sample, num, max(logit[0][0], logit[0][1]), indices))
            if len(candidates) >= k:
                return candidates, eval_cnt
        return candidates, eval_cnt


    def _explain_min_attr(self, instance, cands4attrs, k):
        '''
        Args and Returns are the same above.
        Enumerate the combination w.r.t minimum number of attributes
        ''' 
        eval_cnt = 0
        candidates = []
        perms = []
        tmp_dict = defaultdict(list)
        num_dim = len(cands4attrs)
        for cands in cands4attrs:
            perms.append(range(len(cands)))
        for indices in product(*perms):
            attr_num = num_dim - list(indices).count(0)
            tmp_dict[attr_num].append(indices)
        dimnums = sorted(tmp_dict.keys())
        
        for num in dimnums:
            if num == 0:
                continue
            tmp_dict[num].sort(key = lambda d: sum(d))
            for item in tmp_dict[num]:
                sample = self._format_instance(cands4attrs,item,self.to_df(instance))
                predict, logit = self._eval_model(sample)
                eval_cnt += 1
                if str(predict[0]) != str(instance[2]): # result is flipped
                    candidates.append((sample,num,max(logit[0][0],logit[0][1]), indices))
                if len(candidates) >= k:
                    return candidates,eval_cnt
        return candidates, eval_cnt


    def _explain_binary(self, instance, cands4attrs, k):
        '''
        Use binary search algorithm
        Args:
            N: maximum number of samples to be generated, if the total number of samples is smaller than N, 
            then return all
        Returns:
            samples(Array): an array contains k explainations
        ''' 
        eval_cnt = 0 
        candidates = []
        perms = []
        for cands in cands4attrs:
            perms.append(len(cands))
        num_dim = len(cands4attrs)
        flip_indices = []
        for attr_num in range(1,num_dim+1):
            if attr_num == 1:
                for dim in range(num_dim):
                    cur_indice = [0]*num_dim
                    low = 0
                    high = perms[dim]-1
                    while low < high:
                        mid = (high + low) / 2
                        cur_indice[dim] = int(mid)
                        sample = self._format_instance(cands4attrs,cur_indice,self.to_df(instance))
                        predict, logit = self._eval_model([sample])
                        eval_cnt += 1
                        if str(predict[0]) != str(instance[2]): # result is flipped
                            candidates.append((sample,int(mid),max(logit[0][0],logit[0][1]),cur_indice))
                            if len(candidates) >= k:
                                return candidates,eval_cnt
                            high = mid - 1
                        else:
                            low = mid + 1
            else:
                for group in combinations(range(num_dim), attr_num):
                    lbs = []
                    # print(group)
                    for dim in group:
                        lbs.append(list((1,perms[dim])))
                    local_indice = [0] * attr_num
                    max_time = 1
                    for dim in group:
                        max_time *= perms[dim]
                    enum_time = 0
                    while enum_time < max_time:
                        tag = False
                        for attr_idx in range(attr_num):
                            enum_time += 1
                            if lbs[attr_idx][0] < lbs[attr_idx][1]:
                                tag = True
                                mid = (lbs[attr_idx][1] + lbs[attr_idx][0]) / 2
                                local_indice[attr_idx] = int(mid)
                                if enum_time >= attr_num:
                                    cur_indice = [0]*num_dim
                                    for idx,dim in enumerate(group):
                                        cur_indice[dim] = local_indice[idx]
                                    sample = self._format_ditto_instance(cands4attrs,cur_indice,instance)
                                    predict, logit = self._eval_model([sample])
                                    eval_cnt += 1
                                    if str(predict[0]) != str(instance[2]): # result is flipped
                                        num = sum(cur_indice)
                                        candidates.append((sample,num,max(logit[0][0],logit[0][1]),cur_indice))
                                        if len(candidates) >= k: 
                                            return candidates,eval_cnt
                                        lbs[attr_idx][1] = mid - 1
                                        # continue
                                    else:
                                        lbs[attr_idx][0] = mid + 1
                        if tag == False:
                            break
        '''
        print(perms)
        for item in flip_indices:
            cur_indice = [0]*num_dim
            print(item)
            for idx,dim in enumerate(item[1]):
                    cur_indice[dim] = item[0][idx]
            # start from cur_indice to search the remaining one
            print("Before prediction: "+str(cur_indice))
            for d in item[1]:
                cur_indice[d] += 1
                while cur_indice[d] < perms[d]:
                    print(cur_indice)
                    sample = self._format_ditto_instance(cands4attrs,cur_indice,instance)
                    predict, logit = self._eval_model([sample])
                    eval_cnt += 1
                    if str(predict[0]) != str(instance[2]): # result is flipped
                        num = sum(cur_indice)
                        candidates.append((sample,num,max(logit[0][0],logit[0][1]),cur_indice))
                        if len(candidates) >= k:
                            return candidates, eval_cnt
                    cur_indice[d] += 1
        print("============ End of Case ===============")
        '''
        return candidates, eval_cnt
    
    def _entity2pair_attrs(self, instance):
        attrs = []
        assert(len(instance[0]) == len(instance[1]))
        for idx in range(len(instance[0])):
            item = (instance[0][idx], instance[1][idx])
            attrs.append(item)
        return attrs

    def _min_cost_path(self, cost, operations):
        # operation at the last cell
        path = [operations[cost.shape[0]-1][cost.shape[1]-1]]
    
        # cost at the last cell
        min_cost = cost[cost.shape[0]-1][cost.shape[1]-1]
    
        row = cost.shape[0]-1
        col = cost.shape[1]-1
    
        while row >0 and col > 0:
            if cost[row-1][col-1] <= cost[row-1][col] and cost[row-1][col-1] <= cost[row][col-1]:
                path.append(operations[row-1][col-1])
                row -= 1
                col -= 1
            elif cost[row-1][col] <= cost[row-1][col-1] and cost[row-1][col] <= cost[row][col-1]:
                path.append(operations[row-1][col])
                row -= 1
            else:
                path.append(operations[row][col-1])
                col -= 1                  
        return "".join(path[::-1][1:])
    
    
    def _token_edit_distance(self, str1, str2):
        '''
        Args:
            str1/2: two entities for calculation
        Returns:
            dist(int): the token-level edit distance between two entities
            ops(nparray): the operations to formulate the path
        ''' 
        seq1 = str1.split(' ')
        seq2 = str2.split(' ')
        if len(str1) == 0 and len(str2) == 0:
            return 0, []
        if len(str1) == 0:
            ops = []
            for i in range(len(seq2)):
                ops.append('I')
            return len(seq2), ops
        if len(str2) == 0:
            ops = []
            for i in range(len(seq1)):
                ops.append('D')
            return len(seq1), ops
        matrix = np.zeros((len(seq1)+1, len(seq2)+1))
        matrix[0] = [i for i in range(len(seq2)+1)]
        matrix[:, 0] = [i for i in range(len(seq1)+1)]
        
        ops = np.asarray([['-' for j in range(len(seq2)+1)] \
                                for i in range(len(seq1)+1)])
        ops[0] = ['I' for i in range(len(seq2)+1)]
        ops[:, 0] = ['D' for i in range(len(seq1)+1)]
        ops[0, 0] = '-'
        
        for row in range(1, len(seq1)+1):
            for col in range(1, len(seq2)+1):
                if seq1[row-1] == seq2[col-1]:
                    matrix[row][col] = matrix[row-1][col-1]
                else:           
                    insertion_cost = matrix[row][col-1] + 1
                    deletion_cost = matrix[row-1][col] + 1
                    substitution_cost = matrix[row-1][col-1] + 1
                    matrix[row][col] = min(insertion_cost, deletion_cost, substitution_cost)
                    if matrix[row][col] == substitution_cost:
                        ops[row][col] = 'S'                  
                    elif matrix[row][col] == insertion_cost:
                        ops[row][col] = 'I'
                    else:
                        ops[row][col] = 'D'              
        dist = matrix[len(seq1), len(seq2)]
        operations = self._min_cost_path(matrix, ops)
        return dist,operations
    
    
    def _generate_candidates_from_one_attribute_pos(self, attr1, attr2):
        '''
        If the initial matching label is 1, then it requires to delete tokens from attr1 one by one until empty
        '''
        candidates = [""]
        seq = attr1.split(' ')
        tmp_str = ""
        for token in seq:
            tmp_str += str(token) +" "
            candidates.append(tmp_str.strip())
        return candidates[::-1]
    
    def _generate_candidates_from_one_attribute_neg(self, attr1, attr2):
        candidates = [attr1]
        dist,operations = self._token_edit_distance(attr1, attr2)
        tmp = attr1.split(' ')
        target = attr2.split(' ')
        if len(target) == 0:
            return candidates
        cur = 0 # for seq2
        dnum = 0
        for idx, op in enumerate(operations):
            if op == '-':
                cur += 1
                continue
            else:
                # errors of index out of range happens here when running on server
                # temp patch, will fix later
                if op == 'S':
                    pos1 = idx-dnum
                    pos2 = cur
                    if pos1 >= len(tmp):
                        pos1 = -1
                    if pos2 >= len(target):
                        pos2 = -1
                    tmp[pos1] = target[pos2]
                    cur += 1
                elif op == 'I':
                    pos3 = cur
                    if pos3 >= len(target):
                        pos3 = -1
                    tmp.insert(idx, target[pos3])
                    cur += 1
                else:
                    pos4 = idx-dnum
                    if pos4 >= len(tmp):
                        pos4 = -1
                    del tmp[pos4]
                    dnum += 1
                tmp_str = ""
                for token in tmp:
                    tmp_str += str(token)+" "
                candidates.append(tmp_str.strip())
        return candidates 
    
    
    def _format_ditto_instance(self, attrs, indices, instance):
        '''
        Args:
            attrs (list): The candidates of left entity for each attribute 
            indices (list): The list of indices for choosing candidate
            instance (tuple)
        Returns:
            ditto_item(str): The item that can be fed into the ditto model for testing
        ''' 
        assert len(attrs) == len(indices)
        left = ""
        right = ""
        for i in range(len(attrs)):
            left += " COL "+str(instance[3][i])+" VAL "+str(attrs[i][indices[i]])
            right += " COL "+str(instance[3][i])+" VAL "+str(instance[1][i])
        res = left.strip()+"\t"+right.strip()+"\t"+str(instance[2])
        return res

    def _format_instance(self, attrs, indices, pd_instance):
        '''
        Args:
            attrs (list): The candidates of left entity for each attribute
            indices (list): The list of indices for choosing candidate
            instance (tuple)
        Returns:
            ditto_item(str): The item that can be fed into the ditto model for testing
        '''
        perturbed_df = pd_instance.copy()
        for i in range(len(attrs)):
            perturbed_df[perturbed_df.columns[i]] = str(attrs[i][indices[i]])
        return perturbed_df
    

    def _eval_model_random(self, samples):
        '''
        Generate a random value instead of use ML model for prediction, just for testing
        '''
        results = []
        logits = []
        for sample in samples:
            val = random.random()
            if val > 0.5:
                results.append(1)
            else:
                results.append(0)
            logits.append(val)
        return results, logits

    def _find_explaination(self, candidates):
        '''
        Each candidate is a tuple with 3 attributes: the sentence, # ops, val of logit
        '''
        candidates.sort(key=lambda l: (l[1],-l[2]))
        return candidates[0]
    
    
    def _eval_model(self, samples, prob=True):
        '''
        Args:
            samples(Array): the set of candidate samples
            prob(Boolean): whether return both
        Returns:
            logits(Tensor): the tensor of logits for all samples, shape: N*2, where N is the number of samples
            predication(dataFrame): the predicted class label (0/1)
        '''
            
        predictions = self.model.predict(samples)
        results = np.argmax(predictions[['nomatch_score','match_score']])
        if prob == True:
            logits = [predictions['match_score'].values,
                      predictions['nomatch_score'].values]
            return [results], [logits]
        else:
            return [results]

def formulate_instance(tableA, tableB, inst):
    '''
    Args:
        tableA/tableB(dataFrame): the two tables
        inst(dataFrame): one test instance
    Returns:
        item: a triplet with two entities and label
    '''
    id1 = int(inst[0])
    id2 = int(inst[1])
    header = list(tableA)
    attr_num = len(header)
    left = []
    right = []
    for idx in range(attr_num):
        if pd.isnull(tableA.iloc[id1][idx]):
            left.append("")
        else:
            left.append(str(tableA.iloc[id1][idx]))
        if pd.isnull(tableB.iloc[id2][idx]):
            right.append("")
        else:
            right.append(str(tableB.iloc[id2][idx]))
    item = (left, right, inst[2], header)
    return item