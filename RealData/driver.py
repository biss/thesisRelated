import garrotte_single_scaling as gss
import garrotte_multiple_c as gms

from sklearn.metrics.cluster import adjusted_rand_score
import numpy as np, csv


def set_from_array(array):
    c = [[], []]
    cl, n = array.shape
    for i in range(n):
        for j in range(cl):
            if array[j, i] == 1:
                c[j].append(i)

    return c[0], c[1]

def getUnion(s):
    """ return a union from a set of lists """
    u = set()
    for x in s:
       u = u | set(x)
    return u

def getIntersection(s):
    """ return an intersection from a set of lists """
    i = set(s[0])
    for x in s[1:]:
        i = i & set(x)
    return i

def js(a, b, score):
    combined = [a,b]
    inter = len(getIntersection(combined))
    uni = len(getUnion(combined))
    if score == True:
        return float(inter)/uni
    else:
        return inter, uni

if __name__ == '__main__':

    clusters = 2

    '''
    obj = []
    abs_va = []
    '''
    res = {}
    res_g = {}

    no_of_iter = {}
    no_of_iter_g = {}
    cluster_sim = {}
    #lmd = [.51+0.01*i for i in range(5)]

    #lmd = [0.1*i for i in range(10)]
    #lmd = lmd + [1.0*i for i in range(2,10)]

    lmd = [0.1, 0.2, 0.3, 0.5, 0.7, 1, 2]
    #lmd = [.1]

    for element in lmd:

        sr = gms.SwitchingRegression()
        sr.load_data()
        sr.init(clusters)
        #print sr.regime_param[0].shape
        sr.train(element)

        #specifying limits
        #print 'betas for ',element,' are: ', sr.regime_param
        #print sr.training_iter
        no_of_iter[element] = sr.training_iter
        res[element] = [sr.regime_param, np.array([[sr.rms]]), sr.obj, sr.abs_v]

        #cluster_array = sr.clusters()

        print 'cluster :', sr.U

        first, second = set_from_array(sr.U)

        sr_g = gss.SwitchingRegression_g()
        sr_g.load_data()
        sr_g.init(clusters)
        # print sr.regime_param[0].shape
        sr_g.train(element)

        # specifying limits
        # print 'betas for ',element,' are: ', sr.regime_param
        # print sr.training_iter
        no_of_iter_g[element] = sr_g.training_iter
        res_g[element] = [sr_g.regime_param, np.array([[sr_g.rms]]), sr_g.obj,
                        sr_g.abs_v, sr_g.belittle]

        #cluster_array_g = sr_g.clusters()

        print 'custer_g :', sr_g.U

        first_g, second_g = set_from_array(sr_g.U)

        #obj.append(sr.obj)
        #abs_va.append(sr.abs_v)
        print 'lasso :', first, second
        print 'garrotte :', first_g, second_g

        for_ari = [0]*sr.number # np.zeros(shape=(1,sr.number))
        for_ari_g = [0] * sr_g.number # np.zeros(shape=(1,sr_g.number))

        for i in range(sr.number):
            if i in first:
                for_ari[i] = 0
            elif i in second:
                for_ari[i] = 1
            else:
                print 'error'
                break

        for i in range(sr_g.number):
            if i in first_g:
                for_ari_g[i] = 0
            elif i in second_g:
                for_ari_g[i] = 1
            else:
                print 'error for', i
                break
        print set(first).intersection(second)
        print 'printing ari: ',for_ari, for_ari_g

        cluster_sim[element] = float(adjusted_rand_score(for_ari, for_ari_g))
        print 'hello : ',res
        print 'hello 1 ', res_g

    with open('output/driver_output.csv', 'wb') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res.keys())
        writer.writerow('-')
        for row in zip(*res.values()):
            for row1 in zip(*row):
                for row2 in zip(*row1):
                    writer.writerow(list(row2))
                writer.writerow('-')
        file.close()

    with open('output/driver_output.csv', 'a') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(res_g.keys())
        writer.writerow('-')
        for row in zip(*res_g.values()):
            for row1 in zip(*row):
                for row2 in zip(*row1):
                    writer.writerow(list(row2))
                writer.writerow('-')
        writer.writerow(cluster_sim.keys())
        writer.writerow(cluster_sim.values())
        #writer.writerow(list(abs_va))