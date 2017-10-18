import garrotte_multiple_c as gss
import fcrm_with_lasso as gms

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

    res = {}
    res_g = {}

    no_of_iter = {}
    no_of_iter_g = {}
    cluster_sim = {}

    lmd = [0.01, 0.05, 0.1, 0.2, 0.5, 1]

    for element in lmd:

        sr = gms.SwitchingRegression()
        sr.train(element)
        sr.fill_err()

        no_of_iter[element] = sr.training_iter
        res[element] = [sr.regime_param, np.array([[sr.err]])]

        print 'cluster :', sr.U[:, :5]

        first, second = set_from_array(np.round(sr.U, 0))

        sr_g = gss.SwitchingRegression_g(clusters)
        print 'cluster_g :', sr_g.U[:, :5]
        sr_g.train(element)
        print 'cluster_g :', sr_g.U[:, :5]
        sr_g.fill_err()
        print 'cluster_g :', sr_g.U[:, :5]

        no_of_iter_g[element] = sr_g.training_iter
        res_g[element] = [sr_g.regime_param, np.array([[sr_g.err]]),
                        sr_g.abs_v, sr_g.belittle]

        print 'cluster_g :', sr_g.U[:, :5]

        first_g, second_g = set_from_array(np.round(sr_g.U, 0))

        print 'lasso :', first, second
        print 'garrotte :', first_g, second_g

        for_ari = [0] * sr.number # np.zeros(shape=(1,sr.number))
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