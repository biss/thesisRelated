import fcrm, numpy as np, copy
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import confusion_matrix as cm
import csv

rms_allf = []
rms_lessf = []
ari = []
model = []
for i in range(1, 30):
    u = np.random.uniform(0, 1.0, [2, 804])
    u_r = copy.copy(u)

    sr = fcrm.SwitchingRegression('input/kuiper.csv', u)
    sr_r = fcrm.SwitchingRegression('input/kuiper_3f.csv', u_r)

    sr.train()
    sr.fill_err()

    sr_r.train()
    sr_r.fill_err()

    #print 'u of sr ', sr.U[:, :10]
    #print 'u of sr_r ', sr_r.U[:, :10]

    predicted = list(list(sr.U[:, i]).index(max(list(sr.U[:, i]))) for i in range(sr.number))
    predicted_r = list(list(sr_r.U[:, i]).index(max(list(sr_r.U[:, i]))) for i in range(sr_r.number))

    print sr_r.training_iter
    print sr.training_iter

    print predicted
    print predicted_r

    print 'ari :',adjusted_rand_score(predicted, predicted_r)
    confusion_matrix = cm(predicted, predicted_r)
    np.set_printoptions(precision=2)
    #print 'Confusion matrix, without normalization for lambda = ', element, ' and round no ', i
    print(confusion_matrix)

    #print 'betas are: ', sr.regime_param
    #print 'betas reduced are: ', sr_r.regime_param
    print 'mse mse_r', sr.err, sr_r.err

    rms_allf.append(sr.err)
    rms_lessf.append(sr_r.err)
    ari.append(adjusted_rand_score(predicted, predicted_r))
    model.append([sr.regime_param, sr_r.regime_param])

res = {'ari': ari, 'rmse_all': rms_allf, 'rmse_3f': rms_lessf}
print model

with open('output/tmp_3.csv', 'wb') as file:
    writer = csv.writer(file, delimiter=',')
    writer.writerow(res.keys())
    for row in zip(*res.values()):
        writer.writerow(list(row))
