import numpy as np
import pickle
import statistics

file_name = 'deformed_widowx_train_10hz_100K.pkl'
sim_train = pickle.load(open('sim_' + file_name, 'rb+'))
real_train = pickle.load(open('real_' + file_name, 'rb+'))
print(len(sim_train))
drifts = []
diffs = []

for i in range(4000, 5000):
    real = real_train[i]
    sim = sim_train[i]
    diff = real_train[i][0][-3:]-sim_train[i][0][-3:]
    diffs.append(real_train[i][0][:5]-sim_train[i][0][:5])
    drift = np.linalg.norm(diff)
    drifts.append(drift)
    print(i)

print(np.median(drifts))
print(statistics.stdev(drifts))

print(np.median(diffs, axis=0))
print(np.std(diffs, axis=0))
