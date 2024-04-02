import pathlib
from oniakIO import odats
import numpy as np
from scipy.stats import binom

repetitions = {
    1.0: {
        # rank, algorithm, pd, L, T
        "sift": [
            (4, "sj", 25, 1, 1),
            (4, "jl", 128, 9, 2),
            (9, "sj", 25, 1, 1),
            (9, "jl", 128, 12, 3),
            (16, "sj", 25, 1, 1),
            (16, "jl", 128, 10, 2),
            (32, "sj", 48, 1, 1),
            (32, "jl", 128, 14, 3),
            (64, "sj", 80, 1, 1),
            (64, "jl", 128, 11, 4),
        ],
        "deep": [(9, "sj", 25, 1, 1), (9, "jl", 96, 12, 3)],
        "trevi": [
            (4, "sj", 64, 4, 1),
            (4, "jl", 4096, 86, 10),
            (4, "jlt", 256, 83, 16),
            (9, "sj", 32, 12, 1),
            (9, "jl", 4096, 93, 10),
            (9, "jlt", 256, 91, 15),
            (16, "sj", 64, 9, 1),
            (16, "jl", 4096, 81, 10),
            (16, "jlt", 256, 83, 10),
            (32, "sj", 64, 7, 1),
            (32, "jl", 4096, 87, 10),
            (32, "jlt", 256, 83, 15),
            (64, "sj", 96, 17, 1),
            (64, "jl", 4096, 75, 12),
            (64, "jlt", 256, 90, 10),
            (128, "sj", 192, 12, 1),
            (128, "jl", 4096, 89, 10),
            (128, "jlt", 256, 86, 23),
        ],
        "gist": [(9, "sj", 64, 1, 1), (9, "jl", 960, 45, 7), (9, "jlt", 128, 56, 10)],
        "mnist": [(9, "sj", 64, 1, 1), (9, "jl", 784, 24, 2), (9, "jlt", 128, 25, 2)],
        "enron": [(9, "sj", 64, 1, 1), (9, "jl", 1369, 91, 8), (9, "jlt", 128, 77, 8)],
    },
    1.2: {
        "sift": [(9, "sj", 25, 1, 1), (9, "jl", 128, 10, 3)],
        "trevi": [(9, "sj", 64, 2, 1), (9, "jl", 4096, 42, 7), (9, "jlt", 256, 42, 8)],
    },
}

root_file = pathlib.Path(__file__).parent.parent
l2d_file = str(root_file / "../LSHBundle/data/result/quantile2.odat")
l2d = odats.read_file(l2d_file)[0]

recall_bars = np.linspace(0.7, 0.9, 21)
nexp = len(l2d)+1
nrecalls = len(recall_bars)

# binary search
class ThresholdFinder:
    def __init__(self, L, T):
        # probability of having at least T successes in L trials
        self.transform = lambda x: 1 - binom.cdf(T-1, L, x)
        self.lb_recalls = np.full(nrecalls, 0.0)
        self.ub_recalls = np.full(nrecalls, 1.0)
        self.lb_thresholds = np.full(nrecalls, 0)
        self.ub_thresholds = np.full(nrecalls, nexp-2)
        
    def get_p(self, x):
        return self.transform(x / nexp)
    
    def calc(self):
        cnt = 0
        for i, recall in enumerate(recall_bars):
            ub = self.ub_thresholds[i]
            lb = self.lb_thresholds[i]
            
            while ub - lb > 1:
                cnt += 1
                mid = (ub + lb) // 2
                mid_recall = self.get_p(mid)

                j = i
                while j < nrecalls and mid_recall > recall_bars[j]:
                    j += 1
                # mid is ub for i<j and lb for i>=j
                for k in range(i, j):
                    if mid_recall <= self.ub_recalls[k]:
                        self.ub_thresholds[k] = mid
                        self.ub_recalls[k] = mid_recall
                for k in range(j, nrecalls):
                     if mid_recall >= self.lb_recalls[k]:
                        self.lb_thresholds[k] = mid
                        self.lb_recalls[k] = mid_recall
                
                ub = self.ub_thresholds[i]
                lb = self.lb_thresholds[i]
        print(cnt)
result = []

for lp, lp_dict in repetitions.items():
    for dataset, exp_list in lp_dict.items():
        for rank, solution, proj_dim, bigl, threshold in exp_list:
            if solution == "sj":
                continue
            
            tfinder = ThresholdFinder(bigl, threshold)
            tfinder.calc()
            cur_result = [bigl, threshold]
            cur_result.extend(np.sqrt(l2d[tfinder.ub_thresholds]))
            cur_result.extend(tfinder.ub_recalls)
            print(tfinder.ub_thresholds)
            result.append(cur_result)

result = np.array(result,dtype=np.float32)
odats.write_file("result/thresholds_theory.odat", result)
            
            
