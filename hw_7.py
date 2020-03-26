import matplotlib.pyplot as plt
from surprise import KNNBasic
from surprise import NMF
from surprise import SVD
from surprise import Dataset
from surprise import evaluate, print_perf
import os
from surprise import Reader#load data from a file
file_path = os.path.expanduser('restaurant_ratings.txt')
reader = Reader(line_format='user item rating timestamp', sep='\t')
data = Dataset.load_from_file(file_path, reader=reader)

def number11():
    data.split(n_folds=2)
    svd_algo = SVD()
    perf = evaluate(svd_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    pmf_algo = SVD(biased=False)
    perf = evaluate(pmf_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    nmf_algo = NMF()
    perf = evaluate(nmf_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    ubcf_algo = KNNBasic(sim_options = {
    'user_based': True
    })
    perf = evaluate(ubcf_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    ibcf_algo = KNNBasic(sim_options = {
    'user_based': False
    })
    perf = evaluate(ibcf_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

def number12():
    data.split(n_folds=3)
    svd_algo = SVD()
    perf = evaluate(svd_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    pmf_algo = SVD(biased=False)
    perf = evaluate(pmf_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    nmf_algo = NMF()
    perf = evaluate(nmf_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    ubcf_algo = KNNBasic(sim_options = {
    'user_based': True
    })
    perf = evaluate(ubcf_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    ibcf_algo = KNNBasic(sim_options = {
    'user_based': False
    })
    perf = evaluate(ibcf_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

def number14():
    data.split(n_folds=3)
    ubcf_msd_algo = KNNBasic(sim_options = {
        'name':'MSD',
        'user_based': True
        })
    perf = evaluate(ubcf_msd_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    ubcf_cosine_algo = KNNBasic(sim_options = {
        'name':'cosine',
        'user_based': True
        })
    perf = evaluate(ubcf_cosine_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    ubcf_pearson_algo = KNNBasic(sim_options = {
        'name':'pearson',
        'user_based': True
        })
    perf = evaluate(ubcf_pearson_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    ibcf_msd_algo = KNNBasic(sim_options = {
        'name':'MSD',
        'user_based': False
        })
    perf = evaluate(ibcf_msd_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    ibcf_cosine_algo = KNNBasic(sim_options = {
        'name':'cosine',
        'user_based': False
        })
    perf = evaluate(ibcf_cosine_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

    ibcf_pearson_algo = KNNBasic(sim_options = {
        'name':'pearson',
        'user_based': False
        })
    perf = evaluate(ibcf_pearson_algo, data, measures=['RMSE', 'MAE'])
    print_perf(perf)

def number15():
    data.split(n_folds=3)
    k_num = 1
    ks = []
    ubcf_rmses = []
    ibcf_rmses = []
    while k_num <= 101:
        ubcf_msd_algo = KNNBasic(k = k_num, sim_options = {
            'name':'MSD',
            'user_based': True
            })
        ubcf_perf = evaluate(ubcf_msd_algo, data, measures=['RMSE'])
        print_perf(ubcf_perf)
        for key,value in ubcf_perf.items():
            mean = 0
            for v in value:
                mean = mean + v
            print(mean)
            mean = mean/3
            print(mean)
            ubcf_rmses.append(mean)

        ibcf_msd_algo = KNNBasic(k = k_num, sim_options = {
            'name':'MSD',
            'user_based': False
            })
        ibcf_perf = evaluate(ibcf_msd_algo, data, measures=['RMSE'])
        print_perf(ibcf_perf)
        for key,value in ibcf_perf.items():
            mean = 0
            for v in value:
                mean = mean + v
            print(mean)
            mean = mean/3
            print(mean)
            ibcf_rmses.append(mean)

        print(k_num)
        ks.append(k_num)
        k_num += 10

    plt.bar(ks, ubcf_rmses)
    plt.show()
    plt.bar(ks, ibcf_rmses)
    plt.show()

def main():
    number15()

if __name__=="__main__":
    main()
