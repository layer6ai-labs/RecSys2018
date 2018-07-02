import argparse
from scipy.sparse import *
import numpy as np
from sklearn.utils.extmath import randomized_svd

# Note: This file is for Java command call only, not part of this package at all.

def check_int_positive(value):
    ivalue = int(value)
    if ivalue < 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive int value" % value)
    return ivalue


def check_float_positive(value):
    ivalue = float(value)
    if ivalue <= 0:
         raise argparse.ArgumentTypeError("%s is an invalid positive float value" % value)
    return ivalue

def shape(s):
    try:
        num = int(s)
        return num
    except:
        raise argparse.ArgumentTypeError("Sparse matrix shape must be integer")


def load_csv(path, name, shape):
    data = np.genfromtxt(path + name, delimiter=',')
    matrix = csr_matrix((data[:, 2], (data[:, 0].astype('int32'), data[:, 1].astype('int32'))), shape=shape)
    return matrix

def save_np(matrix, path, name):
    np.savetxt(path + name, matrix, delimiter=',', fmt='%.5f')

def main(args):
    print("Reading CSV")
    matrix_input = load_csv(path=args.path, name=args.train, shape=args.shape)
    print("Perform SVD")
    P, sigma, Qt = randomized_svd(matrix_input,
                                  n_components=args.rank,
                                  n_iter=args.iter,
				  power_iteration_normalizer='LU',
                                  random_state=1)

    PtimesS = np.matmul(P, np.diag(sigma))
    print "computed P*S"

    #Pt = P.T
    save_np(PtimesS, args.path, args.user)
    print "saved P*S"

    save_np(Qt.T, args.path, args.item)
    print "saved Q"

    save_np(sigma, args.path, args.sigm)
    print "saved s"


if __name__ == "__main__":
    # Commandline arguments
    parser = argparse.ArgumentParser(description="SVD")
    parser.add_argument('-i', dest='iter', type=check_int_positive, default=4)
    parser.add_argument('-r', dest='rank', type=check_int_positive, default=100)
    parser.add_argument('-d', dest='path', default="/media/wuga/Storage/python_project/wlrec/IMPLEMENTATION_Projected_LRec/data/")
    parser.add_argument('-f', dest='train', default='matrix.csv')
    parser.add_argument('-u', dest='user', default='U.nd')
    parser.add_argument('-v', dest='item', default='V.nd')
    parser.add_argument('-s', dest='sigm', default='S.nd')
    parser.add_argument('--shape', help="CSR Shape", dest="shape", type=shape, nargs=2)
    args = parser.parse_args()

    main(args)
