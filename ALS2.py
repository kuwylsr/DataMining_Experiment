import numpy as np
import math

def ALS():
    for t in range(iterateNum):

        #更新矩阵U
        for r in range(n):
            for s in range(d):
                temp1 = 0
                temp2 = 0
                for j in range(m):
                    # if 0 not in M[:,j]:
                    if M[r][j] != 0:
                        temp3 = 0
                        for k in range(d):
                            if k != s:
                                temp3 += U[r][k] * V[k][j]
                        temp1 += V[s][j] * (M[r][j] - temp3)
                        temp2 += V[s][j] * V[s][j]
                U[r][s] = temp1 / temp2
        #更新矩阵V
        for r in range(d):
            for s in range(m):
                temp1 = 0
                temp2 = 0
                for i in range(n):
                    # if 0 not in M[i]:
                    if M[i][s] != 0:
                        temp3 = 0
                        for k in range(d):
                            if k != r:
                                temp3 += U[i][k] * V[k][s]
                        temp1 += U[i][r] * (M[i][s] - temp3)
                        temp2 += U[i][r] * U[i][r]
                V[r][s] = temp1 / temp2
        print(cal_RMSE())

def cal_RMSE():
    P = np.dot(U,V)
    sum = 0
    for i in range(n):
        for j in range(m):
            if (i,j) != (2,1) and (i,j) != (4,4):
                sum += (M[i][j] - P[i][j]) ** 2

    return math.sqrt(sum/(n*m-2))

if __name__ == "__main__":

    #M:有空白元素的nxm的效用矩阵
    #U：nxd维表示用户特征的矩阵
    #V：dxm维表示歌曲特征的矩阵
    #令P = UV

    iterateNum = 1000
    n = 5
    m = 5
    d = 3
    M = np.array([[5,2,4,4,3],
         [3,1,2,4,1],
         [2,0,3,1,4],
         [2,5,4,3,5],
         [4,4,5,4,0]])
    # M = np.array([[5,0,4],
    #      [3,1,2],
    #      [2,8,7]])

    U = np.ones([n,d],dtype=float)
    V = np.ones([d,m],dtype=float)

    ALS()
    print(np.dot(U,V))

