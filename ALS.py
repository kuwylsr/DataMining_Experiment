import numpy as np


def als():
    for i in range(iterateNum):
        #固定V，逐个更新所有用户U_r的特征
        for r in range(n):
            U[r] = (np.dot(np.dot(np.linalg.inv(np.dot(V.T, V)), V.T),M[r].T))
            # U[r] = (np.dot(M[r], np.dot(V, (np.linalg.pinv(np.dot(V.T, V))).T)))
            # print(np.dot(V.T, V))
            # print(np.linalg.pinv(np.dot(V.T, V)))
            # print(np.dot(np.linalg.pinv(np.dot(V.T, V)), V.T))
            # print(np.dot(M[r], np.dot(np.linalg.pinv(np.dot(V.T, V)), V.T)))
            # print(U)
            # print("=============================================")

        for s in range(m):
            V[s] = (np.dot(np.dot(np.linalg.inv(np.dot(U.T, U)), U.T), M[:,s]))

            # V[s] = (np.dot(np.dot(np.linalg.pinv(np.dot(U.T, U)), U.T), np.array(M)[s]).T)

        # print(U)
        # print(V)
        err = cal_RMSE()

        print(err)

def cal_RMSE():

    P = np.dot(U,V.T)
    sum = 0
    for i in range(n):
        for j in range(m):
            sum += (M[i][j] - P[i][j]) ** 2
    return sum/(n*m)


if __name__ == "__main__":

    #M:有空白元素的nxm的效用矩阵
    #U：nxd维表示用户特征的矩阵
    #V：dxm维表示歌曲特征的矩阵
    #令P = UV

    iterateNum = 50
    n = 5
    m = 5
    d = 3
    M = np.array([[5,2,4,4,3],
         [3,1,2,4,1],
         [2,0,3,1,4],
         [2,5,4,3,5],
         [4,4,5,4,0]])

    U = np.ones([n,d],dtype=float)
    V = np.ones([d,m],dtype=float).T
    print(M[:,2].shape)
    # print(U)
    # print(np.dot(U.T , U))
    # print(np.linalg.pinv(np.dot(U.T , U)))
    # exit(0)
    # print(np.array(M)[:,2])
    als()
    print(np.dot(U,V.T))
