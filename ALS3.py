import numpy as np
import math
import re
import time
import random
from copy import deepcopy
import matplotlib.pyplot as plt

def ReadHandle_artist_alias(path):
    dict = {} #建立字典，key值为：errorID，value值为：rightID
    with open(path,'r') as f:
        for line in f:
            # 将artist_alias中，小于二维的数据行过滤
            if re.match('\d+\t\d+',line) != None:
                content = line.split('\t')
                dict[content[0]] = content[1].replace("\n","")
    print("Read and Handle artist_alias successfully!")
    return dict

def ReadHandle_artist_data(path):
    dict = {} #建立字典，key值为：artistID，value值为：artistNAME
    with open(path,'r') as f:
        for line in f:
            # 将artist_data中，不符合要求的数据行过滤
            if re.match('\d+\t[\S]',line) != None:
                content = line.split('\t')
                dict[content[0]] = content[1].replace("\n","")
    print("Read and Handle artist_data successfully!")
    return dict

def ReadHandleWrite_user_artist_data(path1,path2):
    avg = 0 #统计所有数据的平均值
    num = 0
    list = []
    with open(path1,'r') as f1:
        for line in f1:
            num += 1
            if num%1000000 == 0:
                print(num)
            content = line.split(' ')
            #过滤不符合要求的数据行
            if len(content) < 3:
                continue
            # 将user_artist_data中，不符合要求的artistID根据artist_alias_dict进行修改
            artistID = content[1]
            rightID = artist_alias_dict.get(artistID)
            if rightID != None:
                # temp = line.replace(artistID,artist_alias_dict[artistID])
                temp = content[0] + " " + rightID + " "+content[2]
            else:
                temp = line
            avg += int(content[2])
            list.append(temp)
            #将正确的user_artist_data写入文件
    with open(path2,"a+") as f2:
        f2.writelines(list)
    print("Read and Handle and Write user_artist_data successfully!")
    return avg/len(list)

def Read_user_artist_list_after(path):
    row_userID_dist = {} #建立字典，key值为：行数，value值为：user的id
    userID_row_dist = {} #建立字典，key值为：user的id，value值为：行数
    col_artistID_dist = {} #建立字典，key值为：列数，value值为：artist的id
    artistID_col_dist = {}  #建立字典，key值为：artist的id，value值为：列数
    row_col_Frequency_dist = {} #建立字典，key值为：行数，value值为：（字典：key：列数，value：频数）
    M_dist = {} #训练集（与上面形式相同）
    T_list = [] #测试集 （列表中保存的是测试集数据在矩阵中的坐标元组（x，y））
    row = 0
    col = 0
    with open(path,'r') as f:
        for line in f:
            content = line.split(' ')
            userID = content[0]
            artistID = content[1]
            frequency = int(content[2])

            if int(frequency) == 0 : #不存播放次数为0的项
                continue
            if userID_row_dist.get(userID) == None:
                userID_row_dist[userID] = row
                row_userID_dist[row] = userID
                row_col_Frequency_dist.update({row : {}})#若该行号没有出现过，则存入字典，并且value为一个空的字典
                # M_dist.update({row : {}})
                row += 1
            if artistID_col_dist.get(artistID) == None:
                artistID_col_dist[artistID] = col
                col_artistID_dist[col] = artistID
                col += 1
            #20%的概率将其加入到测试集，80%的概率加入到训练集
            tempNum = random.uniform(1,10)
            if tempNum <= 2:
                T_list.append((userID_row_dist[userID],artistID_col_dist[artistID]))
            else:
                if M_dist.get(userID_row_dist[userID]) is None:
                    M_dist.update({userID_row_dist[userID] : {}})
                M_dist.get(userID_row_dist[userID]).update({artistID_col_dist[artistID] : frequency})
            #加入到原始矩阵中
            row_col_Frequency_dist.get(userID_row_dist[userID]).update({artistID_col_dist[artistID] : frequency})
    print("Read user_artist_list_after successfully!")
    return row_col_Frequency_dist, M_dist, T_list, row_userID_dist, col_artistID_dist, row, col

def als(M,U,V):
    for t in range(iterateNum):
        #更新矩阵U
        for r in range(n):
            MgetR = M.get(r)
            if MgetR is not None: #如果此行都为空，则直接跳过
                for s in range(d):
                    temp1 = 0
                    temp2 = 0
                    for j in range(m):
                        #判断M_dist[r][j]是否存在
                        if MgetR.get(j) is not None:
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
                    #判断M_dist[i][s]是否存在
                    MgetI = M.get(i)
                    if MgetI is not None:
                        if MgetI.get(s) is not None:
                            temp3 = 0
                            for k in range(d):
                                if k != r:
                                    temp3 += U[i][k] * V[k][s]
                            temp1 += U[i][r] * (M[i][s] - temp3)
                            temp2 += U[i][r] * U[i][r]
                if temp2 == 0 or temp1 == 0:
                    break
                V[r][s] = temp1 / temp2

        print("NO.",t,"end!")
        temp_rmse_test = cal_RMSE_test()
        temp_rmse_train = cal_RMSE_train()
        rmse_test.append(temp_rmse_test)
        rmse_train.append(temp_rmse_train)
        print("RMSE_test: ", temp_rmse_test)
        print("RMSE_train: ", temp_rmse_train)

def cal_RMSE_test():
    P = np.dot(U,V)
    sum = 0
    for (x,y) in T_list:
        sum += (row_col_Frequency_dist[x][y] - P[x][y]) ** 2
    return math.sqrt(sum/len(T_list))

def cal_RMSE_train():
    P = np.dot(U,V)
    sum = 0
    for r,value in M_dist.items():
        for c in value.keys():
            sum += (M_dist[r][c] - P[r][c]) ** 2
    return math.sqrt(sum/(4*len(T_list)))

def recommend(row_col_Frequency_dist, T_list,filepath, row_userID_dist, col_artistID_dist, artist_data_dict, k):
    P = np.dot(U,V)
    #将预测值修改到原始矩阵中
    for (x,y) in T_list:
        row_col_Frequency_dist.get(x).update({y : P[x][y]})
    #对原始矩阵的每一行（是一个map）进行按照value值排序
    with open(filepath,"w+") as f:
        for r in range(row):
            tempMap = row_col_Frequency_dist.get(r)
            sorted(tempMap.items(), key=lambda item:item[1])
            #将每一行（每个用户）的推荐结果写入文件
            tempK = 0
            recommendArtist = ""
            for c in tempMap.keys():
                if tempK >= k :
                    break
                recommendArtist = recommendArtist + " // " + artist_data_dict[col_artistID_dist[c]]
                tempK += 1
            f.write(row_userID_dist[r] + ": " + recommendArtist + "\n")

def plot_rmse():
    x = np.linspace(0, len(rmse_train), len(rmse_train))
    plt.figure()
    plt.plot(x, rmse_train)
    plt.plot(x, rmse_test)
    plt.savefig('mse.jpg')

if __name__ == "__main__":

    user_artist_data_path = "E:\\python-workspace\\DataMining\\user_artist_data_small.txt"
    artist_alias_path = "E:\\python-workspace\\DataMining\\artist_alias.txt"
    artist_data_path = "E:\\python-workspace\\DataMining\\artist_data_small.txt"

    user_artist_data_after_path = "E:\\python-workspace\\DataMining\\user_artist_data_after_small.txt"
    recommend_path = "E:\\python-workspace\\DataMining\\recommend.txt"

    artist_alias_dict = ReadHandle_artist_alias(artist_alias_path) #建立字典，key值为：errorID，value值为：rightID
    artist_data_dict = ReadHandle_artist_data(artist_data_path) #建立字典，key值为：artistID，value值为：artistNAME
    avg = ReadHandleWrite_user_artist_data(user_artist_data_path,user_artist_data_after_path) #写入新的user_artist_data_after.txt

    # print(avg)
    # user_artist_data_after_path = "E:\\python-workspace\\DataMining\\test.txt"
    row_col_Frequency_dist, M_dist, T_list, row_userID_dist, col_artistID_dist, row, col = Read_user_artist_list_after(user_artist_data_after_path)

    rmse_test = []
    rmse_train = []

    iterateNum = 15
    n = row
    m = col
    d = 15

    #初始化分解矩阵
    # U = np.full([n,d],avg,dtype=float)
    # V = np.full([d,m],avg,dtype=float)
    U = np.ones([n,d],dtype=float)
    V = np.ones([d,m],dtype=float)

    als(M_dist,U,V)
    k = 5
    recommend(row_col_Frequency_dist, T_list,recommend_path,row_userID_dist,col_artistID_dist,artist_data_dict, k)
    plot_rmse()
