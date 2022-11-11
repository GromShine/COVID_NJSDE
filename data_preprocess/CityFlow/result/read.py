import pickle

# rb是2进制编码文件，文本文件用r
f = open(r'syn_2x2_gaussian_500_1h_result.pkl','rb')
data = pickle.load(f)
print(data)

f = open(r'syn_2x2_gaussian_500_1h_result2.pkl','rb')
data = pickle.load(f)
print(data)