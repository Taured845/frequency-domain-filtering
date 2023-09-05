# -*- coding: utf-8 -*-
"""
Created on Wed May  3 19:03:22 2023

@author: 86131
"""

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import fft,ifft

# img=cv2.imread('8.jpg')#GBR
img=np.array(Image.open('D:/Tesseract-OCR/程序和图片/3.bmp'))

#转为RGB
tmp=img.copy()
img[:,:,0]=tmp[:,:,2]
img[:,:,2]=tmp[:,:,0]
del tmp

plt.figure()
plt.title('oringnal')
plt.imshow(img)  

# In[1] 预处理
#2.填充
G=img.copy()
G=np.concatenate((G,np.zeros((1,len(G[0]),3))),axis=0)#将行数变为偶数
G=np.concatenate((G,np.zeros((len(G),1,3))),axis=1)#将列数变为偶数
# #填充2倍
# G=np.concatenate((G,np.zeros_like(G)))
# G=np.concatenate((G,np.zeros_like(G)),axis=1)

#3.*(-1)^(x+y)
for i in range(len(G)):
    for j in range(len(G[0])):
        G[i][j][:]=G[i][j][:]*((-1)**(i+j))

# In[2] FFT
N=len(G[0]) #x、u取值范围
M=len(G) #y、v取值范围

#对x方向做FFT
G_e=G[:,[2*i for i in range(int(N/2))],:]#G的偶数列
G_o=G[:,[2*i+1 for i in range(int(N/2))],:]#G的奇数列
F_e=np.zeros((M,int(N/2),3)).astype(complex)#G的偶数列傅里叶变换
F_o=np.zeros((M,int(N/2),3)).astype(complex)#G的奇数列傅里叶变换
for u in range(int(N/2)):
    for k in range(int(N/2)):
        F_e[:,u,:] = F_e[:,u,:] + G_e[:,k,:] * (np.e)**((-2j*np.pi*u*k)/(N/2))
        F_o[:,u,:] = F_o[:,u,:] + G_o[:,k,:] * (np.e)**((-2j*np.pi*u*k)/(N/2))
w=np.array([(np.e)**((-2j*np.pi*u)/N) for u in range(int(N/2))]).reshape(1,-1)
w=np.concatenate((w,w,w)).reshape((1,-1,3))
F_o=w*F_o   
F_1=np.concatenate((F_e+F_o,F_e-F_o),axis=1)#x方向傅里叶变换 

#对y方向做FFT
F_1_e=F_1[[2*i for i in range(int(M/2))],:,:]#F_1的偶数行
F_1_o=F_1[[2*i+1 for i in range(int(M/2))],:,:]#F_1的奇数行
F_e=np.zeros((int(M/2),N,3)).astype(complex)#F_1的偶数行傅里叶变换
F_o=np.zeros((int(M/2),N,3)).astype(complex)#F_1的奇数行傅里叶变换
for v in range(int(M/2)):
    for k in range(int(M/2)):
        F_e[v,:,:] = F_e[v,:,:] + F_1_e[k,:,:] * (np.e)**((-2j*np.pi*v*k)/(M/2))
        F_o[v,:,:] = F_o[v,:,:] + F_1_o[k,:,:] * (np.e)**((-2j*np.pi*v*k)/(M/2))
w=np.array([(np.e)**((-2j*np.pi*v)/M) for v in range(int(M/2))]).reshape(-1,1)
w=np.concatenate((w,w,w)).reshape((-1,1,3))
F_o=w*F_o   
F=np.concatenate((F_e+F_o,F_e-F_o),axis=0)

#掉包实现(检验)
F_x=fft(G,axis=1)#x方向傅里叶变换
F=fft(F_x,axis=0)#G的二维傅里叶变换

plt.figure()
plt.title('(对数处理后)中心化后的频谱')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['axes.labelsize']='x-large'
Spectrum=np.log(1+np.abs(F)) #对数处理后的频谱
plt.imshow( (Spectrum-np.min(Spectrum))/(np.max(Spectrum)-np.min(Spectrum)) ) 

# In[3] 滤波
#1.距离
D_0=int(N/10) #截止频率半径(短边方向周期的1/p)
D=np.zeros_like(F).astype(float)#距离中心的距离
for i in range(int(M/2)):
    for j in range(int(N/2)):
        D[i][j][:] = D[i][N-j-1][:] = D[M-i-1][j][:] = D[M-i-1][N-j-1][:] = ((i-M/2)**2+(j-N/2)**2)**0.5

#2.低通    
n=1.5
H=1/(1+(D/D_0))**(2*n) #巴特沃斯
# H=(np.e)**(-D**2/(2*D_0**2)) #高斯
GF_low=H*F

plt.figure()
plt.title('巴特沃斯低通滤波函数')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['axes.labelsize']='x-large'
plt.imshow(H) 

plt.figure()
plt.title('(对数处理后)巴特沃斯低通滤波_频谱')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['axes.labelsize']='x-large'
Spectrum=np.log(1+np.abs(GF_low))
plt.imshow((Spectrum-np.min(Spectrum))/(np.max(Spectrum)-np.min(Spectrum))) 

# #3.高通
# H=1-H
# GF_high=H*F

# plt.figure()
# plt.title('巴特沃斯高通滤波函数')
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# plt.rcParams['axes.labelsize']='x-large'
# plt.imshow(H) 

# plt.figure()
# plt.title('(对数处理后)巴特沃斯高通滤波_频谱')
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# plt.rcParams['axes.labelsize']='x-large'
# Spectrum=np.log(1+np.abs(GF_high))
# plt.imshow((Spectrum-np.min(Spectrum))/(np.max(Spectrum)-np.min(Spectrum))) 


# In[4] IFFT
#1.低通
G_low=ifft(GF_low,axis=0)
G_low=ifft(G_low,axis=1)
G_low=G_low.real

#*(-1)^(x+y)
for i in range(len(G_low)):
    for j in range(len(G_low[0])):
        G_low[i][j][:]=G_low[i][j][:]*((-1)**(i+j))

G_low=G_low.astype(int)
G_low=G_low[:len(G_low)-1]
G_low=G_low[:,:len(G_low[0])-1]
        
plt.figure()
plt.title('巴特沃斯低通滤波')
plt.rcParams['font.sans-serif']=['SimHei']
plt.rcParams['axes.unicode_minus']=False
plt.rcParams['axes.labelsize']='x-large'
plt.imshow(G_low) 

#保存图片
G_low=Image.fromarray(G_low.astype(np.uint8))
G_low.save('D:/Tesseract-OCR/程序和图片/3_低通滤波.png')


# #2.高通
# G_=ifft
# G_high=ifft(GF_high,axis=0)
# G_high=ifft(G_high,axis=1)
# G_high=G_high.real

# #*(-1)^(x+y)
# for i in range(len(G_high)):
#     for j in range(len(G_high[0])):
#         G_high[i][j][:]=G_high[i][j][:]*((-1)**(i+j))

# G_high=G_high.astype(int)
# G_high=G_high[:len(G_high)-1]
# G_high=G_high[:,:len(G_high[0])-1]
        
# plt.figure()
# plt.title('巴特沃斯高通滤波')
# plt.rcParams['font.sans-serif']=['SimHei']
# plt.rcParams['axes.unicode_minus']=False
# plt.rcParams['axes.labelsize']='x-large'
# G_high=255*(G_high-np.min(G_high))/(np.max(G_high)-np.min(G_high))
# plt.imshow(G_high) 


