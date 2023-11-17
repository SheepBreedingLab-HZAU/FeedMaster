# Copyright (c) 2023 Jiang Xunping and Sun Ling

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# Licensed under the MIT license;

import numpy as np
import tensorflow as tf
# from pulpLineCal import CalSLPFLine
import functools
import sys

np.set_printoptions(suppress=True)# not to output in Scientific notation

lambda1 = 40
lambda2 = 128

class MyRuntimeError(BaseException):#程序运行时抛出的错误
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg
class SentToUserRuntimeError(BaseException):#抛出给用户查看的错误
    def __init__(self, msg):
        self.msg = msg

    def __str__(self):
        return self.msg

def vertify(stand,data):
    resourceYYCF=data[3:,:].transpose(1,0)#饲料营养成分数据
    syxx=data[0,:]/100  #使用下限
    sysx=data[1,:]/100  #使用上限
    slpf=(syxx+sysx)/2
    slpf=slpf/sum(slpf)
    #slpf=np.array([0.67, 0.15, 0.15, 0.01, 0.01, 0.01])
    #syxx=np.array([0.01, 0.01, 0.01, 0.2 , 0.1 , 0.05])
    #sysx=np.array([0.5 , 0.9 , 0.9 , 0.9 , 0.2, 0.2])
    
    if(stand.shape[0]!=resourceYYCF.shape[1]):#判断营养标准的参数数量是否与饲料配方中营养物质含量一致
        raise MyRuntimeError("The parameter quantity set for feeding standards is inconsistent with the parameter quantity of feed ingredients ")
    if(max(syxx-sysx)>0): #饲料原料使用下线大于使用上限
        raise SentToUserRuntimeError("The lower usage of feed ingredients set should be lower than the upper limit of usage")
    if(sum(syxx)>1): #使用下限之合大于1
        raise SentToUserRuntimeError("The sum of lower usage limits should be less than 1")  
    if(sum(sysx)<1): #使用下限之合大于1
        raise SentToUserRuntimeError("The sum of upper usage limits should be greater than 1")
        
def build_optimizer(global_step=None,optim = 'RMSprop'):
    """
        global_step: A variable representing the current step.
    """
    
    optimizer = None
    if global_step is None:
        global_step = tf.compat.v1.train.get_or_create_global_step()
    learning_rate_base = 0.001
    total_steps = 1000
    def learning_rate_fn():
        learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
                np.pi *
                (tf.cast(global_step, tf.float32) 
                ) / float(total_steps)))
     
        return tf.where(global_step > total_steps, 0.0, learning_rate,
                                        name='learning_rate')
    def learning_rate_fn2():
        learning_rate = 0.5 * learning_rate_base * (1 + tf.cos(
                np.pi *
                (tf.cast(global_step, tf.float32) 
                ) / float(total_steps)))*10
     
        return tf.where(global_step > total_steps, 0.0, learning_rate,
                                        name='learning_rate')
    if optim=='Adadelta':
        optimizer = tf.keras.optimizers.Adadelta(
            learning_rate_fn)
    elif optim== 'Adagrad':
        optimizer = tf.keras.optimizers.Adagrad(
        learning_rate_fn)
    elif optim==      'Adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate_fn)
    elif optim==      'Adamax':
        optimizer = tf.keras.optimizers.Adamax(
            learning_rate_fn)
    elif optim==      'Ftrl':
        optimizer = tf.keras.optimizers.Ftrl(
            learning_rate_fn)
    elif optim==      'Nadam':
        optimizer = tf.keras.optimizers.Nadam(
            learning_rate_fn)
    elif optim==    'RMSprop':
          optimizer = tf.keras.optimizers.RMSprop(
              learning_rate_fn)
    elif optim==    'SGD':
          optimizer = tf.keras.optimizers.SGD(
              learning_rate_fn2)
    
    

    return optimizer, learning_rate_fn
class SLPF_Model(tf.keras.Model):
    def __init__(self,data,stand,ini_formula):
        super(SLPF_Model, self).__init__()
        resourceYYCF=data[3:,:].transpose(1,0)#饲料营养成分数据
        syxx=data[0,:]/100  #使用下限
        sysx=data[1,:]/100  #使用上限
        #price=data[2,:]     #原料价格 
        self.stand=tf.constant(stand,dtype=tf.float64)
        self.resourceYYCF=tf.constant(resourceYYCF,dtype=tf.float64)
        self.syxx=tf.constant(syxx,dtype=tf.float64)
        self.sysx=tf.constant(sysx,dtype=tf.float64)
        
        #从数据库中存储的配方中设置初始值 
        #大数据 配方库
        #每种营养标准 记录各个地区的配方
        #跟原系统比   精度提高了多少、速度提高了多少
        #self.price=tf.convert_to_tensor(price,dtype=tf.float64)
        # price = data[2,:]
        # Nutrition = data[3:,:]
        
        # slpfin = CalSLPFLine(price,Nutrition,stand)
        if ini_formula:
            slpfin = ini_formula
        else:
            slpfin=tf.Variable((syxx+sysx)/2,dtype=tf.float64,name='slpf') 
        self.slpfin=tf.Variable(slpfin,dtype=tf.float64,name='slpf') 
        #self.slpfin=tf.Variable([ 0.52978666,  0.48079258,  0.51442608, -0.01448196, -0.95694631,
        #       -1.86566855],dtype=tf.float64,name='slpf')
    def call(self,a):
        """
        Parameters
        ----------
        slpf : TYPE array
            DESCRIPTION.
        Returns
        -------
        None.
    
        """    
        #slpf=tf.where(self.slpfin<0,0,self.slpfin) 
        slpf=tf.divide(self.slpfin,tf.reduce_sum(self.slpfin))
        #yycfQs=tf.subtract(pfyycf,stand)                    #存储配方中缺少哪些营养成分 
        pfyycf=tf.matmul(tf.reshape(slpf,[1,-1]),self.resourceYYCF)
        yycfQs=pfyycf-self.stand
        return yycfQs,slpf
    #model=models.Model(inputs=[stand,resourceYYCF,syxx,sysx],outputs=[yycfQs,slpf_result])    
    
    

class modifySLPF(object):
    def __init__(self,model):
        self.optimizer=tf.keras.optimizers.SGD(learning_rate=0.01)
        self.model=model
        self.need=1 #调整饲料配方总含量到100%
    def minargNotZero(self,arr):  #取出除0以外最接近0的值所在的序号
        arr1=np.where(arr==0,100,arr)#不要修改原数组
        #arr1[arr1==0.]=100#将0变大
        return np.argmin(arr1)
    def modify(self):
        slpf            =self.model.variables[0].numpy()
        syxx            =np.array(self.model.syxx) 
        sysx            =np.array(self.model.sysx) 
        slpf_Only_lt    =np.where(slpf>syxx,slpf,syxx)    #将低于使用下限的饲料原料调整到使用下限
        slpf            =np.where(slpf_Only_lt<sysx,slpf_Only_lt,sysx)    #按照使用上下限调整配方
        arg_lt          =np.where(slpf==syxx)[0] #计算配方中达到使用下限的营养物质序号
        arg_gt          =np.where(slpf==sysx)[0] #计算配方中达到使用上限的营养物质序号
        if(sum(slpf)>1):
            #比例调低饲料配方
            slpfLtCount=sum(slpf[arg_lt])#计算达到使用下限的营养物质总含量
            slpfCount=sum(slpf)#饲料配方总百分比
            rato=(slpfCount-slpfLtCount)/(self.need-slpfLtCount)
            #计算调低比例
            slpf[slpf!=syxx]=slpf[slpf!=syxx]/rato  #将该原料使用量还原，以便调参
        if(sum(slpf)<1):
            #按比例调高饲料配方
            slpfGtCount=sum(slpf[arg_gt])#计算达到使用上限的营养物质总含量
            slpfCount=sum(slpf)#饲料配方总百分比
            rato=(slpfCount-slpfGtCount)/(self.need-slpfGtCount)
            #计算调低比例
            slpf[slpf!=sysx]=slpf[slpf!=sysx]/rato  #将该原料使用量还原，以便调参
        return slpf
    #先计算原料使用量是调高还是调低
    def autoModify(self):
        i=0
        slpf=self.modify()
        while(abs(sum(slpf)-1)>0.001):
            i=i+1
            slpf=self.modify()
            if i>500:
                raise SentToUserRuntimeError("Parameter adjustment failed, please verify the input data")
                break
        grad=(self.model.variables[0].numpy()-slpf)/self.optimizer.lr.numpy()
        gradtensor=tf.constant(grad,self.model.variables[0].dtype)
        self.optimizer.apply_gradients(grads_and_vars=zip([gradtensor], self.model.variables))  

def calSLPF(data,stand,standWeight,ini_formula,show=0,process=[0,0]):
    global_step=tf.Variable(0,trainable=False,name='global_step',dtype=tf.dtypes.int64)
    resultArr=[]  
    if type(stand) != np.ndarray or type(standWeight) != np.ndarray :
        return '',''
    
    vertify(stand,data)
    price=tf.constant(data[2])
    
    model = SLPF_Model(data,stand,ini_formula)  
    # optimizer = tf.keras.optimizers.Nadam(learning_rate=0.0001)  #用Nadam好，其它几个容易陷入局部最优解陷阱
    optimizer,_ = build_optimizer(global_step)
    modify=modifySLPF(model)
    stand=tf.constant(stand,tf.float64)#营养标准
    for i in range(1000):
        with tf.GradientTape() as tape:
            yycfQs,slpf=model(i)
            lossu=tf.where(yycfQs>0,yycfQs,0)
            lossl=tf.where(yycfQs<0,tf.abs(yycfQs),0)
            # loss=tf.reduce_sum(tf.divide(lossu,stand))+tf.reduce_sum(tf.divide(lossl,stand))*50+tf.reduce_sum(tf.multiply(price,slpf))/tf.reduce_mean(price)
            loss1 = tf.reduce_sum(standWeight*lossu/stand)+tf.reduce_sum(standWeight*lossl/stand)*lambda1
            loss2 = (tf.reduce_sum(standWeight*lossu)+tf.reduce_sum(standWeight*lossl)*lambda1)/tf.reduce_sum(stand)
            loss3 = tf.reduce_sum(price*slpf)
            
            loss=loss1 + loss2 + lambda2 * loss3/np.mean(price)
            resultArr.append([sum(np.array(price*slpf)),np.array(slpf),np.array(yycfQs/stand)])
        grads = tape.gradient(loss, model.variables)
        # print("i={},loss={}".format(i,loss))
        if show>0 and type(show)==int:#默认不显示
            if i % 200==0:
                print("{}/{} setp:{},nutrition loss={:.2f} price={:.2f}\n".format(
                      process[0],process[1],i,loss1,sum(np.array(price*slpf))))
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables)) 
        # if i % 1==0:
        #     print("\tgrads   =\t{}\n\tNewmodeVar=\t{}".format(np.array(grads),np.array(model.variables[0])))
        modify.autoModify()
        global_step.assign_add(1)
        # if i % 1==0:
        #     print("\tModifyModeVar=\t{}\n\n\n\n\n\n".format(np.array(model.variables[0])))
    
        
        # print(next(lt))
    # except MyRuntimeError as e:
    #     print("错误类型是{}\n发生错误的文件是{}\n发生错误的行数是{}\n".format(e.__class__,e.__traceback__.tb_frame.f_globals["__file__"],e.__traceback__.tb_lineno))
    # except SentToUserRuntimeError as e:
    #     aaa.append(e)
    #     print(str(e.__class__))
    #     print(e)
    
        
    # for i in range(200):
    #     print(next(lt))
    #比对，并挑出符合要求，且价格最低的饲料配方
    def comp(x,y):
        return x[0]-y[0]
    resultArr.sort(key=functools.cmp_to_key(comp))
    retSLPF=[]
    price=0
    for i in range(len(resultArr)):
        if(np.all(resultArr[i][2]>-0.01)):
            retSLPF=resultArr[i][1]
            price=resultArr[i][0]
            break
    for k in range(1,15):
        if len(retSLPF)!=0:
            break
        for i in range(len(resultArr)):
            if(np.all(resultArr[i][2]>-0.01*k)):
                retSLPF=resultArr[i][1]
                price=resultArr[i][0]
                break
    if len(retSLPF)==0:
        retSLPF=resultArr[0][1]
        price=resultArr[0][0]
    return retSLPF,price
    #print("最终的饲料配方是：{}\n配方价格是：{}".format(retSLPF,price))
