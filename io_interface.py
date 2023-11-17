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

import glob
from conf_Reader import conf_Reader
import os
import numpy as np
np.set_printoptions(suppress=True)#
class io_interface(object):
    def __init__(self,iotype='conf'):
        self._allconfFiles = glob.glob('files/*.conf') 
        self._currentIndex = -1 #
        self.iotype = iotype
        self._error = '' # store errors during calculation
        self._resourceName = []
        self._stand = []
        self._data  = []
        self._standTitle=[]
        self._hasFormula = False
    def getNext(self):
        self._currentIndex +=1
        if(len(self._allconfFiles)>self._currentIndex): 
            return self.__getNextconf()
        else:
            return False
    def setError(self,error):
        self._error += '\n'+ error
    def getProgress(self):
        return  self._currentIndex,len(self._allconfFiles),self._allconfFiles[self._currentIndex]
    def __ljust(self,txt):
        return txt.ljust(20," ")
    def __writeResource(self,io_hander):
        try:
            if self._error != '':
                io_hander.write("*************There are some errors during program execution********\n")
                io_hander.write(self._error)
                io_hander.write("\n********************************************************************************\n")
            io_hander.write('Resource\n')
            io_hander.write("_____________________________________\n")
            io_hander.write('\t'.join([self.__ljust('')]+self._resourceName)+"\n")
            resourceStr =  [ '\t'.join([str(b)for b in a]) for a in self._data]
            io_hander.write(self.__ljust('Lower usage limit') +'\t' + resourceStr[0] +"\n")
            io_hander.write(self.__ljust('Upper usage limit') +'\t' + resourceStr[1] +"\n")
            io_hander.write(self.__ljust('Price') + '\t' + resourceStr[2]+"\n")
            for i in range(3,len(resourceStr)):
                standName=self._standTitle[i-3] if (len(self._standTitle)>i-3) else ""
                io_hander.write(self.__ljust(standName) + '\t' + resourceStr[i] +"\n")
            io_hander.write("\n\n\n")
            
            io_hander.write("Stand\n")
            io_hander.write("_____________________________________\n")
            io_hander.write("\t".join(self._standTitle)+"\n")
            io_hander.write("\t".join([str(b)for b in self._stand]))
            io_hander.write("\n\n")
           
            io_hander.write("\n\n\n")    
        except Exception:
            pass
    def __writeFormula(self,io_hander,SLPF,price):
        io_hander.write("Formula\n")
        io_hander.write("_____________________________________\n")
        io_hander.write("Resource\tPrice\tUseage\n")
        # io_hander.write("_____________________________________\n")
        pricestr = [str(a) for a in self._data[2]]
        slpfstr = ['{:.2f}'.format(a)+"%" for a in SLPF*100]
        formulastr = np.concatenate([[self._resourceName],[pricestr],[slpfstr]],axis=0).transpose().tolist()
        io_hander.write('\n'.join('\t'.join(a) for a in formulastr))
        io_hander.write("\n_____________________________________\n")
        io_hander.write('Total\t'+'{:.2f}'.format(price)+'\t'+'{:.2f}'.format(np.sum(SLPF)*100))
        io_hander.write("\n\n\n")
    def __writeNuturntion(self,io_hander,nuturntion,difference):
        io_hander.write("Nutrition\n")
        io_hander.write("_____________________________________\n")
        io_hander.write("StandName\tStand\tFormula\tDifference\n")
        nuturntion = ['{:.2f}'.format(a)for a in nuturntion]
        difference = ['{:.2f}'.format(a)for a in difference]
        stand_str= np.concatenate([[self._standTitle],[self._stand],[nuturntion],[difference]],axis=0).transpose().tolist()
        io_hander.write('\n'.join('\t'.join(a) for a in stand_str))
        
    def __calNutruntion(self,SLPF):
        nuturntion = np.matmul(self._data[3:],SLPF.reshape([-1,1])).reshape([-1])
        difference = nuturntion - self._stand
        if np.any(difference/self._stand<-0.05):
            self.setError("The nutritional composition of the formula does not meet \
the feeding standards, please adjust the feed ingredients.")
        return nuturntion,difference
    def saveFormula(self,SLPF,price):
        if type(SLPF)==np.ndarray:
            self._hasFormula = True
        if self._hasFormula:
            self.__calNutruntion(SLPF)
        fileName = self._allconfFiles[self._currentIndex]
        resultUrl = fileName.replace('files','result').replace('.conf','.txt')
        if not os.path.exists(os.path.dirname(resultUrl)):
            os.makedirs(os.path.dirname(resultUrl))
        resultHander = open (resultUrl,'w')
        self.__writeResource(resultHander)
        if self._hasFormula: #if failed on formula, not show err
            self.__writeFormula(resultHander,SLPF,price)
            nuturntion, difference=self.__calNutruntion(SLPF)
            self.__writeNuturntion(resultHander, nuturntion, difference)
    def __getNextconf(self):
        conf_url = self._allconfFiles[self._currentIndex]
        oneResource = conf_Reader(conf_url)
        data = oneResource.getData()
        stand = oneResource.getStand()
        standWeight = oneResource.getStandWeight()
        ini_formula = oneResource.getIniFormula()
        self._stand = stand
        self._standWeight = standWeight
        self._data  = data
        self._error = oneResource.getError()
        self._resourceName = oneResource.getResourceName()
        self._standTitle = oneResource.getStandTitle()
        return data,stand,standWeight,ini_formula



