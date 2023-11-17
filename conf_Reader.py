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
import configparser
import numpy as np

class conf_Reader(object):
    def __init__(self,fileUrl):
        self._fileName = fileUrl
        self._conf = configparser.ConfigParser()
        self._error = ''
        self._stand =[]
        self._resourceName=[]
        self._standTitle=[]
        self._standWeight =[]
        self._ini_formula=False
        # data=np.array([[down limit],
        #                 [upper limit],
        #                 [price],
        #                 [nutrition 1],
        #                 [nutrition 2],
        #                 [......]])
        self._data = np.array([[]]) 
        self._readFile()
    def __can_all_convert_to_digits(self, lst):  
        for elem in lst:  
            if not elem.isdigit():  
                return False  
        return True
    def _readFile(self):
        try:
            self._conf.read(self._fileName,encoding='gb2312')
            sections = self._conf.sections()
            if 'Standard' not  in sections:
                self._error = 'Error, No Stand Found'
            sections = [a for a in sections if 'Resource' in a]
            
            stand = self._conf.get('Standard','Standard').split(',')
            self._standTitle = self._conf.get('Standard','StandTitle').split(',')
            self._stand = np.array([float(a) for a in stand])
            
            standWeight =  self._conf.get('Standard','StandWeight').split(',')
            self._standWeight = np.array([float(a) for a in standWeight])
            
           
            
            if len( self._standTitle) != len(self._stand):
                self._error = self._error +f'\nthe length of standTitle is {len(self._standTitle)} \
                    but the length of stand is {len(self._stand)}'
            if len( self._stand) != len(self._standWeight):
                 self._error = self._error +f'\nthe length of stand is {len(self._stand)} \
                     but the length of standWeight is {len(self._standWeight)}'
            data=[]
            for section in sections:
                one_Resource = self._get_One_Resource(section)
                if one_Resource:
                    name,price,nutrition,limit=one_Resource 
                    self._resourceName +=[name]
                    data.append( [limit+[price]+nutrition])
            self._data=np.concatenate(data,axis=0).transpose()
            
            ini_formula=self._conf.get('Initial Feed Formula','Formula').split(',')
            if self.__can_all_convert_to_digits(ini_formula) and len(ini_formula)==len(self._resourceName):
                self._ini_formula = np.array([float(a) for a in ini_formula])
                    
        except Exception as e:
            self._error = str(e)
    def getData(self):
        if self._data.size >0 :
            return self._data
        else:
            return 'No data'
    def getStand(self):
        if self._stand.size >0 :
            return self._stand
        else:
            return 'No stand'
    def getStandWeight(self):
        if self._standWeight.size >0 :
            return self._standWeight
        else:
            return 'No stand'
    def getResourceName(self):
        if len(self._resourceName) >0 :
            return self._resourceName
        else:
            return 'No resource'
    def getIniFormula(self):
        return self._ini_formula
    def getError(self):
        return self._error
    def getStandTitle(self):
        return self._standTitle
    def _get_One_Resource(self,section):
        try:
            name = self._conf.get(section,'Name')
            price  = float(self._conf.get(section,'Price'))
            nutrition= self._conf.get(section,'Nutrition Content').split(',')
            limit = self._conf.get(section,'Usage Limit').split(',') 
            nutrition = [float(a) for a in nutrition]
            limit = [float(a) for a in limit]
            if (len(nutrition)!=len(self._stand)):
                self._error += '\n'+ f'the nutrition lenth of resource {section} is inconsistent with stand ' 
            if (len(limit)!=2):
                self._error += '\n'+ f'the limit lenth of resource {section} should be 2 '
            return name,price,nutrition,limit
        except Exception as e:
            self._error += '\n'+ f'error on read resource {section} ' +str(e)
        return False

        