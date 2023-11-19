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
import os
from Formula_Class import calSLPF
from io_interface import io_interface
from Formula_Class import SentToUserRuntimeError,MyRuntimeError
import time
import numpy as np

if __name__=='__main__':

    allData = io_interface()
    while(True):
        singleDate = allData.getNext()
        if not singleDate: # 所有列表均已计算完成
            break
        data,stand,standWeight,ini_formula = singleDate
        try:
            tstart = time.time()
            currentIndex,total_Query,_ = allData.getProgress()
            SLPF,price=calSLPF(data, stand,standWeight,ini_formula, 0, [currentIndex,total_Query])
            
            allData.saveFormula(SLPF,price)
            time.sleep(2)
            currentt=time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            print(f"{currentt} complete {currentIndex}/{total_Query}")
        except SentToUserRuntimeError as e:
            SLPF=np.zeros([len(data[0])])
            price=0
            allData.setError(str(e))
            allData.saveFormula(SLPF,price)
        except MyRuntimeError as e:
            print(str(e))
        except BaseException as e:
            print("\n\n\n\n\n")
            #print(e.__traceback__.tb_frame.f_globals)
            print("\n\n\n\n\n")
            print("错误类型是{}\n\n发生错误的行数是{}\n错误信息{}".format(e.__class__,e.__traceback__.tb_lineno,e.__str__))  
        except Exception:
            pass
