import math
import numpy as np
import matplotlib.pyplot as plt

######################INPUTS#################
i1 = float(raw_input("input1: "))
i2 = float(raw_input("input2: "))
#############################################

#################TARGET######################
target1 = float(raw_input("target1: "))
target2 = float(raw_input("target2: "))
#############################################

################LEARNING RATE################
n = float(raw_input("n: "))
#############################################

#################WEIGHTS#####################
w1 = 0.15
w2 = 0.25
w3 = 0.40
w4 = 0.50
w5 = 0.20
w6 = 0.30
w7 = 0.45
w8 = 0.55
#############################################
iteration = list()
output1 = list()
output2 = list()
for i in range(1000000):
        iteration.append(i)
        neth1 = i1*w1 + i2*w5 + 0.35
        neth2 = i2*w6 + i1*w2 + 0.35

        outh1 = (1)/(1+math.exp(-neth1))
        outh2 = (1)/(1+math.exp(-neth2))

        neto1 = outh1*w3 + outh2*w7 + 0.60
        neto2 = outh2*w8 + outh1*w4 + 0.60

        outo1 = (1)/(1+math.exp(-neto1))
        outo2 = (1)/(1+math.exp(-neto2))

        output1.append(outo1)
        output2.append(outo2)

        eo1 = math.pow((target1 - outo1),2)*0.5
        eo2 = math.pow((target2 - outo2),2)*0.5

        #update weights
        roc = (( ( (outo1 - target1)*outo1*(1-outo1)*w3) + ((outo2 - target2)*outo2*(1-outo2)*w4) )*outh1*(1-outh1)*i1)
        w1 = w1 - n*(( ( (outo1 - target1)*outo1*(1-outo1)*w3) + ((outo2 - target2)*outo2*(1-outo2)*w4) )*outh1*(1-outh1)*i1)
        w5 = w5 - n*(( ( (outo1 - target1)*outo1*(1-outo1)*w3) + ((outo2 - target2)*outo2*(1-outo2)*w4) )*outh1*(1-outh1)*i2)

        w2 = w2 - n*(( ( (outo1 - target1)*outo1*(1-outo1)*w7) + ((outo2 - target2)*outo2*(1-outo2)*w8) )*outh2*(1-outh2)*i1)
        w6 = w6 - n*(( ( (outo1 - target1)*outo1*(1-outo1)*w7) + ((outo2 - target2)*outo2*(1-outo2)*w8) )*outh2*(1-outh2)*i2)

        w3 = w3 - (n*((outo1 - target1)*outo1*(1-outo1)*outh1))
        w4 = w4 - (n*((outo2 - target2)*outo2*(1-outo2)*outh1))
      
        w8 = w8 - (n*((outo2 - target2)*outo2*(1-outo2)*outh2))
        w7 = w7 - (n*((outo1 - target1)*outo1*(1-outo1)*outh2))
        #print w1,w2,w3,w4,w5,w6,w7,w8,"...",roc
        if (round(outo1,3) == target1) and (round(outo2,3) == target2):
                break
#print "\n"
print "ITERATION ",i,"\nERROR1:",eo1,"PREDICTED OUTPUT1:",round(outo1,3),"\nERROR2:",eo2,"PREDICTED OUTPUT2:",round(outo2,3)
print "HIDDEN OUTPUT1:",outh1,"HIDDEN OUTPUT2",outh2
#plt.plot(iteration,output1,'k',iteration,output2,'k')
#plt.show()
    

