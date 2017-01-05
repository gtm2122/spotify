import numpy as np
import scipy as sc
import csv
from glob import glob
#user_data={}
import csv
import os
from scipy.cluster.vq import kmeans2
#print os.getcwd()

#print glob('.')


"""

user_data=[]
song_data=[]
with open('user_data_sample.csv', 'rb') as f:
    reader = csv.reader(f,delimiter=',')
    
        
    for row in reader:
        
        user_data.append(row)
        if(row[0]=="unknown"):
            #print row
#print len(user_data)
#print "here"
with open('end_song_sample.csv', 'rb') as f:
    reader = csv.reader(f,delimiter=',')
    for row in reader:
        song_data.append(row)

user_features=user_data[0]
song_features=song_data[0]
del user_data[0]
del song_data[0]

female_ms=0.
male_ms=0.
female_count=0.
male_count=0.
unknown_ms=0.
unknown_count=0.


user_dict = {}
for row in user_data:
   
    user_dict[row[4]]=[row[0],row[1],row[2],row[3]]


num_female=0
num_male=0
num_unknown=0


for i in user_data:

    if i[0]=="male":
        num_male+=1 
    elif i[0]=="female":
        num_female+=1
    else:
        num_unknown+=1


for row in song_data:
    if(user_dict[row[5]][0]=='female'):
        #print user_dict[row[5]]
        #print row[0]
        #print 1+float(row[0])
        #break
        female_ms+=float(row[0])
        female_count+=1
        
    elif(user_dict[row[5]][0]=='male'):
        male_ms+=float(row[0])
        male_count+=1
       
    else:
        #print user_dict[row[5]]
        unknown_ms+=float(row[0])
        unknown_count+=1
        
        


print "overall female songtime ",female_ms
print "overall female track listens ",female_count
print "time per female per track listens",female_ms/(female_count*num_female)





print "overall male songtime ",male_ms
print "overall male track listens ",male_count
print "time per male per track listens",male_ms/(male_count*num_male)



print "from above data, on an average a female clicks on a track and listens to it for 28 seconds"
print "whereas a male listens to a track for 25 seconds average"
print "but there are 26 people who have not listed their gender"
"""

K = range(1,101)
male_dict={}
male_train_arr = np.zeros((num_male,2))
female_train_arr = np.zeros((num_female,2))
count_m = 0
count_f = 0
male_id = []
female_id = []
for i,j in user_dict.iteritems():
    if (j[0]=='male' and len(j[1][0:2])>0):
        #print j[1][0:2]
        #print j[3]
        male_train_arr[count_m,:]=np.array([np.float(j[1][0:2]),np.float(j[3])])
        count_m+=1
        male_id.append(i)
    elif(j[0]=='female' and len(j[1][0:2])>0):                
        female_train_arr[count_f,:]=np.array([np.float(j[1][0:2]),np.float(j[3])])
        count_f+=1        
        female_id.append(i)
        

def SSE(cent,clust,data):
    
    err = 0
    for i in range(0,cent.shape[0]):
        np.sum(cent[i,:]-)
    
    return err



