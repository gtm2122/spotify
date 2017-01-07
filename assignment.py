import numpy as np
import scipy as sc
import csv
from glob import glob

import csv
import os
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

np.random.seed(20)
###Q1
###HERE
'''
user_data=[]
song_data=[]
with open('user_data_sample.csv', 'rb') as f:
    reader = csv.reader(f,delimiter=',')
    
        
    for row in reader:
        
        user_data.append(row)
        #if(row[0]=="unknown"):
            #print row

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
'''
### Q2
'''
def elbowp(sqerr):
    testline = np.zeros((len(sqerr),2))
    testline[:,0]=np.arange(0,len(sqerr))
   
    
    sqerr_coord=np.zeros((len(sqerr),2))
    sqerr_coord[:,0]=np.arange(0,len(sqerr))
    sqerr_coord[:,1]=sqerr 
    
    testline[0:len(np.arange(sqerr[0],sqerr[-1],(sqerr[-1]-sqerr[0])/(len(sqerr)-1) )),1]= np.arange(sqerr[0],sqerr[-1],(sqerr[-1]-sqerr[0])/(len(sqerr)-1) )
    
    #plt.plot(testline)
    dist = (sqerr_coord - testline)[:,0]**2.0 + (sqerr_coord - testline)[:,1]**2.0
       
    return np.argmax(dist)


def SSE(cent,clust,data):
    
    err = 0
    for i in range(0,cent.shape[0]):
        
        err +=(np.linalg.norm(cent[i,:]-data[np.where(clust==i),:]))**2
    return err

male_dict={}
#male_train_arr = np.zeros((num_male,2))
male_train_arr = []

female_train_arr = np.zeros((num_female,2))
count_m = 0
count_f = 0
male_id = []
female_id = []

country = []
for i,j in user_dict.iteritems():
    if (j[0]=='male' and len(j[1][0:2])>0):
        #print j[1][0:2]
        #print j[3]
        #male_train_arr[count_m,:]=np.array([np.float(j[1][0:2]),np.float(j[3])])
        male_train_arr.append(np.array([np.float(j[1][0:2]),np.float(j[3])]))
        count_m+=1
        male_id.append(i)
    elif(j[0]=='female' and len(j[1][0:2])>0):                
        female_train_arr[count_f,:]=np.array([np.float(j[1][0:2]),np.float(j[3])])
        count_f+=1        
        female_id.append(i)
    if(j[2] not in country):
        country.append(j[2])
male_train_arr=np.array(male_train_arr)
error_f = []
error_m = []
error_f1 = []
error_m1 = []

      
K = range(1,50)


### Doing gender-specific clustering

for i in K:
    centroidf,cf = kmeans2(female_train_arr,i,minit='points')
    centroidm,cm = kmeans2(male_train_arr,i,minit='points')
    a = SSE(centroidm,cm,male_train_arr)
    error_m.append(a)
    b = SSE(centroidf,cf,female_train_arr)
    error_f.append(b)
    
elbow_m = elbowp(error_m)
elbow_f = elbowp(error_f)

#break
fig1 = plt.figure()
plt.plot(error_f)
fig1.savefig('project/female_clusters_elbow.png')
plt.close()
fig2 = plt.figure()
plt.plot(error_m)
fig2.savefig('project/male_clusters_elbow.png')
plt.close()
print "From graph, the elbow point for female clusters occurs at k = ",elbow_f

print "From graph, the elbow point for male clusters occurs at k =  ",elbow_m

cent_f,clus_f = kmeans2(female_train_arr,elbow_f-1,minit='points')

cent_m,clus_m = kmeans2(male_train_arr,elbow_m-1,minit='points')


### I plot the scatter plot using the features as the age group and account age
### This is regardless of their respective countries, but this is gender-specific

fig3 = plt.figure()
plt.scatter(female_train_arr[:,0],female_train_arr[:,1],c=clus_f)
fig3.savefig('project/scatter_plt_f.png')
plt.close()
fig4 = plt.figure()
plt.scatter(male_train_arr[:,0],male_train_arr[:,1],c=clus_m)
fig4.savefig('project/scatter_plt_m.png')
plt.close()
'''
### Doing country specific clustering
'''
user_country_dict = {}
user_id_country_ord= {}

count = -1
for i,j in user_dict.iteritems():
    if(len(j[1])>0):
        count+=1


for i in country:
    user_country_dict[i]=[]
    user_id_country_ord[i] = []
count = -1

for i,j in user_dict.iteritems():
    if(len(j[1])>0):
        
        user_country_dict[j[2]].append(np.array([np.float(j[1][0:2]),np.float(j[3])]))
        user_id_country_ord[j[2]].append(i)

user_country_clusts = {}

for i,j in user_country_dict.iteritems():
    user_country_dict[i]=np.array(user_country_dict[i]).reshape((len(user_country_dict[i]),2))
    if(user_country_dict[i].shape[0]>15):
        sqerror = []
        #print (user_country_dict[i].shape[0])/2 
        for k in range(2,min((user_country_dict[i].shape[0])/2,100) ):
            centroids,clusts = kmeans2(user_country_dict[i],k)
            sqerror.append(SSE(centroids,clusts,user_country_dict[i]))
        elbow_c = elbowp(sqerror)
        centroids,clusts = kmeans2(user_country_dict[i],elbow_c)
        user_country_clusts[i] = clusts
        
        #print elbow_c

### Performing reorganization of song data

song_dict_usr = {}
song_dict_song = {}
context = []
product = []
for i in song_data:
    if i[5] not in song_dict_usr :
        song_dict_usr[i[5]] = []
    song_dict_usr[i[5]].append([i[0],i[1],i[3],i[4]])
    
    if i[3] not in product:
        product.append(i[3])
    if i[1] not in context:
        context.append(i[1])
        
    if(i[2] not in song_dict_song ):
        song_dict_song[i[2]] = []
    song_dict_song[i[2]].append([i[0],i[1],i[3],i[4],i[5]])


### Performing gender specific analysis
'''
male_clus_avgtime = np.zeros((1,max(clus_m)+1))
male_clus_context =np.zeros((max(clus_m)+1,len(context)))
male_clus_product =np.zeros((max(clus_m)+1,len(product)))

female_clus_avgtime = np.zeros((1,max(clus_f)+1))
female_clus_context =np.zeros((max(clus_f)+1,len(context)))
female_clus_product =np.zeros((max(clus_f)+1,len(product)))
'''

def get_feature_count_clus(dict_usr,clust,clust_id,cont,prod):
    avgtime = np.zeros((1,max(clust)+1))
    context_count=np.zeros((max(clust)+1,len(cont)))
    prod_count = np.zeros((max(clust)+1,len(prod)))
    for i in range(0,clust.shape[0]):
        if clust_id[i] in dict_usr:
            for j in dict_usr[clust_id[i]]:
                
                avgtime[0,clust[i]]+=np.float(j[0])/clust.shape[0]
                context_count[clust[i],cont.index(j[1])]+=1
                prod_count[clust[i],prod.index(j[2])]+=1
    return avgtime,context_count,prod_count
'''
'''
male_clus_avgtime,male_clus_context,male_clus_product = get_feature_count_clus(song_dict_usr,clus_m,male_id,context,product)
female_clus_avgtime,female_clus_context,female_clus_product = get_feature_count_clus(song_dict_usr,clus_f,female_id,context,product)
'''
'''
### Performing country specific analysis
### I chose to perform this analysis on the top 3 countries from the dataset

'''
top_3_country = {}

a = user_country_clusts

len_clusts = []
key_clusts = []

for i in a.iterkeys():
    len_clusts.append(a[i].shape[0])
    key_clusts.append(i)
len_clusts = np.argsort(np.array(len_clusts))[::-1]

country_context_clus={}
country_product_clus={}
country_avgtime_clus={}


for i in len_clusts[0:3]:
    
    country_context_clus[key_clusts[i]]=[]
    country_product_clus[key_clusts[i]]=[]
    country_avgtime_clus[key_clusts[i]]=[]
    x,y,z = get_feature_count_clus(song_dict_usr,a[key_clusts[i]],user_id_country_ord[key_clusts[i]],context,product)
    
    country_context_clus[key_clusts[i]].append(y)
    country_product_clus[key_clusts[i]].append(z)
    country_avgtime_clus[key_clusts[i]].append(x)
    
#top_3_country = 


'''
def get_fave_song_clus(dict_song,clus_id,clus):
    fave_song_clus = []
    song_ids = []
    max_count = []
    for i in range(0,max(clus)+1):
        idx = np.where(clus==i)[0]
        clus_spec_id=[]
        for a in idx:
            #print a
            clus_spec_id.append(clus_id[a])
        count = []
        #print clus_spec_id
        #break
        list_clus = [dict_song[x] for x in clus_spec_id]
        for j,l in dict_song.iteritems():
            if j not in count:
                count.append(0)
                song_ids.append(j)                
            if clus_spec_id in l:
                count[j]+=1
        count=np.array(count)
        max_count.append(max(count))
        fave_song_clus.append(song_ids[np.argmax(count)])
    return fave_song_clus,max_count

fave_song_male_clus,fave_song_count_male = get_fave_song_clus(song_dict_song,male_id,clus_m)
'''           

        

        
        
        


