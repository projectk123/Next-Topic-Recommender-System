#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
from random import randint
import streamlit as st


# In[ ]:





# In[2]:


assignment_attributes = ['score','attempts','duration','result','problem_check',                        'problem_check_correct','problem_check_incorrect']
access_info_attributes = ['number_of_times_accessed','play_video',                          'pause_video','reset_problem','seek_video',                          'stop_video']
ils_mapping_attributes = ['processing_a','processing_r','perception_s'                          ,'perception_i','reception_vi','reception_ve',                          'understanding_s','understanding_g']
forum_attributes = ['click_forum','create_thread','delete_thread','create_comment','delete_comment']

# columns = ['user_id','cumulative_score']
# topic_indices = []
# end_topic_indices = []
# def get_topics_from_csv():
#     '''
#     per topic : 16 attributes,
#     per topic with assignment : 23 attributes 
    
#     '''
#     global columns, topic_indices, end_topic_indices
#     topics_df = pd.read_csv('topics.csv')
#     for index, row in topics_df.iterrows():
#         topic_id = row[0]
#         topic_indices+=[topics_df.columns.get_loc(topic_id)]
#         topic_title = row[1]
#         columns+=[topic_id]
#         columns+=[topic_title]
#         columns+=access_info_attributes
#         columns+=ils_mapping_attributes
        
#         if 'a' in topic_id:
#             columns+=assignment_attributes
#             end_topic_indices+=[topic_indices[-1]+23]
#         else:
#             end_topic_indices+=[topic_indices[-1]+16]
        
# def get_course_specific_attributes():
#     global columns
#     columns+=forum_attributes
#     columns+=['duration']
#     columns+=['percentage_of_completion']
# def get_user_info():
#     global columns
#     columns+=ils_mapping_attributes
#     columns+=['number_of_assignments_completed','number_of_topics_completed','number_of_courses_completed','success_rate','average_time_spent_per_assignment']
# get_topics_from_csv()
# get_course_specific_attributes()
# get_user_info()
# print(columns[:50])
# print(topic_indices)
# print(end_topic_indices)
columns = ['user_id','cumulative_score','topic_id','topic_title']
columns+=assignment_attributes
columns+=access_info_attributes
columns+=ils_mapping_attributes
columns+=forum_attributes
columns+=['duration','percentage_of_completion']
columns+=ils_mapping_attributes
columns+=['number_of_assignments_completed','number_of_topics_completed','NA','success_rate','average_time_spent_per_assignment']


# In[3]:


def get_cumulative_score(row):
    return random.uniform(0,1)

problem_check_correct_g = 0
problem_check_g = 0
def get_success_rate():

    global problem_check_correct_g, problem_check_g
    if problem_check_g==0:
        return 1
    
    return problem_check_correct_g/problem_check_g
def get_user_info(topic_number,assignment_number,duration):
    userinfo = dict()
    userinfo['ils_mapping'] = get_ils_mapping()

    if topic_number==0:
        userinfo['average_time_spent_per_assignment'] = 0
    else:
        userinfo['average_time_spent_per_assignment'] = duration/(topic_number*79)
    
    userinfo['number_of_assignments_completed'] = assignment_number
    userinfo['number_of_topics_completed'] = topic_number
    # ONLY ONE COURSE ON KANNADA AS OF NOW
    userinfo['number_of_courses_completed'] = 0
    userinfo['success_rate'] = get_success_rate()
    l = list(userinfo['ils_mapping'])
    del userinfo['ils_mapping']
    l+=list(userinfo.values())
    return l
    
def get_create_thread_count(click_forum):
    return randint(1,click_forum) if click_forum>1 else 0
def get_delete_thread_count(create_thread):
    return randint(1,create_thread) if create_thread>1 else 0
def get_create_comment(create_thread):
    return randint(create_thread,create_thread+50) if create_thread>1 else 0
def get_delete_comment(create_comment):
    return randint(0,create_comment) if create_comment>1 else 0

def get_forum():

    forum = dict()
    click_forum = randint(0,100)
    create_thread = get_create_thread_count(click_forum)
    create_comment = get_create_comment(create_thread)

    forum = {
        'click_forum':click_forum,
        'create_thread': create_thread,
        'delete_thread':get_delete_thread_count(create_thread),
        'create_comment':create_comment,
        'delete_comment':get_delete_comment(create_comment)

    }
    return list(forum.values())

def get_grade_from_score(score):
    return 1 if score>=0.9 else 2 if score>=0.8 else 3 if score>=0.7 else 4 if score>=0.6 else  5 if score>=0.5 else 6
def get_assignment():
    global problem_check_correct_g, problem_check_g
    assignment_instance=dict()
    assignment_score=random.uniform(0.4,1.0)
    attempts=random.randint(1,100)
    problem_check_correct=random.randint(0,attempts)
    problem_check_incorrect=random.randint(0,attempts)
    assignment_instance={
        'score':assignment_score,
        'attempts':attempts,
        'duration':attempts*random.randint(10*60*100,360000),
        'result':get_grade_from_score(assignment_score),
        'problem_check':problem_check_correct+problem_check_incorrect+random.randint(1,10),
        'problem_check_correct':problem_check_correct,  
        'problem_check_incorrect':problem_check_incorrect
    }
    problem_check_correct_g+=assignment_instance['problem_check_correct']
    problem_check_g+=assignment_instance['problem_check']
    return list(assignment_instance.values())
def get_access_info():
    access_info = dict()
    play_video = randint(1,100)
    access_info['number_of_times_accessed'] = randint(1,10)
    access_info['play_video'] = play_video
    access_info['pause_video'] = play_video
    access_info['reset_problem'] = randint(1,100)
    access_info['seek_video'] = randint(1,100)
    access_info['stop_video'] = play_video - randint(0,play_video)

    return list(access_info.values())

def get_ils_mapping():
    ils_mapping = dict()
    ils_mapping = {
        'processing_a': bool(random.getrandbits(1)),
        'processing_r': bool(random.getrandbits(1)),
        'perception_s':bool(random.getrandbits(1)),
        'perception_i':bool(random.getrandbits(1)),
        'reception_ve':bool(random.getrandbits(1)),
        'understanding_s':bool(random.getrandbits(1)),
        'understanding_g':bool(random.getrandbits(1)),
        'reception_vi':bool(random.getrandbits(1))

    }
    
    return list(ils_mapping.values())

row = [0]*45
topics_df = pd.read_csv('topics.csv')
def get_topics_done():
    return random.randint(1,79)
def get_topic_id_from_csv(i):
#     print(i,topics_df.iloc[i,0])
    return topics_df.iloc[i,0]
def get_topic_from_csv(i):
    
#     print(i,topics_df.iloc[i,1])
    return topics_df.iloc[i,1]
    
def get_rows(user_id,s):
    global row, problem_check_g, problem_check_correct_g

    problem_check=0
    problem_check_correct=0
    row[0]=user_id
    assignment_number = 0 
    if len(s)==0:
        return []
    
    
    topic_number = random.choices(list(s),k = len(s))[0]
    s.remove(topic_number)
    
    row[2] = get_topic_id_from_csv((int(topic_number)-1))
    row[3] = get_topic_from_csv(int(topic_number)-1)
#     topic_start = topic_indices[topic_number-1]
#     topic_end = end_topic_indices[topic_number-1]
    if  'a' in get_topic_id_from_csv(int(topic_number)-1):
        assignment = get_assignment()
        assignment_number+=1
        row[4:11] = assignment
        access_info = get_access_info()
        ils_mapping = get_ils_mapping()
        row[11:17] = access_info
        row[17:25] = ils_mapping
    else:
        row[4:11]=[0]*7
        access_info = get_access_info()
        ils_mapping = get_ils_mapping()
        row[11:17] = access_info
        row[17:25] = ils_mapping
    row[1]=get_cumulative_score(row)
    row[25:30]=get_forum()
    row[30]=topic_number*random.randint(100*60*10,100*60*10+100*60*5)
    row[31] = topic_number/79
    row[32:] = get_user_info(topic_number-assignment_number,assignment_number,row[28])

    return row
    


# In[4]:


from ipywidgets import IntProgress
from IPython.display import display
import time, sys

total_users = 20
df = pd.DataFrame(columns = columns)
k = 0


total_users_progress_bar = IntProgress(min=0, max=total_users,description='Progress bar') # instantiate the bar
display(total_users_progress_bar)

for user_id in range(1,total_users+1):
    
    total_users_progress_bar.value+=1
    
    topics_done = get_topics_done()
    
    
    s = set(random.sample(range(1,80),topics_done))
    
    topics_for_user_progress_bar = IntProgress(min=1, max=topics_done+1,description=f'User:{user_id}')
    display(topics_for_user_progress_bar)
    
    for i in range(topics_done):
        topics_for_user_progress_bar.value+=1
        row_x = get_rows(user_id,s)
        if row_x == []:
            break
        df.loc[k]=row_x
        k+=1
#     topics_for_user_progress_bar.layout.display = 'none' # Hide the progress bar
    topics_for_user_progress_bar.close()
#     if user_id%250==0:
#         print(f'Data Till {user_id} done.')
# df.head(len(df))


# In[5]:


user_course_path_df = pd.DataFrame(columns=['user_id','course_path'])
# user_course_path_df
for index, row in df.iterrows():

    try:
        exist = user_course_path_df.iloc[row['user_id']-1,0]
        temp = user_course_path_df.iloc[row['user_id']-1,1]
        temp.append(row.topic_id)
        row_x = [row['user_id'],temp]
        user_course_path_df.loc[row['user_id']-1]=row_x
    except:
        row_x = [row.user_id,[row.topic_id]]
        user_course_path_df.loc[row['user_id']-1]=row_x
        
#     user_course_path_df.loc[user_course_path_df['user_id'] == row['user_id']]    
# user_course_path_df.head(20)


# In[6]:


# UNDERSTADING WHICH PARAMETERS TO USE TO APPLY KNN
# TRY PAIRPLOT FROM SEABORN LATER
# for index,row in df.iterrows():
#     if row.user_id==1:
#         print(row)
        


# In[7]:


df.to_csv('mock_dataset_final.csv')
# new_user = get_rows(99)
df = pd.read_csv('mock_dataset_final.csv')


# In[8]:


# iterate over columns
# df.loc[colname].astype(float)
for col in df:
    if 'topic' in col: continue
    df[col] = df[col].astype('float')


# In[9]:


some_df = df.pivot_table(index='user_id',columns = 'topic_id',values = ['cumulative_score','processing_a','processing_r','perception_s','perception_i','reception_vi','reception_ve','understanding_s','understanding_g','duration','percentage_of_completion']).fillna(0)
# some_df = some_df.T
# some_df
# for idx, row in some_df.iterrows():
#     sectorName = idx
#     print(row[0])
    # sectorCount = row['topic_id']
    # print(sectorName, sectorCount)


# In[10]:


from scipy.sparse import csr_matrix

an_df = csr_matrix(some_df.values)

from sklearn.neighbors import NearestNeighbors


model_knn = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
model_knn.fit(an_df)


# In[11]:


query_index = int(st.text_input("Enter a User ID"))
query_index-=1
# print(query_index)
distances, indices = model_knn.kneighbors(some_df.iloc[query_index,:].values.reshape(1, -1), n_neighbors = 6)


# In[12]:


def get_topics_done_by_user(user_id):
    topics_list = {}
    for index, row in df.iterrows():
        if row.user_id==user_id:
            topics_list[row.topic_id]=row.topic_title
    return topics_list
             
# print(get_topics_done_by_user(1.0))


# In[13]:


def get_common_topics(for_user,from_user):
    d3 = {}
    d1 =get_topics_done_by_user(for_user)
    d2 =get_topics_done_by_user(from_user)
    for key in d1:
        if key in d2:
                d3[key]=d1[key]
    return d3


d = get_common_topics(19.0,5.0)
for index, row in df.iterrows():
    if row.user_id==19.0 or row.user_id==5.0:
        c=0
        if row.topic_id in d:
            print(f'COMMON {row.user_id}->{row.topic_id}')
            c+=1
        else:
            print(f'{row.user_id}->{row.topic_id}')
            

    
    


# In[14]:
topic_id_list = []
def get_topic_id_list():
    topics_df = pd.read_csv('topics.csv')
    for index, row in topics_df.iterrows():
        topic_id_list.append(row.topic_id)
get_topic_id_list()
# print(topic_id_list)


# In[15]:


course_path_cluster = {
    '1.0':[0],
    '1.1':[0],
    '1.3':[0,['1.0','1.1'],'1.2'],
    '1.2':[0,['2.0','2a1'],'2.1'],
    '2.0':[0],
    '2a1':[0],
    '2.1':[0,['2a2','2a3'],'2.3'],
    '2.3':[0,'2a4','2a5',['3.0','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11','3.12','3.13','3.14','3.15','3.16'],'4.0'],
    '3.1':[0,['3a1','3a2','3a3','3a4']],
    '3.2':[0,['3.2.1','3a5'],['3.2.2','3a6'],['3.2.3','3a7'],['3.2.4','3a8'],'3a9','3a10'],
    '3.3':[0,'3a12'],
    '3.4':[0,'3a13'],
    '3.5':[0,'3a14'],
    '3.6':[0,'3a15','3.6.1'],
    '3.7':[0,'3.7.1'],
    '3.8':[0,'3a16'],
    '3.10':[0],
    '3.11':[0],
    '3.12':[0,'3a17'],
    '3.13':[0,'3a18'],
    '3.14':[0],
    '3.15':[0],
    '3.16':[0],
    '4.0':[0,['4.1','4.2','4.2.1'],['4.3','4a1'],['4.4','4a2'],['4.5','4a3'],['4.6','4a4'],['4.7','4a5'],['4.8','4.9','4a6'],'4.10'],
    '4.10':[0,'5.0'],
    '5.0':[0,['5.1','5.2','5.3'],'5a1','5a2','5a3','5a4','5a5'],    
}


# In[16]:


# Make a path graph to define chronology as adjacency list
course_path = {
    '1.0':[0],
    '1.1':[0],
    '1.3':[0,'1.0','1.1','1.2'],
    '1.2':[0,'2.0','2a1','2.1'],
    '2.0':[0],
    '2a1':[0],
    '2.1':[0,'2a2','2a3','2.3'],
    '2.3':[0,'2a4','2a5','3.0','3.1','3.2','3.3','3.4','3.5','3.6','3.7','3.8','3.9','3.10','3.11','3.12','3.13','3.14','3.15','3.16','4.0'],
    '3.1':[0,'3a1','3a2','3a3','3a4'],
    '3.2':[0,'3.2.1','3a5','3.2.2','3a6','3.2.3','3a7','3.2.4','3a8','3a9','3a10'],
    '3.3':[0,'3a12'],
    '3.4':[0,'3a13'],
    '3.5':[0,'3a14'],
    '3.6':[0,'3a15','3.6.1'],
    '3.7':[0,'3.7.1'],
    '3.8':[0,'3a16'],
    '3.10':[0],
    '3.11':[0],
    '3.12':[0,'3a17'],
    '3.13':[0,'3a18'],
    '3.14':[0],
    '3.15':[0],
    '3.16':[0],
    '4.0':[0,'4.1','4.2','4.2.1','4.3','4a1','4.4','4a2','4.5','4a3','4.6','4a4','4.7','4a5','4.8','4.9','4a6','4.10'],
    '4.10':[0,'5.0'],
    '5.0':[0,'5.1','5.2','5.3','5a1','5a2','5a3','5a4','5a5'],    
}
def create_all_keys():
    old_course_path = {}
    for k,v in course_path.items():
        old_course_path[k]=v
    for k,v in old_course_path.items():
        for i in v[1:]:
            if i not in old_course_path:
                course_path[i]=[0]
    # print(course_path)
create_all_keys()





# In[17]:


KNN_users = []
# target_user = some_df.index[query_index]
# target_user = st.text_input("Enter a User ID")
target_user = query_index+1
for i in range(0, len(distances.flatten())):

    if i == 0:
        # print('Recommendations for {0}:\n'.format(some_df.index[query_index]))
        st.write(f" # Recommendations for User {target_user}:\n")
    else:
        KNN_users.append(some_df.index[indices.flatten()[i]])
        # print('{0}: {1}, with distance of {2}:'.format(i, some_df.index[indices.flatten()[i]], distances.flatten()[i]))


# In[19]:


def get_ils_mapping_from_topic_id(from_user,topic_id):
    for index, row in df.iterrows():
        if row.topic_id==topic_id and row.user_id == from_user:
            return row[18:26]
        
        
def next_topic_to_be_recommended(for_user,from_user):
    common_topics_dict = get_common_topics(for_user,from_user)
    common_topics = list(common_topics_dict.keys())
    last_topic = common_topics[-1]
    d = get_topics_done_by_user(from_user)
    c=0
    for topic_id,topic_title in d.items():
        if c==1:
            ils = dict(get_ils_mapping_from_topic_id(from_user,topic_id))
            return [topic_id,ils]
        if topic_id==last_topic:
            c=1
    return -1
# next_topic_to_be_recommended(10.0,19.0) 
results = []
    
for user in KNN_users:
    x = next_topic_to_be_recommended(target_user,user)
    if x!=-1:
        results.append(x)
for topic in results:
    print(topic)

    # st.write(topic)
    
    #     common_topics_dict = get_common_topics(user,target_user)
#     common_topics = list(common_topics_dict.keys())
#     common_topics.sort()
#     print(common_topics)
#     print("-------------------------------******************-------------------------")
    


# In[23]:


topics_to_be_recommended = pd.DataFrame(columns=["topic","ils_mapping"])
# topics_to_be_recommended
for k,topic in enumerate(results):

    topics_to_be_recommended.loc[k] = topic
topics_to_be_recommended.to_csv('../web/topics_to_be_recommended.csv')



def path_string(target_user,topic):
    course_path_of_user = user_course_path_df.iloc[int(target_user)-1][1]
    l = []
    # for i in range(len(course_path_of_user)-1):
    #     l.append(f'{course_path_of_user[i]} [label="{course_path_of_user[i]}"]')
    for i in range(len(course_path_of_user)-1):
        l.append(f'"{course_path_of_user[i]}" -> "{course_path_of_user[i+1]}"')
    if topic!=-1:
        l.append(f'"{topic}" [color="red"]')
        l.append(f'"{course_path_of_user[len(course_path_of_user)-1]}" -> "{topic}" [color="red"]')
    s = '\n'.join(l)
    s = 'digraph{\n'+s
    s+="\n}"
    return s
first, second, third, fourth, fifth, sixth = st.columns(6)
first.graphviz_chart(path_string(target_user,-1))
try:
    second.graphviz_chart(path_string(target_user,results[0][0]))
    third.graphviz_chart(path_string(target_user,results[1][0]))
    fourth.graphviz_chart(path_string(target_user,results[2][0]))
    fifth.graphviz_chart(path_string(target_user,results[3][0]))
    sixth.graphviz_chart(path_string(target_user,results[4][0]))
except:
    st.write("There does not exist five neighbors for this particular user")
