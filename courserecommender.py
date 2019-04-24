import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt
'''
courses = pd.read_csv('movies.csv')
courses['year'] = courses.title.str.extract('(\(\d\d\d\d\))',expand = False)
courses['year'] = courses.title.str.extract('(\d\d\d\d)',expand = False)
courses['title'] = courses.title.str.replace('(\(\d\d\d\d\))','')
courses['title'] = courses['title'].apply(lambda x:x.strip())
courses['genres'] = courses.genres.str.split('|')

'''
df_ = pd.DataFrame()
df2 = pd.DataFrame(index=movies_df.index,columns = ['name'])

for i in range(34208):
    s = ""
    for j in range(4):
        x = np.random.randint(26)+65
        x = chr(x)
        s+=x
    df2[i,0] = s
    
courses1_df  = pd.read_csv('courses.csv')
#courses1_df = movies_df.copy()

courseswithgenres = courses1_df.copy()

for index,row in courseswithgenres.iterrows():
    for genre in row['genres']:
        courseswithgenres.at[index,genre] = 1
courseswithgenres = courseswithgenres.fillna(0)

#Reccomender System

userinput = [
            {'name':'Breakfast Club, The', 'rating':5},
            {'name':'Toy Story', 'rating':3.5},
            {'name':'Jumanji', 'rating':2},
            {'name':"Pulp Fiction", 'rating':5},
            {'name':'Akira', 'rating':4.5}
         ] 
input = pd.DataFrame(userinput)

inputid = courses1_df[courses1_df['name'].isin(input['name'].tolist())]

input = pd.merge(inputid,input)
input = input.drop('genres',1).drop('year',1)
input

usercourses = courseswithgenres[courseswithgenres['name'].isin(input['name'].tolist())]

usercourses = usercourses.reset_index(drop= True)
usercourses = usercourses.drop('courseId',1).drop('name',1).drop('genres',1).drop('year',1)

userprofile = usercourses.transpose().dot(input['rating'])

genretable =courseswithgenres.set_index(courseswithgenres['courseId'])
genretable = genretable.drop('courseId',1).drop('name',1).drop('genres',1).drop('year',1)

recommendation_table = genretable*userprofile

x = recommendation_table.sum(axis=1)/(userprofile.sum())

x = x.sort_values(ascending =False)
y=  x.keys().values


z = []

for i in range(5):
    if y[i] in courses1_df['courseId']:
        z.append( courses1_df[courses1_df['courseId']==y[i]])


#collaborative filtering 
        
ratings_df = pd.read_csv('ratings.csv')
ratings_df.head()

ratings_df = ratings_df.drop('timestamp',1)
courses_df = courses1_df.copy()
#courses_df = courses_df.drop('genres', 1)
courses_df.head()


userInput = [
            {'name':'TXTU', 'rating':5},
            {'name':'KXYU', 'rating':3.5},
            {'name':'MVFG', 'rating':2},
            {'name':"GBXC", 'rating':5},
            {'name':'SIBT', 'rating':4.5}
         ] 
inputcourses = pd.DataFrame(userInput)
inputcourses

inputId = courses_df[courses_df['name'].isin(inputcourses['name'].tolist())]
inputcourses = pd.merge(inputId,inputcourses)
#inputcourses = inputcourses.drop('year',1)
inputcourses

userSubset = ratings_df[ratings_df['courseId'].isin(inputcourses['courseId'].tolist())]
userSubset.head()

userSubsetGroup = userSubset.groupby(['userId'])
userSubsetGroup.head()
userSubsetGroup.get_group(1130)
userSubsetGroup = sorted(userSubsetGroup,  key=lambda x: len(x[1]), reverse=True)
userSubsetGroup
userSubsetGroup[0:3]

userSubsetGroup = userSubsetGroup[0:100]

#using pearson Corrrelation coefficient to find similarity
pearsonCorrelationDict = {}
for name, group in userSubsetGroup:
    group = group.sort_values(by='courseId')
    inputcourses = inputcourses.sort_values(by='courseId')
    nRatings = len(group)
    temp_df = inputcourses[inputcourses['courseId'].isin(group['courseId'].tolist())]
    tempRatingList = temp_df['rating'].tolist()
    tempGroupList = group['rating'].tolist()
    Sxx = sum([i**2 for i in tempRatingList]) - pow(sum(tempRatingList),2)/float(nRatings)
    Syy = sum([i**2 for i in tempGroupList]) - pow(sum(tempGroupList),2)/float(nRatings)
    Sxy = sum( i*j for i, j in zip(tempRatingList, tempGroupList)) - sum(tempRatingList)*sum(tempGroupList)/float(nRatings)
    
    if Sxx != 0 and Syy != 0:
        pearsonCorrelationDict[name] = Sxy/sqrt(Sxx*Syy)
    else:
        pearsonCorrelationDict[name] = 0

pearsonCorrelationDict.items()

pearsonDF = pd.DataFrame.from_dict(pearsonCorrelationDict, orient='index')
pearsonDF.columns = ['similarityIndex']
pearsonDF['userId'] = pearsonDF.index
pearsonDF.index = range(len(pearsonDF))
pearsonDF.head()

topUsers=pearsonDF.sort_values(by='similarityIndex', ascending=False)[0:50]
topUsers.head()
topUsersRating=topUsers.merge(ratings_df, left_on='userId', right_on='userId', how='inner')
topUsersRating.head()
topUsersRating['weightedRating'] = topUsersRating['similarityIndex']*topUsersRating['rating']
topUsersRating.head()
tempTopUsersRating = topUsersRating.groupby('courseId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()

recommendation_df = pd.DataFrame()

recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['courseId'] = tempTopUsersRating.index
recommendation_df.head()

recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(10)

x = courses_df.loc[courses_df['courseId'].isin(recommendation_df.head(10)['courseId'].tolist())]

y = x.iloc[:,[1,2]].values

