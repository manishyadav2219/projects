import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from math import sqrt

movies_df = pd.read_csv('movies.csv')
movies_df['year'] = movies_df.title.str.extract('(\(\d\d\d\d\))',expand = False)
movies_df['year'] = movies_df.title.str.extract('(\d\d\d\d)',expand = False)
movies_df['title'] = movies_df.title.str.replace('(\(\d\d\d\d\))','')
movies_df['title'] = movies_df['title'].apply(lambda x:x.strip())
movies_df['genres'] = movies_df.genres.str.split('|')

movies_df.to_csv('out.csv')

movieswithgenres = movies_df.copy()

for index,row in movieswithgenres.iterrows():
    for genre in row['genres']:
        movieswithgenres.at[index,genre] = 1
movieswithgenres = movieswithgenres.fillna(0)

#Reccomender System

userinput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputmovies = pd.DataFrame(userinput)

inputid = movies_df[movies_df['title'].isin(inputmovies['title'].tolist())]

inputmovies = pd.merge(inputid,inputmovies)
inputmovies = inputmovies.drop('genres',1).drop('year',1)
inputmovies

usermovies = movieswithgenres[movieswithgenres['title'].isin(inputmovies['title'].tolist())]

usermovies = usermovies.reset_index(drop= True)
usermovies = usermovies.drop('movieId',1).drop('title',1).drop('genres',1).drop('year',1)

userprofile = usermovies.transpose().dot(inputmovies['rating'])

genretable =movieswithgenres.set_index(movieswithgenres['movieId'])
genretable = genretable.drop('movieId',1).drop('title',1).drop('genres',1).drop('year',1)

recommendation_table = genretable*userprofile

x = recommendation_table.sum(axis=1)/(userprofile.sum())

x = x.sort_values(ascending =False)
y=  x.keys().values


z = []

for i in range(5):
    if y[i] in movies_df['movieId']:
        z.append( movies_df[movies_df['movieId']==y[i]])


#collaborative filtering 
        
ratings_df = pd.read_csv('ratings.csv')
ratings_df.head()

ratings_df = ratings_df.drop('timestamp',1)
courses_df = movies_df.copy()
courses_df = courses_df.drop('genres', 1)
courses_df.head()
courses_df.to_csv('out1.csv')

userInput = [
            {'title':'Breakfast Club, The', 'rating':5},
            {'title':'Toy Story', 'rating':3.5},
            {'title':'Jumanji', 'rating':2},
            {'title':"Pulp Fiction", 'rating':5},
            {'title':'Akira', 'rating':4.5}
         ] 
inputcourses = pd.DataFrame(userInput)
inputcourses

inputId = courses_df[courses_df['title'].isin(inputcourses['title'].tolist())]
inputcourses = pd.merge(inputId,inputcourses)
inputcourses = inputcourses.drop('year',1)
inputcourses

userSubset = ratings_df[ratings_df['movieId'].isin(inputcourses['movieId'].tolist())]
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
    group = group.sort_values(by='movieId')
    inputcourses = inputcourses.sort_values(by='movieId')
    nRatings = len(group)
    temp_df = inputcourses[inputcourses['movieId'].isin(group['movieId'].tolist())]
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
tempTopUsersRating = topUsersRating.groupby('movieId').sum()[['similarityIndex','weightedRating']]
tempTopUsersRating.columns = ['sum_similarityIndex','sum_weightedRating']
tempTopUsersRating.head()

recommendation_df = pd.DataFrame()

recommendation_df['weighted average recommendation score'] = tempTopUsersRating['sum_weightedRating']/tempTopUsersRating['sum_similarityIndex']
recommendation_df['movieId'] = tempTopUsersRating.index
recommendation_df.head()

recommendation_df = recommendation_df.sort_values(by='weighted average recommendation score', ascending=False)
recommendation_df.head(10)

courses_df.loc[courses_df['movieId'].isin(recommendation_df.head(10)['movieId'].tolist())]

