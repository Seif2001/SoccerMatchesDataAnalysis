# Project: Investigate a Dataset - ٍSoccer Matches

## Table of Contents
<ul>
<li><a href="#intro">Introduction</a></li>
<li><a href="#wrangling">Data Wrangling</a></li>
<li><a href="#eda">Exploratory Data Analysis</a></li>
<li><a href="#conclusions">Conclusions</a></li>
</ul>

<a id='intro'></a>
## Introduction

### Dataset Description 

In this report, the soccer matches dataset from Kaggle was selected. I have chosen this dataset in order to mix, for the first time ever, my passion for soccer and my passion for data analysis. The selected dataset includes the data of matches played amongst european teams between 2008 and 2016, in addition the data includes the attributes of how those teams played and also the attributes of the players on those teams.



### Question(s) for Analysis

#### 1) What is the best mix of team attributes that leads to the highest precentages of wins?
From the datasets, the dataset of the matches played and the goals scored from the participating teams is given along with a data set with the attributes of these teams. These two datasets can be combined in order to answer this question.
##### 2) Are left footed players "naturally better" than right footed players as some may claim?
This question can also be answered from the combined dataset mentioned above.



```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
%matplotlib inline
```

<a id='wrangling'></a>
## Data Wrangling




### General Properties



```python
df_country = pd.read_csv('Country.csv')
df_league = pd.read_csv('League.csv')
df_match = pd.read_csv('Match.csv')
df_player = pd.read_csv('Player.csv')
df_player_att = pd.read_csv('Player_Attributes.csv')
df_team = pd.read_csv('Team.csv')


# loading data into pandas dataframes

```


```python
df_team_att = pd.read_csv('Team_attributes.csv')
```

<span style="color:blue">Displaying the null and non null counts for each dataframe</span>


```python
df_country.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11 entries, 0 to 10
    Data columns (total 2 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   id      11 non-null     int64 
     1   name    11 non-null     object
    dtypes: int64(1), object(1)
    memory usage: 304.0+ bytes
    


```python
df_league.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11 entries, 0 to 10
    Data columns (total 3 columns):
     #   Column      Non-Null Count  Dtype 
    ---  ------      --------------  ----- 
     0   id          11 non-null     int64 
     1   country_id  11 non-null     int64 
     2   name        11 non-null     object
    dtypes: int64(2), object(1)
    memory usage: 392.0+ bytes
    


```python
df_match.isnull().sum()
```




    id                0
    country_id        0
    league_id         0
    season            0
    stage             0
                  ...  
    GBD           11817
    GBA           11817
    BSH           11818
    BSD           11818
    BSA           11818
    Length: 115, dtype: int64




```python
df_player.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 11060 entries, 0 to 11059
    Data columns (total 7 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   id                  11060 non-null  int64  
     1   player_api_id       11060 non-null  int64  
     2   player_name         11060 non-null  object 
     3   player_fifa_api_id  11060 non-null  int64  
     4   birthday            11060 non-null  object 
     5   height              11060 non-null  float64
     6   weight              11060 non-null  int64  
    dtypes: float64(1), int64(4), object(2)
    memory usage: 605.0+ KB
    


```python
df_player_att.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 183978 entries, 0 to 183977
    Data columns (total 42 columns):
     #   Column               Non-Null Count   Dtype  
    ---  ------               --------------   -----  
     0   id                   183978 non-null  int64  
     1   player_fifa_api_id   183978 non-null  int64  
     2   player_api_id        183978 non-null  int64  
     3   date                 183978 non-null  object 
     4   overall_rating       183142 non-null  float64
     5   potential            183142 non-null  float64
     6   preferred_foot       183142 non-null  object 
     7   attacking_work_rate  180748 non-null  object 
     8   defensive_work_rate  183142 non-null  object 
     9   crossing             183142 non-null  float64
     10  finishing            183142 non-null  float64
     11  heading_accuracy     183142 non-null  float64
     12  short_passing        183142 non-null  float64
     13  volleys              181265 non-null  float64
     14  dribbling            183142 non-null  float64
     15  curve                181265 non-null  float64
     16  free_kick_accuracy   183142 non-null  float64
     17  long_passing         183142 non-null  float64
     18  ball_control         183142 non-null  float64
     19  acceleration         183142 non-null  float64
     20  sprint_speed         183142 non-null  float64
     21  agility              181265 non-null  float64
     22  reactions            183142 non-null  float64
     23  balance              181265 non-null  float64
     24  shot_power           183142 non-null  float64
     25  jumping              181265 non-null  float64
     26  stamina              183142 non-null  float64
     27  strength             183142 non-null  float64
     28  long_shots           183142 non-null  float64
     29  aggression           183142 non-null  float64
     30  interceptions        183142 non-null  float64
     31  positioning          183142 non-null  float64
     32  vision               181265 non-null  float64
     33  penalties            183142 non-null  float64
     34  marking              183142 non-null  float64
     35  standing_tackle      183142 non-null  float64
     36  sliding_tackle       181265 non-null  float64
     37  gk_diving            183142 non-null  float64
     38  gk_handling          183142 non-null  float64
     39  gk_kicking           183142 non-null  float64
     40  gk_positioning       183142 non-null  float64
     41  gk_reflexes          183142 non-null  float64
    dtypes: float64(35), int64(3), object(4)
    memory usage: 59.0+ MB
    


```python
df_team.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 299 entries, 0 to 298
    Data columns (total 5 columns):
     #   Column            Non-Null Count  Dtype  
    ---  ------            --------------  -----  
     0   id                299 non-null    int64  
     1   team_api_id       299 non-null    int64  
     2   team_fifa_api_id  288 non-null    float64
     3   team_long_name    299 non-null    object 
     4   team_short_name   299 non-null    object 
    dtypes: float64(1), int64(2), object(2)
    memory usage: 11.8+ KB
    


```python
df_team_att.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1458 entries, 0 to 1457
    Data columns (total 25 columns):
     #   Column                          Non-Null Count  Dtype  
    ---  ------                          --------------  -----  
     0   id                              1458 non-null   int64  
     1   team_fifa_api_id                1458 non-null   int64  
     2   team_api_id                     1458 non-null   int64  
     3   date                            1458 non-null   object 
     4   buildUpPlaySpeed                1458 non-null   int64  
     5   buildUpPlaySpeedClass           1458 non-null   object 
     6   buildUpPlayDribbling            489 non-null    float64
     7   buildUpPlayDribblingClass       1458 non-null   object 
     8   buildUpPlayPassing              1458 non-null   int64  
     9   buildUpPlayPassingClass         1458 non-null   object 
     10  buildUpPlayPositioningClass     1458 non-null   object 
     11  chanceCreationPassing           1458 non-null   int64  
     12  chanceCreationPassingClass      1458 non-null   object 
     13  chanceCreationCrossing          1458 non-null   int64  
     14  chanceCreationCrossingClass     1458 non-null   object 
     15  chanceCreationShooting          1458 non-null   int64  
     16  chanceCreationShootingClass     1458 non-null   object 
     17  chanceCreationPositioningClass  1458 non-null   object 
     18  defencePressure                 1458 non-null   int64  
     19  defencePressureClass            1458 non-null   object 
     20  defenceAggression               1458 non-null   int64  
     21  defenceAggressionClass          1458 non-null   object 
     22  defenceTeamWidth                1458 non-null   int64  
     23  defenceTeamWidthClass           1458 non-null   object 
     24  defenceDefenderLineClass        1458 non-null   object 
    dtypes: float64(1), int64(11), object(13)
    memory usage: 284.9+ KB
    


### Data Cleaning

 


```python
# Adding the country of each league using their id
df_league['country'] = df_country[df_country['id'] == df_league['id']]['name']
```


```python
df_league
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>country_id</th>
      <th>name</th>
      <th>country</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>Belgium Jupiler League</td>
      <td>Belgium</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1729</td>
      <td>1729</td>
      <td>England Premier League</td>
      <td>England</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4769</td>
      <td>4769</td>
      <td>France Ligue 1</td>
      <td>France</td>
    </tr>
    <tr>
      <th>3</th>
      <td>7809</td>
      <td>7809</td>
      <td>Germany 1. Bundesliga</td>
      <td>Germany</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10257</td>
      <td>10257</td>
      <td>Italy Serie A</td>
      <td>Italy</td>
    </tr>
    <tr>
      <th>5</th>
      <td>13274</td>
      <td>13274</td>
      <td>Netherlands Eredivisie</td>
      <td>Netherlands</td>
    </tr>
    <tr>
      <th>6</th>
      <td>15722</td>
      <td>15722</td>
      <td>Poland Ekstraklasa</td>
      <td>Poland</td>
    </tr>
    <tr>
      <th>7</th>
      <td>17642</td>
      <td>17642</td>
      <td>Portugal Liga ZON Sagres</td>
      <td>Portugal</td>
    </tr>
    <tr>
      <th>8</th>
      <td>19694</td>
      <td>19694</td>
      <td>Scotland Premier League</td>
      <td>Scotland</td>
    </tr>
    <tr>
      <th>9</th>
      <td>21518</td>
      <td>21518</td>
      <td>Spain LIGA BBVA</td>
      <td>Spain</td>
    </tr>
    <tr>
      <th>10</th>
      <td>24558</td>
      <td>24558</td>
      <td>Switzerland Super League</td>
      <td>Switzerland</td>
    </tr>
  </tbody>
</table>
</div>



### Cleaning the matches data set
Removed irrelevent columns that will not help in data explorations and only kept the relevent ones


Added new columns that will help in calculating the precentage won for each team in a given time period later on

##### id: The id of each match played, may be of use later in answering the first question

##### country_id and league_id: Not really needed, but may be of use later if another question about the data was to be answered

##### season and	date: The seaoson and date in which the game was played will help in differntiating each game because the team 
##### attributes dataset also includes the date of each team attribute

##### home_team_api_id	away_team_api_id : The apis of each team will help in distinguishing each team on the team attributes data set

##### home_team_goal	away_team_goal: These will help in determaining which team won



```python
#gettig only the relevent data that will help with answering the questions

df_match = df_match[['id', 'country_id','league_id','season','date','match_api_id','home_team_api_id','away_team_api_id','home_team_goal','away_team_goal']]
df_match
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>country_id</th>
      <th>league_id</th>
      <th>season</th>
      <th>date</th>
      <th>match_api_id</th>
      <th>home_team_api_id</th>
      <th>away_team_api_id</th>
      <th>home_team_goal</th>
      <th>away_team_goal</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-17 00:00:00</td>
      <td>492473</td>
      <td>9987</td>
      <td>9993</td>
      <td>1</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16 00:00:00</td>
      <td>492474</td>
      <td>10000</td>
      <td>9994</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16 00:00:00</td>
      <td>492475</td>
      <td>9984</td>
      <td>8635</td>
      <td>0</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-17 00:00:00</td>
      <td>492476</td>
      <td>9991</td>
      <td>9998</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16 00:00:00</td>
      <td>492477</td>
      <td>7947</td>
      <td>9985</td>
      <td>1</td>
      <td>3</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25974</th>
      <td>25975</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-22 00:00:00</td>
      <td>1992091</td>
      <td>10190</td>
      <td>10191</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25975</th>
      <td>25976</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23 00:00:00</td>
      <td>1992092</td>
      <td>9824</td>
      <td>10199</td>
      <td>1</td>
      <td>2</td>
    </tr>
    <tr>
      <th>25976</th>
      <td>25977</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23 00:00:00</td>
      <td>1992093</td>
      <td>9956</td>
      <td>10179</td>
      <td>2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25977</th>
      <td>25978</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-22 00:00:00</td>
      <td>1992094</td>
      <td>7896</td>
      <td>10243</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25978</th>
      <td>25979</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23 00:00:00</td>
      <td>1992095</td>
      <td>10192</td>
      <td>9931</td>
      <td>4</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
<p>25979 rows × 10 columns</p>
</div>




```python
# adding a column for the winner and loser for each match
# -1 if a draw occurs

df_match['winner_api'] = ''
df_match['loser_api'] = ''
for i in range(df_match.shape[0]):
    if df_match['home_team_goal'][i] > df_match['away_team_goal'][i]:
        df_match['winner_api'][i] = df_match['home_team_api_id'][i]
        df_match['loser_api'][i] = df_match['away_team_api_id'][i]
    elif df_match['home_team_goal'][i] < df_match['away_team_goal'][i]:
        df_match['winner_api'][i] = df_match['away_team_api_id'][i]
        df_match['loser_api'][i] = df_match['home_team_api_id'][i]
    else:
        df_match['winner_api'][i] = -1
        df_match['loser_api'][i] = -1
df_match
```

    <ipython-input-14-4c7d6d24e59f>:14: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_match['winner_api'][i] = -1
    <ipython-input-14-4c7d6d24e59f>:15: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_match['loser_api'][i] = -1
    <ipython-input-14-4c7d6d24e59f>:11: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_match['winner_api'][i] = df_match['away_team_api_id'][i]
    <ipython-input-14-4c7d6d24e59f>:12: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_match['loser_api'][i] = df_match['home_team_api_id'][i]
    <ipython-input-14-4c7d6d24e59f>:8: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_match['winner_api'][i] = df_match['home_team_api_id'][i]
    <ipython-input-14-4c7d6d24e59f>:9: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_match['loser_api'][i] = df_match['away_team_api_id'][i]
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>country_id</th>
      <th>league_id</th>
      <th>season</th>
      <th>date</th>
      <th>match_api_id</th>
      <th>home_team_api_id</th>
      <th>away_team_api_id</th>
      <th>home_team_goal</th>
      <th>away_team_goal</th>
      <th>winner_api</th>
      <th>loser_api</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-17 00:00:00</td>
      <td>492473</td>
      <td>9987</td>
      <td>9993</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16 00:00:00</td>
      <td>492474</td>
      <td>10000</td>
      <td>9994</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16 00:00:00</td>
      <td>492475</td>
      <td>9984</td>
      <td>8635</td>
      <td>0</td>
      <td>3</td>
      <td>8635</td>
      <td>9984</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-17 00:00:00</td>
      <td>492476</td>
      <td>9991</td>
      <td>9998</td>
      <td>5</td>
      <td>0</td>
      <td>9991</td>
      <td>9998</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16 00:00:00</td>
      <td>492477</td>
      <td>7947</td>
      <td>9985</td>
      <td>1</td>
      <td>3</td>
      <td>9985</td>
      <td>7947</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25974</th>
      <td>25975</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-22 00:00:00</td>
      <td>1992091</td>
      <td>10190</td>
      <td>10191</td>
      <td>1</td>
      <td>0</td>
      <td>10190</td>
      <td>10191</td>
    </tr>
    <tr>
      <th>25975</th>
      <td>25976</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23 00:00:00</td>
      <td>1992092</td>
      <td>9824</td>
      <td>10199</td>
      <td>1</td>
      <td>2</td>
      <td>10199</td>
      <td>9824</td>
    </tr>
    <tr>
      <th>25976</th>
      <td>25977</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23 00:00:00</td>
      <td>1992093</td>
      <td>9956</td>
      <td>10179</td>
      <td>2</td>
      <td>0</td>
      <td>9956</td>
      <td>10179</td>
    </tr>
    <tr>
      <th>25977</th>
      <td>25978</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-22 00:00:00</td>
      <td>1992094</td>
      <td>7896</td>
      <td>10243</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>25978</th>
      <td>25979</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23 00:00:00</td>
      <td>1992095</td>
      <td>10192</td>
      <td>9931</td>
      <td>4</td>
      <td>3</td>
      <td>10192</td>
      <td>9931</td>
    </tr>
  </tbody>
</table>
<p>25979 rows × 12 columns</p>
</div>




```python
#turning the date into datetime object for easier access to the year in which each game was played later on.
df_match['date'] = pd.to_datetime(df_match['date'])
df_match
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>country_id</th>
      <th>league_id</th>
      <th>season</th>
      <th>date</th>
      <th>match_api_id</th>
      <th>home_team_api_id</th>
      <th>away_team_api_id</th>
      <th>home_team_goal</th>
      <th>away_team_goal</th>
      <th>winner_api</th>
      <th>loser_api</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-17</td>
      <td>492473</td>
      <td>9987</td>
      <td>9993</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16</td>
      <td>492474</td>
      <td>10000</td>
      <td>9994</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16</td>
      <td>492475</td>
      <td>9984</td>
      <td>8635</td>
      <td>0</td>
      <td>3</td>
      <td>8635</td>
      <td>9984</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-17</td>
      <td>492476</td>
      <td>9991</td>
      <td>9998</td>
      <td>5</td>
      <td>0</td>
      <td>9991</td>
      <td>9998</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16</td>
      <td>492477</td>
      <td>7947</td>
      <td>9985</td>
      <td>1</td>
      <td>3</td>
      <td>9985</td>
      <td>7947</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25974</th>
      <td>25975</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-22</td>
      <td>1992091</td>
      <td>10190</td>
      <td>10191</td>
      <td>1</td>
      <td>0</td>
      <td>10190</td>
      <td>10191</td>
    </tr>
    <tr>
      <th>25975</th>
      <td>25976</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23</td>
      <td>1992092</td>
      <td>9824</td>
      <td>10199</td>
      <td>1</td>
      <td>2</td>
      <td>10199</td>
      <td>9824</td>
    </tr>
    <tr>
      <th>25976</th>
      <td>25977</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23</td>
      <td>1992093</td>
      <td>9956</td>
      <td>10179</td>
      <td>2</td>
      <td>0</td>
      <td>9956</td>
      <td>10179</td>
    </tr>
    <tr>
      <th>25977</th>
      <td>25978</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-22</td>
      <td>1992094</td>
      <td>7896</td>
      <td>10243</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
    </tr>
    <tr>
      <th>25978</th>
      <td>25979</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23</td>
      <td>1992095</td>
      <td>10192</td>
      <td>9931</td>
      <td>4</td>
      <td>3</td>
      <td>10192</td>
      <td>9931</td>
    </tr>
  </tbody>
</table>
<p>25979 rows × 12 columns</p>
</div>




```python
# making a years column for and more general access for the date of each match

df_match['year'] = df_match['date'].dt.year
df_match
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>country_id</th>
      <th>league_id</th>
      <th>season</th>
      <th>date</th>
      <th>match_api_id</th>
      <th>home_team_api_id</th>
      <th>away_team_api_id</th>
      <th>home_team_goal</th>
      <th>away_team_goal</th>
      <th>winner_api</th>
      <th>loser_api</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-17</td>
      <td>492473</td>
      <td>9987</td>
      <td>9993</td>
      <td>1</td>
      <td>1</td>
      <td>-1</td>
      <td>-1</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16</td>
      <td>492474</td>
      <td>10000</td>
      <td>9994</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16</td>
      <td>492475</td>
      <td>9984</td>
      <td>8635</td>
      <td>0</td>
      <td>3</td>
      <td>8635</td>
      <td>9984</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-17</td>
      <td>492476</td>
      <td>9991</td>
      <td>9998</td>
      <td>5</td>
      <td>0</td>
      <td>9991</td>
      <td>9998</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>1</td>
      <td>2008/2009</td>
      <td>2008-08-16</td>
      <td>492477</td>
      <td>7947</td>
      <td>9985</td>
      <td>1</td>
      <td>3</td>
      <td>9985</td>
      <td>7947</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>25974</th>
      <td>25975</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-22</td>
      <td>1992091</td>
      <td>10190</td>
      <td>10191</td>
      <td>1</td>
      <td>0</td>
      <td>10190</td>
      <td>10191</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>25975</th>
      <td>25976</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23</td>
      <td>1992092</td>
      <td>9824</td>
      <td>10199</td>
      <td>1</td>
      <td>2</td>
      <td>10199</td>
      <td>9824</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>25976</th>
      <td>25977</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23</td>
      <td>1992093</td>
      <td>9956</td>
      <td>10179</td>
      <td>2</td>
      <td>0</td>
      <td>9956</td>
      <td>10179</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>25977</th>
      <td>25978</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-22</td>
      <td>1992094</td>
      <td>7896</td>
      <td>10243</td>
      <td>0</td>
      <td>0</td>
      <td>-1</td>
      <td>-1</td>
      <td>2015</td>
    </tr>
    <tr>
      <th>25978</th>
      <td>25979</td>
      <td>24558</td>
      <td>24558</td>
      <td>2015/2016</td>
      <td>2015-09-23</td>
      <td>1992095</td>
      <td>10192</td>
      <td>9931</td>
      <td>4</td>
      <td>3</td>
      <td>10192</td>
      <td>9931</td>
      <td>2015</td>
    </tr>
  </tbody>
</table>
<p>25979 rows × 13 columns</p>
</div>




```python
#turning the date into datetime object for easier access to the year in which each game was played later on.

df_team_att['date'] = pd.to_datetime(df_team_att['date'])
df_team_att
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>team_fifa_api_id</th>
      <th>team_api_id</th>
      <th>date</th>
      <th>buildUpPlaySpeed</th>
      <th>buildUpPlaySpeedClass</th>
      <th>buildUpPlayDribbling</th>
      <th>buildUpPlayDribblingClass</th>
      <th>buildUpPlayPassing</th>
      <th>buildUpPlayPassingClass</th>
      <th>...</th>
      <th>chanceCreationShooting</th>
      <th>chanceCreationShootingClass</th>
      <th>chanceCreationPositioningClass</th>
      <th>defencePressure</th>
      <th>defencePressureClass</th>
      <th>defenceAggression</th>
      <th>defenceAggressionClass</th>
      <th>defenceTeamWidth</th>
      <th>defenceTeamWidthClass</th>
      <th>defenceDefenderLineClass</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>434</td>
      <td>9930</td>
      <td>2010-02-22</td>
      <td>60</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>...</td>
      <td>55</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>50</td>
      <td>Medium</td>
      <td>55</td>
      <td>Press</td>
      <td>45</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>434</td>
      <td>9930</td>
      <td>2014-09-19</td>
      <td>52</td>
      <td>Balanced</td>
      <td>48.0</td>
      <td>Normal</td>
      <td>56</td>
      <td>Mixed</td>
      <td>...</td>
      <td>64</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>47</td>
      <td>Medium</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>434</td>
      <td>9930</td>
      <td>2015-09-10</td>
      <td>47</td>
      <td>Balanced</td>
      <td>41.0</td>
      <td>Normal</td>
      <td>54</td>
      <td>Mixed</td>
      <td>...</td>
      <td>64</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>47</td>
      <td>Medium</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>77</td>
      <td>8485</td>
      <td>2010-02-22</td>
      <td>70</td>
      <td>Fast</td>
      <td>NaN</td>
      <td>Little</td>
      <td>70</td>
      <td>Long</td>
      <td>...</td>
      <td>70</td>
      <td>Lots</td>
      <td>Organised</td>
      <td>60</td>
      <td>Medium</td>
      <td>70</td>
      <td>Double</td>
      <td>70</td>
      <td>Wide</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>77</td>
      <td>8485</td>
      <td>2011-02-22</td>
      <td>47</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>52</td>
      <td>Mixed</td>
      <td>...</td>
      <td>52</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>47</td>
      <td>Medium</td>
      <td>47</td>
      <td>Press</td>
      <td>52</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>1454</td>
      <td>15005</td>
      <td>10000</td>
      <td>2011-02-22</td>
      <td>52</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>52</td>
      <td>Mixed</td>
      <td>...</td>
      <td>53</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>46</td>
      <td>Medium</td>
      <td>48</td>
      <td>Press</td>
      <td>53</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>1455</td>
      <td>15005</td>
      <td>10000</td>
      <td>2012-02-22</td>
      <td>54</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>50</td>
      <td>Normal</td>
      <td>Organised</td>
      <td>44</td>
      <td>Medium</td>
      <td>55</td>
      <td>Press</td>
      <td>53</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>1456</td>
      <td>15005</td>
      <td>10000</td>
      <td>2013-09-20</td>
      <td>54</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>32</td>
      <td>Little</td>
      <td>Organised</td>
      <td>44</td>
      <td>Medium</td>
      <td>58</td>
      <td>Press</td>
      <td>37</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>1457</td>
      <td>15005</td>
      <td>10000</td>
      <td>2014-09-19</td>
      <td>54</td>
      <td>Balanced</td>
      <td>42.0</td>
      <td>Normal</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>32</td>
      <td>Little</td>
      <td>Organised</td>
      <td>44</td>
      <td>Medium</td>
      <td>58</td>
      <td>Press</td>
      <td>37</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>1458</td>
      <td>15005</td>
      <td>10000</td>
      <td>2015-09-10</td>
      <td>54</td>
      <td>Balanced</td>
      <td>42.0</td>
      <td>Normal</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>32</td>
      <td>Little</td>
      <td>Organised</td>
      <td>44</td>
      <td>Medium</td>
      <td>58</td>
      <td>Press</td>
      <td>37</td>
      <td>Normal</td>
      <td>Cover</td>
    </tr>
  </tbody>
</table>
<p>1458 rows × 25 columns</p>
</div>




```python
# Also making a years column for and more general access for the date of each team's attributes

df_team_att['year'] = df_team_att['date'].dt.year
```


```python
# getting the number of games played and the number of games won and the number of games lost for each team in 
# and then combining it with the team attribute data frame


df_team_att['played'] = ''
df_team_att['won'] = ''
df_team_att['lost'] = ''

done = 0
z = 0
while z < (df_team_att.shape[0]):
    for i in range(df_team_att['team_api_id'].value_counts()[df_team_att['team_api_id'][z]]):
        played = 0
        won = 0
        lost = 0
        for j in range(df_match.shape[0]):
            if i < df_team_att['team_api_id'].value_counts()[df_team_att['team_api_id'][z]] - 1: 
                if df_match['year'][j] >= df_team_att['year'][i+done] and df_match['year'][j] < df_team_att['year'][i+1 +done]:
                    if (df_match['home_team_api_id'][j] == df_team_att['team_api_id'][i+done] or df_match['away_team_api_id'][j] == df_team_att['team_api_id'][i+done]):
                        played+=1
                    if(df_match['winner_api'][j]== df_team_att['team_api_id'][i+done]):
                        won+=1
                    if(df_match['loser_api'][j]== df_team_att['team_api_id'][i+done]):
                        lost+=1

                
            else:
                if df_match['year'][j] >= df_team_att['year'][i+done]:
                    if (df_match['home_team_api_id'][j] == df_team_att['team_api_id'][i+done] or df_match['away_team_api_id'][j] == df_team_att['team_api_id'][i+done]):
                        played+=1
                    if(df_match['winner_api'][j]== df_team_att['team_api_id'][i+done]):
                        won+=1
                    if(df_match['loser_api'][j]== df_team_att['team_api_id'][i+done]):
                        lost+=1
        df_team_att['played'][i+done] = played
        df_team_att['won'][i+done] = won
        df_team_att['lost'][i+done] = lost
        #print(played)
    done += df_team_att['team_api_id'].value_counts()[df_team_att['team_api_id'][z]]
    z = done
    print(z, end = '\r')
df_team_att
```

    <ipython-input-19-aa180d5ec99f>:35: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_team_att['played'][i+done] = played
    <ipython-input-19-aa180d5ec99f>:36: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_team_att['won'][i+done] = won
    <ipython-input-19-aa180d5ec99f>:37: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_team_att['lost'][i+done] = lost
    

    1458




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>team_fifa_api_id</th>
      <th>team_api_id</th>
      <th>date</th>
      <th>buildUpPlaySpeed</th>
      <th>buildUpPlaySpeedClass</th>
      <th>buildUpPlayDribbling</th>
      <th>buildUpPlayDribblingClass</th>
      <th>buildUpPlayPassing</th>
      <th>buildUpPlayPassingClass</th>
      <th>...</th>
      <th>defencePressureClass</th>
      <th>defenceAggression</th>
      <th>defenceAggressionClass</th>
      <th>defenceTeamWidth</th>
      <th>defenceTeamWidthClass</th>
      <th>defenceDefenderLineClass</th>
      <th>year</th>
      <th>played</th>
      <th>won</th>
      <th>lost</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>434</td>
      <td>9930</td>
      <td>2010-02-22</td>
      <td>60</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>...</td>
      <td>Medium</td>
      <td>55</td>
      <td>Press</td>
      <td>45</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2010</td>
      <td>35</td>
      <td>10</td>
      <td>21</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>434</td>
      <td>9930</td>
      <td>2014-09-19</td>
      <td>52</td>
      <td>Balanced</td>
      <td>48.0</td>
      <td>Normal</td>
      <td>56</td>
      <td>Mixed</td>
      <td>...</td>
      <td>Medium</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2014</td>
      <td>37</td>
      <td>10</td>
      <td>17</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>434</td>
      <td>9930</td>
      <td>2015-09-10</td>
      <td>47</td>
      <td>Balanced</td>
      <td>41.0</td>
      <td>Normal</td>
      <td>54</td>
      <td>Mixed</td>
      <td>...</td>
      <td>Medium</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2015</td>
      <td>18</td>
      <td>3</td>
      <td>10</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>77</td>
      <td>8485</td>
      <td>2010-02-22</td>
      <td>70</td>
      <td>Fast</td>
      <td>NaN</td>
      <td>Little</td>
      <td>70</td>
      <td>Long</td>
      <td>...</td>
      <td>Medium</td>
      <td>70</td>
      <td>Double</td>
      <td>70</td>
      <td>Wide</td>
      <td>Cover</td>
      <td>2010</td>
      <td>40</td>
      <td>11</td>
      <td>23</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>77</td>
      <td>8485</td>
      <td>2011-02-22</td>
      <td>47</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>52</td>
      <td>Mixed</td>
      <td>...</td>
      <td>Medium</td>
      <td>47</td>
      <td>Press</td>
      <td>52</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2011</td>
      <td>41</td>
      <td>11</td>
      <td>20</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>1454</td>
      <td>15005</td>
      <td>10000</td>
      <td>2011-02-22</td>
      <td>52</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>52</td>
      <td>Mixed</td>
      <td>...</td>
      <td>Medium</td>
      <td>48</td>
      <td>Press</td>
      <td>53</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2011</td>
      <td>28</td>
      <td>3</td>
      <td>14</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>1455</td>
      <td>15005</td>
      <td>10000</td>
      <td>2012-02-22</td>
      <td>54</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>Medium</td>
      <td>55</td>
      <td>Press</td>
      <td>53</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2012</td>
      <td>33</td>
      <td>17</td>
      <td>8</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>1456</td>
      <td>15005</td>
      <td>10000</td>
      <td>2013-09-20</td>
      <td>54</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>Medium</td>
      <td>58</td>
      <td>Press</td>
      <td>37</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2013</td>
      <td>8</td>
      <td>6</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>1457</td>
      <td>15005</td>
      <td>10000</td>
      <td>2014-09-19</td>
      <td>54</td>
      <td>Balanced</td>
      <td>42.0</td>
      <td>Normal</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>Medium</td>
      <td>58</td>
      <td>Press</td>
      <td>37</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2014</td>
      <td>21</td>
      <td>7</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>1458</td>
      <td>15005</td>
      <td>10000</td>
      <td>2015-09-10</td>
      <td>54</td>
      <td>Balanced</td>
      <td>42.0</td>
      <td>Normal</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>Medium</td>
      <td>58</td>
      <td>Press</td>
      <td>37</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2015</td>
      <td>39</td>
      <td>13</td>
      <td>16</td>
    </tr>
  </tbody>
</table>
<p>1458 rows × 29 columns</p>
</div>




```python
#getting the win precentage by dividing the number of games won over total games played and multiplying by 100

df_team_att['Won Percent'] = 0
for i in range(df_team_att.shape[0]):
    if df_team_att['played'][i]>0:
        df_team_att['Won Percent'][i] = (df_team_att['won'][i]/df_team_att['played'][i]) * 100
    else:
        df_team_att['Won Percent'][i] = 0


```

    <ipython-input-21-afe1f53e8bf0>:4: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_team_att['Won Percent'][i] = (df_team_att['won'][i]/df_team_att['played'][i]) * 100
    <ipython-input-21-afe1f53e8bf0>:6: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      df_team_att['Won Percent'][i] = 0
    


```python
df_team_att
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>team_fifa_api_id</th>
      <th>team_api_id</th>
      <th>date</th>
      <th>buildUpPlaySpeed</th>
      <th>buildUpPlaySpeedClass</th>
      <th>buildUpPlayDribbling</th>
      <th>buildUpPlayDribblingClass</th>
      <th>buildUpPlayPassing</th>
      <th>buildUpPlayPassingClass</th>
      <th>...</th>
      <th>defenceAggression</th>
      <th>defenceAggressionClass</th>
      <th>defenceTeamWidth</th>
      <th>defenceTeamWidthClass</th>
      <th>defenceDefenderLineClass</th>
      <th>year</th>
      <th>played</th>
      <th>won</th>
      <th>lost</th>
      <th>Won Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>434</td>
      <td>9930</td>
      <td>2010-02-22</td>
      <td>60</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>50</td>
      <td>Mixed</td>
      <td>...</td>
      <td>55</td>
      <td>Press</td>
      <td>45</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2010</td>
      <td>35</td>
      <td>10</td>
      <td>21</td>
      <td>28</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>434</td>
      <td>9930</td>
      <td>2014-09-19</td>
      <td>52</td>
      <td>Balanced</td>
      <td>48.0</td>
      <td>Normal</td>
      <td>56</td>
      <td>Mixed</td>
      <td>...</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2014</td>
      <td>37</td>
      <td>10</td>
      <td>17</td>
      <td>27</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>434</td>
      <td>9930</td>
      <td>2015-09-10</td>
      <td>47</td>
      <td>Balanced</td>
      <td>41.0</td>
      <td>Normal</td>
      <td>54</td>
      <td>Mixed</td>
      <td>...</td>
      <td>44</td>
      <td>Press</td>
      <td>54</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2015</td>
      <td>18</td>
      <td>3</td>
      <td>10</td>
      <td>16</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>77</td>
      <td>8485</td>
      <td>2010-02-22</td>
      <td>70</td>
      <td>Fast</td>
      <td>NaN</td>
      <td>Little</td>
      <td>70</td>
      <td>Long</td>
      <td>...</td>
      <td>70</td>
      <td>Double</td>
      <td>70</td>
      <td>Wide</td>
      <td>Cover</td>
      <td>2010</td>
      <td>40</td>
      <td>11</td>
      <td>23</td>
      <td>27</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>77</td>
      <td>8485</td>
      <td>2011-02-22</td>
      <td>47</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>52</td>
      <td>Mixed</td>
      <td>...</td>
      <td>47</td>
      <td>Press</td>
      <td>52</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2011</td>
      <td>41</td>
      <td>11</td>
      <td>20</td>
      <td>26</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>1453</th>
      <td>1454</td>
      <td>15005</td>
      <td>10000</td>
      <td>2011-02-22</td>
      <td>52</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>52</td>
      <td>Mixed</td>
      <td>...</td>
      <td>48</td>
      <td>Press</td>
      <td>53</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2011</td>
      <td>28</td>
      <td>3</td>
      <td>14</td>
      <td>10</td>
    </tr>
    <tr>
      <th>1454</th>
      <td>1455</td>
      <td>15005</td>
      <td>10000</td>
      <td>2012-02-22</td>
      <td>54</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>55</td>
      <td>Press</td>
      <td>53</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2012</td>
      <td>33</td>
      <td>17</td>
      <td>8</td>
      <td>51</td>
    </tr>
    <tr>
      <th>1455</th>
      <td>1456</td>
      <td>15005</td>
      <td>10000</td>
      <td>2013-09-20</td>
      <td>54</td>
      <td>Balanced</td>
      <td>NaN</td>
      <td>Little</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>58</td>
      <td>Press</td>
      <td>37</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2013</td>
      <td>8</td>
      <td>6</td>
      <td>1</td>
      <td>75</td>
    </tr>
    <tr>
      <th>1456</th>
      <td>1457</td>
      <td>15005</td>
      <td>10000</td>
      <td>2014-09-19</td>
      <td>54</td>
      <td>Balanced</td>
      <td>42.0</td>
      <td>Normal</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>58</td>
      <td>Press</td>
      <td>37</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2014</td>
      <td>21</td>
      <td>7</td>
      <td>10</td>
      <td>33</td>
    </tr>
    <tr>
      <th>1457</th>
      <td>1458</td>
      <td>15005</td>
      <td>10000</td>
      <td>2015-09-10</td>
      <td>54</td>
      <td>Balanced</td>
      <td>42.0</td>
      <td>Normal</td>
      <td>51</td>
      <td>Mixed</td>
      <td>...</td>
      <td>58</td>
      <td>Press</td>
      <td>37</td>
      <td>Normal</td>
      <td>Cover</td>
      <td>2015</td>
      <td>39</td>
      <td>13</td>
      <td>16</td>
      <td>33</td>
    </tr>
  </tbody>
</table>
<p>1458 rows × 30 columns</p>
</div>



<a id='eda'></a>
## Exploratory Data Analysis




### Which mix of attributes had the most average wins?


```python
# Use this, and more code cells, to explore your data. Don't forget to add
#   Markdown cells to document your observations and findings.
df_team_att.columns
```




    Index(['id', 'team_fifa_api_id', 'team_api_id', 'date', 'buildUpPlaySpeed',
           'buildUpPlaySpeedClass', 'buildUpPlayDribbling',
           'buildUpPlayDribblingClass', 'buildUpPlayPassing',
           'buildUpPlayPassingClass', 'buildUpPlayPositioningClass',
           'chanceCreationPassing', 'chanceCreationPassingClass',
           'chanceCreationCrossing', 'chanceCreationCrossingClass',
           'chanceCreationShooting', 'chanceCreationShootingClass',
           'chanceCreationPositioningClass', 'defencePressure',
           'defencePressureClass', 'defenceAggression', 'defenceAggressionClass',
           'defenceTeamWidth', 'defenceTeamWidthClass', 'defenceDefenderLineClass',
           'year', 'played', 'won', 'lost', 'Won Percent'],
          dtype='object')




```python
# seperating each attribute with Won Percent to see which class among the attribute has the highest win percentage

df_BU_speed = df_team_att[['buildUpPlaySpeedClass', 'Won Percent' ]]
df_BU_dribble = df_team_att[['buildUpPlayDribblingClass', 'Won Percent' ]]
df_BU_pass = df_team_att[['buildUpPlayPassingClass', 'Won Percent' ]]
df_BU_playpos = df_team_att[['buildUpPlayPositioningClass', 'Won Percent' ]]
df_CC_pass = df_team_att[['chanceCreationPassingClass', 'Won Percent' ]]
df_CC_cross = df_team_att[['chanceCreationCrossingClass', 'Won Percent' ]]
df_CC_shoot = df_team_att[['chanceCreationShootingClass', 'Won Percent' ]]
df_CC_pos = df_team_att[['chanceCreationPositioningClass', 'Won Percent' ]]
df_D_press = df_team_att[['defencePressureClass', 'Won Percent' ]]
df_D_agress = df_team_att[['defenceAggressionClass', 'Won Percent' ]]
df_D_width = df_team_att[['defenceTeamWidthClass', 'Won Percent' ]]
df_D_line = df_team_att[['defenceDefenderLineClass', 'Won Percent' ]]

```


```python
# created a function which calculates the average won percentage for each class in the attribute

def func(df):
    df[df['Won Percent'] != 0]
    df = df.reset_index(drop = True)
    unique_vals = df.iloc[:,0].unique()
    z=0
    list_of_avgs = []
    for i in range(len(unique_vals)):
        total = 0
        number_of_occurs = df.iloc[:,0].value_counts()[unique_vals[i]]
        for j in range(z,z+number_of_occurs):
            total+=df.iloc[:,1][j]
        z+=number_of_occurs
        avg = total/number_of_occurs
        list_of_avgs.append(avg)
    df_counted_vals = pd.DataFrame([unique_vals,list_of_avgs])
    df_counted_vals = df_counted_vals.transpose()
    df_counted_vals.columns = ['Class', 'Average Win Percent']
    return df_counted_vals
```


```python
df_BU_Speed = func(df_BU_speed)
df_BU_dribble = func(df_BU_dribble)
df_BU_pass = func(df_BU_pass)
df_BU_playpos = func(df_BU_playpos)
df_CC_pass = func(df_CC_pass)
df_CC_cross = func(df_CC_cross)
df_CC_shoot = func(df_CC_shoot)
df_CC_pos = func(df_CC_pos)
df_D_press = func(df_D_press)
df_D_agress = func(df_D_agress)
df_D_width = func(df_D_width)
df_D_line = func(df_D_line)
```


```python
df_BU_Speed
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Balanced</td>
      <td>29.362331</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Fast</td>
      <td>31.77907</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Slow</td>
      <td>25.862745</td>
    </tr>
  </tbody>
</table>
</div>



This table shows that the build up speed class that yields the most wins is fast


```python
df_BU_dribble
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Little</td>
      <td>29.210159</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Normal</td>
      <td>29.690531</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Lots</td>
      <td>32.666667</td>
    </tr>
  </tbody>
</table>
</div>



This table shows that the build up dribble class that yields the most wins is lots


```python
df_BU_pass
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mixed</td>
      <td>29.462783</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Long</td>
      <td>30.531915</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Short</td>
      <td>27.992188</td>
    </tr>
  </tbody>
</table>
</div>



This table shows that the build up passing class that yields the most wins is long passing


```python
df_BU_playpos
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organised</td>
      <td>29.504329</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Free Form</td>
      <td>27.444444</td>
    </tr>
  </tbody>
</table>
</div>



This table shows that the build up player positioning that yields the most wins is an organised one


```python
df_CC_pass
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Normal</td>
      <td>29.457352</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Risky</td>
      <td>29.602339</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Safe</td>
      <td>27.589286</td>
    </tr>
  </tbody>
</table>
</div>



This table shows that risky passing in chance creation gets the most wins


```python
df_CC_cross
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Normal</td>
      <td>29.420921</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lots</td>
      <td>29.331754</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Little</td>
      <td>29.269231</td>
    </tr>
  </tbody>
</table>
</div>



This table shows very little difference between the class of crossing in chance creation


```python
df_CC_shoot
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Normal</td>
      <td>29.514706</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Lots</td>
      <td>29.324873</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Little</td>
      <td>26.108108</td>
    </tr>
  </tbody>
</table>
</div>



This table shows that a normal shooting amount yields the highest wins, however lots of shooting also gets a very similar amount of wins


```python
df_CC_pos
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Organised</td>
      <td>29.604278</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Free Form</td>
      <td>27.630872</td>
    </tr>
  </tbody>
</table>
</div>



The best positioning in chance creation is organised


```python
df_D_press
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Medium</td>
      <td>29.543846</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Deep</td>
      <td>29.175325</td>
    </tr>
    <tr>
      <th>2</th>
      <td>High</td>
      <td>27.098361</td>
    </tr>
  </tbody>
</table>
</div>



A medium defense press yeild the highest number of wins


```python
df_D_agress
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Press</td>
      <td>29.404239</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Double</td>
      <td>31.070707</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Contain</td>
      <td>27.435294</td>
    </tr>
  </tbody>
</table>
</div>



Double defense agression yields the highest number of wins


```python
df_D_width
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Normal</td>
      <td>29.525661</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Wide</td>
      <td>29.243243</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Narrow</td>
      <td>27.098361</td>
    </tr>
  </tbody>
</table>
</div>



Playing with normal width when defending yeilds the highest number of wins


```python
df_D_line
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Class</th>
      <th>Average Win Percent</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Cover</td>
      <td>29.639501</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Offside Trap</td>
      <td>26.041667</td>
    </tr>
  </tbody>
</table>
</div>



Playing cover on the defense line yields the highest number of wins

<span style="color:blue">In the next 12 cells each team attribute was displayed with the precentage won for each class in the team attribute</span>


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_BU_Speed)
plt.xlabel('Build up Speed Class')
plt.ylabel('Win Percentage')
plt.title('Build up Speed vs Win Percentage')
plt.show()
```


    
![png](output_58_0.png)
    


This bar chart shows that the build up speed class that yields the most wins is fast


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_BU_dribble)
plt.xlabel('Build up Dribble Class')
plt.ylabel('Win Percentage')
plt.title('Build up Dribble vs Win Percentage')
plt.show()
```


    
![png](output_60_0.png)
    


This bar chart shows that the build up dribble class that yields the most wins is lots


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_BU_pass)
plt.xlabel('Build up Passing Class')
plt.ylabel('Win Percentage')
plt.title('Build up Passing vs Win Percentage')
plt.show()
```


    
![png](output_62_0.png)
    


This bar chart shows that the build up passing class that yields the most wins is long passing


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_BU_playpos)
plt.xlabel('Build up Player Positioning Class')
plt.ylabel('Win Percentage')
plt.title('Build up Player Positioning vs Win Percentage')
plt.show()
```


    
![png](output_64_0.png)
    


This bar chart shows that the build up player positioning that yields the most wins is an organised one


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_CC_pass)
plt.xlabel('Chance Creation Passing Class')
plt.ylabel('Win Percentage')
plt.title('Chance Creation Passing vs Win Percentage')
plt.show()
```


    
![png](output_66_0.png)
    


This bar chart shows that risky passing in chance creation gets the most wins


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_CC_cross)
plt.xlabel('Chance Creation Crossing Class')
plt.ylabel('Win Percentage')
plt.title('Chance Creation Crossing vs Win Percentage')
plt.show()
```


    
![png](output_68_0.png)
    


This bar chart shows very little difference between the class of crossing in chance creation


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_CC_shoot)
plt.xlabel('Chance Creation Shooting Class')
plt.ylabel('Win Percentage')
plt.title('Chance Creation Shooting vs Win Percentage')
plt.show()
```


    
![png](output_70_0.png)
    


This bar chart shows that a normal shooting amount yields the highest wins, however lots of shooting also gets a very similar amount of wins


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_CC_pos)
plt.xlabel('Chance Creation Positioning Class')
plt.ylabel('Win Percentage')
plt.title('Chance Creation Positioning vs Win Percentage')
plt.show()
```


    
![png](output_72_0.png)
    


The best positioning in chance creation is organised


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_D_press)
plt.xlabel('Defence Press Class')
plt.ylabel('Win Percentage')
plt.title('Defence Press vs Win Percentage')
plt.show()
```


    
![png](output_74_0.png)
    


A medium defense press yeild the highest number of wins


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_D_agress)
plt.xlabel('Defence Aggression Class')
plt.ylabel('Win Percentage')
plt.title('Defence Aggression vs Win Percentage')
plt.show()
```




    <BarContainer object of 3 artists>




    
![png](output_76_1.png)
    


Double defense agression yields the highest number of wins


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_D_width)
plt.xlabel('Defence Width Class')
plt.ylabel('Win Percentage')
plt.title('Defence Width vs Win Percentage')
plt.show()
```


    
![png](output_78_0.png)
    


Playing with normal width when defending yeilds the highest number of wins


```python
plt.bar(x = 'Class',height ='Average Win Percent', data = df_D_line)
plt.xlabel('Defence Line Class')
plt.ylabel('Win Percentage')
plt.title('Defence Line vs Win Percentage')
plt.show()
```


    
![png](output_80_0.png)
    


Playing cover on the defense line yields the highest number of wins

###  Are left footed players "naturally better" than right footed players as some may claim?


```python
#breaking the df_player_att into two data frames one for right footed players and one for left footed players

df_right_foot = df_player_att[df_player_att['preferred_foot'] == 'right']
df_left_foot = df_player_att[df_player_att['preferred_foot'] == 'left']
```

#### The distribution of overall rating among right footed players


```python
df_right_foot.hist(column = 'overall_rating')  
#visualizing the distribution of overall rating and right footed players using a histogram
```




    array([[<AxesSubplot:title={'center':'overall_rating'}>]], dtype=object)




    
![png](output_85_1.png)
    



```python
ax = sb.boxplot(x=df_right_foot['overall_rating'])
#visualizing the distribution of overall rating and right footed players using a boxplot
```


    
![png](output_86_0.png)
    


#### The distribution of overall rating among left footed players


```python
df_left_foot.hist(column = 'overall_rating')
#visualizing the distribution of overall rating and left footed players using a histogram
```




    array([[<AxesSubplot:title={'center':'overall_rating'}>]], dtype=object)




    
![png](output_88_1.png)
    



```python
ax = sb.boxplot(x=df_left_foot['overall_rating'])
#visualizing the distribution of overall rating and left footed players using a boxplot
```


    
![png](output_89_0.png)
    



```python
df_right_foot['overall_rating'].mean()
#getting the average overall rating for right footed players
```




    68.59155835241928




```python
df_left_foot['overall_rating'].mean()
#getting the average overall rating for left footed players
```




    68.6261820132788



#### The average of the overall rating among right footed players was: 68.59155835241928 and for left footed players was: 68.6261820132788

<a id='conclusions'></a>
## Conclusions

### Results

#### Question 1

Apperantly from the visualisations and the calculated data the best mix of attributes that will yeild the highest win precentages are:

Build up speed: slow

Build up dribbling: lots

Build up passing: short

Build up player positioning: free-form

Chance creation passing: risky

Chance creation crossing: lots

Chance creation shooting: little

Chance creation positioning: free-form

Defense press: high

Defense aggression: double

Defense width: wide

Defense line: offside trap

#### Question 2

There is no real difference between overall rating distribution for right footed players and left footed players.

The histograms for both are very similar, both exhbiting a normal curve.

The averages are also very similar.

### Limitations

In the first question a few limitations occured, first not all the types within each playing class had the same number of games which may have altered the average slightly. Secondly, it was not specified how each type within the playing class was measuered, for a more accurate representation, the number of attacks or the number of passes per game could be added. The possesion for each team could be also added for each game. This could result in further analysis to find if a corralation exists between these values and the number of wins.

In the second question, the only limitation is that the number of left footed players was less than the number of right footed players, which is expected. However this could have affected the average slightly.



```python
from subprocess import call
call(['python', '-m', 'nbconvert', 'Investigate_a_Dataset.ipynb'])
```


```python

```


```python

```
