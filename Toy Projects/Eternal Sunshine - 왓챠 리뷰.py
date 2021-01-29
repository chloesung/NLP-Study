#!/usr/bin/env python
# coding: utf-8

# ## 1. Import

# In[90]:


import time
import csv
from selenium import webdriver
from bs4 import BeautifulSoup
import warnings
warnings.filterwarnings("ignore")


# In[2]:


chrome_path = r'/usr/local/bin/chromedriver'
driver = webdriver.Chrome(executable_path=chrome_path)


# ## 2. Data Crawling

# In[3]:


watcha_url = 'https://pedia.watcha.com/ko-KR/contents/m85X9LW/comments' #<이터널 선샤인> 영화 리뷰
driver.get(watcha_url)

#밑으로 자동 스크롤 해주기! 1.8초에 한번씩 스크롤 됨
#===참고: https://velog.io/@devmin/selenium-crawling-infinite-scroll-click ==
SCROLL_PAUSE_TIME = 1.8

# Get scroll height
last_height = driver.execute_script("return document.body.scrollHeight")

while True:
    # Scroll down to bottom
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait to load page
    time.sleep(SCROLL_PAUSE_TIME)
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight-50);")
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate new scroll height and compare with last scroll height
    new_height = driver.execute_script("return document.body.scrollHeight")

    if new_height == last_height:
        break

    last_height = new_height


# In[21]:


# 쉽게 뽑아내기 위해 soup에 넣어주기
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')


# In[70]:


people = soup.find_all(class_ = 'css-bawlbm') #각 고객별 데이터 다 가져오기
rate_list = []
review_list = []
for i in people:
    #평점이 없으면 그냥 None을 달고 아니면 평점을 넣기
    try:
        rate_list.append(i.find(class_ = 'css-yqs4xl').get_text()) #별점 데이터
    except:
        rate_list.append(None)
    try:
        review_list.append(i.find(class_ = "css-aintwb-Text e1xxz10x0").get_text()) #리뷰 데이터
    except:
        review_list.append(None)


# In[71]:


print(len(review_list))
print(len(rate_list))


# In[73]:


#데이터셋으로 합쳐주기
import pandas as pd
df = pd.DataFrame({'review': review_list, 'rate': rate_list})
df.head()


# In[80]:


#평점 나와있지 않은 데이터는 적절하게 바꿔주기
df = df.replace('보고싶어요',10)
df = df.replace('보는중',10)
df['rate'] = df['rate'].astype(float)


# In[81]:


df.head()


# In[82]:


#네단계로 평점 데이터 분배
df.loc[(df['rate'] >= 0) & (df['rate'] < 3.5), 'grad'] = '그저 그래요'
df.loc[(df['rate'] >= 3.5) & (df['rate'] < 4.5), 'grad'] = '재밌어요'
df.loc[(df['rate'] >= 4.5) & (df['rate'] < 10), 'grad'] = '인생영화'
df.loc[df['rate'] == 10, 'grad'] = '기대중'


# In[84]:


df.to_csv('eternal sunshine.csv') #엄청 오래걸렸으니까 저장해주기,,ㅎㅎ


# ## 3. EDA & Data Preprocessing

# In[94]:


import matplotlib as plt
plt.rcParams['font.family'] = 'AppleGothic'
df['grad'].value_counts().plot(kind = 'pie')


# In[96]:


#중복되는 리뷰는 날려주기
df.drop_duplicates(subset=['review'], inplace=True)


# In[99]:


#결측치 확인
print(df.isnull().sum())


# In[101]:


df = df.dropna(how = 'any') #결측치도 날려주기


# In[102]:


#한글과 공백을 제외하고 모두 삭제함
df['review'] = df['review'].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")


# In[104]:


df.head()


# In[106]:


#결측치 재확인
import numpy as np
df['review'].replace('', np.nan, inplace=True)
print(df.isnull().sum())


# In[107]:


df = df.dropna(how = 'any') #결측치 삭제


# In[117]:


df = df.to_csv('watcha.csv')


# ## 4. Tokenizing

# In[ ]:


from konlpy.tag import Okt
from tensorflow.keras.preprocessing.text import Tokenizer


# In[25]:


stopwords = ['않다','에서','있다','없다','그렇다','아니다','것','이다','의','가','이','은','들','는','좀','잘','걍','과','도','을','를','으로','자','에','와','한','하다','휴','수'] #불용어


# In[26]:


#konlpy로 토큰화를 진행합니다
okt = Okt()
token = []
for sentence in df['review']:
    temp_X = []
    temp_X = okt.morphs(sentence, stem=True) # 토큰화
    temp_X = [word for word in temp_X if not word in stopwords] # 불용어 제거
    token.append(temp_X)


# In[27]:


token[:3]


# In[28]:


df['token'] = token


# In[29]:


tokenizer = Tokenizer()
tokenizer.fit_on_texts(token)


# In[30]:


print(tokenizer.word_index)


# In[31]:


threshold = 3
total_cnt = len(tokenizer.word_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('단어 집합(vocabulary)의 크기 :',total_cnt)
print('등장 빈도가 %s번 이하인 희귀 단어의 수: %s'%(threshold - 1, rare_cnt))
print("단어 집합에서 희귀 단어의 비율:", (rare_cnt / total_cnt)*100)
print("전체 등장 빈도에서 희귀 단어 등장 빈도 비율:", (rare_freq / total_freq)*100)


# In[32]:


from collections import Counter


# In[33]:


words = np.hstack(df['token'].values)
word_count = Counter(words)
print(word_count.most_common(20))


# In[45]:


life_words = np.hstack(df[df.grad == '인생영화']['token'].values)
positive_words = np.hstack(df[df.grad == '재밌어요']['token'].values)
negative_words = np.hstack(df[df.grad == '그저 그래요']['token'].values)


# ## 5. Visualization

# In[87]:


# 워드클라우드를 그리는 함수 만들기
from wordcloud import WordCloud, ImageColorGenerator
from PIL import Image
def displayWordCloud(data = None, backgroundcolor = 'white'):
  stopwords_kr = ['영화','보다','같다','되다','에게','되어다','까지']
  pic = np.array(Image.open('eternal-sunshine.png'))
  image_colors = ImageColorGenerator(pic)
  import matplotlib.pyplot as plt
  get_ipython().run_line_magic('matplotlib', 'inline')
  get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
  wordcloud = WordCloud(font_path = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf',max_words=2000,mask=pic,background_color ='white',stopwords = stopwords_kr).generate(data)
  plt.figure(figsize = (15 , 10))
  plt.imshow(wordcloud.recolor(color_func=image_colors), interpolation='bilinear')
  plt.axis("off")
  plt.show()


# In[82]:


# 결과 출력하기. 출력하는 시간을 보기 위해 %time을 붙였다.
get_ipython().run_line_magic('time', "displayWordCloud(' '.join(life_words))")


# In[83]:


displayWordCloud(' '.join(positive_words))


# In[88]:


displayWordCloud(' '.join(negative_words))


# In[129]:


get_ipython().run_line_magic('pylab', 'inline')
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
img = mpimg.imread('eternal-sunshine.png')
imgplot = plt.imshow(img)
plt.show()

