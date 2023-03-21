#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.display import display, Image
print("슝=3")


# In[2]:


import pandas as pd
print("슝=3")


# In[3]:


macbook = pd.read_csv('~/aiffel/bike_regression/data/macbook.csv')
print(macbook.shape)
macbook.head()


# In[4]:


# 데이터시각화
import matplotlib.pyplot as plt
# 실행한 브라우저에서 바로 그림을 볼 수 있게 해준다.
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina' # 더 높은 해상도로 출력한다.")
print("슝=3")


# In[5]:


# x축에는 사용연수(used_years), y축에는 중고가(price)
plt.scatter(macbook['used_years'], macbook['price'])
plt.show()


# ### 상관관계
# Pearson 상관 계수로 표현한다.
# ![image.png](attachment:image.png)

# 상관관계가 없으면 상관계수는 0이고, 상관관계가 높으면 상관관계의 절대값이 커진다.
# 상관계수가 +이면 데이터 분포의 기울기가 양이고, 한 변수가 커질수록 다른 변수의 값이 커진다. 상관계수가 -이면 데이터분포의 기울기가 음이고, 한변수가 커질수록 다른 변수가 작아진다.
# 데이터분포가 직선에 가까울수록 -1또는 1에 가까워지고, 데이터분포가 넓게 퍼지며 원에 가까워질수록 상관계수의 값은 0에 가까워진다.
# 

# In[6]:


plt.scatter(macbook['used_years'], macbook['price'])
plt.show()


# ### 상관계수 구하기

# In[7]:


import numpy as np

# 상관계수를 구하는 함수 np.corrcoef(x,y)를 사용
np.corrcoef(macbook['used_years'], macbook['price'])


# 상관계수는 약 -0.79 정도로 강한 음의 상관계수를 보였다.

# ### 모델 만들기
# 
# * 모델(model) : 특정정보를 입력받아서 그 정보에 따라 원하는 값을 예측하여 값을 출력하는 함수
# 
# x에 used_years 정보 입력, y에는 출력에 해당하는 price 정보를 담는다.
# 

# In[8]:


x = macbook["used_years"].values
y = macbook["price"].values
print("슝=3")


# ### 일차함수 모델
# 
# * 기울기 : w
# * y절편 : b

# In[9]:


def model(x,w,b):
    y = w * x + b
    return y
print("슝=3")


# In[10]:


# y=2x + 1 함수에 x=5 대입
model(x=5, w=2, b=1)


# ### 모델학습
# 모델학습 : 모델이 입력을 받았을 때 정답값에 가까운 출력을 낼 수 있는 최적의 '매개변수' 혹은 'parameter'를 찾는다, 방정식을 푼다
# 

# In[11]:


# y = -20x + 140, (2,100), (5, 40), (6, 20)

# x축, y축 그리기
plt.axvline(x=0, c='black')
plt.axhline(y=0, c='black')

# y = wx + b 일차함수 그리기
x = np.linspace(0, 8, 9)
y = model(x, w=-20, b=140) # y = -20x + 140
plt.plot(y)

# (x, y) 점찍기
x_data = [2, 5, 6]
y_data = [100, 40, 20]
plt.scatter(x_data, y_data, c='r', s=50)

plt.show()


# ### 오차를 최소화하는 모델

# In[12]:


plt.scatter(macbook['used_years'], macbook['price'])
plt.show


# y = wx + b 에서 최적의 직선, 추세선이 될 수 있는 w와 b를 찾는다.
# w와 b는 매개변수, 파라미터 혹은 가중치라고 한다.
# 정확한 방정식이 아닌 최적의 방정식을 찾는다.

# ---
# 손실함수, 아직 불완전한 현재의 모델이 출력하는 값과 실제 정답간의 차이
# 1. 모델의 출력값과 실제 정답간의 차이를 계산
# 2. 그 차이를 단계적으로 줄여나가는 순서로 모델 학습

# In[13]:


# 예시
w = 3.1
b = 2.3
print("슝=3")


# w와 b는 랜덤한 초깃값
# y = 3.1x + 2.3

# In[14]:


x = np.linspace(0, 5, 6)
y = model(x, w, b) # y = 3.1x + 2.3
plt.plot(y, c='r')

plt.scatter(macbook['used_years'], macbook['price'])
plt.show()


# In[15]:


# 맞지 않는직선
# used_years 출력

x = macbook["used_years"].values
x


# In[16]:


prediction = model(x, w, b) #현재는 w = 3.1, b = 2.3
prediction


# In[17]:


# 위 값들을 macbook 데이터프레임에 넣어서 실제 값과 얼마나 다른지 확인할 수 있다.
macbook['prediction'] = prediction
macbook.head()


# In[18]:


# 정답과 예측값 간의 사이
# price와 prediction의 차이 = error컬럼 생성
macbook['error'] = macbook['price'] - macbook['prediction']
macbook.head()


# In[19]:


# RMSE함수 사용
def RMSE(a, b):
    mse = ((a - b) ** 2).mean()
    rmse = mse ** 0.5
    return rmse
print("슝=3")


# In[20]:


x = macbook["used_years"].values
y = macbook["price"].values

predictions = model(x, w, b)
print(predictions)


# In[21]:


rmse = RMSE(predictions, y)
rmse


# ### 손실함수(비용함수) Loss function, Cost fuction

# ---
# 모델의 예측값과 정답값에 대한 차이를 계산하는 함수
# 현재 모델이 얼마나 손실을 내고 있는지 나타내는 개념
# 
# Loss(손실)이 크면 모델이 정답과 먼 예측을 하고 있다는 뜻, Loss가 작으면 모델이 정답과 가까운, 올바른 예측을 하고 있다고 해석. 작을수록 좋다.

# In[22]:


def loss(x, w, b, y):
    predictions = model(x, w, b)
    L = RMSE(predictions, y)
    return L
print("슝=3")


# ### 기울기와 경사하강법(Gradient Descent)
# 
# ![image.png](attachment:image.png)

# * x축 : w
# * y축 : loss
# 
# 현재의 w를 최적의 w로 옮기는 방법 --> Gradient Descent
# 
# * w가 최적의 w보다 작다면 w를 늘린다.
# * w가 최적의 w보다 크다면 w를 늘린다.

# ![image.png](attachment:image.png)
# 
# * w가 최적값보다 작을때, 커질수록 loss가 점점 작아지기 때문에 접선의 기울기가 음수
# * w가 최적값보다 클 때, 커질수록 loss가 점점 커지므로 접선의 기울기가 양수
# 
# \to 특정 점에서의 기울기는 항상 자기자신보다 함수값이 작아지는 방향을 알려줄 수 있다.
# 
# 1. 현재 w에서 기울기를 구한다.
# 2. 기울기가 음수라면, 현재 w를 키운다.
# 3. 기울기가 양수라면, 현재 w를 줄인다.
# 
# $$w\prime = w -\eta g$$
# * w\prime : 새로운 w
# * w : 현재 w
# * $\eta$ : 학습률(얼마나 업데이트할지를 결정하는 함수)
# * g : gradienet, 기울기
# 
# w에서 기울기를 빼주면 원하는 대로 동작할수있다.
# g가 음수면 w-g의 값이 커진다.
# g가 양수면 w-g의 값이 작아진다.
# 
# $\eta$는 고정되어있는 상수로서 w를 변화시키는 양을 조절한다.
# g는 키울지, 줄일지에 대한 방향과 크기를 $\eta$는 얼마나 키울지 줄일지 배율을 결정한다.
# 
# #### 수치미분
# 0에 무한히 가까워져야 극한을 구할 수 있지만 0.0001같이 매우 작을 값을 두고 계산하여 근사값을 구할 수 있다.
# 수학적으로 정확한 값은 아니지만 근사한 미분계수를 찾는 방법을 수치미분이라고 한다.

# In[23]:


def gradient(x, w, b, y):
    dw = (loss(x, w+0.0001, b, y) - loss(x, w, b, y)) / 0.0001
    db = (loss(x, w, b+0.0001, y) - loss(x, w, b, y)) / 0.0001
    return dw, db
print("슝=3")


# ### 정의된 손실함수와 기울기 함수로 모델을 학습시켜 최적화하기
# #### 하이퍼 파라미터
# 모델이 스스로 학습해나가는 파라미터가 아니라, 사람이 직접 사전에 정하고 시작해야하는 파라미터를 하이퍼 파라미터라고 한다.

# In[24]:


LEARNING_RATE = 1 # 학습률
print("슝=3")


# 1. 입력데이터 x와 정답데이터 y 준비
# 2. w, b 랜덤하게 선택
# 3. 현재 w,b로 모델의 prediction을 구하고, y값과 비교해서 loss funciton계산
# 4. loss function, 그 점에서 gradient 계산
# 5. 계산된 gradient로 w,b 업데이트
# 6. 3~5의 과정을 손실함수가 줄어들때까지 반복

# In[25]:


x = macbook["used_years"].values
y = macbook["price"].values
print("슝=3")


# In[26]:


# 초기 가중치 랜덤 설정
w = 3.1
b = 2.3
w,b


# In[27]:


# 손실함수값이 단계별로 얼마정도인지를 저장할 losses라는 빈리스트 생성
losses = []
print("슝=3")


# In[28]:


for i in range(1,2001):
    dw, db =gradient(x, w, b, y)
    w -= LEARNING_RATE * dw
    b -= LEARNING_RATE * db
    L = loss(x, w, b, y)
    losses.append(L)
    if i % 100 == 0:
        print('Iteration %d : Loss %0.4f' % (i, L))


# In[29]:


plt.plot(losses)
plt.show()


# In[30]:


w, b


# In[31]:


# 모델에 넣을 x값들 준비
x = np. linspace(0, 5, 6)

# x, w, b를 모델에 넣어 y값 출력
y = model(x, w, b)

# 일차함수 y 그리기
plt.plot(y, c='r')

# 원본 데이터 점찍기
plt.scatter(macbook['used_years'], macbook['price'])
plt.show()


# In[32]:


test = pd.read_csv("~/aiffel/bike_regression/data/macbook_test.csv")
print(test.shape)
test.head()


# In[33]:


test_x = test['used_years'].values
test_y = test['price'].values


# In[34]:


prediction = model(test_x, w, b)
test['prediction'] = prediction
test


# In[35]:


test['error'] = test['price'] - test['prediction']
test


# In[36]:


rmse = ((test['error'] ** 2).sum() / len(test)) ** 0.5
rmse


# In[37]:


# 모델 일차함수 그리기
x = np.linspace(0, 5, 6)
y = model(x, w, b)
plt.plot(y, c="r")

# 실제 데이터 값
plt.scatter(test['used_years'], test['price'])

# 모델이 예측한 값
plt.scatter(test['used_years'], test['prediction'])
plt.show()


# ### 다변수 선형회귀

# In[38]:


import seaborn as sns

sns.get_dataset_names()


# In[39]:


tips = sns.load_dataset("tips")
print(tips.shape)
tips.head()


# In[40]:


import pandas as pd
tips = pd.get_dummies(tips, columns=['sex', 'smoker', 'day', 'time'])
tips.head()


# In[41]:


tips = tips[['total_bill', 'size', 'sex_Male', 'sex_Female', 'smoker_Yes', 'smoker_No',
             'day_Thur', 'day_Fri', 'day_Sat', 'day_Sun', 'time_Lunch', 'time_Dinner', 'tip']]
tips.head()


# In[42]:


X = tips[['total_bill', 'size', 'sex_Male', 'sex_Female', 'smoker_Yes', 'smoker_No',
          'day_Thur', 'day_Fri', 'day_Sat', 'day_Sun', 'time_Lunch', 'time_Dinner']].values
y = tips['tip'].values
print("슝=3")


# In[43]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)


# In[44]:


import numpy as np
W = np.random.rand(12)
b = np.random.rand()
print("슝=3")


# In[45]:


W


# In[46]:


b


# In[47]:


def model(X, W, b):
    predictions = 0
    for i in range(12):
        predictions += X[:, i] * W[i]
    predictions += b
    return predictions
print("슝=3")
    
    


# In[48]:


def MSE(a,b):
    mse = ((a - b) ** 2).mean()
    return mse
print("슝=3")


# In[49]:


def loss(X, W, b, y):
    predictions = model(X, W, b)
    L = MSE(predictions, y)
    return L


# In[50]:


def gradient(X, W, b, y):
    # N은 데이터 포인터 개수
    
    N = len(y)
    
    # y_pred 준비
    y_pred = model(X, W, b)
    
    # gradient 계산
    dW = 1/N * 2 * X.T.dot(y_pred - y)
    
    # b의 gradient 계산
    db = 2 * (y_pred - y).mean()
    return dW, db
print("슝=3")


# In[51]:


dW, db = gradient(X, W, b, y)
print("dW:", dW)
print("db:", db)


# In[52]:


LEARNING_RATE = 0.0001


# In[53]:


losses = []

for i in range(1, 1001):
    dW, db = gradient(X_train, W, b, y_train)
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    L = loss(X_train, W, b, y_train)
    losses.append(L)
    if i % 10 ==0:
        print('Iteration %d : Loss %0.4f' %(i,L))


# In[54]:


import matplotlib.pyplot as plt
plt.plot(losses)
plt.show


# In[55]:


W, b


# In[56]:


prediction = model(X_test, W, b)
mse = loss(X_test, W, b, y_test)
mse


# In[57]:


# prediction과 y_test 비교
plt.scatter(X_test[:, 0], y_test)
plt.scatter(X_test[:, 0], prediction)
plt.show()


# #### 모델설계, 손실함수 정의, 기울기 계산 및 최적화 과정을 sklearn으로 진행하기

# In[62]:


# 데이터준비
tips = sns.load_dataset("tips")
tips = pd.get_dummies(tips, columns=['sex', 'smoker', 'day', 'time'])
tips = tips[['total_bill', 'size', 'sex_Male', 'sex_Female', 'smoker_Yes', 'smoker_No',
            'day_Thur', 'day_Fri', 'day_Sat', 'day_Sun', 'time_Lunch', 'time_Dinner','tip']]
print("슝=3")


# In[58]:


X = tips[['total_bill', 'size', 'sex_Male', 'sex_Female', 'smoker_Yes', 'smoker_No',
          'day_Thur', 'day_Fri', 'day_Sat', 'day_Sun', 'time_Lunch', 'time_Dinner']].values
y = tips['tip'].values
print("슝=3")


# In[59]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("슝=3")


# In[60]:


# 준비된 모델을 가져다 쓰기
from sklearn.linear_model import LinearRegression
model = LinearRegression()
print("슝=3")


# In[61]:


# 입력데이터 X_train과 y_train을 넣어 fit 시킨다.
model.fit(X_train, y_train)


# In[62]:


predictions = model.predict(X_test)
predictions


# In[63]:


# 정답데이터와 비교
from sklearn.metrics import mean_squared_error

mse = mean_squared_error(y_test, predictions)
mse


# In[64]:


plt.scatter(X_test[:, 0], y_test, label="true")
plt.scatter(X_test[:, 0], predictions, label="pred")
plt.legend()
plt.show()


# * 입력데이터 X, 정답데이터 y, 모델의 출력인 prediction 각각의 개념을 알고, 정답데이터에 가까운 출력을 낼 수 있도록 학습한다.
# * 손실함수 적용, 손실함수의 미분값은 손실함수를 줄이기 위해서 구하며, 수치미분과 해석미분 두가지 방법으로 진행할 수 있다.
# * 선형회귀의 전체 프로세스
# * 사이킷 런을 활용하여 빠르게 선형회귀를 진행할 수 있다.

# ## 프로젝트1 : 당뇨병 수치 예측

# ### (1) 데이터가져오기

# In[285]:


from sklearn.datasets import load_diabetes
diabetes = load_diabetes()

df_X = diabetes.data
df_y = diabetes.target

print(type(df_X))
print(type(df_y))


# ### (2) 모델에 입력할 데이터 X 준비하기

# In[286]:


# df_X에 있는 값들을 numpy array로 변환해서 저장
df_X = np.array(df_X)
print(df_X)


# ### (3) 모델에 예측할 데이터 y 준비하기

# In[287]:


# df_y에 있는 값들을 numpy array로 변환해서 저장
df_y = np.array(df_y)
print(df_y)


# ### (4) train 데이터와 test 데이터로 분리하기

# In[375]:


X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.25, random_state=23)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# ### (5) 모델 준비하기

# In[376]:


# 입력데이터 개수에 맞는 가중치 w와 b 준비, 모델함수 구현
W = np.random.rand(10)
b = np.random.rand()
print(W)
print(b)


# In[377]:


def regression_model(X, W, b):
    y = np.dot(X, W) + b
    return y


# ### (6) 손실함수 loss 정의하기

# In[380]:


# 손실함수를 MSE로 정의
def MSE(s,t):
    MSE = ((s - t) ** 2).mean()
    return MSE


# In[381]:


def loss(X, W, b, y):
    predictions = regression_model(X, W, b)
    L = MSE(predictions, y)
    return L


# ### (7) 기울기를 구하는 gradient 함수 구현

# In[382]:


def gradient(X, W, b, y):
    # N은 데이터 포인터 개수
    
    N = len(y)
    
    # y_pred 준비
    y_pred = regression_model(X, W, b)
    
    # gradient 계산
    dW = 1/N * 2 * X.T.dot(y_pred - y)
    
    # b의 gradient 계산
    db = 2 * (y_pred - y).mean()
    return dW, db


# ### (8) 학습률(하이퍼파라미터) 설정

# In[383]:


LEARNING_RATE = 0.008


# ### (9) 모델 학습하기

# In[386]:


losses = []

for i in range(1, 27501):
    dW, db = gradient(X_train, W, b, y_train)
   
    W -= LEARNING_RATE * dW
    b -= LEARNING_RATE * db
    L = loss(X_train, W, b, y_train)
    losses.append(L)
    if i % 10 ==0:
        print('Iteration %d : Loss %0.4f' %(i,L))


# ### (10) test 데이터에 대한 성능 확인하기

# In[387]:


test_prediction = regression_model(X_test, W, b)
final_loss = MSE(test_prediction, y_test)
final_loss


# * MSE : 2959.36

# ### (11) 정답 데이터와 예측한 데이터 시각화 하기

# In[220]:


fig = plt.figure()

plt.scatter(df_X[:,3], df_y, label= 'labeled')
plt.scatter(df_X[:,3], regression_model(df_X,W,b), label = 'predicted')
plt.legend()

plt.show()


# 파란색은 실제값, 주황색은 predicted 값을 나타낸다. 

# ## 프로젝트2 : 월요일 오후 세시, 자전거 타는 사람은?

# * 캐글 경진대회에서 제공한 데이터셋
# * 시간, 온도, 습도, 계절 등의 정보가 담긴 데이터를 통해 자전거의 대여량을 예측
# * sklearn(사이킷런)의 LinearRegression 모델 활용

# ### (1) 데이터 가져오기

# In[223]:


import pandas as pd

train = pd.read_csv("~/data/data/bike-sharing-demand/train.csv")
train.head()


# ### (2) datetime 컬럼을 datetime 자료형으로 반환하고, 
# ###      연, 월, 일, 시, 분, 초까지 6가지 컬럼 생성하기

# In[225]:


train['datetime'] = pd.to_datetime(train['datetime'])
train.info()


# In[226]:


train['year'] = train['datetime'].dt.year
train['month'] = train['datetime'].dt.month
train['day'] = train['datetime'].dt.day
train['hour'] = train['datetime'].dt.hour
train['minute'] = train['datetime'].dt.minute
train['second'] = train['datetime'].dt.second


# In[227]:


train.info()


# In[232]:


# 'season'과 'weather'은 dummy 변수 처리한다.
train_dummy = pd.get_dummies(train, columns = ['season', 'weather'])


# ### (3) year, month, day, hour, minute, second 데이터 개수 시각화

# * sns.countplot으로 시각화
# * subplot으로 한번에 6개 그래프로 시각화

# In[233]:


datetime_cols = ['year', 'month', 'day', 'hour', 'minute', 'second']


# In[234]:


fig = plt.figure(figsize = (15, 9))

for i, col in enumerate(datetime_cols):
    ax = fig.add_subplot(2, 3, i+1)
    sns.countplot(data = train_dummy, x = col)
    
plt.suptitle('Contplot')
plt.tight_layout()
plt.show()


# * 'minute'과 'second'는 유의미한 연관성을 보이지 않았다.

# ### (4) X, y 컬럼 선택 및 train, test 데이터분리

# In[235]:


train_dummy.corr()['count'].abs().sort_values(ascending=False)


# In[236]:


# 상관계수가 높은 컬럼을 제거해준다.
df_X = train_dummy.drop(['count', 'registered', 'casual', 'minute', 'second', 'datetime'], axis=1)
df_X


# In[237]:


# y변수에 'count' 컬럼 넣기
df_y = train['count']
df_y


# In[241]:


print('df_X : ',type(df_X), df_X.shape)


# In[242]:


print('df_y: ', type(df_y), df_y.shape)


# In[244]:


# train 데이터와 test데이터로 분리
X_train, X_test, y_train, y_test = train_test_split(df_X, df_y, test_size=0.2, random_state=40)

print('X_train', X_train.shape)
print('X_test', X_test.shape)
print('y_train', y_train.shape)
print('y_test', y_test.shape)


# ### (5) LinearRegression 모델 학습

# In[245]:


from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)


# ### (6) 학습된 모델로 X_test에 대한 예측값 출력 및 손실함수값 계산

# In[250]:


# MSE, RMSE
from sklearn.metrics import mean_squared_error

y_pred = model.predict(X_test)
MSE = mean_squared_error(y_test, y_pred)
RMSE = MSE ** 0.5
print('MSE : ', MSE, ',RMSE : ', RMSE)


# * RMSE가 140정도의 값을 보였다.

# ### (7) x축은 temp 또는 humidity로, y축은 count로 예측결과 시각화하기

# In[252]:


df_X.head(3)


# In[272]:





# * 'temp'는 index가 2이고, humidity는 index가 4이다.

# In[283]:


plt.scatter(X_test['temp'], y_test, label='actual')
plt.scatter(X_test['temp'], y_pred, label='predic')
plt.xlabel('temp')
plt.ylabel('count')
plt.legend()
plt.show()


# In[284]:


plt.scatter(X_test['humidity'], y_test, label='actual')
plt.scatter(X_test['humidity'], y_pred, label='predic')
plt.xlabel('temp')
plt.ylabel('count')
plt.legend()
plt.show()


# ## 회고

# * 생각보다 변수가 많아서 특히 시각화를 하는데 어려움을 겪었다. matplotlib의 더 많은 기능을 공부해서, 더 효과적이고 보기 좋은 그래프(산점도)를 그려보고 싶다.
# 
# * 데이터 튜닝 과정에서 상당히 오랜 시간이 걸렸다.

# ## Reference
# 
# * https://yhyun225.tistory.com/11
# * https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.to_datetime.html#pandas.to_datetime
# 

# In[ ]:




