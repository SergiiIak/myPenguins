import streamlit as st
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

#загружаем натренированную модель из файла и делаем сайт
# для предсказания использую загруженную и натренированную модель

st.title('Penguin Classifier')
st.write("This app uses 6 inputs to predict the species of penguin using \
         a model built on the Palmer's Penguin's dataset. Use the form below to get started!")

#загружаем из файла модель и ключ кодирования
rf_pickle = open('RFmodel_penguin.pickle', 'rb')
map_pickle = open('output_penguin.pickle', 'rb')
rfc = pickle.load(rf_pickle)
unique_penguin_mapping = pickle.load(map_pickle)
rf_pickle.close()
map_pickle.close()

#создаем элементы ввода
#для текстовых переменных - это список из нескольких вариантов
#для числовых переменных - поле численного ввода
island = st.selectbox('Penguin Island', options=['Biscoe', 'Dream', 'Torgerson'])
sex = st.selectbox('Sex', options=['Female', 'Male'])
bill_length = st.number_input('Bill Length (mm)', value=39.1, format='%f')
bill_depth = st.number_input('Bill Depth (mm)', value=18.7, format='%f')
flipper_length = st.number_input('Flipper Length (mm)', value=181.0, format='%f')
body_mass = st.number_input('Body Mass (g)', value=3750.0, format='%f')

st.write('the user inputs are {}'.format([island, sex, bill_length, bill_depth, flipper_length, body_mass]))

#подгатавливаем dataset для рапознования
#преобразуем вводимые пользователем данные из строкового значения в числовой
island_biscoe, island_dream, island_torgerson = 0, 0, 0
if island == 'Biscoe':
    island_biscoe = 1
elif island == 'Dream':
    island_dream = 1
elif island == 'Torgerson':
    island_torgerson = 1

sex_female, sex_male = 0, 0
if sex == 'Female':
    sex_female = 1
elif sex == 'Male':
    sex_male = 1

#делаем предсказание вида пингвина по загруженным данным
new_prediction = rfc.predict([[bill_length, bill_depth, flipper_length, body_mass,
                               island_biscoe, island_dream, island_torgerson, sex_female, sex_male]])

prediction_species = unique_penguin_mapping[new_prediction][0]

st.write('We predict your penguin is of the {} species'.format(prediction_species))

#делаем визуализацию предсказание на графике
#добавляем одну сроку вниз с дефолтным пингвином/тем который задается пользователем
penguins_df = pd.read_csv('penguins.csv')
addPenguin = ['new', island, bill_length, bill_depth, flipper_length, body_mass, sex, 2007]
penguins_df.loc[len(penguins_df)] = addPenguin

selected_x_var = 'bill_length_mm'
selected_y_var = 'bill_depth_mm'

# за счет параметра hue='species' разные виды пингвинов отображаются разным цветом
sns.set_style('darkgrid')
fig2, ax2 = plt.subplots()
ax2 = sns.scatterplot(data=penguins_df, x=selected_x_var, y=selected_y_var,
                      hue='species')
plt.xlabel(selected_x_var)
plt.ylabel(selected_y_var)
plt.title(f'Predicted penguin is: {prediction_species}')
st.pyplot(fig2)

#выводим график feature importance, который загружаем с диска
st.write('Below is a feature importance plot')
st.image('feature_importance.png')


st.write('Ниже график который показывает как распределена длина клюва у пингвинов\
          то есть - мы видим по оси y количество пингвинов с определенной длиной клюва,\
          при этом это еще и дифференцировано по виду пингвина. \
          А вертикальная линия показывает какого пингвина мы задали')

#удаляем последнюю строку с дефолтным пингвином
penguins_df.drop(penguins_df.index[-1], axis=0, inplace=True)
#penguins_df.loc[len(penguins_df)] = addPenguin

fig, ax = plt.subplots()
ax = sns.displot(x=penguins_df['bill_length_mm'], hue=penguins_df['species'])
plt.axvline(bill_length)
plt.title('Bill Length by Species')
st.pyplot(ax)

st.write('а здесь мы не дифференцируем по видам пингвина, поэтому разные виды суммируются')


fig, ax = plt.subplots()
ax = sns.displot(x=penguins_df['bill_length_mm'])
plt.axvline(bill_length)
plt.title('Bill Length by Species 2')
st.pyplot(ax)

# fig, ax = plt.subplots()
# ax = sns.displot(x=penguins_df['bill_depth_mm'], hue=penguins_df['species'])
# plt.axvline(bill_depth)
# plt.title('Bill Depth by Species')
# st.pyplot(ax)
#
#
# fig, ax = plt.subplots()
# ax = sns.displot(x=penguins_df['flipper_length_mm'], hue=penguins_df['species'])
# plt.axvline(flipper_length)
# plt.title('Flipper Length by Species')
# st.pyplot(ax)


ggg = 1