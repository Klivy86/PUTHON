import pandas as pd
import random
from sklearn.preprocessing import OneHotEncoder

# Генерация данных
lst = ['robot'] * 10
lst += ['human'] * 10
random.shuffle(lst)
data = pd.DataFrame({'whoAmI': lst})

# Инициализация OneHotEncoder
encoder = OneHotEncoder(sparse_output=False)  # Для новых версий sklearn
# encoder = OneHotEncoder(sparse=False)  # Для старых версий sklearn

# Преобразование данных в one-hot представление
one_hot_encoded_data = encoder.fit_transform(data[['whoAmI']])

# Преобразование в DataFrame
one_hot_encoded_df = pd.DataFrame(one_hot_encoded_data, columns=encoder.get_feature_names_out(['whoAmI']))

# Вывод результата
print(one_hot_encoded_df.head())
