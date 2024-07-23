#!/usr/bin/env python
# coding: utf-8

# In[2]:


# import tensorflow as tf
# import tensorflow_hub as hub
# import tensorflow_text as text    

# !pip install --user tensorflow_text


# !pip install tensorflow_hub

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text    


import pandas as pd

df = pd.read_csv('Data.csv')
# "C:\Users\akash\Python_DL\project_data\Data.csv"
df.head(5)

df.groupby('Category').describe()


# In[3]:


# Separate data frames for individual classes (Class A and Class B)
spam = df[df['Category'] == 'spam']
ham = df[df['Category'] == 'ham']

# Display the data frames
print("DataFrame for Class spam:")
print(spam)

print("\nDataFrame for Class ham:")
print(ham)


# In[4]:


# Downsample Class B to have the same number of samples as Class A
ham_downsampled = ham.sample(n=len(spam), random_state=42)

# Display the downsampled DataFrame for Class B
print("DataFrame for Class ham (Downsampled):")
print(ham_downsampled)


# In[5]:


all_downsampled = pd.concat([spam, ham_downsampled], ignore_index=True)

# Display the concatenated DataFrame
print("Concatenated DataFrame:")
print(all_downsampled)


# In[6]:


label_mapping = {'spam': 1, 'ham': 0}
all_downsampled['Label'] = all_downsampled['Category'].apply(lambda x: label_mapping[x])

# Display the updated DataFrame with the 'Label' column
print(all_downsampled)


# In[7]:


from sklearn.model_selection import train_test_split



# Features (X) are all the columns except 'Category' and 'Label'
X = all_downsampled.drop(columns=['Category', 'Label'])

# Labels (y) are contained in the 'Label' column
y = all_downsampled['Label']

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
print("Training set shape - X_train:", X_train.shape, "y_train:", y_train.shape)
print("Testing set shape - X_test:", X_test.shape, "y_test:", y_test.shape)


# In[8]:


X_train.head()


# In[9]:


# Downloading the bert models for pre processing and encoding


bert_preprocess = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
bert_encoder = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/4")


# In[10]:


# Bert layers
text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name='text')
preprocessed_text = bert_preprocess(text_input)
outputs = bert_encoder(preprocessed_text)

# Neural network layers
l = tf.keras.layers.Dropout(0.1, name="dropout")(outputs['pooled_output'])
l = tf.keras.layers.Dense(1, activation='sigmoid', name="output")(l)

# Use inputs and outputs to construct a final model
model = tf.keras.Model(inputs=[text_input], outputs = [l])


# In[11]:


model.summary()


# In[12]:


model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])


# In[13]:


model.fit(X_train, y_train, epochs=1)


# In[14]:


model.evaluate(X_test, y_test)


# In[15]:


reviews = [
    'Reply to win Â£100 weekly! Where will the 2006 FIFA World Cup be held? Send STOP to 87239 to end service',
    'You are awarded a SiPix Digital Camera! call 09061221061 from landline. Delivery within 28days. T Cs Box177. M221BP. 2yr warranty. 150ppm. 16 . p pÂ£3.99',
    'it to 80488. Your 500 free text messages are valid until 31 December 2005.',
    'Hey Sam, Are you coming for a cricket game tomorrow',
    "Why don't you wait 'til at least wednesday to see if you get your ."
]


model.predict(reviews)


# In[16]:


import pickle


# In[18]:


filename = 'balanced_models/balanced.pickle'
pickle.dump(model, open(filename, 'wb'))


# In[19]:


h5 = 'balanced_models/balanced.h5'
model.save(h5)


# In[22]:


model.save('balanced_models/balanced.keras')


# In[ ]:




