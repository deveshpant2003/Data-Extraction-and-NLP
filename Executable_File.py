#!/usr/bin/env python
# coding: utf-8

# # Data Extraction

# In[1]:


import pandas as pd
import numpy as np
input_file=pd.read_csv('Input.xlsx - Sheet1.csv')
import requests
from bs4 import BeautifulSoup

a=0

for i in input_file['URL']:
  b=input_file['URL_ID'][a]
  a=a+1

  url=str(i)

  response=requests.get(url)

  if response.status_code==200:
    soup=BeautifulSoup(response.text,'html.parser')
    
    content_div=soup.find('div',class_='td-post-content tagdiv-type')
    Title=soup.find('title')

    if content_div:
      children=content_div.find_all(['h1','p','ul','ol'])

      with open(str(b)+'.txt','w',encoding='utf-8') as file:
        
        file.write(f'{Title.getText().strip()}\n') 
        
        for child in children:

            if child.name=='h1':
              file.write(f'\n{child.get_text().strip()}\n')

            elif child.name=='p':
              file.write(f'{child.get_text().strip()}\n')

            elif child.name in ['ul','ol']:
              list_items=child.find_all('li')
              for item in list_items:
                file.write(f'{item.get_text().strip()}\n')
            
            print(f'{str(b)+".txt"} saved successfully\n')
      
    else:
      print('content not found')

  else:print('failed to retrieve the page. Status code = {response.status_code}')
        


# # Sentimental Analysis

# ## Cleaning using Stop Words Lists

# In[2]:


import os

folder_path=r'C:\Users\deves\Downloads\drive-download-20241019T130625Z-001'
words_to_remove=[]

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        file_path=os.path.join(folder_path, filename)
        try:
            with open(file_path,'r',encoding='ISO-8859-1') as file:
                for line in file:
                    stop_word=line.split('|')[0].strip()
                    words_to_remove.append(stop_word.lower())
        except UnicodeDecodeError:
            print(f'Error in reading the file {filename}')
            


# In[3]:


print(words_to_remove)


# In[4]:


import os
import re

a = 0

clean_folder_path = r"C:\Users\deves\Documents\Intern Project\Clean text"  # Replace with your desired folder path

if not os.path.exists(clean_folder_path):
    os.makedirs(clean_folder_path)

folder_path = r"C:\Users\deves\Documents\Intern Project"

for filename in os.listdir(folder_path):
    if filename.endswith('.txt'):
        try:
            
            b = input_file['URL_ID'][a]
            a += 1

            file_path = os.path.join(folder_path, filename)

            with open(file_path, 'r', encoding='ISO-8859-1') as file:
                extracted_text = file.read()

            clean_extracted_text = extracted_text.lower()
            for word in words_to_remove:
                
                clean_extracted_text = re.sub(rf'\b{word}\b', '', clean_extracted_text)

            clean_file_path = os.path.join(clean_folder_path, f'clean_{b}.txt')

            with open(clean_file_path, 'w', encoding='ISO-8859-1') as clean_file:
                clean_file.write(clean_extracted_text)

            print(f"File saved: {clean_file_path}")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")


# ## Creating a dictionary of Positive and Negative words

# In[5]:


with open(r"C:\Users\deves\Downloads\Master Dictionary\negative-words.txt",'r',encoding='ISO-8859-1') as file:
    negative_words=file.read().splitlines()
with open(r"C:\Users\deves\Downloads\Master Dictionary\positive-words.txt",'r',encoding='ISO-8859-1') as file:
    positive_words=file.read().splitlines()


# In[6]:


get_ipython().system('pip install nltk')


# ## Extracting Derived variables

# In[7]:


import os
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

folder_path = r'C:\Users\deves\Documents\Intern Project\Clean text'

positive_score_list=[]
negative_score_list=[]
total_words_list=[]

for file in os.listdir(folder_path):
    
    if not file.endswith('.txt'):
        continue
    
    file_path = os.path.join(folder_path, file)
    
    with open(file_path, 'r', encoding='ISO-8859-1') as clean_file: 
        extracted_text = clean_file.read()
        
        tokens = word_tokenize(extracted_text)
        
        positive_score = 0
        negative_score = 0
        
        for token in tokens:
            if token in positive_words:
                positive_score += 1
            if token in negative_words:
                negative_score += 1

        print(f"Negative Score for {file}:", negative_score)
        positive_score_list.append(positive_score)
        negative_score_list.append(negative_score)
        total_words_list.append(len(tokens))


# In[8]:


print(positive_score_list)


# In[9]:


print(total_words_list)


# In[12]:


import pandas as pd
import numpy as np
positive_score_list=np.array(positive_score_list)
negative_score_list=np.array(negative_score_list)
total_words_list=np.array(total_words_list)
polarity_score_list=(positive_score_list-negative_score_list)/((positive_score_list+negative_score_list)+0.000001)
subjectivity_score_list=(positive_score_list + negative_score_list)/ ((total_words_list) + 0.000001).reshape(147,1)

subjectivity_score_list= pd.DataFrame(subjectivity_score_list, columns=['subjectivity_score'])
positive_score_list= pd.DataFrame(positive_score_list, columns=['positive_score']) 
negative_score_list= pd.DataFrame(negative_score_list, columns=['negative_score']) 
polarity_score_list= pd.DataFrame(polarity_score_list, columns=['polarity_score'])


# In[24]:


total_words_list=pd.DataFrame(total_words_list,columns=['total_words'])


# In[13]:


print(polarity_score_list)


# # Analysis of Readability

# ## Average Sentence Length

# In[14]:


from nltk.tokenize import word_tokenize, sent_tokenize

folder_path=r'C:\Users\deves\Documents\Intern Project'

average_sentence_length_list=[]

for file in os.listdir(folder_path):
    if not file.endswith('.txt'):
        continue
    
    file_path = os.path.join(folder_path, file)
    
    with open(file_path, 'r', encoding='ISO-8859-1') as file: 
        extracted_text = file.read()

    sentences = sent_tokenize(extracted_text)
    words = word_tokenize(extracted_text)
    
    num_sentences = len(sentences)
    num_words = len(words)
    
    if num_sentences > 0:  
        average_sentence_length = num_words / num_sentences
        average_sentence_length_list.append(average_sentence_length)
        print(f"Average Sentence Length: {average_sentence_length}")
    else:
        print("No sentences found in the text.")

average_sentence_length_list=np.array(average_sentence_length_list)


# ## Percentage of Complex words

# In[15]:


from nltk.corpus import cmudict

nltk.download('cmudict')

d = cmudict.dict()

def count_syllables(word):
    word = word.lower()
    if word in d:

        return [len([phoneme for phoneme in pronunciation if phoneme[-1].isdigit()]) for pronunciation in d[word]][0]
    else:
        
        return 1

def percentage_complex_words(text):
    words = word_tokenize(text)
    num_words = len(words)
    num_complex_words = sum(1 for word in words if count_syllables(word) > 2)
    
    if num_words > 0:
        percentage_complex = num_complex_words / num_words 
        return percentage_complex
    else:
        return 0  

folder_path=r'C:\Users\deves\Documents\Intern Project'

complex_word_percentage_list=[]

for file in os.listdir(folder_path):

    if not file.endswith('.txt'):
        continue
    
    file_path = os.path.join(folder_path, file)
    

    with open(file_path, 'r', encoding='ISO-8859-1') as file: 
        extracted_text = file.read()

    complex_word_percentage = percentage_complex_words(extracted_text)
    complex_word_percentage_list.append(complex_word_percentage)
    print(f"Percentage of Complex Words: {complex_word_percentage:.2f}")


# In[16]:


fog_index= 0.4*(average_sentence_length_list+complex_word_percentage_list)
print(fog_index)


# In[17]:


average_sentence_length_list= pd.DataFrame(average_sentence_length_list, columns=['average_sentence_length'])
complex_word_percentage_list= pd.DataFrame(complex_word_percentage_list, columns=['complex_word_percentage'])
fog_index_list=pd.DataFrame(fog_index, columns=['fog_index'])


# ## Complex Word Count

# In[18]:


folder_path=r'C:\Users\deves\Documents\Intern Project'

complex_word_list=[]

for file in os.listdir(folder_path):
    # Skip any system files/folders like .ipynb_checkpoints
    if not file.endswith('.txt'):
        continue
    
    file_path = os.path.join(folder_path, file)
    
    # Open the file and read its contents
    with open(file_path, 'r', encoding='ISO-8859-1') as file: 
        extracted_text = file.read()
    words = word_tokenize(extracted_text)
    num_words = len(words)
    num_complex_words = sum(1 for word in words if count_syllables(word) > 2)
    complex_word_list.append(num_complex_words)
complex_word_list=pd.DataFrame(complex_word_list,columns=['complex_word'])


# ## Word Count

# In[19]:


import string

folder_path=r"C:\Users\deves\Documents\Intern Project\Clean text"

word_list=[]

for file in os.listdir(folder_path):
    # Skip any system files/folders like .ipynb_checkpoints
    if not file.endswith('.txt'):
        continue
    
    file_path = os.path.join(folder_path, file)
    
    # Open the file and read its contents
    with open(file_path, 'r', encoding='ISO-8859-1') as file: 
        extracted_text = file.read()

    # Remove punctuation using str.translate
    cleaned_text = extracted_text.translate(str.maketrans('', '', string.punctuation))
    words = word_tokenize(cleaned_text)
    num_words = len(words)
    word_list.append(num_words)

word_list=pd.DataFrame(word_list,columns='word')


# ## Syllable Count Per Word

# In[41]:


import re

def count_syllables(word):

    word = word.lower()
    
    if word.endswith("es") or word.endswith("ed"):
        # If word has only one syllable, "es" or "ed" should not reduce syllables
        if len(re.findall(r'[aeiouy]', word[:-2])) > 1:
            word = word[:-2]
    
    # Find all vowels in the word (including 'y' as a vowel)
    vowels = re.findall(r'[aeiouy]', word)
    
    # Return the number of vowels (syllables)
    return max(1, len(vowels))  # Ensure at least 1 syllable per word

def total_syllables_in_text(text):
    # Tokenize text into words (removing punctuation)
    words = re.findall(r'\b\w+\b', text)
    
    # Calculate the total syllables by summing syllables in each word
    total_syllables = sum(count_syllables(word) for word in words)
    
    return total_syllables

folder_path=r'C:\Users\deves\Documents\Intern Project'

syllable_list=[]

for file in os.listdir(folder_path):
    # Skip any system files/folders like .ipynb_checkpoints
    if not file.endswith('.txt'):
        continue
    
    file_path = os.path.join(folder_path, file)
    
    with open(file_path, 'r', encoding='ISO-8859-1') as file: 
        extracted_text = file.read()

    total_syllables = total_syllables_in_text(extracted_text)
    syllable_list.append(total_syllables)

    print(f"Total syllables in the text: {total_syllables}")

syllable_list=pd.DataFrame(syllable_list,columns=['syllable'])


# ## Personal Pronouns

# In[35]:


import re

def count_personal_pronouns(text):
    # Define the list of personal pronouns
    personal_pronouns = r'\b(I|we|my|ours|us)\b'


    
    # Find all matches for the personal pronouns (case insensitive)
    matches = re.findall(personal_pronouns, text, re.IGNORECASE)
    
    # Filter out "US" (uppercase) from the matches
    filtered_matches = [match for match in matches if match.lower() != 'us' or match != 'US']
    
    return len(filtered_matches)

personal_pronouns_list=[]

for file in os.listdir(folder_path):
    
    if not file.endswith('.txt'):
        continue
    
    file_path = os.path.join(folder_path, file)
    
    with open(file_path, 'r', encoding='ISO-8859-1') as file: 
        extracted_text = file.read()

    personal_pronoun_count = count_personal_pronouns(extracted_text)

    personal_pronouns_list.append(personal_pronoun_count)
    
    print(f"Total personal pronouns in the text: {personal_pronoun_count}")

personal_pronoun_list=pd.DataFrame(personal_pronouns_list,columns=['personal_pronoun'])


# ## Average Word Length

# In[23]:


def avg_characters_per_word(text):
    # Tokenize the text into words (removing punctuation)
    words = re.findall(r'\b\w+\b', text)
    
    total_characters = sum(len(word) for word in words)
    
    total_words = len(words)
    
    if total_words == 0:
        return 0
    return total_characters / total_words

average_characters_list=[]

for file in os.listdir(folder_path):
    # Skip any system files/folders like .ipynb_checkpoints
    if not file.endswith('.txt'):
        continue
    
    file_path = os.path.join(folder_path, file)
    
    with open(file_path, 'r', encoding='ISO-8859-1') as file: 
        extracted_text = file.read()


    average_characters = avg_characters_per_word(extracted_text)

    average_characters_list.append(average_characters)
    
    print(f"Average number of characters per word: {average_characters:.2f}")

average_characters_list=pd.DataFrame(average_characters_list, columns=['average_characters'])


# In[44]:


import pandas as pd

dataframes_or_lists = [
    positive_score_list,
    negative_score_list,
    polarity_score_list,
    subjectivity_score_list,
    average_sentence_length_list,
    complex_word_percentage_list,
    fog_index_list,
    average_sentence_length_list,  
    complex_word_list,
    word_list,
    syllable_list,
    personal_pronoun_list,
    average_characters_list
]


columns = [
    'positive_score',
    'negative_score',
    'polarity_score',
    'subjectivity_score',
    'average_sentence_length',
    'complex_word_percentage',
    'fog_index',
    'average_sentence_length',  
    'complex_word',
    'word',
    'syllable',
    'personal_pronoun',
    'average_characters'
]

# Convert lists to DataFrames if needed, and ensure column names are assigned
dataframes_converted = [
    pd.DataFrame(item, columns=[col]) if isinstance(item, list) else item
    for item, col in zip(dataframes_or_lists, columns)
]

Final = pd.concat(dataframes_converted, axis=1)

print(Final.columns)


# In[45]:


Final.columns=['POSITIVE SCORE',
               'NEGATIVE SCORE',
               'POLARITY SCORE',
               'SUBJECTIVITY SCORE',
               'AVG SENTENCE LENGTH',
               'PERCENTAGE OF COMPLEX WORDS',
               'FOG INDEX',
               'AVG NUMBER OF WORDS PER SENTENCE',
               'COMPLEX WORD COUNT',
               'WORD COUNT',
               'SYLLABLE PER WORD',
               'PERSONAL PRONOUNS',
               'AVG WORD LENGTH']


# In[46]:


input_file=pd.DataFrame(input_file)


# In[48]:


input_file.columns


# In[49]:


Final=pd.concat([input_file[['URL_ID','URL']], Final], axis=1)


# In[50]:


Final.tail()


# In[52]:


Final.to_csv('Output Data Structure', index=False)

