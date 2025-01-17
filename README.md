# Data-Extraction-and-NLP

## Objective
The objective is to extract textual data articles from the given URL and perform text analysis to compute variables that are used in NLP (Natural Language Processing).

## Data Extraction
For each of the URLs, given in the **Input.xlsx** file, the executable file ** extract the article text and save the extracted article in a text file with URL_ID as its file name. While extracting text, ther program extracts only the article title and the article text but does not extract the website header, footer, or anything other than the article text. The programming language I used is **python** and the library used for data crawling is **BeautifulSoup**.

## Data Analysis
For each of the extracted texts from the article, the program perform textual analysis and compute variables, given in the output structure excel file **'Output Data Structure.xlsx'**. The output is saved in the exact order as given in this file.

## Variables
The definition of each of the variables are given in the **'Text Analysis.docx'** file. Look for these variables in the analysis document (Text Analysis.docx):
1. POSITIVE SCORE
2. NEGATIVE SCORE
3. POLARITY SCORE
4. SUBJECTIVITY SCORE
5. AVG SENTENCE LENGTH
6. PERCENTAGE OF COMPLEX WORDS
7. FOG INDEX
8. AVG NUMBER OF WORDS PER SENTENCE
9. COMPLEX WORD COUNT
10. WORD COUNT
11. SYLLABLE PER WORD
12. PERSONAL PRONOUNS
13. AVG WORD LENGTH

## Overview of the contents of this repository

- **Master Dictionary :** The folder consists of two text files **negative_words.txt** and **positive_words.txt**. Both these documents are used to calculate positive and negative scores ( look for these terms in the “Text Analysis.docx” file for more information).
- **Stop Words :** The folder consists of different text file containg a list of stop words which are used to clean the text so that Sentiment Analysis can be performed by excluding the words found in these lists.
- **Input.xlsx :** Contains a list of URLs which are used to extract the text from the respevtive webpage.
- **Output Data Structure.xlxs :** Demonstrates how the output variables will be stored with respect to the given URL.
- **Text Analysis.docx :** Objective of this document is to explain methodology adopted to perform text analysis to drive sentimental opinion, sentiment scores, readability, passive words, personal pronouns etc.
- **Things to Consider :** The text file explains the changes a person need to make in the code before executing it.
- **Executable_File :** Python Script




