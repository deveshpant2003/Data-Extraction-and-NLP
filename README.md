# Data Extraction and NLP

## Objective
The goal is to extract textual data articles from the provided URL, conduct text analysis, and compute variables used in NLP (Natural Language Processing).

## Data Extraction
For each of the URLs, given in the **Input.xlsx** file, the executable file **'Executable_File.py'** extracts the article text and saves the extracted article in a text file with URL_ID as its file name. During the text extraction process, the program only extracts the article title and text, leaving out the website header and footer. I utilized **Python** as the programming language and **BeautifulSoup** as the data crawling library.

## Data Analysis
The program performs textual analysis and computes variables for each extracted text from the website, as specified in the file **'Output Data Structure.xlsx'**. This file precisely specifies the order in which to save the output.

## Variables
The **'Text Analysis.docx'** file provides the definition of each of these variables:
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

-**Master Dictionary:** The folder consists of two text files, negative_words.txt and positive_words.txt. Both these documents are used to calculate positive and negative scores (look for these terms in the “Text Analysis.docx” file for more information).
-**Stop Words:** The folder consists of different text files containing a list of stop words that are used to clean the text so that sentiment analysis can be performed by excluding the words found in these lists.
-**Input.xlsx:** This file contains a list of URLs used to extract text from the corresponding webpage.
-**Output Data Structure.xlxs:** Demonstrates how the output variables will be stored with respect to the given URL.
-**Text Analysis.docx:** The objective of this document is to explain the methodology adopted to perform text analysis to drive sentimental opinion, sentiment scores, readability, passive words, personal pronouns, etc.
-**Things to Consider:** The text file explains the changes a person needs to make in the code before executing it.
-**Executable_File:** Python Script




