import os
import numpy as np
import pandas as pd
from datetime import datetime
import json
import matplotlib.pyplot as plt
import re
import nltk
# nltk.download('punkt')
from nltk.tokenize import sent_tokenize, word_tokenize, regexp_tokenize

##############################################################################################################
# Detect stories related to racism, critical race theory, DEI, woke in the closed captioning of news with    #
# predefined sentence boundaries                                                                             #
##############################################################################################################

class Story:
    
    def __init__(self, text, k=10):
        
        self.k = k
        self.sentences = sent_tokenize(text)
        self.length = len(self.sentences)
      
    @staticmethod    
    def racism_test(sentence):
        return bool(re.search(r'(?<!conspi)racis[mt]', sentence))
    
    @staticmethod
    def crt_test(sentence):
        return 'critical race theory' in sentence
    
    @staticmethod
    def dei_test(sentence):
        dei_pattern = 'diversity.+?equity.+?inclusion'
        return bool(re.search(dei_pattern, sentence))
    
    @staticmethod
    def woke_test(sentence):
        woke_pattern = '(?!woken)(?!woker)(?!wokep)(?!woked)(?!wokend)(?!woke\W(?:[\w+\W+]){0,25}?(up|by|from|to|him|his|her|you|u|your|me|us|my|our|them|their|it|everyone|something))(?<!\w)(?!woke\W+(?:\w+\W+){1,50}?(morning|a.?m))(?<!(got|get|had)\s)(?<!(have|just)\s)(?<!haven\'t\s)(?<!i\W)((?<!he\W))(?<!they\W)(?<!you\W)(?<!last\W)woke'
        return bool(re.search(woke_pattern, sentence))
        
    def extract_stories(self, test, tok_pattern=r"[\w|\"'|,|.]+|\$[\d\.]+|[ap].?m$"):
        '''
        Params:
            test: a function that returns a boolean, from one of the four @staticmethod tests (racism/crt/dei/woke). 
            tok_pattern: a regular expression pattern for word tokenization.    
        Returns:
            stories: a list of stories (str). An empty list if no stories are identified
        '''
        
        stories = []
        self.examined = -1

        self.test = test
        
        for i, sent in enumerate(self.sentences):
            
            # if the sentence has been checked when searching for the right boundary of a previous sentence, skip 
            if self.examined >= i:
                continue
                     
        
            if self.test(sent):

                left_bound, right_bound = self.search_boundaries(i)
                
                story_elements = self.sentences[left_bound:right_bound]
                        
                story = ''

                for ele in story_elements: 

                    ele = ' '.join(regexp_tokenize(ele, tok_pattern))
                    story += ele
                    story += ' '

                stories.append(story.rstrip())
        
        return stories
                
    
    def search_boundaries(self, i):        
       
        # Initialize story boundaries: 
        # Window the k sentences before and after the keywords-occurring sentence
        left_bound = max(0, i-self.k)
        right_bound = min(i+self.k+1, self.length)
        
        # Extend the story boundaries if another keyword is encountered within the story boundary
        boundry_found = False

        search_new_encounters = True

        while boundry_found is False:      

            while search_new_encounters: 
                marker = right_bound-1

                for j in range(marker, i, -1):

                    if self.test(self.sentences[j]):
                        right_bound = min(j+self.k+1, self.length)

                        break

                if marker == right_bound - 1:

                    search_new_encounters = False
                    boundry_found = True

        self.examined = right_bound - 1
        
        return left_bound, right_bound


if __name__ == "__main__":
    
    SOURCE_PATH = '/home/shared_files/tveyes_racism/'
    DESTINATION_PATH = '/home/shared_files/tveyes_stories/'
    
    
    for file in sorted(os.listdir(SOURCE_PATH)):
        
        print(f"Start extracting stories from\n{file}:")
    
        df = pd.read_csv(os.path.join(SOURCE_PATH, file))
    
        all_stories_racism = []
        all_stories_crt = []

        for i in df.index:
            
            full_text = df.loc[i, 'program_text']


            story_detector = Story(full_text)
            racism_stories = story_detector.extract_stories(Story.racism_test)
            crt_stories = story_detector.extract_stories(Story.crt_test)
            
            if racism_stories:
                for story_r in racism_stories:
                    all_stories_racism.append((i, story_r))

            if crt_stories:
                for story_c in crt_stories:
                    all_stories_crt.append((i, story_c))
          
        outfile = f'story_{file[-11:]}'
        
        racism_story_df = pd.DataFrame(all_stories_racism, columns=['original_index', 'story'])        
        final_racism_stories = df.drop('program_text', axis=1).merge(racism_story_df, how='right', left_index=True, right_on='original_index')
        final_racism_stories.to_csv(os.path.join(DESTINATION_PATH, 'racism', outfile), index=False)
        
        if all_stories_crt:
            crt_story_df = pd.DataFrame(all_stories_crt, columns=['original_index', 'story'])        
            final_crt_stories = df.drop('program_text', axis=1).merge(crt_story_df, how='right', left_index=True, right_on='original_index')
            final_crt_stories.to_csv(os.path.join(DESTINATION_PATH, 'crt', outfile), index=False)
           
        print('completed!\n--------------------')
    
    
