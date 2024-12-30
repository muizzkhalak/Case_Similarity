import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
from data_collection import EurLexCollection
from typing import Tuple, Optional, Union, Literal

class CasePreprocessing(EurLexCollection):

    def __init__(self):
        super()

    @staticmethod
    def _preprocessing(doc: str) -> str:

        '''
        Clean and preprocess a document string by removing legal references, special characters, and redundancies.
        Parameters:
            - doc (str): Input text document as a string.
        Returns:
            - str: Cleaned and preprocessed text.
        '''

        a = '\(([a-z]{1,2}|[0-9]{1,2})\)' # (1) | (a) | (vi) | (12)
        b = 'Order No \d+\/\d+' # Order No 902/2089
        c = '(P|p)aragraph(s)? \d+((, \d+)*)?(,)?( (and|to) \d+)?' # paragraph 1 | paragraphs 1 and 9 | paragraphs 1, 2, and 3 | paragraphs 1 to 4
        d = '(Council )?Regulation (\((EC|EEC)\) )?(\[)?(No )?\d+\/\d+(\])?' # Regulation 90/234 | Regulation No 90/234 | Council Regulation (EEC) No 90/234 | Regulation (EC) No 90/234
        e = 'Article(s)? \d+([a-z])?((, \d+([a-z])?)*)?(,)?( and \d+([a-z])?)?' # Article 6 | Article 6e | Articles 6 and 7 | Articles 6, 7e, and 8
        f = '(R|r)ecital(s)? \d+((, \d+)*)?(,)?( (and|to) \d+)?' # recital 6 | recitals 6 and 7 | recitals 6, 7 and 8
        g = '(Council )?Directive(s)?( [A-Z]+)? \d+\/\d+(\/[A-Z]+)?((, \d+\/\d+(\/[A-Z]+)?)*)?(,)?( and \d+\/\d+(\/[A-Z]+)?)?' # Directive 990/2013 | Directives 990/2013/EEC and 990/2013 | Directives 990/2013/EEC, 970/2014/EEC and 870/2013/EC
        h = '(Case(s)? )?C(-|‑)\d+\/\d+' # C-90/123 (Cases are refereced as such)
        i = '\d{1,2} (January|February|March|April|May|June|July|August|September|October|November|December) \d{4}' # Date (21 July 2024)
        j = 'EU:[A-Z]:\d{4}:\d+' # ECLI identifier (EU:C:2023:908)
        k = 'OJ \d{4} (L|C) \d+, p\. \d+' # OJ 2006 L 328, p. 59
        l = '\d{1,2}\.' # 1.
        m = '[^\w\.\s]' # replace all everything except words, a fullstop, and a space
        n = r'\b(\w+)(?:\s+\1)+\b' # replace repeating words (at least 2 occurrences)

        doc = doc.replace('\xa0',' ')

        doc = re.sub(a,'',doc)
        doc = re.sub(b,'',doc)
        doc = re.sub(c,'',doc)
        doc = re.sub(d,'',doc)
        doc = re.sub(e,'',doc)
        doc = re.sub(f,'',doc)
        doc = re.sub(g,'',doc)
        doc = re.sub(h,'',doc)
        doc = re.sub(i,'',doc)
        doc = re.sub(j,'',doc)
        doc = re.sub(k,'',doc)
        doc = re.sub(l,'',doc)
        doc = doc.replace('…','') # Remove ellipsis
        doc = re.sub(m,'',doc) # Remove special characters
        doc = re.sub('\d+','',doc) # Remove all numbers
        doc = doc.replace('TFEU','') # Remove specific keywords

        # Remove extra spaces and newlines
        doc = re.sub('\n', ' ', doc)
        doc = re.sub('  +',' ',doc)

        # Remove duplicate words
        doc = re.sub(n, r'\1', doc)
        doc = doc.strip() # Trim whitespace

        return doc
    
    # Static method to extract content from HTML (type 1 structure)
    @staticmethod
    def _html_extractor_type_1(full_html_text: bytes) -> pd.DataFrame:

        '''
        Extracts paragraph numbers and text from HTML content (type 1 format).
        Parameters:
            - full_html_text (bytes): Full HTML content of the document.
        Returns:
            - DataFrame containing paragraph numbers and text.
        '''

        soup = BeautifulSoup(full_html_text, 'lxml') # Parse HTML with BeautifulSoup
        tables = soup.find('body').findAll('table', recursive=False) # Find tables at the first level in the body

        all_content = []
        for table in tables:

            cols = table.findAll('col', recursive=False) # Find column definitions
            paragraph = table.find('tr').findAll('td', recursive=False) # Find text within table rows

            # Check table structure to classify its content
            if len(cols) > 2 or all([True if col.attrs.get('width') not in ['5%', '95%'] else False for col in cols]):
                for row in paragraph:
                    all_content[-1]['Text'] += ' ' + row.text.strip() # Append text to the last paragraph
            else:
                temp = {'ParaNum' : paragraph[0].text.strip(), 'Text' : paragraph[1].text.strip()}
                all_content.append(temp) # Add new paragraph data

        all_content = pd.DataFrame(all_content) # Convert content to a DataFrame
        # Remove irrelevant rows and reset indices
        all_content = all_content[(all_content['ParaNum'] != '–') & (all_content['ParaNum'] != '—') & (all_content['Text'] != '[Signatures]')]
        all_content.reset_index(drop=True, inplace=True)
        
        # Retain content starting from paragraph 1
        first_index = all_content[all_content['ParaNum'] == '1'].index

        all_content = all_content.iloc[first_index[0]:]
        all_content.reset_index(drop=True, inplace=True)

        return all_content

    # Static method to extract content from HTML (type 2 structure)
    @staticmethod
    def _html_extractor_type_2(full_html_text: bytes) -> pd.DataFrame:

        '''
        Extracts paragraph numbers and text from HTML content (type 2 format).
        Parameters:
            - full_html_text (bytes): Full HTML content of the document.
        Returns:
            - DataFrame containing paragraph numbers and text.
        '''

        soup = BeautifulSoup(full_html_text, 'lxml') # Parse HTML with BeautifulSoup
        tags = soup.findAll('p', attrs={'class' : 'C01PointnumeroteAltN'}) # Find specific paragraph tags

        all_content = []
        last_tag = tags[-1] # Get the last paragraph tag
        for i in range(len(tags)):

            # Logic to group text blocks and extract paragraph details
            content = []

            current_tag = tags[i]
            if current_tag == last_tag:
                content.append(current_tag)
            else:
                end_tag = tags[i+1]
                content.append(current_tag)
                while True:
                    current_tag = current_tag.find_next_sibling()
                    if current_tag == end_tag:
                        break
                    else:
                        if current_tag.attrs.get('class'):
                            if 'Titre' in current_tag.attrs.get('class')[0]:
                                continue
                            else:
                                content.append(current_tag)
                        else:
                            continue
            
            # Combine all extracted tags into a paragraph
            paragraph = ''
            for tag in content:
                if tag.attrs.get('class')[0] == 'C01PointnumeroteAltN':
                    try:
                        para_number = tag.find('a').text # Extract paragraph number
                        tag.select('a')[0].extract()
                        paragraph += tag.text.strip() + '  '
                    except:
                        check_para_number = re.match('^\d+',tag.text.strip())
                        if check_para_number:
                            para_number = check_para_number.group(0)
                            paragraph += re.sub('^\d+','',tag.text.strip()) + '  '
                        else:
                            paragraph += tag.text.strip() + '  '
                else:
                    paragraph += tag.text.strip() + '  '

            # Append paragraph details to the results
            if 'para_number' in locals():
                if para_number.isnumeric():
                    temp = {'ParaNum' : para_number, 'Text' : paragraph}
                    all_content.append(temp)
                else:
                    if len(all_content) > 0:
                        all_content[-1]['Text'] += ' ' + paragraph

                del para_number
                del paragraph
            else:
                if len(all_content) > 0:
                    all_content[-1]['Text'] += ' ' + paragraph

        # Extract operative parts
        operative = soup.findAll('p', attrs={'class' : re.compile(r'Disposit')})
        for paragraph in operative:
            temp = {'ParaNum' : np.nan , 'Text' : paragraph.text.strip()}
            all_content.append(temp)

        return pd.DataFrame(all_content)

    def get_preprocessed_case(self, 
                        source_celex: str, 
                        result: Literal['paragraph', 'article'],
                        text: Optional[bytes] = None) -> Union[pd.DataFrame, str]:
        
        '''
        Retrieve and preprocess a case document, returning structured paragraphs or full text.
        Parameters:
            - source_celex (str): CELEX number of the document.
            - result (Literal['paragraph', 'article']): Specify the output format:
                * 'paragraph' for structured paragraphs as a dictionary.
                * 'article' for a concatenated full text as a string.
            - text (Optional[bytes]): Pre-fetched raw HTML content of the case document (optional).
        Returns:
            - Union[pd.DataFrame, str]: Either a dictionary of paragraphs or a full concatenated string.
        '''
        # If HTML text is not provided, fetch it using the CELEX number
        if text:
            pass
        else:
            text = self.get_full_text(source_celex=source_celex)

        if isinstance(text, bytes):
            text = text.decode('utf-8')

        # Check if the text contains an error message
        if 'None of the requests returned' in text:
            return None
        # Detect the HTML structure type and extract content accordingly
        if 'C01PointnumeroteAltN' in text: # Type 2 structure
            content = self._html_extractor_type_2(text)
        elif ('class="coj-sum-title-1"' in text) or ('class="sum-title-1"' in text): # Type 1 structure
            content = self._html_extractor_type_1(text)
        else:
            # Return None if the HTML structure is unrecognized
            return None

        # Apply preprocessing to the extracted text content
        content['Text'] = content['Text'].apply(self._preprocessing)

        # Handle paragraph numbers (cleaning, forward-filling, and conversion to integer)
        content['ParaNum'] = np.where((content['ParaNum'].isna()) | ((content['ParaNum'] == '')), np.nan, content['ParaNum'])
        content['ParaNum'] = content['ParaNum'].astype(float)
        content['ParaNum'] =  content['ParaNum'].ffill()
        content['ParaNum'] =  content['ParaNum'].astype(int)

        # Create a structured dictionary for the paragraphs
        final = {}
        for para in content['ParaNum'].unique():
            para_df = content[content['ParaNum'] == para]
            if para_df.shape[0] > 1: # Handle cases where multiple rows exist for a paragraph
                for i in range(para_df.shape[0]):
                    final[str(para) + '_' + str(i+1) + '_' + source_celex] = para_df.iloc[i]['Text']
            else:
                final[str(para) + '_' + source_celex] = para_df.iloc[0]['Text']

        # Return the output based on the specified result format
        if result=='paragraph':
            return final # Return structured paragraphs as a dictionary
        elif result=='article':
            return ' '.join(list(final.values())) # Concatenate all paragraphs into a single string
