import nltk
from nltk.corpus import stopwords
stop_words = set(stopwords.words("english"))
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Make custom list for words that should not be included as uppercase. These are often abbreviations or the description of the condition 
# the person has, abbreviation for dosage, store where they picked up the medication, etc.

mh_list = ['AA','AD','AAF','AAP','PCP','AB','BP','AC','ACL','OCD','ADHD',
           'ADHDADD','ADD','ADDADHD','ADHA','ADHDIA','ADHDPI','AHHD','AMPM','ARNP',
           'GAD','GI','GF','IBS','XR','ER','MG','MD','DR','PTSD','SSRI','SR','GP',
           'BPII','BUTI','CDPHP','CPAP','CPEP','CFSME','GADOCD','MHNI','MS','BDD',
           'XL', 'CC','PT','SNRI','HCL','CVS','IBS','IBSD','CBT','CFS','MAO','MI','PMDD',
           'SSRIS','SNRIS','OSA','PMS','BPD','II','DNA','PM','TMJ','SRI','OBGYN','MBA',
           'SNRI','USA','DX', 'DSM']


def clean_text(df): 
    
    """
    Cleans the review text field
    """
    
    #Remove stop words
    df['review_clean'] = df['review'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    
    #Strip punctuation
    df['review_clean'] = df['review_clean'].str.replace('[^\w\s]','')

    #Strip numbers
    df['review_clean'] = df['review_clean'].str.replace('\d+', '')
    
    #Remove extra white space
    df['review_clean'] = df['review_clean'].str.strip()
    
    #Remove all space characters
    df['review_clean'] = df['review_clean'].str.replace("  ", " ")
    
    
    
def text_length(text):
    
    """
    Count number of words in the text
    """
    
    text = text.strip()
    
    total = 0
    
    for word in text.split():
        total += 1
        
    return total
        
def uppercase(text):
    
    """
    Returns a list of only selected uppercase words
    and are not in custom list of words specified only for mental health
    """
    
    text = text.strip()
    
    total = []
    
    for word in text.split():
        if word.isupper() and len(word) > 1 and word not in mh_list:
            total.append(word)
    return ', '.join(total)


def uppercase_count(text):
    
        
    """
    Returns a count of only selected uppercase words
    and are not in custom list of words specified only for mental health
    """
    
    text = text.strip()
    
    total = 0
    
    for word in text.split():
        if word.isupper() and len(word) > 1 and word not in mh_list:
            total += 1
    return total


def PosSentimentAnalyzer(text):
    """
    Returns count of capitalized words frm ntlk sentiment polarity that are positively
    charged (i.e > .50)
    """
    
    sia = SentimentIntensityAnalyzer()
    
    text = text.strip()
    
    total = []
    
    for word in text.split():
        if (sia.polarity_scores(word)['compound']) >= 0.5:
            total.append(word)  
    return', '.join(total)


def PosSentimentAnalyzerCount(text):
    """
    Returns count of capitalized words frm ntlk sentiment polarity that are positively
    charged (i.e > .50)
    """
    
    sia = SentimentIntensityAnalyzer()
    
    text = text.strip()
    
    total = 0
    
    for word in text.split():
        if (sia.polarity_scores(word)['compound']) >= 0.5:
            total += 1  
    return total

def NegSentimentAnalyzer(text):
    """
    Returns count of capitalized words from ntlk sentiment polarity that are negatively
    charged (i.e > .50)
    """
    
    sia = SentimentIntensityAnalyzer()
    
    text = text.strip()
    
    total = []
    
    for word in text.split():
        if (sia.polarity_scores(word)['compound']) <= -0.5:
            total.append(word)  
    return', '.join(total)


def NegSentimentAnalyzerCount(text):
    """
    Returns count of capitalized words frm ntlk sentiment polarity that negatively charged
    charged (i.e > .50)
    """
    
    sia = SentimentIntensityAnalyzer()
    
    text = text.strip()
    
    total = 0
    
    for word in text.split():
        if (sia.polarity_scores(word)['compound']) <= -0.5:
            total += 1  
    return total