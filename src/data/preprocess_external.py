# Taken from: https://gist.github.com/ghadj/507e53effcf7fa9e873b3ed485723527

import string
import re
import html

# reference: https://github.com/NeelShah18/emot 
# pip install emot --upgrade
from emot.emo_unicode import UNICODE_EMOJI, EMOTICONS_EMO

# reference: https://github.com/nltk/nltk
# pip install nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')

# reference: https://github.com/psf/requests
# pip install requests
import requests

# reference: https://pypi.org/project/beautifulsoup4/
# pip install beautifulsoup4 
from bs4 import BeautifulSoup

# reference: https://pypi.org/project/inflect/
# pip install inflect
import inflect

# reference: https://pypi.org/project/pyspellchecker/
# pip install pyspellchecker
from spellchecker import SpellChecker

# reference https://gist.github.com/gruber/8891611
# from urlmarker import URL_REGEX 
URL_REGEX = re.compile("https?:\\/\\/(?:www\\.)?[-a-zA-Z0-9@:%._\\+~#=]{1,256}\\.[a-zA-Z0-9()]{1,6}\\b(?:[-a-zA-Z0-9()@:%_\\+.~#?&\\/=]*)")


# ============================================================================ #
#                          URL related functions                               #
# ============================================================================ #

def removeURLs(tweet):
    """Removes URLs in the tweet given."""
    tweet = re.sub(URL_REGEX, '', tweet)
    return tweet


def listURLs(tweet):
    """Returns a list of URLs contained in the given tweet."""
    return re.findall(URL_REGEX, tweet)


def extractTextFromURLs(urls):
    """Returns text from the given list of URL filtering out some HTML tags."""
    extracted = ''
    for url in urls:
        try:
            res = requests.get(url)
        except Exception as e:
            print(e)
            continue

        html_page = res.content
        soup = BeautifulSoup(html_page, 'html.parser')
        text = soup.find_all(text=True)

        undesired = ['[document]', 'noscript',
                     'header', 'html',
                     'meta', 'head',
                     'input', 'script',
                     'style', 'title']
        for t in text:
            if t.parent.name not in undesired:
                extracted += '{} '.format(t)

    return extracted


# ============================================================================ #
#                        Remove unwanted elements                              #
# ============================================================================ #

def replaceHTMLChar(tweet):
    """Convert all named and numeric character references 
    (e.g. &gt;, &#62;, &#x3e;) in the string s to the 
    corresponding Unicode characters."""
    return html.unescape(tweet)


def removeNonAscii(tweet):
    """Removes non ascii characters from given string."""
    return tweet.encode('ascii', 'ignore').decode('ascii')


def removeNonPrintable(tweet):
    """Removes non printable characters from given string."""
    return ''.join(filter(lambda x: x in string.printable, tweet))


def removePunctuation(tweet):
    """Removes punctuations (removes # as well)."""
    translator = str.maketrans(string.punctuation, ' '*len(string.punctuation))
    return tweet.translate(translator)


def removeNums(tweet):
    """Removes numeric values from the given string."""
    return ''.join([char for char in tweet if not char.isdigit()])


def replaceUsernames(tweet, text='USER'):
    """Removes usernames from given tweet."""
    return re.sub('@[^\s]+', text, tweet)


def removeUsernames(tweet):
    return replaceUsernames(tweet, text='')


def removeRepeatedChars(tweet):
    """Reduces repeated consecutive characters from given tweet to only two."""
    return re.sub(r'(.)\1+', r'\1\1', tweet)

def removeStopWords(tweet_list):
    """Removes stop-words from the given tweet."""
    return [word for word in tweet_list if word not in stopwords.words('english')]


# ============================================================================ #
#                           Format related functions                           #
# ============================================================================ #

def toLowerCase(tweet):
    """Separate camelCase to space delimited and convert tweet to lower-case."""
    tweet = re.sub(r'((?<=[a-z])[A-Z]|(?<!\A)[A-Z](?=[a-z]))', r' \1', tweet)
    tweet = tweet.lower()
    return tweet


# ============================================================================ #
#                           Meaning related functions                          #
# ============================================================================ #

def replaceEmojis(tweet):
    """Replace emojis in the text with their correspinding meaning."""
    for emot in UNICODE_EMOJI:
        tweet = tweet.replace(emot, "_".join(
            UNICODE_EMOJI[emot].replace(",", "").replace(":", "").split()))
    return tweet


def replaceEmoticons(tweet):
    """Replace emoticons in the text with their correspinding meaning."""
    for emot in EMOTICONS_EMO:
        tweet = re.sub(
            u'('+emot+')', "_".join(EMOTICONS_EMO[emot].replace(",", "").split()), tweet)
    return tweet


def replaceNums(tweet):
    """Replace numerical values with their textual representation."""
    infeng = inflect.engine()
    processed_tweet = []
    for word in tweet.split():
        processed_tweet.append(infeng.number_to_words(
            word) if word.isdigit() else word)
    return ' '.join(processed_tweet)


def correctSpelling(tweet_list):
    """Corrects spelling in the given string."""
    spell = SpellChecker()
    spell.word_frequency.load_words(['url'])  # add url to the dictionary
    # find those words that may be misspelled
    misspelled = spell.unknown(tweet_list)
    processed_tweet = []
    for word in tweet_list:
        # Replaced misspelled with the one most likely answer
        processed_tweet.append(spell.correction(
            word) if word in misspelled else word)
    return processed_tweet


def replaceFromDictionary(tweet_list, dictionary):
    """Replaces words included in the dictionary given with their corresponding value."""
    processed_list = []
    for word in tweet_list:
        if word in dictionary:
            if len(dictionary.get(word).split()) > 1:  # in case of multiple words in value
                processed_list.extend(dictionary.get(word).split())
            else:
                processed_list.append(dictionary.get(word))
        else:
            processed_list.append(word)
    return processed_list


def stemming(tweet_list):
    """Stemming - reduces the word-forms by removing suffixes."""
    return [PorterStemmer().stem(word) for word in tweet_list]


def lemmatization(tweet_list):
    """Lemmatization - reduces the word-forms to linguistically valid lemmas."""
    return [WordNetLemmatizer().lemmatize(word) for word in tweet_list]
