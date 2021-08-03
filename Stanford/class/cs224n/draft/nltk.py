'''
pip install nltk

# nltk.tokenize
# nltk.tag
'''
import nltk
nltk.download('punkt')

sentence = """Клен ты мой опавший, клен заледенелый,
Что стоишь нагнувшись под метелью белой?
Или что увидел? Или что услышал?
Словно за деревню погулять ты вышел."""
tokens = nltk.word_tokenize(sentence)
print(tokens)

from nltk.stem.snowball import SnowballStemmer
stemmer = SnowballStemmer("russian")
l=[stemmer.stem(word) for word in tokens]
print(l)
# ['клен', 'ты', 'мо', 'опа', ',', 'клен', 'заледенел', ',', 'что', 'сто', 'нагнувш', 'под', 'метел', 'бел', '?', 'ил', 'что', 'увидел', '?', 'ил', 'что', 'услыша', '?', 'словн', 'за', 'деревн', 'погуля', 'ты', 'вышел', '.']
