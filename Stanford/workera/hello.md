### 🎉 Congratulations 🎉

If you want your model to produce the minimum possible number of false positives, which of the following metrics would you maximize?
(TP means “True Positive”, FN means “False Negative”, FP means “False Positive”, TN means “True Negative”.)

Если вы хотите, чтобы ваша модель давала минимально возможное количество ложных срабатываний, какой из следующих показателей вы бы максимизировали?

- TN/(TP+FN)
- TP/(TP+FP) , точность (precision)
- TP/(TP+FN) , полнота (recall)

+ (TP+TN)/(TP+TN+FP+FN) = (TP+TN)/|Dataset| , [доля правильных ответов](https://raw.githubusercontent.com/esokolov/ml-course-hse/d70ea03bcd41b6e2e77c481553cfbfcaf7a81304/2020-fall/lecture-notes/lecture04-linclass.pdf) (accuracy)
- (2P * R)/(P+R) , (F1-score)

Точность показывает, какая доля объектов, выделенных классификатором как положительные, действительно является положительными. 
Полнота показывает, какая часть положительных объектов была выделена классификатором.

### coursera
- [bcaffo/courses](https://github.com/bcaffo/courses)
- [Statistical Inference](https://leanpub.com/LittleInferenceBook/read#leanpub-auto-question) [videos](https://www.youtube.com/playlist?list=PLpl-gQkQivXiBmGyzLrUjzsblmQsLtkzJ)

expected values = mean and variance

```python
import  nltk.translate.bleu_score as bleu

# Setting the two different candidate translation that we will compare with two reference translations

reference_translation=['The cat is on the mat.'.split(),
                       'There is a cat on the mat.'.split()
                      ]
candidate_translation_1='the the the mat on the the.'.split()
candidate_translation_2='The cat is on the matrix.'.split()

# Calculating the BLEU score for candidate translation 1

print("BLEU Score: ",bleu.sentence_bleu(reference_translation, candidate_translation_1))
# The hypothesis contains 0 counts of 3-gram overlaps.
# BLEU Score:  6.968148412761692e-155

print("BLEU Score: ",bleu.sentence_bleu(reference_translation, candidate_translation_2))
# BLEU Score:  0.7598356856515925

```
