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
