# 6 Простые решения

В этой главе вводим понятие простых решений, где мы принимаем единое решение при неопределенности.1 Мы изучим проблему принятия решений с точки зрения теории полезности, которая включает в себя моделирование предпочтений агента как реальную функцию по сравнению с неопределенными результатами.2 Эта глава начинается с обсуждения того, как небольшой набор ограничений на рациональные предпочтения может привести к существованию функции полезности (utility function). Эта функция полезности может быть выведена из последовательности запросов предпочтений (sequence of preference queries). Затем мы представляем максимальный принцип ожидаемой полезности как определение рациональности, центральную концепцию в теории решений, которая будет использоваться в качестве принципа управления принятием решений в этой книге.3 Мы показываем, как проблемы с принятием решений могут быть представлены в качестве сетей решений, и показываем алгоритм для нахождения оптимального решения. Внедрена концепция стоимости информации (value of information), которая измеряет полезность, полученную посредством наблюдения за дополнительными переменными. Глава завершается кратким обсуждением того, как принятие человеческих решений не всегда соответствует максимальному ожидаемому принципу полезности.

## 6.1 Ограничения на Рациональные Предпочтения

Мы начали нашу дискуссию о неопределенности в главе 2, определив необходимость сравнения нашей степени веры в различные заявления. Эта глава требует возможности сравнить степень желательности двух разных результатов. Мы заявляем наши предпочтения, используя следующие операторы:

- A ≻ B если мы предпочитаем A больше B.
- A ∼ B если нам безразлично между A и B.
- A ≻∼ B если мы предпочитаем A больше B или безразлично.

Точно так же, как убеждения могут быть субъективными, так и предпочтения.

В дополнение к сравнению событий, наши операторы предпочтения могут быть использованы для сравнения предпочтений по сравнению с неопределенными результатами. Лотерея - это набор вероятностей, связанных с набором результатов. Например, если $S_{1:n}$ - это набор результатов, а $P_{1:n}$ - их связанные с ними вероятности, то лотерея, включающая эти результаты и вероятности, записывается как

$[ S_1 : p_1 ; . . . ; S_n : p_n ]$

Существование реальной меры полезности возникает из набора предположений о предпочтениях.4. Из этой функции полезности можно определить, что значит принимать рациональные решения при неопределенности. Так же, как мы наложили набор ограничений на убеждения, мы будем накладывать некоторые ограничения на предпочтения:5

- Полнота. Только одно из следующих данных может быть: A ≻ B, B ≻ A, or A ∼ B.
- Транзитивность. Если A ≻∼ B и B ≻∼ C, тогда A ≻∼ C.
- Непрерывность. Если A ≻∼ C ≻∼ B, тогда существует вероятность p такая, что [ A : p; B : 1 − p] ∼ C.
- Независимость. Если A ≻ B, тогда для любого C и вероятности p, [ A : p; C : 1 − p] ≻∼ [ B : p; C : 1 − p].

Это ограничения на рациональные предпочтения.Они ничего не говорят о предпочтениях реальных людей; на самом деле, есть убедительные доказательства того, что люди не всегда рациональны (точка обсуждается далее в разделе 6.7). Наша цель в этой книге - понять рациональное принятие решений с вычислительной точки зрения, чтобы мы могли создавать полезные системы. Возможное расширение этой теории к пониманию принятия человеческих решений представляет только вторичный интерес.

## 6.2 Функции полезности

Подобно тому, как ограничения на сравнение правдоподобия различных утверждений приводят к существованию реальной меры вероятности, ограничения на рациональные предпочтения приводят к существованию реальной меры полезности. Это следует из наших ограничений на рациональные предпочтения, что существует реальная функция полезности U, так что

- U(A) > U (B) если и только если A ≻ B, и
- U(A) = U (B) если и только if A ∼ B.

Функция полезности уникальна до положительной аффинной трансформации. Другими словами, для любых констант m > 0 и b, U′ (S) = mU (S) + b тогда и только тогда, когда предпочтения вызваны U′ такой же, как U. Полезности похожи на температуры: вы можете сравнивать температуру, используя Кельвина, Цельсия или Фаренгейта, все из которых являются аффинными преобразованием друг друга.

Это следует из ограничений на рациональные предпочтения, что полезность лотереи определяется

$U([S_1 : p_1 ; . . . ; S_n : p_n ]) = \sum_{i=1}^n p_i U(S_i)$ (6.2)

Пример 6.1 применяет это уравнение для вычисления полезности результатов, связанных с системой избегания столкновений.


-----------
Пример 6.1. Лотерея, включающая результаты системы избегания столкновений.

Предположим, что мы строим систему избегания столкновений.Результат встречи самолета определяется, предупреждает ли система (A) и происходит ли столкновение (C). Поскольку A и C являются бинарными, есть четыре возможных результата. До тех пор, пока наши предпочтения рациональны, мы можем написать нашу функцию полезности в пространстве возможных лотерей с точки зрения четырех параметров:

$U ( a^0 , c^0 ), U ( a^1 , c^0 ), U ( a^0 , c^1 ),$ и $U ( a^1, c^1 )$. Например,

$U ([ a^0 , c^0 : 0.5; a^1 , c^0 : 0.3; a^0 , c^1 : 0.1; a^1 , c^1 : 0.1])$

равно

$0.5U ( a^0 , c^0 ) + 0.3U ( a^1 , c^0 ) + 0.1U ( a^0 , c^1 ) + 0.1U ( a^1 , c^1 )$

-----------

Если функция полезности ограничена, то мы можем определить *нормализованную функцию полезности*, где наилучшим возможным результатом является полезность 1, а наихудшим возможным результатом является полезность 0. Полезность каждого из других результатов масштабируется и переносится по мере необходимости.

## 6.3 Выявление полезности