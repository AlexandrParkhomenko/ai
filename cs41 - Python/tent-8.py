import sys
# print(sorted(['Ш', 'А', 'Т', 'Ё', 'Р']))
let = 'АЁРТШ'
counter=1
for a in let:
    for b in let:
        for c in let:
            for d in let:
                for e in let:
                    l=a+b+c+d+e
                    #print(l,counter,l.count('А'))
                    if l.count('А') == 1:
                        # m = l.index('А')
                        # n = m-1
                        # if n < 0: n=0
                        # m=m+1
                        # if m > 4: m=4
                        # if l[n]!='Ё' and l[m]!='Ё':
                        if 'ЁЁ' not in l:
                            print(l, counter)
                            sys.exit()
                    counter+=1
#183
