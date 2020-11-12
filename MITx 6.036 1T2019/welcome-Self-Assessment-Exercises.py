# simple routines
def add_two_lists(a, b):
    return [x*y for x,y in zip(a,b)]

def dot(v1, v2):
    return sum(x*y for x,y in zip(v1,v2))

# dot([1,2,3], [4,5,6])
# 32
