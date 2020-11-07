# 1A) What is the margin γ of this data set with respect to that separator (up to 3 decimal places)?

data = np.array([[200, 800, 200, 800],
#             [-0.3,  -0.3,  0.3,  0.3],
             [0.2,  0.2,  0.8,  0.8],
             [1,  1,  1,  1]])
labels = np.array([[-1, -1, 1, 1]])
th = np.array([[0.,1,-0.5]])

def gamma(x, y, th):
    print("x=",x)
    print("x.shape",x.shape)
    print("y=",y)
    print("th=",th)
    print("th.shape=",th.shape)
    #th_multy = np.ones([th.shape[1],x.shape[1]]) * th.T
    #print("th_multy=",th_multy)
    a = y*np.dot(th,x)
    print("y*(np.dot(th_multy.T,x))=",a)
    b = np.dot(th[0].T,th[0])**0.5
    print("||θ||=",b)
    b = np.linalg.norm(th.T)
    print("||θ||=",b) #//
    c = a/b
    print("a/b=",c)
    return(c[0][np.argmin(c)])

g = gamma(data, labels, th)
print("gamma=",g)
# 0.2683281572999747

# 1B) What is the theoretical bound on the number of mistakes perceptron will make on this problem?
#print(np.dot(data.T,data)**0.5) # oops, unable get matrix with lengths
R = 0
for i in data.T:
   R = max( R, np.dot(i.T,i)**0.5 )
print("R=",R)
print("(R/g)**2=",(R/g)**2)
# 8888911.666666672
