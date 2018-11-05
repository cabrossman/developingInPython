#http://flennerhag.com/2017-01-08-Recursive-Override/

""" image you are using StandardScaler, but want it to return pandas DF instead of numpy array"""

import numpy as np
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

df = DataFrame(np.array([[1, 0], [2, 1], [0, 1]]),columns=['a', 'b'], dtype='float')
print(df)

nd = StandardScaler().fit_transform(df)
print(nd)

#rename the standard scaler class to foo
class foo(StandardScaler):
	pass

fnd = foo().fit_transform(df)
fnd

#override the transform class
class foo(StandardScaler):
	def transform(self, x): #this overwrides the transform method
		print(type(x))

foo().fit_transform(df)

class foo(StandardScaler):

    def transform(self, x):
        z = super(foo, self).transform(x) #overrides and turns to dataframe using Standard
        return DataFrame(z, index=x.index, columns=x.columns)

dff = foo().fit_transform(df)
dff

"""final class"""
class StandardScalerDf(StandardScaler):
    """DataFrame Wrapper around StandardScaler"""

    def __init__(self, copy=True, with_mean=True, with_std=True):
        super(StandardScalerDf, self).__init__(copy=copy, with_mean=with_mean,with_std=with_std)

    def transform(self, X, y=None):
        z = super(StandardScalerDf, self).transform(X.values)
        return DataFrame(z, index=X.index, columns=X.columns)

dff = StandardScalerDf().fit_transform(df)
dff


"""final class"""
class StandardScalerDf(StandardScaler):
    """shouldnt need initializer since we are not altering it"""

    def transform(self, X, y=None):
        z = super().transform(X.values)
        return DataFrame(z, index=X.index, columns=X.columns)

dff = StandardScalerDf().fit_transform(df)
dff