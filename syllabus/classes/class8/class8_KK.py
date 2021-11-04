#1) Calculate the dot product between two word embedding which you believe are similar
import gensim.downloader as api
import numpy as np
from torch import nn
model = api.load("glove-wiki-gigaword-50")

similar = np.dot(model['chair'], model['armchair'], out = None)

#2) Calculate the dot product between the two word and a word which you believe is dissimilar

dissimilar = np.dot(model['chair'], model['dog'], out = None)


#3) make the three words into a matrix $E$ and multiply it with its own transpose using matrix multiplication. So $E \cdot E^T$
 '''what does the values in matrice (E?) correspond to? What do you imagine the dot product is? 
 *Hint*, similarity between vectors (cosine similarity) is exactly the same as the dot product assuming you normalize the lenghth of each vector to 1.'''

E = np.array([model['chair'], model['dog'], model['cow']])
np.dot(E, E.transpose())

'''It is the mechanism of attention, i.e. how highly each word is connected to other words.'''

#4) Examine the attention formula from Vaswani et al. (2017), you have now implemented $Q\cdot K^T$
'''
$$
Attention(Q, K, V) = softmax(\frac{Q\cdot K^T}{\sqrt{d}}) \cdot V
$$
Where $d$ is the dimension of of the embedding and Q, K, V stands for queries, keys and values.
'''
d = len(model['chair'])
Q = E
K = E
V = np.dot(E, E.transpose())



attention = nn.Softmax(np.dot(np.dot(Q, K.transpose())/np.sqrt(d), V))

#4.1) Now add the softmax. Examining the outcome, how come that the matrix is no longer symmetric?
nn.Softmax(np.dot(Q, K.transpose()))
'''It is still symmetric? '''

#4.2) Now normalize the using the $\sqrt{d}$, how does this change the outcome?
nn.Softmax(np.dot(Q, K.transpose())/np.sqrt(d))
#The similarities are normalised (becoming smaller)

#1) the matrix resulting from the softmax is referred to as the attention 
# matrix and is how much each matrix should pay attention to the others 
# when we multiply our attention matrix by our matrix $E$ (corresponding 
# to $V$). 
# Try it out:

