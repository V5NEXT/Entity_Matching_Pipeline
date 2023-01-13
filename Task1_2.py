import Levenshtein
import numpy as np
from recordlinkage.index import Block


data_prep = __import__('Task1_1')
try:
    attrlist = data_prep.__all__
except AttributeError:
    attrlist = dir(data_prep)
for attr in attrlist:
    globals()[attr] = getattr(data_prep, attr)

# With preprocessing
df_DBLP, df_ACM = data_prep.preprocessing()
# Without preprocessing
# df_DBLP, df_ACM = data_prep.evaluating_dataset()


v_1=df_ACM['venue'].value_counts()
v_2=df_DBLP['venue'].value_counts()

v_1_index=v_1.index
v_2_index=v_2.index

def similarities():
    lista = []
    lista_2 = []
    maksimumi = np.zeros((len(v_1_index), len(v_2_index)))
    for i in range(len(v_1_index)):
        for j in range(len(v_2_index)):
            maksimumi[i][j] = Levenshtein.ratio(v_1_index[i], v_2_index[j])
            #print(maksimumi)
    from numpy import unravel_index
    for i in range(len(v_1_index)):
        (u, v) = unravel_index(maksimumi.argmax(), maksimumi.shape)
        lista.append(v_1_index[u])
        lista_2.append(v_2_index[v])
        maksimumi[:, v] = np.zeros(len(v_1_index))
        maksimumi[u, :] = np.zeros(len(v_2_index))
        #print(maksimumi)
    #print(lista)
    #print(lista_2)

    dictionary = dict(zip(lista_2, lista))
    dictionary

    df_DBLP['venue'] = df_DBLP['venue'].map(dictionary)
   
    return df_DBLP

df_DBLP=similarities()  #processed

def blocking():
    indexer = Block(left_on=['year', 'venue'],
                    right_on=['year', 'venue'])
    candidate_links = indexer.index(df_ACM, df_DBLP)


    return df_ACM,df_DBLP,candidate_links

#run if you dont want task 1_3
#df_ACM, df_DBLP, candidate_links = blocking()






