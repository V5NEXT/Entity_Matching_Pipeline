
import pandas as pd
import recordlinkage


data_blocking = __import__('Task1_2')
try:
    attrlist = data_blocking.__all__
except AttributeError:
    attrlist = dir(data_blocking)
for attr in attrlist:
    globals()[attr] = getattr(data_blocking, attr)


df_perfect_Match = pd.read_csv('DBLP-ACM_perfectMapping.csv', header=0, encoding="ISO-8859-1")

df_ACM, df_DBLP, candidate_links = data_blocking.blocking()


def feature_scores():
    compare_cl = recordlinkage.Compare()
    compare_cl.exact("venue", "venue", label="venue")
    compare_cl.string("title", "title", method="jarowinkler", label="title")
    compare_cl.exact("year", "year", label="year")
    compare_cl.string("authors", "authors", method="jarowinkler", label="authors")
    features = compare_cl.compute(candidate_links, df_ACM, df_DBLP)
    
    return features

features=feature_scores()

def matching_filter_step():    
    matches = features[features[['title','authors']].sum(axis=1)>= 1.6]
    matches.index.names = ['ACM','DBLP']
    matches=matches.reset_index()
    
    return  matches

matches = matching_filter_step()

#print(matches)

def matching_refinement_step():
    
    matches_new=matches.loc[matches.groupby(['ACM'])['DBLP'].idxmax()]
    links_pred=matches_new.set_index(['ACM', 'DBLP'])
    
    return links_pred 


links_pred=matching_refinement_step() 

#print(links_pred.index)  

def translating_true_links():
    
    #print(df_perfect_Match)
    d_1 = df_ACM['id'].to_dict()
    d_1_flip = {y: x for x, y in d_1.items()}
    #print(d_1_flip)
    d_2 = df_DBLP['id'].to_dict()
    d_2_flip = {y: x for x, y in d_2.items()}
    #print(d_2_flip)
    df_perfect_Match['idACM'] = df_perfect_Match['idACM'].map(d_1_flip)
    df_perfect_Match['idDBLP'] = df_perfect_Match['idDBLP'].map(d_2_flip)

    perfectMapping = df_perfect_Match[['idACM', 'idDBLP']]
    #print(perfectMapping)
    links_true = pd.MultiIndex.from_frame(perfectMapping)
    
    return links_true

links_true=translating_true_links() 

#print(links_true)


def evaluation():

    f_score=recordlinkage.fscore(links_true, links_pred.index)

    return f_score

f_score=evaluation()

print(f_score)











