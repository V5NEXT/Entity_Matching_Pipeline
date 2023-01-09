
import recordlinkage
import pandas as pd

data_binning = __import__('Task1_2')
try:
    attrlist = data_binning.__all__
except AttributeError:
    attrlist = dir(data_binning)
for attr in attrlist:
    globals()[attr] = getattr(data_binning, attr)

df_perfect_Match = pd.read_csv('DBLP-ACM_perfectMapping.csv', header=0, encoding="ISO-8859-1")

df_ACM, df_DBLP, candidate_links = data_binning.binning()


def matching():
    compare_cl = recordlinkage.Compare()
    compare_cl.exact("venue", "venue", label="venue")
    compare_cl.string("title", "title", method="jarowinkler", label="title")
    compare_cl.exact("year", "year", label="year")
    compare_cl.string("authors", "authors", method="jarowinkler", label="authors")

    features = compare_cl.compute(candidate_links, df_ACM, df_DBLP)

    matches = features[features[['title', 'authors']].sum(axis=1) >= 1.6]
    matches.index.names = ['ACM', 'DBLP']
    matches = matches.reset_index()
    matches_new = matches.loc[matches.groupby(['ACM'])['DBLP'].idxmax()]
    links_pred = matches_new.set_index(['ACM', 'DBLP'])

    return links_pred


def evaluation():
    d_1 = df_ACM['id'].to_dict()
    d_1_flip = {y: x for x, y in d_1.items()}

    d_2 = df_DBLP['id'].to_dict()
    d_2_flip = {y: x for x, y in d_2.items()}

    df_perfect_Match['idACM'] = df_perfect_Match['idACM'].map(d_1_flip)
    df_perfect_Match['idDBLP'] = df_perfect_Match['idDBLP'].map(d_2_flip)

    perfectMapping = df_perfect_Match[['idACM', 'idDBLP']]

    links_true = pd.MultiIndex.from_frame(perfectMapping)

    f_score = recordlinkage.fscore(links_true, links_pred.index)

    return f_score


links_pred = matching()
f_score = evaluation()
print(links_pred)
print(f_score)










