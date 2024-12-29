from graph_models import HECO

meta_paths_dict = {
    'case-legislation-case': [('case', 'cites', 'legislation'), ('legislation', 'cited_by', 'case')],
    'case-subject-case': [('case', 'has_subject', 'subject_matter'), ('subject_matter', 'related_to', 'case')],
    'case-legislation-legislation-case' : [('case', 'cites', 'legislation') , ('legislation', 'cites', 'legislation'), ('legislation' , 'cited_by', 'case')],
    'case-subject-legislation-subject-case' : [('case', 'has_subject', 'subject_matter'), ('subject_matter', 'related_to', 'legislation'), ('legislation', 'has_subject', 'subject_matter'), ('subject_matter', 'related_to', 'case')],
    'case-legislation-subject-legislation-case': [('case', 'has_subject', 'subject_matter'), ('subject_matter', 'related_to', 'legislation'), ('legislation', 'has_subject', 'subject_matter'), ('subject_matter', 'related_to', 'case')],
}

network_schema = [
    ('legislation', 'cited_by', 'case'),
    ('subject_matter', 'related_to', 'case')
]

sample_rate = {
    'legislation': 2,
    'subject_matter': 2
}

model = HECO(
    feature_dir = '/Users/muizzkhalak/Drive/Github/Case_Similarity/data/titles/',
    feature_model = 'nlpaueb/legal-bert-base-uncased',
    feature_pooling = 'mean',
    meta_paths_dict = meta_paths_dict,
    network_schema = network_schema,
    category = 'case',
    sample_rate = sample_rate,
)
