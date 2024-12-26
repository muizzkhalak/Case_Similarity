from HeCo import HeCo
from pandas import concat
import torch
import torch.optim as optim
import dgl
import networkx as nx
import random
from gensim.models import Word2Vec
from sentence_bert import SentenceBERT
from preprocessing import CasePreprocessing
import json
import warnings
warnings.filterwarnings('ignore')


class HECO(CasePreprocessing):

    def __init__(self,
                 feature_dir: str,
                 feature_model: str, 
                 feature_pooling: str,
                 meta_paths_dict: dict,
                 network_schema: dict,
                 category: str,
                 sample_rate: dict,
                 feat_drop: float = 0.1,
                 attn_drop: float = 0.1,
                 tau: float = 0.5,
                 lam: float = 0.5,
                 learning_rate: float = 0.01,
                 weight_decay: float = 1e-5,
                 num_epochs: int = 10,
                 device: str = None):

        super()

        if device:
            self.device = device
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print('Loading Features')
        with open(feature_dir + 'cases_titles.json', 'r') as file:
            self.feature_cases = json.load(file)
            self.feature_cases = {i : self._preprocessing(j) for i,j in self.feature_cases.items()}

        with open(feature_dir + 'legislation_titles.json', 'r') as file:
            self.feature_legislations = json.load(file)
            self.feature_legislations = {i : self._preprocessing(j) for i,j in self.feature_legislations.items()}

        with open(feature_dir + 'subject_matter_mapping.json', 'r') as file:
            self.feature_subject_matters = json.load(file)

        print('Building Graph')
        self.G = self.build_graph()
        print('Creating Feature Projections')
        # self.feature_dict = self._get_entity_features(feature_dir,feature_model,feature_pooling)
        self.meta_paths_dict = meta_paths_dict
        self.network_schema = network_schema

        self.category = category
        self.feat_drop = feat_drop
        self.attn_drop = attn_drop
        self.tau = tau
        self.lam = lam
        self.sample_rate = sample_rate
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs

        self.pos = torch.eye(self.G.num_nodes(category), device=device)
    
    def build_graph(self):

        judgement_celex_numbers = self.get_case_law_judgement_celex(year=None, celex_limit=None, preliminary_ruling=True)
        legislations_celex_numbers = self.get_celex_by_doc_type('3') + self.get_celex_by_doc_type('1')

        judgement_citations = self.get_citation_by_doc_type('6')
        legislation_citations = concat([self.get_citation_by_doc_type('3'), self.get_citation_by_doc_type('1')])
        
        cross_reference_citations = concat([judgement_citations, legislation_citations])
        subject_matter_citations = concat([self.get_subject_matter_by_doc_type('1'), self.get_subject_matter_by_doc_type('3'), self.get_subject_matter_by_doc_type('6')])



        case_case_edges = cross_reference_citations[(cross_reference_citations['source'].str.startswith('6')) & (cross_reference_citations['target'].str.startswith('6'))]
        case_case_edges = case_case_edges[(case_case_edges['source'].isin(list(self.feature_cases.keys()))) & (case_case_edges['target'].isin(list(self.feature_cases.keys())))]

        case_leg_edges = cross_reference_citations[(cross_reference_citations['source'].str.startswith('6')) & (cross_reference_citations['target'].str.startswith('3') | cross_reference_citations['target'].str.startswith('1'))]
        case_leg_edges = case_leg_edges[case_leg_edges['source'].isin(list(self.feature_cases.keys()))]
        case_leg_edges = case_leg_edges[case_leg_edges['target'].isin(list(self.feature_legislations.keys()))]

        case_subject_edges = subject_matter_citations[subject_matter_citations['source'].str.startswith('6')]
        case_subject_edges = case_subject_edges[case_subject_edges['source'].isin(list(self.feature_cases.keys()))]

        leg_leg_edges = cross_reference_citations[(cross_reference_citations['source'].str.startswith('3') | cross_reference_citations['source'].str.startswith('1')) & (cross_reference_citations['target'].str.startswith('3') | cross_reference_citations['target'].str.startswith('1'))]
        leg_leg_edges = leg_leg_edges[(leg_leg_edges['source'].isin(list(self.feature_legislations.keys()))) & (leg_leg_edges['target'].isin(list(self.feature_legislations.keys())))]

        leg_subject_edges = subject_matter_citations[(subject_matter_citations['source'].str.startswith('3')) | (subject_matter_citations['source'].str.startswith('1'))]
        leg_subject_edges = leg_subject_edges[leg_subject_edges['source'].isin(list(self.feature_legislations.keys()))]

        case_case_edges.drop_duplicates(inplace = True)
        case_leg_edges.drop_duplicates(inplace = True)
        leg_leg_edges.drop_duplicates(inplace = True)
        case_subject_edges.drop_duplicates(inplace = True) 
        leg_subject_edges.drop_duplicates(inplace = True) 


        # self.cases = sorted(judgement_celex_numbers)
        self.cases = sorted(list(set((case_leg_edges['source'].unique().tolist()) + 
                                    (case_case_edges['source'].unique().tolist()) + 
                                    (case_case_edges['target'].unique().tolist()) +
                                    (case_subject_edges['source'].unique().tolist()))))
        self.cases = {cas : idx for idx,cas in enumerate(self.cases)}

        self.legislations = sorted(list(set((case_leg_edges['target'].unique().tolist()) + 
                                            (leg_leg_edges['source'].unique().tolist()) + 
                                            (leg_leg_edges['target'].unique().tolist()) +
                                            (leg_subject_edges['source'].unique().tolist()))))
        self.legislations = {leg : idx for idx,leg in enumerate(self.legislations)}

        self.subjects = sorted(subject_matter_citations['target'].unique().tolist())
        self.subjects = {sub : idx for idx,sub in enumerate(self.subjects)}

        self.rev_cases = {idx:cas for cas, idx in self.cases.items()}
        self.rev_legislations = {idx:leg for leg, idx in self.legislations.items()}
        self.rev_subjects = {idx:sub for sub, idx in self.subjects.items()}


        case_case_edges['source'] = case_case_edges['source'].apply(lambda x: self.cases[x])
        case_case_edges['target'] = case_case_edges['target'].apply(lambda x: self.cases[x])

        case_leg_edges['source'] = case_leg_edges['source'].apply(lambda x: self.cases[x])
        case_leg_edges['target'] = case_leg_edges['target'].apply(lambda x: self.legislations[x])

        case_subject_edges['source'] = case_subject_edges['source'].apply(lambda x: self.cases[x])
        case_subject_edges['target'] = case_subject_edges['target'].apply(lambda x: self.subjects[x])

        leg_leg_edges['source'] = leg_leg_edges['source'].apply(lambda x: self.legislations[x])
        leg_leg_edges['target'] = leg_leg_edges['target'].apply(lambda x: self.legislations[x])

        leg_subject_edges['source'] = leg_subject_edges['source'].apply(lambda x: self.legislations[x])
        leg_subject_edges['target'] = leg_subject_edges['target'].apply(lambda x: self.subjects[x])


        case_case_src = torch.tensor(case_case_edges['source'].tolist())  
        case_case_dst = torch.tensor(case_case_edges['target'].tolist())  

        legis_legis_src = torch.tensor(leg_leg_edges['source'].tolist())   
        legis_legis_dst = torch.tensor(leg_leg_edges['target'].tolist())   

        case_legis_src = torch.tensor(case_leg_edges['source'].tolist())    
        case_legis_dst = torch.tensor(case_leg_edges['target'].tolist())    

        legis_case_src = case_legis_dst
        legis_case_dst = case_legis_src


        case_subj_src = torch.tensor(case_subject_edges['source'].tolist())   
        case_subj_dst = torch.tensor(case_subject_edges['target'].tolist())

        subj_case_src = case_subj_dst
        subj_case_dst = case_subj_src


        legis_subj_src = torch.tensor(leg_subject_edges['source'].tolist())    
        legis_subj_dst = torch.tensor(leg_subject_edges['target'].tolist())   

        subj_legis_src = legis_subj_dst
        subj_legis_dst = legis_subj_src

        # Build the heterograph using canonical etypes
        hg = dgl.heterograph({
            # Case - Case
            ('case', 'cites', 'case'): (case_case_src, case_case_dst),
            # Legislation - Legislation
            ('legislation', 'cites', 'legislation'): (legis_legis_src, legis_legis_dst),
            # Case - Legislation and reverse
            ('case', 'cites', 'legislation'): (case_legis_src, case_legis_dst),
            ('legislation', 'cited_by', 'case'): (legis_case_src, legis_case_dst),
            # Case - Subject matter and reverse
            ('case', 'has_subject', 'subject_matter'): (case_subj_src, case_subj_dst),
            ('subject_matter', 'related_to', 'case'): (subj_case_src, subj_case_dst),
            # Legislation - Subject matter and reverse
            ('legislation', 'has_subject', 'subject_matter'): (legis_subj_src, legis_subj_dst),
            ('subject_matter', 'related_to', 'legislation'): (subj_legis_src, subj_legis_dst)
            })
        
        # Remove legislations without any features
        # legislation_nodes_hg = [self._key_from_value(self.rev_legislations,i) for i in hg.nodes('legislation').tolist()]
        # available_feature_legislation = list(self.feature_legislations.keys())
        # missing_legislation_features = list(set(legislation_nodes_hg) - set(available_feature_legislation))
        # missing_legislation_features = [self.legislations[leg] for leg in missing_legislation_features]
        # print(len(missing_legislation_features))
        # print(hg.num_nodes('legislation'))
        # hg = dgl.remove_nodes(hg, torch.tensor(missing_legislation_features), ntype='legislation')
        # print(hg.num_nodes('legislation'))

        # # Remove cases without any features
        # case_nodes_hg = [self._key_from_value(self.rev_cases,i) for i in hg.nodes('case').tolist()]
        # available_feature_case = list(self.feature_cases.keys())
        # missing_case_features = list(set(case_nodes_hg) - set(available_feature_case))
        # missing_case_features = [self.cases[cas] for cas in missing_case_features]
        # print(missing_case_features)
        # print(hg.num_nodes('case'))
        # hg = dgl.remove_nodes(hg, torch.tensor(missing_case_features), ntype='case')
        # print(hg.num_nodes('case'))

        hg = hg.to(self.device)

        return hg
    
    @staticmethod
    def _key_from_value(rev_dictionary: dict, idx: int) -> str:

        return rev_dictionary[idx]
    
    def _get_feature_embedding(self, feature_dir, feature_model, feature_type):

        if feature_type == 'case':
            codes = [self._key_from_value(self.rev_cases,i) for i in self.G.nodes('case').tolist()]
            # code_text = [self.get_title(code) for code in codes]
            code_text = [self.feature_cases[code] for code in codes]
            return feature_model.encode(code_text)
        elif feature_type == 'legislation':
            codes = [self._key_from_value(self.rev_legislations,i) for i in self.G.nodes('legislation').tolist()]
            # code_text = [self.get_title(code) for code in codes]
            code_text = [self.feature_legislations[code] for code in codes]
            return feature_model.encode(code_text)
        elif feature_type == 'subject_matter':
            codes = [self._key_from_value(self.rev_subjects,i) for i in self.G.nodes('subject_matter').tolist()]
            # code_text = [self.get_subject_matter_text(code) for code in codes] 
            code_text = [self.feature_subject_matters[code] for code in codes]
            return feature_model.encode(code_text)

    def _get_entity_features(self,  feature_dir: str, feature_model: str,feature_pooling: str):

        feature_model = SentenceBERT(feature_model,feature_pooling, device=self.device)

        print('Generating Case Features')
        case_features = self._get_feature_embedding(feature_dir,feature_model,'case')
        print('Generating Legislation Features')
        legis_features = self._get_feature_embedding(feature_dir,feature_model,'legislation')
        print('Generating Subject Matter Features')
        subj_features = self._get_feature_embedding(feature_dir,feature_model,'subject_matter')

            
        h_dict = {
            'cases' : torch.Tensor(case_features, device=self.device),
            'legislation' : torch.Tensor(legis_features, device=self.device),
            'subject_matter' : torch.Tensor(subj_features, device=self.device)
        }

        return h_dict
    
    def train(self):
        
        self.model = HeCo(
                        meta_paths_dict=self.meta_paths_dict,
                        network_schema=self.network_schema,
                        category=self.category,
                        hidden_size=self.feature_dict.values()[0].shape[1],
                        feat_drop=self.feat_drop,
                        attn_drop=self.attn_drop,
                        sample_rate=self.sample_rate,
                        tau=self.tau,
                        lam=self.lam
                    ).to(self.device)
        
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        self.model.train()
        for epoch in range(self.num_epochs):
            optimizer.zero_grad()
            loss = self.model(self.G, self.feature_dict, self.pos)
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}: Loss {loss.item():.4f}")

    def get_embeddings(self):
        
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embeds(self.G, self.feature_dict)
        
        return embeddings
    


class MetaPath2Vec(CasePreprocessing):

    def __init__(self, 
                 metapaths, 
                 number_of_walks,
                 vector_size=128,
                 window=5, 
                 min_count=1, 
                 sg=1, 
                 workers=4, 
                 epochs=10):

        super()

        self.metapaths = metapaths
        self.number_of_walks = number_of_walks
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.sg = sg
        self.workers = workers
        self.epochs = epochs

        self.G = self.build_graph()

        self.neighbors_by_type = self._organize_neignbors()


    def build_graph(self):

        judgement_celex_numbers = self.get_case_law_judgement_celex(year=None, celex_limit=None, preliminary_ruling=True)
        legislations_celex_numbers = self.get_celex_by_doc_type('3') + self.get_celex_by_doc_type('1')

        judgement_citations = self.get_citation_by_doc_type('6')
        legislation_citations = concat([self.get_citation_by_doc_type('3'), self.get_citation_by_doc_type('1')])
        
        cross_reference_citations = concat([judgement_citations, legislation_citations])
        subject_matter_citations = concat([self.get_subject_matter_by_doc_type('1'), self.get_subject_matter_by_doc_type('3'), self.get_subject_matter_by_doc_type('6')])



        case_case_edges = cross_reference_citations[(cross_reference_citations['source'].str.startswith('6')) & (cross_reference_citations['target'].str.startswith('6'))]
        case_case_edges = case_case_edges[(case_case_edges['source'].isin(judgement_celex_numbers)) & (case_case_edges['target'].isin(judgement_celex_numbers))]

        case_leg_edges = cross_reference_citations[(cross_reference_citations['source'].str.startswith('6')) & (cross_reference_citations['target'].str.startswith('3') | cross_reference_citations['target'].str.startswith('1'))]
        case_leg_edges = case_leg_edges[case_leg_edges['source'].isin(judgement_celex_numbers)]

        case_subject_edges = subject_matter_citations[subject_matter_citations['source'].str.startswith('6')]
        case_subject_edges = case_subject_edges[case_subject_edges['source'].isin(judgement_celex_numbers)]

        leg_leg_edges = cross_reference_citations[(cross_reference_citations['source'].str.startswith('3') | cross_reference_citations['source'].str.startswith('1')) & (cross_reference_citations['target'].str.startswith('3') | cross_reference_citations['target'].str.startswith('1'))]

        leg_subject_edges = subject_matter_citations[(subject_matter_citations['source'].str.startswith('3')) | (subject_matter_citations['source'].str.startswith('1'))]


        case_case_edges.drop_duplicates(inplace = True)
        case_leg_edges.drop_duplicates(inplace = True)
        leg_leg_edges.drop_duplicates(inplace = True)
        case_subject_edges.drop_duplicates(inplace = True) 
        leg_subject_edges.drop_duplicates(inplace = True) 

        case_nodes = list(set(case_case_edges['source'].tolist() + 
                      case_case_edges['target'].tolist() + 
                      case_leg_edges['source'].tolist() + 
                      case_subject_edges['source'].tolist()
                      )
                )

        leg_nodes = list(set(case_leg_edges['target'].tolist() + 
                            leg_leg_edges['source'].tolist() + 
                            leg_leg_edges['target'].tolist() + 
                            leg_subject_edges['source'].tolist()
                            ) 
                        )

        subject_matter_nodes = list(set(case_subject_edges['target'].tolist() + 
                                        leg_subject_edges['target'].tolist()
                                        )
                                    )
        
        G = nx.Graph()

        for i in case_nodes:
            G.add_node(i, type='case')

        for i in leg_nodes:
            G.add_node(i, type='legislation')

        for i in subject_matter_nodes:
            G.add_node(i, type='subject_matter')

        for _,i in case_leg_edges.iterrows():
            G.add_edge(i['source'], i['target'])

        for _,i in case_case_edges.iterrows():
            G.add_edge(i['source'], i['target'])

        for _,i in leg_leg_edges.iterrows():
            G.add_edge(i['source'], i['target'])

        for _,i in case_subject_edges.iterrows():
            G.add_edge(i['source'], i['target'])

        for _,i in leg_subject_edges.iterrows():
            G.add_edge(i['source'], i['target'])

        return G
    
    def _organize_neignbors(self):

        neighbors_by_type = {}
        for node in self.G.nodes():
            neighbors_by_type[node] = {}
            for neighbor in self.G.neighbors(node):
                neighbor_type = self.G.nodes[neighbor]['type']
                neighbors_by_type[node].setdefault(neighbor_type, []).append(neighbor)

        return neighbors_by_type
    
    @staticmethod
    def _metapath_random_walk(start_node, metapath, neighbors_by_type):
        walk = [start_node]
        current_node = start_node
        for node_type in metapath[1:]:
            neighbors = neighbors_by_type[current_node].get(node_type, [])
            if not neighbors:
                break  # No further nodes to walk to
            next_node = random.choice(neighbors)
            walk.append(next_node)
            current_node = next_node
        return walk
    
    def _generate_walks(self):

        self.walks = []
        num_walks = self.number_of_walks  # Number of walks per starting node

        for metapath in self.metapaths:
            start_nodes = [node for node, data in self.G.nodes(data=True) if data['type'] == metapath[0]]
            for start_node in start_nodes:
                for _ in range(num_walks):
                    walk = self._metapath_random_walk(start_node, metapath, self.neighbors_by_type)
                    self.walks.append(walk)
    
    def train(self):

        self.model = Word2Vec(sentences=self.walks, 
                              vector_size=self.vector_size, 
                              window=self.window, 
                              min_count=self.min_count, 
                              sg=self.sg, 
                              workers=self.workers, 
                              epochs=self.epochs)
        
    def get_embeddings(self):

        all_celex_judgement = self.get_case_law_judgement_celex(year=None, celex_limit=None, preliminary_ruling=True)
        embeddings = {node: self.model.wv[node] for node in self.G.nodes() if node in all_celex_judgement}
        return embeddings