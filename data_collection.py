import pandas as pd
from SPARQLWrapper import SPARQLWrapper, JSON
from bs4 import BeautifulSoup
import requests
from urllib.parse import quote_plus
from typing import Tuple, Optional, Union

# Define a class for interacting with EUR-Lex SPARQL endpoint and related tasks
class EurLexCollection:

    # Function to retrieve CELEX numbers for case law judgments
    @staticmethod
    def get_case_law_judgement_celex(year: Optional[str] = None,
                                     celex_limit: Optional[int] = None,
                                     preliminary_ruling: Optional[bool] = True) -> list:
        '''
        Retrieve CELEX numbers for case law judgments.
        Parameters:
            - year (Optional[str]): Filter case law by the year of publication.
            - celex_limit (Optional[int]): Limit the number of CELEX numbers returned.
            - preliminary_ruling (Optional[bool]): Include preliminary rulings if True.
        Returns:
            - List of CELEX numbers for case law judgments.
        '''
        # Initialize SPARQL wrapper
        sparql = SPARQLWrapper('https://publications.europa.eu/webapi/rdf/sparql')
        sparql.setReturnFormat(JSON)  # Set output format to JSON

        # SPARQL query to retrieve CELEX numbers for case law
        sparql.setQuery('''
                PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT DISTINCT ?celex
                WHERE {
                    ?doc cdm:resource_legal_id_celex ?celex ;
                        cdm:resource_legal_id_sector "6"^^xsd:string ;
                        cdm:resource_legal_year %s ;
                        cdm:resource_legal_type ?type ;
                        cdm:case-law_has_procjur ?protype ;
                        cdm:work_has_resource-type ?form .
                    %sFILTER(?protype = <http://publications.europa.eu/resource/authority/procjur/REFER_PREL>)
                    FILTER(?form = <http://publications.europa.eu/resource/authority/resource-type/JUDG>)
                    FILTER(?type IN ("CJ"^^xsd:string, "TJ"^^xsd:string, "FJ"^^xsd:string)) .
                } %s
            ''' % (f'"{year}"^^xsd:gYear' if year else '?year', 
                   '' if preliminary_ruling else '#',
                   f'LIMIT {str(celex_limit)}' if celex_limit else ''))

        # Execute the SPARQL query and convert results to JSON
        ret = sparql.queryAndConvert()

        # Extract CELEX numbers from the results
        results = [result['celex']['value'] for result in ret['results']['bindings']]
        return results

    # Function to retrieve CELEX numbers for legislation
    @staticmethod
    def get_legislation_celex(year: Optional[str] = None,
                              celex_limit: Optional[int] = None) -> list:
        '''
        Retrieve CELEX numbers for legislation.
        Parameters:
            - year (Optional[str]): Filter legislation by the year of publication.
            - celex_limit (Optional[int]): Limit the number of CELEX numbers returned.
        Returns:
            - List of CELEX numbers for legislation.
        '''
        # Initialize SPARQL wrapper
        sparql = SPARQLWrapper('https://publications.europa.eu/webapi/rdf/sparql')
        sparql.setReturnFormat(JSON)  # Set output format to JSON

        # SPARQL query to retrieve CELEX numbers for legislation
        sparql.setQuery('''
                PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT DISTINCT ?celex
                WHERE {
                    ?doc cdm:resource_legal_id_celex ?celex ;
                         cdm:resource_legal_year %s ;
                         cdm:resource_legal_id_sector "3"^^xsd:string .
                } %s
            ''' % (f'"{year}"^^xsd:gYear' if year else '?year', 
                   f'LIMIT {str(celex_limit)}' if celex_limit else ''))

        # Execute the SPARQL query and convert results to JSON
        ret = sparql.queryAndConvert()

        # Extract CELEX numbers from the results
        results = [result['celex']['value'] for result in ret['results']['bindings']]
        return results

    # Function to retrieve citations for a given document
    @staticmethod
    def get_citations(source_celex: str, 
                      cites_depth: Optional[int] = 1, 
                      cited_depth: Optional[int] = 1) -> list:
        '''
        Retrieve citations for a document, both those it cites and those citing it.
        Parameters:
            - source_celex (str): CELEX number of the source document.
            - cites_depth (Optional[int]): Depth of outgoing citations to include.
            - cited_depth (Optional[int]): Depth of incoming citations to include.
        Returns:
            - List of CELEX numbers for documents citing or cited by the source document.

        Additional Information:
            - Any numbers higher than 1 denote that new source document citing a document of its own.

            - This specific implementation does not care about intermediate steps, it simply finds
              anything X or fewer hops away without linking those together.
        '''
        # Initialize SPARQL wrapper
        sparql = SPARQLWrapper('https://publications.europa.eu/webapi/rdf/sparql')
        sparql.setReturnFormat(JSON)  # Set output format to JSON

        # SPARQL query to retrieve citations
        sparql.setQuery('''
            PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
            PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

            SELECT DISTINCT * WHERE
            {
            {
                SELECT ?name2 WHERE {
                    ?doc cdm:resource_legal_id_celex "%s"^^xsd:string .
                    ?doc cdm:work_cites_work{1,%i} ?cited .
                    ?cited cdm:resource_legal_id_celex ?name2 .
                }
            } UNION {
                SELECT ?name2 WHERE {
                    ?doc cdm:resource_legal_id_celex "%s"^^xsd:string .
                    ?cited cdm:work_cites_work{1,%i} ?doc .
                    ?cited cdm:resource_legal_id_celex ?name2 .
                }
            }
            }''' % (source_celex, cites_depth, source_celex, cited_depth))

        # Execute the SPARQL query and convert results to JSON
        ret = sparql.queryAndConvert()

        # Extract unique CELEX numbers from the results
        targets = {bind['name2']['value'] for bind in ret['results']['bindings']}
        return list(targets)

    # Function to retrieve directory codes for a specific case law
    @staticmethod
    def get_directory_codes_for_judgement(source_celex: str) -> list:
        '''
        Retrieve directory codes for a specific case law.
        Parameters:
            - source_celex (str): CELEX number of the case law.
        Returns:
            - List of directory codes associated with the case law.
        '''
        # Initialize SPARQL wrapper
        sparql = SPARQLWrapper('https://publications.europa.eu/webapi/rdf/sparql')
        sparql.setReturnFormat(JSON)  # Set output format to JSON

        # SPARQL query to retrieve directory codes
        sparql.setQuery('''
                PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT DISTINCT ?code
                WHERE {
                    ?doc cdm:resource_legal_id_celex  "%s"^^xsd:string .
                    ?doc cdm:case-law_is_about_concept_new_case-law ?code .
                }
            ''' % (source_celex))

        # Execute the SPARQL query and convert results to JSON
        ret = sparql.queryAndConvert()

        # Extract directory codes from the results
        results = [result['code']['value'].split('/')[-1] for result in ret['results']['bindings']]
        return results

    # Function to retrieve subject matters for a specific case law
    @staticmethod
    def get_subject_matter(source_celex: str) -> list:
        '''
        Retrieve subject matters for a specific case law.
        Parameters:
            - source_celex (str): CELEX number of the case law.
        Returns:
            - List of subject matters associated with the case law.
        '''
        # Initialize SPARQL wrapper
        sparql = SPARQLWrapper('https://publications.europa.eu/webapi/rdf/sparql')
        sparql.setReturnFormat(JSON)  # Set output format to JSON

        # SPARQL query to retrieve subject matters
        sparql.setQuery('''
                PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT DISTINCT ?subject_matter
                WHERE {
                    ?doc cdm:resource_legal_id_celex  "%s"^^xsd:string .
                    ?doc cdm:resource_legal_is_about_subject-matter ?subject_matter .
                }
            ''' % (source_celex))

        # Execute the SPARQL query and convert results to JSON
        ret = sparql.queryAndConvert()

        # Extract subject matters from the results
        results = [result['subject_matter']['value'].split('/')[-1] for result in ret['results']['bindings']]
        return results
    
    # Function to retrieve title for a specific case law or legislation
    @staticmethod
    def get_title(source_celex: str) -> list:
        '''
        Retrieve subject matters for a specific case law.
        Parameters:
            - source_celex (str): CELEX number of the case law.
        Returns:
            - List of subject matters associated with the case law.
        '''
        # Initialize SPARQL wrapper
        sparql = SPARQLWrapper('https://publications.europa.eu/webapi/rdf/sparql')
        sparql.setReturnFormat(JSON)  # Set output format to JSON

        # SPARQL query to retrieve subject matters
        sparql.setQuery('''
                PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>

                SELECT DISTINCT ?title
                    WHERE {
                        ?doc cdm:expression_title ?title .
                        ?doc owl:sameAs ?w . 
                        FILTER (?w = <http://publications.europa.eu/resource/celex/%s.ENG>)
                }
            ''' % (quote_plus(source_celex)))

        # Execute the SPARQL query and convert results to JSON
        ret = sparql.queryAndConvert()

        # Extract subject matters from the results
        results = [result['title']['value'] for result in ret['results']['bindings']][0]
        return results

    # Function to fetch the full text of a document using its CELEX number
    @staticmethod
    def get_full_text(source_celex: str) -> bytes:
        '''
        Retrieve the full text of a legal document using its CELEX number.
        Parameters:
            - source_celex (str): CELEX number of the document.
        Returns:
            - The full content of the document as bytes.
        '''
        # If the CELEX number is a URL, extract the last part as the CELEX identifier
        if source_celex.startswith('http'):
            source_celex = source_celex.split('/')[-1]

        # HTTP headers to mimic a browser request
        headers = {
            'Accept': 'text/html,application/xhtml+xml,application/xml',
            'Accept-Language': 'en',
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        }

        # Parameters for the GET request, specifying the language
        params = {'language': 'en'}

        # Send a GET request to the specified URL to fetch the document content
        response = requests.get(f'http://publications.europa.eu/resource/celex/{source_celex}', 
                                headers=headers, params=params)

        # Return the raw content of the response
        return response.content

    # Function to retrieve subject matters for a specific document type
    @staticmethod
    def get_subject_matter_by_doc_type(doc_type: str) -> pd.DataFrame:
        '''
        Retrieve subject matters associated with a specific document type.
        Parameters:
            - doc_type (str): Type of document to filter by (e.g., "6" for case law).
        Returns:
            - A pandas DataFrame with source CELEX numbers and their associated subject matters.
        '''
        # Initialize SPARQL wrapper
        sparql = SPARQLWrapper('https://publications.europa.eu/webapi/rdf/sparql')
        sparql.setReturnFormat(JSON)  # Set output format to JSON

        # SPARQL query to retrieve subject matters for the given document type
        sparql.setQuery('''
                PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT ?source ?target
                WHERE {
                    ?doc cdm:resource_legal_id_celex ?source .
                    ?doc cdm:resource_legal_id_sector "%s"^^xsd:string .
                    ?doc cdm:resource_legal_is_about_subject-matter ?target .
                }
            ''' % doc_type)

        # Execute the SPARQL query and convert results to JSON
        ret = sparql.queryAndConvert()

        # Convert the results into a pandas DataFrame
        results = pd.DataFrame([{'source': row['source']['value'], 
                                 'target': row['target']['value'].split('/')[-1]} 
                                for row in ret['results']['bindings']])
        return results

    # Function to retrieve subject matters text from subject matter codes 
    @staticmethod
    def get_subject_matter_text(code: str) -> str:
        '''
        Retrieve subject matters text from subject matter codes.
        Parameters:
            - code (str): Subject matter code.
        Returns:
            - Textual description of subject matters.
        '''

        r = requests.get(f"https://publications.europa.eu/resource/authority/subject-matter/{code}")
        soup = BeautifulSoup(r.content, 'xml')
        text = soup.find('skos:prefLabel', {'xml:lang' : 'en'}).text

        return text

    # Function to retrieve directory codes for all judgments
    @staticmethod
    def get_directory_codes_for_all_judgements() -> pd.DataFrame:
        '''
        Retrieve directory codes associated with all judgments.
        Returns:
            - A pandas DataFrame with CELEX numbers (source) and their associated directory codes (target).
        '''
        # Initialize SPARQL wrapper
        sparql = SPARQLWrapper('https://publications.europa.eu/webapi/rdf/sparql')
        sparql.setReturnFormat(JSON)  # Set output format to JSON

        # SPARQL query to retrieve directory codes for all judgments
        sparql.setQuery('''
                PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT ?source ?target
                WHERE {
                    ?doc cdm:resource_legal_id_celex ?source .
                    ?doc cdm:resource_legal_id_sector "6"^^xsd:string .
                    ?doc cdm:case-law_is_about_concept_new_case-law ?target .
                }
            ''')

        # Execute the SPARQL query and convert results to JSON
        ret = sparql.queryAndConvert()

        # Convert the results into a pandas DataFrame
        results = pd.DataFrame([{'source': row['source']['value'], 
                                 'target': row['target']['value'].split('/')[-1]} 
                                for row in ret['results']['bindings']])
        return results

    # Function to retrieve citations for a specific document type
    @staticmethod
    def get_citation_by_doc_type(doc_type: str) -> pd.DataFrame:
        '''
        Retrieve citations for a specific document type.
        Parameters:
            - doc_type (str): Type of document to filter by (e.g., "6" for case law).
        Returns:
            - A pandas DataFrame with CELEX numbers for source and target citations.
        '''
        # Initialize SPARQL wrapper
        sparql = SPARQLWrapper('https://publications.europa.eu/webapi/rdf/sparql')
        sparql.setReturnFormat(JSON)  # Set output format to JSON

        # SPARQL query to retrieve citations for the specified document type
        sparql.setQuery('''
                PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT ?source ?target
                WHERE {
                    ?doc cdm:resource_legal_id_celex ?source .
                    ?doc cdm:resource_legal_id_sector "%s"^^xsd:string .
                    ?doc cdm:work_cites_work ?cited .
                    ?cited cdm:resource_legal_id_celex ?target .
                }
            ''' % doc_type)

        # Execute the SPARQL query and convert results to JSON
        ret = sparql.queryAndConvert()

        # Convert the results into a pandas DataFrame
        results = pd.DataFrame([{'source': row['source']['value'], 
                                 'target': row['target']['value']} 
                                for row in ret['results']['bindings']])
        return results
    
    @staticmethod
    def get_celex_by_doc_type(doc_type: str) -> list:
        '''
        Retrieve all celex numbers for a specific document type.
        Parameters:
            - doc_type (str): Type of document to filter by (e.g., "6" for case law).
        Returns:
            - A list of CELEX numbers for the entered document type.
        '''
        sparql = SPARQLWrapper('https://publications.europa.eu/webapi/rdf/sparql')
        sparql.setReturnFormat(JSON)
        sparql.setQuery('''
                PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
                PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

                SELECT DISTINCT ?celex  WHERE {
                    ?doc cdm:resource_legal_id_celex ?celex.
                    ?doc cdm:resource_legal_id_sector "%s"^^xsd:string .
                }
            '''% (doc_type))
        ret = sparql.queryAndConvert()

        results = [result['celex']['value'] for result in ret['results']['bindings']]

        return results

    # Function to build a network of directory codes
    @staticmethod
    def get_directory_codes_network(return_nodes: bool) -> Union[list, pd.DataFrame]:
        '''
        Build a network of directory codes, showing parent-child relationships.
        Parameters:
            - return_nodes (bool): Whether to return just the nodes or both nodes and edges.
        Returns:
            - pandas DataFrame of edges and a list of nodes (if return_nodes is True).
        '''

        # HTTP headers to mimic a browser request
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36',
        }

        # Fetch the directory codes from the specified URL
        r = requests.get("http://publications.europa.eu/resource/authority/fd_578", headers = headers)
        soup = BeautifulSoup(r.content, 'xml')  # Parse the response content as XML

        # Extract directory codes from the parsed XML
        direc_codes = []
        for code in soup.findAll('rdf:Description'):
            direc_codes.append(code['rdf:about'].split('/')[-1])

        # Filter the directory codes and remove duplicates
        codes_list = list(set([code for code in direc_codes if code.split('.')[0].isnumeric()]))

        # Build a set of node codes for quick lookup
        node_set = set(sorted(codes_list))

        # Initialize an empty list to store edges (parent-child relationships)
        edges = []

        # Iterate over each node code to determine parent-child relationships
        for code in sorted(codes_list):
            parts = code.split('.')  # Split the code into parts
            if len(parts) == 1:  # Root node (no parent)
                continue
            else:
                parent_code = '.'.join(parts[:-1])  # Parent code is derived by removing the last part
                if parent_code in node_set:
                    edges.append((parent_code, code))  # Add an edge (parent -> child)

        # Return the appropriate data based on the return_nodes parameter
        if return_nodes:
            return codes_list, pd.DataFrame(edges, columns=['source', 'target'])
        else:
            return pd.DataFrame(edges, columns=['source', 'target'])