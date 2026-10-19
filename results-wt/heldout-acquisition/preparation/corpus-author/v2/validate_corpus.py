"""Offline receipts and shape checks. Incomplete requirements stay false."""
import collections,json,pathlib,re,sys,unittest,unicodedata
from source_tools import ROOT,PRIVATE,save,sha
HELD=ROOT.parents[2];INV=HELD/'preparation/inventory/v2';sys.path[:0]=[str(INV),str(HELD)]
from harness import queries,identities
from registry_tools import read_registry

def shape(row):
 assert set(row)=={'id','concept','lenses'}
 assert re.fullmatch('h[0-9]{2}',row['id'])
 qs=queries({k:row[k] for k in ['concept','lenses']})
 normalized=[' '.join(re.findall('[a-z0-9]+',unicodedata.normalize('NFKC',q).casefold())) for _,q in qs]
 assert len(set(normalized))==12
 return normalized

def validate():
 md=json.loads((ROOT/'BLINDED-METADATA.json').read_text());s=json.loads((PRIVATE/'sources.json').read_text())['sources'];disp=json.loads((PRIVATE/'candidate-dispositions.json').read_text());pool,reg=read_registry(INV)
 assert len(disp)==24 and [x['selection_index'] for x in disp]==list(range(1,25))
 assert len(md)==len(s)==11
 allq=[]
 for row in md:allq+=shape(row)
 assert len(allq)==len(set(allq))==132
 checks=[]
 for row,src in zip(md,s):
  assert row['id']==src['card_id'];assert not (identities({'title':src['candidate_identity']['title'],**({'arxiv_id':src['candidate_identity']['arxiv']} if src['candidate_identity']['arxiv'] else {}),**({'doi':src['candidate_identity']['doi']} if src['candidate_identity']['doi'] else {})})&set(pool.owners))
  raw=(ROOT/src['raw_body_path']).read_bytes();text=(ROOT/src['text_extraction_path']).read_bytes();assert sha(raw)==src['raw_body_sha256'];assert sha(text)==src['text_extraction_sha256'];assert raw.startswith(b'%PDF')
  t=text.decode();a,b=src['excerpt_span_codepoints'];assert t[a:b]==src['short_literal_excerpt'];assert src['short_literal_excerpt'] in (PRIVATE/f"candidate-{src['selection_index']:02}"/'inspected.txt').read_text()
  assert src['inspection']['selected_words']<=1200
  public=json.dumps(row).casefold();assert src['candidate_identity']['title'].casefold() not in public
  for author in src['candidate_identity']['authors']:assert author.casefold() not in public
  assert not re.search(r'(?:arxiv:|doi:|10\.\d{4,9}/|https?://)',public)
  checks.append({'id':row['id'],'body_hash':True,'text_hash':True,'literal_span':True,'selected_span_words':src['inspection']['selected_words'],'known_public_alias_disjoint':True,'query_shape':True})
 for p,h in json.loads((ROOT/'INPUT-PINS.json').read_text()).items():assert sha(pathlib.Path(p).read_bytes())==h
 # All returned lookup/error bodies are retained too; transport exceptions have no response bytes.
 access=[]
 for p in sorted(PRIVATE.glob('candidate-*/access.json')):
  rs=json.loads(p.read_text());counts=collections.Counter(r['kind'] for r in rs)
  for r in rs:
   if 'raw_path' in r:assert sha((ROOT/r['raw_path']).read_bytes())==r['raw_sha256']
  access.append({'candidate':int(p.parent.name[-2:]),'lookup_attempts':counts['lookup'],'body_attempts':counts['body'],'strict_attempt_limits_pass':counts['lookup']<=2 and counts['body']<=1})
 counts=dict(collections.Counter(x['family'] for x in s));result={'status':'INCOMPLETE_NOT_READY','source_grounded_provisional_units':len(s),'expected_units':12,'metadata_structure_and_queries_pass':True,'queries_per_unit':12,'unique_normalized_queries_total':len(allq),'family_counts':counts,'four_families_three_each_pass':all(v==3 for v in counts.values()) and len(counts)==4,'expected_twelve_candidate_sources_pass':len(s)==12,'candidate_source_records_are_labels':False,'independent_reference_vault_present':False,'source_authentication_checks':checks,'all_24_dispositions_present':True,'strict_request_attempt_limits_pass':all(x['strict_attempt_limits_pass'] for x in access),'access_counts':access,'bounded_saved_inspection_spans_pass':True,'complete_historical_inspection_accounting_verified':False,'full_known_inventory_complete':reg['full_known_repo_scope_complete'],'future_acquisition_ready':False,'study_acquisition_run':False,'policy_unchanged':True,'no_global_independence_claim':True}
 save(ROOT/'VALIDATION.json',result);return result

class Synthetic(unittest.TestCase):
 def fixture(self):return {'id':'h99','concept':'synthetic scalar state','lenses':[{'name':str(i),'phrases':[f'phrase {i} {j}' for j in range(3)]} for i in range(4)]}
 def test_shape(self):self.assertEqual(len(shape(self.fixture())),12)
 def test_extra_identity_rejected(self):
  r=self.fixture();r['title']='invented target'
  with self.assertRaises(AssertionError):shape(r)
 def test_duplicate_normalized_rejected(self):
  r=self.fixture();r['lenses'][0]['phrases']=['x y','x-y','z']
  with self.assertRaises(AssertionError):shape(r)
 def test_missing_phrase_rejected(self):
  r=self.fixture();r['lenses'][0]['phrases'].pop()
  with self.assertRaises(ValueError):shape(r)
 def test_empty_concept_rejected(self):
  r=self.fixture();r['concept']=' '
  with self.assertRaises(ValueError):shape(r)
if __name__=='__main__':
 suite=unittest.defaultTestLoader.loadTestsFromTestCase(Synthetic);res=unittest.TextTestRunner(verbosity=2).run(suite);assert res.wasSuccessful();r=validate();print(json.dumps({k:v for k,v in r.items() if k not in ['source_authentication_checks','access_counts']},indent=2))
