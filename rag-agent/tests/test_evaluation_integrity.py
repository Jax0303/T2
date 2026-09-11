import hashlib
import json
from copy import deepcopy
import pytest
from analysis.population_accounting import account
from analysis.result_sources import validate_pair
from analysis import result_sources as sources

def fixture_rows():
    # Data questions have a very different conditional accuracy from headers.
    records=[]; answers=[]; gold=[]
    for q,mode,hit,a,g in [("a","all",1,1,1),("b","all",1,0,0),
                           ("c","all",0,0,0),("d","all",0,1,1),
                           ("h","any",1,1,1),("i","any",1,1,1)]:
        records.append({"query_id":q,"mode":mode,"correct":hit,"answer":[q]})
        answers.append({"query_id":q,"answer_correct":a,"retrieval_correct":hit,"answer":[q]})
        gold.append({"query_id":q,"answer_correct":g,"answer":[q]})
    return records,answers,gold

def test_population_denominator_and_bucket_overlap():
    records,answers,gold=fixture_rows()
    s=account(records,answers,gold,lambda r:r["mode"]=="all")
    assert (s["n"],s["n_hit"],s["n_miss"])==(4,2,2)
    assert s["conditional_hit_em"]==0.5
    assert s["conditional_miss_em"]==0.5
    assert s["buckets"]["reader_ceiling"]==1  # excludes the retrieval miss
    assert s["both_wrong"]==2

def test_paired_gap_allows_gold_to_hurt():
    records,answers,gold=fixture_rows()
    gold[0]["answer_correct"]=0
    s=account(records,answers,gold,lambda r:r["mode"]=="all")
    assert s["hit_paired_gap"]==-0.25
    assert s["gold_em"]-s["answer_em"]==s["hit_paired_gap"]+s["miss_paired_gap"]

def test_population_mismatch_is_not_silently_intersected():
    records,answers,gold=fixture_rows()
    with pytest.raises(ValueError,match="populations differ"):
        account(records,answers[:-1],gold,lambda r:True)

def pair():
    h=hashlib.md5(b"visible value 42").hexdigest()[:12]
    r={"query_id":"q","correct":1,"mode":"all","context":["visible value 42"],"context_sha":h,"answer":[42]}
    a={"query_id":"q","retrieval_correct":1,"mode":"all","context_sha":h,"answer":[42]}
    return r,a

def test_correct_score_cannot_mask_wrong_reader_context():
    r,a=pair();a["context_sha"]="wrong"
    with pytest.raises(ValueError,match="context mismatch"):
        validate_pair([r],[a])

def test_missing_or_duplicate_answer_is_rejected():
    r,a=pair()
    with pytest.raises(ValueError,match="populations differ"):
        validate_pair([r],[])
    with pytest.raises(ValueError,match="duplicate"):
        validate_pair([r],[a,a])

def test_retrieval_metadata_must_match():
    r,a=pair();a["retrieval_correct"]=0
    with pytest.raises(ValueError,match="metadata mismatch"):
        validate_pair([r],[a])

def test_incomplete_corrected_leg_never_falls_back(tmp_path,monkeypatch):
    old=tmp_path/"old";new=tmp_path/"new"
    old.mkdir();new.mkdir()
    monkeypatch.setattr(sources,"ORIGINAL",old)
    monkeypatch.setattr(sources,"ANSWERS",new)
    tag="t_rowcol_values"
    _,a=pair()
    for folder in (old,new):
        (folder/(tag+"_answer_retrieved.jsonl")).write_text(json.dumps(a)+"\n")
    assert sources.current_answer_rows(tag)==[]

def test_committed_primary_and_data_accounting():
    root=sources.ROOT
    read=sources.read_jsonl
    records=read(root/"results/retrieval_accuracy/t_s3c_hybrid_records.jsonl")
    answers=read(root/"results/retrieval_accuracy/t_s3c_hybrid_answer_retrieved.jsonl")
    gold=read(root/"results/retrieval_accuracy/t_s3c_hybrid_answer_gold_v2.jsonl")
    p=account(records,answers,gold,lambda r:r["mode"]=="all" and r["m"]==1 and (r.get("aggregation") or "none")=="none")
    assert (p["n"],p["retrieved_correct"],p["gold_correct"])==(991,710,948)
    assert p["buckets"]=={"both_correct":692,"gold_only_fail":18,"retrieval_miss":81,"reader_ceiling":19,"distractor":181}
    d=account(records,answers,gold,lambda r:r["mode"]=="all")
    assert (d["n_hit"],d["hit_answer_correct"],d["n_miss"],d["miss_answer_correct"])==(1108,748,137,7)
