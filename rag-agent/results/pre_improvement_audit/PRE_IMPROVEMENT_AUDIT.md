# 사전 개선 진단 (PRE_IMPROVEMENT_AUDIT)

모델/랭킹 미변경, 신규 API 호출 없음. 기존 retrieval 결과(results/evaluation_v2/s3c_v2_records.jsonl, results/bottleneck_root_cause/*)와 data/hitab 테이블 파일만 사용.

- 전체 query 1584건, top20_detail 행 31623건
- global: R@1=0.5752 R@5=0.7982 R@20=0.9142 MRR=0.6814 (n=991, unknown_rank=6)
- top1 성공 570건, wrong_table 102건, wrong_row 113건, wrong_column 143건
- gold_rank>20 79건, unknown 6건

## 생성 파일

- top20_detail.jsonl / top20_detail.csv
- error_examples.md
- oracle_diagnosis.json
- integrity_audit.md
