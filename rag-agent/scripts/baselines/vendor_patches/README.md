# vendor/ 복원 기록 (2026-10-23)

`vendor/` 는 .gitignore 라 집 PC 에만 있었고 연구실 PC 에는 없었다. 결과 README 에 적힌 커밋으로 다시 받았다.

| 디렉터리 | 원본 | 커밋 | 가져온 방법 | 패치 |
|---|---|---|---|---|
| `vendor/raptor` | github.com/parthsarthi03/raptor | `7da1d48a7e1d7dec61a63c9d9aae84e2dfaa5767` | 커밋 tar.gz | 0건 |
| `vendor/tablerag_yu` | github.com/yxh-y/TableRAG | `f8d798fc60fb44736d7285ca3ca1dbcb7b3e984d` | git clone + checkout | 0건(MySQL→SQLite 문구는 `scripts/baselines/tablerag_yu.py` 가 바꿈) |
| `vendor/table_rag` | github.com/google-research/google-research `table_rag/` | `08a8d6736475776f42ffac23b2c13111a28e5795` | sparse checkout | **3개 파일**: 이 디렉터리의 `table_rag/agent/` 사본 참조 |

`table_rag/agent/local_backend.py` 는 우리가 쓴 파일(집 PC 원본 유실 → 2026-10-23 README "원설정과 다른 것" 1·2·6 행과 `tablerag_chen.py` 호출면대로 재작성):
LocalQwen(OpenAI 클라이언트 모양, 정지열 적용), HuggingFaceEmbeddings(sentence-transformers), FAISS/Document/VectorRetriever(IndexFlatL2, vecs.npy+docs.json 캐시).
`model.py`: `local:` 공급자 분기·외부 API 모듈 지연 import. `retriever.py`: langchain 클래스 → local_backend 대체.
`rag_agent/llm/local_qwen.py` `complete(..., stop=None)` 추가(기본 None 이면 기존과 같음).

새로 설치: tiktoken 0.14.0, umap-learn 0.5.12(README 기록 그대로), openai 3.36.1·tenacity(raptor 패키지 import 에 필요).
복원: `cp -r scripts/baselines/vendor_patches/table_rag/agent/* vendor/table_rag/agent/` (원본 sparse checkout 뒤).
