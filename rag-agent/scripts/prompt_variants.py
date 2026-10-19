#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""답변 지시문 변형 목록 — PREREG-2026-10-15-prompt-sweep.md (결과 전에 고정).

지시문 = 축마다 하나씩 고른 문장을 정해진 순서로 이어 붙인 사용자 메시지 하나(system 메시지 없음, 튜토리얼과 같음).
모든 축을 0 으로 두면 LangChain rlm/rag-prompt 원문과 글자 단위로 같다(`assert` 로 확인).

축(순서대로):
  R 역할        R0 원문 / R1 재무 분석가
  X 입력 설명   X0 없음 / X1 "표 셀, 값의 뜻은 행·열 머리글" 한 문장 / X2 "표 또는 표의 일부" (2026-10-19 추가) / X3 X1 + " > " 경로 표기 설명 (2026-10-19 추가)
  F 거절        F0 원문("모르면 모른다") / F1 "항상 답하라"
  B 풀이        B0 원문("세 문장 이내") / B1 단계별(셀 찾기→계산→답) / B2 계산식을 먼저 쓰기
  D 규칙        D0 없음 / D1 증가율·평균·개수·부호·머리글 일치·이름 답 규칙 6개 / D2 D1 + (7) 여러 표면 머리글 맞는 표만, 표끼리 합산 금지 (2026-10-19 추가)
  C 답 형식     C0 없음 / C1 'Final answer:' 값만 / C2 'Final answer:' 유형별(숫자·머리글 글·Yes/No)
  E 예시        E0 없음 / E1 풀이 예시 7개(train 에서 882 밖 문항, dev 에서도 뺌)
변형 이름 = "R0X1F1B2D1C2E1" 처럼 축 글자+수준.
"""
from __future__ import annotations

import itertools
import re

AXES = {
    "R": ["You are an assistant for question-answering tasks.",
          "You are a financial analyst answering questions about tables from a company's annual report."],
    "X": ["",
          " The context contains cells from the report's tables; the meaning of each value is given by its row and column headers.",
          " The context contains tables or parts of tables from the report; the meaning of each value is given by its row and column headers.",   # X2 (2026-10-19, PREREG-2026-10-19-prompt-table-rule.md)
          " The context contains cells from the report's tables; the meaning of each value is given by its row and column headers. "
          "Header paths are written from the top-level header down to the sub-header, separated by ' > ' (for example 'Revenues > 2019' means the 2019 column under Revenues)."],   # X3 (2026-10-19, PREREG-2026-10-19-sentence-form.md: 꺽쇠 설명)
    "F": ["If you don't know the answer, just say that you don't know.",
          "The context contains what is needed to answer; do not say that you don't know, always give your best answer."],
    "B": ["Use three sentences maximum and keep the answer concise.",
          "Think step by step: first identify the cells needed and state their values, then carry out any calculation, then give the answer.",
          "Before answering, write out the calculation as an explicit arithmetic expression using the values from the context "
          "(for example (2,253 - 2,057) / 2,057 = 0.0953) and then evaluate it. If no calculation is needed, state the value directly."],
    "D": ["",
          " Rules: (1) Percentage change or growth rate = (new - old) / old. (2) Average = sum / count; total or sum = add all the items asked. "
          "(3) 'How many' asks for a count of the items meeting the condition. (4) Use values exactly as shown, keeping their sign "
          "(values in parentheses are negative); do not convert units. (5) Choose the cell whose row and column headers both match the question. "
          "(6) For 'which' or 'what' questions asking for a name, answer with the exact text from the headers.",
          " Rules: (1) Percentage change or growth rate = (new - old) / old. (2) Average = sum / count; total or sum = add all the items asked. "
          "(3) 'How many' asks for a count of the items meeting the condition. (4) Use values exactly as shown, keeping their sign "
          "(values in parentheses are negative); do not convert units. (5) Choose the cell whose row and column headers both match the question. "
          "(6) For 'which' or 'what' questions asking for a name, answer with the exact text from the headers. "
          "(7) If the context contains several tables, use the table whose row and column headers match the question; "
          "do not add up values from different tables unless the question asks for it."],   # D2 = D1 + (7) (2026-10-19)
    "C": ["",
          " On the last line write 'Final answer: ' followed by only the final value, with no units or explanation.",
          " On the last line write 'Final answer: ' followed by the answer only: a number for numeric questions (no units or words), "
          "the exact header text for questions asking which item or what name, or Yes or No for yes-or-no questions."],
    "E": ["", "examples"],
}
ORDER = "RXFBDCE"
CONTEXT_SENTENCE = " Use the following pieces of retrieved context to answer the question."
ORIGINAL = ("You are an assistant for question-answering tasks. Use the following pieces of retrieved "
            "context to answer the question. If you don't know the answer, just say that you don't know. "
            "Use three sentences maximum and keep the answer concise.")          # langchain_chunk.RAG_PROMPT 원문
TEMPLATE = "{instr}\n{examples}Question: {question} \nContext: {context} \nAnswer:"

# 풀이 예시 7개 — MultiHiertt train 중 882 평가 문항 밖(dev 에서도 뺀다, results/prompt_sweep_20261015/fewshot_ids.json).
# (질문, 풀이, 최종값). 풀이는 사람이 썼고 값은 정답 셀·공식 정답에서 옮겼다.
EXAMPLES = [
    ("a1133e58b65041d88b0c203a4e3770eb", "Which year the Average commission per trade is the highest?",
     "The Average commission per trade is $10.98 in 2008, $11.33 in 2009, $11.21 in 2010, $11.01 in 2011 and $11.01 in 2012. "
     "The highest is $11.33, in 2009.", "2009"),
    ("c8924496a03242e99b8d6d805f3862d6", "How many Provision for loan and lease losses exceed the average of Provision for loan and lease losses in 2011?",
     "The 2011 values of Provision for loan and lease losses are 10,300, 4,025, -696 and 13,629. "
     "Their average is (10,300 + 4,025 - 696 + 13,629) / 4 = 6,814.5. Two of them, 10,300 and 13,629, exceed the average.", "2"),
    ("360606a963af44f5a69b2f36402c8e71", "Does the value of Residential in 2004 greater than that in 2003?",
     "No. Residential is 9,487 in 2004 and 10,810 in 2003, so the 2004 value is smaller.", "No"),
    ("8e8526fc0ee84df68982f93f0eed1318", "In which years is cash greater than operating income?",
     "Cash is $103.4 in 2008 and $357.4 in 2009; operating income is $7.3 in 2008 and $50.8 in 2009. Cash is greater in both years.",
     "2008 and 2009"),
    ("6a8f67b99b68445ebb0247402f2791b3", "What is the growth rate of Total operating revenues between 2007 Ended December 31 and 2008 Ended December 31?",
     "Total operating revenues are 1,535 in 2007 and 1,019 in 2008. Growth rate = (1,019 - 1,535) / 1,535 = -0.3362, that is -33.62%.",
     "-0.3362"),
    ("1567e3f8abd549e0af7e0dff9f40b92d", "What will Total revenue be like in 2019 ended December 31 if it develops with the same increasing rate as in 2018 ended December 31? (in million)",
     "Total revenue is 16,329 in 2017 and 17,132 in 2018. The 2018 increasing rate is (17,132 - 16,329) / 16,329 = 0.04918. "
     "Applying the same rate to 2018: 17,132 x (1 + 0.04918) = 17,974.49.", "17974.49"),
    ("b79881fd342f41bfa42595d7aee14a6e", "what's the total amount of Cost of Product Sales of Years Ended Change Amount, and Interest rate contracts of 2013 ?",
     "Cost of Product Sales (Change, Amount) is $267,612 and Interest rate contracts in 2013 is 2,400. Total = 267,612 + 2,400 = 270,012.",
     "270012"),
]


def parse(name: str) -> dict:
    m = re.fullmatch(r"R(\d)X(\d)F(\d)B(\d)D(\d)C(\d)E(\d)", name)
    if not m:
        raise ValueError(f"변형 이름이 아님: {name}")
    lv = dict(zip(ORDER, map(int, m.groups())))
    for k, v in lv.items():
        if v >= len(AXES[k]):
            raise ValueError(f"{name}: 축 {k} 수준 {v} 없음")
    return lv


def all_variants(axes: str = ORDER) -> list:
    """`axes` 에 든 축만 모든 수준을 돌고, 나머지 축은 0."""
    ranges = [range(len(AXES[k])) if k in axes else [0] for k in ORDER]
    return ["".join(f"{k}{v}" for k, v in zip(ORDER, combo)) for combo in itertools.product(*ranges)]


def instruction(name: str) -> str:
    lv = parse(name)
    s = AXES["R"][lv["R"]] + CONTEXT_SENTENCE + AXES["X"][lv["X"]] + " " + AXES["F"][lv["F"]] + " " + AXES["B"][lv["B"]]
    s += AXES["D"][lv["D"]] + AXES["C"][lv["C"]]
    return s


def examples_block(name: str) -> str:
    lv = parse(name)
    if lv["E"] == 0:
        return ""
    final = lv["C"] != 0
    parts = []
    for _uid, q, sol, ans in EXAMPLES:
        parts.append(f"Question: {q} \nAnswer: {sol}" + (f"\nFinal answer: {ans}" if final else ""))
    return "Examples:\n\n" + "\n\n".join(parts) + "\n\nNow answer the following question.\n"


def build(name: str, question: str, context: str) -> str:
    return TEMPLATE.format(instr=instruction(name), examples=examples_block(name), question=question, context=context)


def prompt_text(name: str) -> str:
    """결과 요약에 적는 틀(질문·검색 결과 자리는 중괄호)."""
    return TEMPLATE.format(instr=instruction(name), examples=examples_block(name), question="{question}", context="{context}")


assert instruction("R0X0F0B0D0C0E0") == ORIGINAL, "모든 축 0 은 원문과 같아야 한다"
assert build("R0X0F0B0D0C0E0", "Q", "C") == f"{ORIGINAL}\nQuestion: Q \nContext: C \nAnswer:"

if __name__ == "__main__":
    import sys
    for n in (sys.argv[1:] or ["R0X0F0B0D0C0E0", "R1X1F1B2D1C2E1"]):
        print(f"===== {n}\n{prompt_text(n)}\n")
    print(len(all_variants()), "variants in full factorial;", len(all_variants("XFBDCE")), "without R")
