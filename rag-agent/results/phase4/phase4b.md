## Task A -- P4_path_cell vs P1_fixed_512, McNemar exact (paired)

| pool | n | P1 EM | P4 EM | b (P4만) | c (P1만) | delta | 95% CI | p |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| hitab_lookup | 60 | 0.4500 | 0.5833 | 17 | 9 | +0.1333 | [-0.0333, +0.3000] | 0.1686 |
| hitab_arith | 60 | 0.0167 | 0.0167 | 1 | 1 | +0.0000 | [-0.0500, +0.0500] | 1.0000 |
| aitqa | 60 | 0.3333 | 0.3000 | 8 | 10 | -0.0333 | [-0.1667, +0.1000] | 0.8145 |
| rhb_fact | 60 | 0.3167 | 0.3333 | 10 | 9 | +0.0167 | [-0.1333, +0.1667] | 1.0000 |
| rhb_num | 54 | 0.1296 | 0.1296 | 3 | 3 | +0.0000 | [-0.0926, +0.0926] | 1.0000 |

p는 다중비교 보정 전 값이다. 보정하지 않았다.


## Task B -- gold_in_topk=False 인데 정답인 행 전건

n = 12

- [P1_fixed_512 / rhb_num] query_id=1461
  query: How many types of Effectiveness of Health Facility Mitigation Measures are listed?
  gold_answer='3'  pred_parsed='3'
  gold_table_id=society-table45  gold_cell=[["society-table45", 8, 0]]
  같은 값 셀 컨텍스트 내 존재: Y  (n=8)
  위치: [('society-table22', 0, 1), ('society-table22', 0, 2), ('society-table22', 10, 3), ('society-table22', 10, 4), ('society-table37', 0, 1), ('society-table37', 0, 2), ('society-table37', 10, 3), ('society-table37', 10, 4)]

- [P1_fixed_512 / rhb_num] query_id=1378
  query: How many different types of effectiveness measures for health facility mitigation measures are there in the table?
  gold_answer='3'  pred_parsed='3'
  gold_table_id=society-table25  gold_cell=[["society-table25", 9, 0]]
  같은 값 셀 컨텍스트 내 존재: Y  (n=8)
  위치: [('society-table37', 0, 1), ('society-table37', 0, 2), ('society-table37', 10, 3), ('society-table37', 10, 4), ('society-table22', 0, 1), ('society-table22', 0, 2), ('society-table22', 10, 3), ('society-table22', 10, 4)]

- [P1_fixed_512 / rhb_num] query_id=1462
  query: How many types of Infection Fatality Rates (IFRs) are listed?
  gold_answer='3'  pred_parsed='3'
  gold_table_id=society-table45  gold_cell=[["society-table45", 8, 0]]
  같은 값 셀 컨텍스트 내 존재: N  (n=0)

- [P4_path_cell / rhb_num] query_id=1586
  query: How many countries in Asia experienced a decrease in the number of deaths due to musculoskeletal disorders from 1986 to 2011?
  gold_answer='1'  pred_parsed='1'
  gold_table_id=society-table55  gold_cell=[["society-table55", 67, 6]]
  같은 값 셀 컨텍스트 내 존재: N  (n=0)

- [P4_path_cell / rhb_num] query_id=1461
  query: How many types of Effectiveness of Health Facility Mitigation Measures are listed?
  gold_answer='3'  pred_parsed='3'
  gold_table_id=society-table45  gold_cell=[["society-table45", 8, 0]]
  같은 값 셀 컨텍스트 내 존재: N  (n=0)

- [P4_path_cell / rhb_num] query_id=202
  query: What is the combined time spent by mothers and fathers with 'Own household children under 18' on 'Caring for and helping household children' activities' subcategories when both spouses work full-time? Round the answer to two decimal places.
  gold_answer='2.23'  pred_parsed='2.23'
  gold_table_id=activitytime-table05  gold_cell=[["activitytime-table05", 78, 1]]
  같은 값 셀 컨텍스트 내 존재: N  (n=0)

- [P1_fixed_512 / aitqa] query_id=q-473
  query: How much money did Alaska airlines spend on medical benefits of its employees in 2017 ?
  gold_answer='216'  pred_parsed='216'
  gold_table_id=tab-105  gold_cell=[["tab-105", 1, 0]]
  같은 값 셀 컨텍스트 내 존재: Y  (n=1)
  위치: [('tab-53', 3, 1)]

- [P1_fixed_512 / aitqa] query_id=q-289
  query: Find consolidated CASM for United Airlines in the year of 2014
  gold_answer='14.85'  pred_parsed='14.85'
  gold_table_id=tab-61  gold_cell=[["tab-61", 20, 3]]
  같은 값 셀 컨텍스트 내 존재: Y  (n=4)
  위치: [('tab-1', 8, 4), ('tab-84', 11, 0), ('tab-65', 7, 1), ('tab-82', 7, 0)]

- [P4_path_cell / aitqa] query_id=q-207
  query: In 2017 what was the fuel gallon consumption of American Airlines ?
  gold_answer='4,352'  pred_parsed='4,352'
  gold_table_id=tab-39  gold_cell=[["tab-39", 0, 1]]
  같은 값 셀 컨텍스트 내 존재: N  (n=0)

- [P1_fixed_512 / rhb_fact] query_id=1565
  query: What is the total number of cells for Mouse 1 Luminal at 14 days?
  gold_answer='115'  pred_parsed='115'
  gold_table_id=science-table20  gold_cell=[["science-table20", 0, 0]]
  같은 값 셀 컨텍스트 내 존재: Y  (n=2)
  위치: [('science-table41', 0, 0), ('science-table35', 0, 0)]

- [P4_path_cell / rhb_fact] query_id=59
  query: What is the difference between the number of unemployed individuals looking for full-time work and those looking for part-time work for the 'Total, 16 years and over' category?
  gold_answer='3810'  pred_parsed='3810'
  gold_table_id=employment-table08  gold_cell=[["employment-table08", 30, 2]]
  같은 값 셀 컨텍스트 내 존재: N  (n=0)

- [P4_path_cell / rhb_fact] query_id=2191
  query: What was the employment rate for Buddhists in 2018? (Final answer should be an integer.)
  gold_answer='69'  pred_parsed='69'
  gold_table_id=religion-table01  gold_cell=[["religion-table01", 8, 4]]
  같은 값 셀 컨텍스트 내 존재: N  (n=0)


## Task C -- hitab_arith / gold_cell 오답 전건

n = 24 / 31

| query_id | gold | pred_answer_raw | 주입 셀 값 | 분류 |
|---|---|---|---|---|
| eb98069c | [5.389691] | '5.39' | ['485', '2614'] | (ii) 계산했으나 틀림 |
| 9825bef7 | [0.44, 1.2] | '0.44 to 1.2' | ['0.44', '0.63', '0.76', '0.91', '1.2'] | (iii) 형식 이상 |
| 277d629d | [0.222462] | '22.3%' | ['18855593', '4194659'] | (ii) 계산했으나 틀림 |
| 0b50ea01 | [44] | '-44%' | ['-44'] | (i) 셀 값 그대로 |
| 48c1fb4a | [0.070572] | '78.1%' | ['355821.0', '11509.0', '13602.0'] | (ii) 계산했으나 틀림 |
| 3c224552 | [-0.072612] | '8.0%' | ['29513.0', '27370.0'] | (ii) 계산했으나 틀림 |
| 3a7bfc48 | [2.457831] | '2.45' | ['20.4', '8.3'] | (ii) 계산했으나 틀림 |
| 27fda0d4 | [61] | '3.44' | ['86.0', '25.0'] | (ii) 계산했으나 틀림 |
| f03b343f | [0.331364] | '1.04%' | ['106.0', '3260.0', '10158.0'] | (ii) 계산했으나 틀림 |
| 2518e450 | [96.5, 88.8] | '88.8% to 96.5%' | ['96.5', '91.2', '88.8', '90.4', '91.0', '91.4', '96.0'] | (iii) 형식 이상 |
| a3c2b6d4 | [0.089947] | '9.0%' | ['595.0', '6615.0'] | (ii) 계산했으나 틀림 |
| e29d9d1a | [0.4] | '-0.4' | ['-0.4'] | (i) 셀 값 그대로 |
| 7cda8ec1 | [25.908333] | '26.3%' | ['24.2', '24.6', '25.0', '25.2', '25.4', '25.5', '25.4', '25.5', '26.0', '27.4', '28.1', '28.6'] | (ii) 계산했으나 틀림 |
| 82c81421 | [21] | '0.0' | ['21.0', '21.0'] | (ii) 계산했으나 틀림 |
| 67c2942e | [4.583333] | '4.63' | ['7.2', '33.0'] | (ii) 계산했으나 틀림 |
| f42a76ef | [0.054256] | '5.43%' | ['3260.0', '60085.0'] | (ii) 계산했으나 틀림 |
| c45a627a | [0.834291] | '25.7%' | ['964865', '199523', '269493', '212223', '72374', '51365'] | (ii) 계산했으나 틀림 |
| e8bc7f60 | [71.8] | '28.2' | ['28.2'] | (i) 셀 값 그대로 |
| 7b9e03f8 | [0.060265] | '6.03%' | ['3621.0', '60085.0'] | (ii) 계산했으나 틀림 |
| a080b456 | [0.681511] | '4.3%' | ['238768', '64104', '13659', '12920', '12712', '12156', '11977', '10764', '9141', '9018', '6272'] | (ii) 계산했으나 틀림 |
| d20de665 | [0.173998] | '17.3%' | ['1151.0', '6615.0'] | (ii) 계산했으나 틀림 |
| 647b2025 | [0.161149] | '16.1%' | ['1066.0', '6615.0'] | (ii) 계산했으나 틀림 |
| 6f9de467 | [3.142857] | '3.2' | ['10.5', '33.0'] | (ii) 계산했으나 틀림 |
| 4e07d458 | [96.95] | '2.6' | ['3.1', '3.2', '3.1', '3.3', '3.5', '3.2', '3.6', '3.6', '3.5', '2.9', '2.3', '1.3'] | (ii) 계산했으나 틀림 |

- (i) 셀 값 그대로: 3
- (ii) 계산했으나 틀림: 19
- (iii) 형식 이상: 2
