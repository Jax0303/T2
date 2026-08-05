# 셀 문장 템플릿 차이 (측정만, 통일 안 함)

표 20개 / 셀 683개, HiTab dev. 생성: `scripts/diag_template_divergence.py`.

구현: (a) `rag_agent/serialization/caption.py` (배포 설정 = S3/long/cell), (b) `rag_agent/serialize/verbalize.py`, (c) `scripts/operand_collision_multihiertt.py:174-198` 내부 사본.

## 구현별 길이

| 구현 | 평균 문자수 | 평균 토큰수 |
|---|---|---|
| a_caption_long | 216.2 | 33.3 |
| a_caption_medium | 72.1 | 11.0 |
| b_verbalize_long | 185.9 | 28.3 |
| c_script_S3 | 72.1 | 11.0 |

## 쌍별 차이

| 쌍 | 완전일치 | 토큰 자카드 중앙값 |
|---|---|---|
| a_caption_long vs b_verbalize_long | 0.000 | 0.815 |
| a_caption_long vs c_script_S3 | 0.000 | 0.333 |
| b_verbalize_long vs c_script_S3 | 0.000 | 0.421 |
| a_caption_medium vs c_script_S3 | 1.000 | 1.000 |

## 값 문자열

- `serialize/verbalize._fmt` vs `serialization/base.fmt_value` 불일치: 147/683 (0.215) — 정수형 float를 `21` / `21.0`로 서로 다르게 렌더한다. 토큰 단위 차이라 BM25 매칭에 영향.

## 차이 예시 10건 (불일치 683/683건 중 균등 추출)

**0_1_nsf21326-tab001 r0c0**

- (a) `In the table 'table 1 federal r&d spending by mandatory and discretionary budget authority, by agency: fys 2017-19', among department of defensea, the value of 2017 actual is 49197.`
- (b) `In table 1 federal r&d spending by mandatory and discretionary budget authority, by agency: fys 2017-19, 2017 actual for department of defensea is 49197.`
- (c) `For department of defensea, 2017 actual is 49197.`

**1008 r3c4**

- (a) `In the table 'percentage distribution of black male workers and other male workers aged 25 to 59, by industry sector and generation, canada, 2016', among percent > construction, the value of other male workers > immigrant is 10.2.`
- (b) `In percentage distribution of black male workers and other male workers aged 25 to 59, by industry sector and generation, canada, 2016, other male workers, immigrant for percent, construction is 10.2.`
- (c) `For percent > construction, other male workers > immigrant is 10.2.`

**1017 r2c0**

- (a) `In the table 'hirings and departures of police officers, by province and territory, canada, 2017/2018', among nova scotia, the value of hirings > total > number is 80.`
- (b) `In hirings and departures of police officers, by province and territory, canada, 2017/2018, hirings, total, number for nova scotia is 80.`
- (c) `For nova scotia, hirings > total > number is 80.`

**1029 r0c4**

- (a) `In the table 'average counts of adults under correctional supervision, by type of supervision and jurisdiction, 2013/2014', among newfoundland and labrador, the value of remand > rate is 21.0.`
- (b) `In average counts of adults under correctional supervision, by type of supervision and jurisdiction, 2013/2014, remand, rate for newfoundland and labrador is 21.`
- (c) `For newfoundland and labrador, remand > rate is 21.0.`

**102_totto12065-2 r1c14**

- (a) `In the table 'return', among 2011 > min, the value of punt return > +20 is 3.0.`
- (b) `In return, punt return, +20 for 2011, min is 3.`
- (c) `For 2011 > min, punt return > +20 is 3.0.`

**1043 r6c2**

- (a) `In the table 'industrial distribution of private businesses, by ownership and two-digit naics industry code, canada, 2010', among percent distribution > industry > manufacturing, the value of primarily self-employed > immigrant-owned is 1.6.`
- (b) `In industrial distribution of private businesses, by ownership and two-digit naics industry code, canada, 2010, primarily self-employed, immigrant-owned for percent distribution, industry, manufacturing is 1.6.`
- (c) `For percent distribution > industry > manufacturing, primarily self-employed > immigrant-owned is 1.6.`

**104_54_nsf19316-tab002 r7c0**

- (a) `In the table 'table 2. domestic r&d intensity for companies located in the united states that performed or funded r&d, by company size: 2008-15', among medium companies > 100-249, the value of 2008 is 6.0.`
- (b) `In table 2. domestic r&d intensity for companies located in the united states that performed or funded r&d, by company size: 2008-15, 2008 for medium companies, 100-249 is 6.`
- (c) `For medium companies > 100-249, 2008 is 6.0.`

**1062 r4c5**

- (a) `In the table 'cisgender and transgender health risk behaviours, by behaviour, canada, 2018', among used drugs or alcohol to cope > with abuse or violence that occurred in lifetime, the value of transgender > 95% confidence interval > to is 52.6.`
- (b) `In cisgender and transgender health risk behaviours, by behaviour, canada, 2018, transgender, 95% confidence interval, to for used drugs or alcohol to cope, with abuse or violence that occurred in lifetime is 52.6.`
- (c) `For used drugs or alcohol to cope > with abuse or violence that occurred in lifetime, transgender > 95% confidence interval > to is 52.6.`

**1079 r3c2**

- (a) `In the table 'enterprise size class transition matrix for the period 2008 to 2014', among % > between 20 and 49, the value of enterprise size class, 2014 > between 10 and 19 is 13.3.`
- (b) `In enterprise size class transition matrix for the period 2008 to 2014, enterprise size class, 2014, between 10 and 19 for %, between 20 and 49 is 13.3.`
- (c) `For % > between 20 and 49, enterprise size class, 2014 > between 10 and 19 is 13.3.`

**1097 r1c4**

- (a) `In the table 'socio-economic and demographic characteristics of the 2006 census-cmdb cohort (census long-form questionnaire, non-institutionalized population), all ages, canada, 2006 to 2011', among average > average age, the value of weighted estimates > non-aboriginal is 38.6.`
- (b) `In socio-economic and demographic characteristics of the 2006 census-cmdb cohort (census long-form questionnaire, non-institutionalized population), all ages, canada, 2006 to 2011, weighted estimates, non-aboriginal for average, average age is 38.6.`
- (c) `For average > average age, weighted estimates > non-aboriginal is 38.6.`

