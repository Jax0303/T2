# 정답 라벨 감사 — 질문과 gold 셀이 공통 단어 0개인 건 전수

생성: `scripts/gold_label_audit.py`. 원본 JSONL: `results/gold_label_audit_{realhitbench,aitqa}.jsonl`

**플래그 기준:** 질문의 내용어와 gold 셀의 **헤더 경로**(값 제외)가 한 단어도 안 겹침.
불용어(the/what/total/number/average/sum 등)는 세지 않음. 값을 빼는 이유는 gold가 값으로 매칭된 것이라 값을 세면 항상 겹치기 때문.

**플래그 = 라벨이 틀렸다는 증명이 아니라, 사람이 봐야 할 집합.**

## RealHitBench

| | 건수 | 플래그 | 비율 |
|---|---|---|---|
| 전체 | 231 | 61 | 26.4% |
| 검색 성공 | 135 | 10 | 7.4% |
| 검색 실패 | 96 | 51 | 53.1% |

### 플래그된 61건 전수

**#180** · Fact Checking · OSC=1 · m=1
- Q: Which sector (Whole economy, Manufacturing, or Services) had the highest percentage change in Output per hour worked in 2020 Q2?
- 정답: `Whole economy`
- gold 셀: `Revisions since previously published estimates: Whole economy`

**#271** · Fact Checking · OSC=0 · m=1
- Q: Which state has the highest death percent?
- 정답: `Punjab`
- gold 셀: `Recovered > Lakshadweep > Least > States: Punjab`

**#330** · Data Analysis · OSC=0 · m=6
- Q: Can you summarize the key features of the table, including the main variables and any evident relationships or trends?
- 정답: `The table presents Pearson correlation coefficients between environmental variables like PAR, Ta, VPD, Sal, and WL. It h`
- gold 셀: `WL > Pearson correlation coefficients: Ta`
- gold 셀: `WL > Pearson correlation coefficients: VPD`
- gold 셀: `WL > Pearson correlation coefficients: Sal`
- gold 셀: `WL > Pearson correlation coefficients: Ta`
- gold 셀: `WL > Pearson correlation coefficients: VPD`
- gold 셀: `WL > Pearson correlation coefficients: Sal`

**#375** · Fact Checking · OSC=0 · m=1
- Q: What is the total number of learners who participated in English programs across all institution types?
- 정답: `860200`
- gold 셀: `of which EnglishR > Participation: 860,200`

**#474** · Fact Checking · OSC=1 · m=1
- Q: Compare the national disposable incomes of Australia and Austria for the year 2020. Which country had a higher income?
- 정답: `Australia`
- gold 셀: `Time period > Reference area: Australia`

**#513** · Fact Checking · OSC=0 · m=1
- Q: Between “March 31 2013” and “March 31 2016”, which period recorded the higher number of Black, Asian, and Minority Ethnic intakes for Army Reserve?
- 정답: `31 Mar 16`
- gold 셀: `Glossary of terms: 31 Mar 16`

**#679** · Fact Checking · OSC=0 · m=1
- Q: What are the job descriptions assigned to the Electrical Department and expected to be completed in 2 hours?
- 정답: `Thermostate Switch Change on DG-2`
- gold 셀: `Meeting Date: 26 Oct, 24 > DAILY MORNING DISCUSSION: Thermostate Switch Change on DG-2`

**#680** · Fact Checking · OSC=0 · m=2
- Q: Identify the task locations where jobs raised by the Mechanical Department have been marked as "Work in Progress."
- 정답: `JL-1, Muffler Room`
- gold 셀: `Meeting Date: 26 Oct, 24 > DAILY MORNING DISCUSSION: JL-1`
- gold 셀: `Meeting Date: 26 Oct, 24 > DAILY MORNING DISCUSSION: Muffler Room`

**#702** · Fact Checking · OSC=0 · m=2
- Q: Which product has the lowest "Price" under the "Galvalume" material, and what is its corresponding "CostLabor"?
- 정답: `W - Valley, 2.60`
- gold 셀: `W - Valley > 21 > 0.51219512195122 > LABOR: $2.60`
- gold 셀: `Deluxe Ridge Cap > 20.44 > 0.49853658536585 > LABOR: $2.60`

**#779** · Numerical Reasoning · OSC=0 · m=1
- Q: Identify the team with the highest goals scored in a single match and specify the number of goals.
- 정답: `4`
- gold 셀: `21 > Venezuela > 2 > 2 > Actual Results: 4`

**#813** · Numerical Reasoning · OSC=0 · m=1
- Q: How many members have zero attendance for operations (Ops) during the month?
- 정답: `18`
- gold 셀: `March 2016 > F3 > F: 18`

**#869** · Numerical Reasoning · OSC=1 · m=3
- Q: Rank the top three ingredients with the highest quantities in the Baked Caprese Chicken recipe.
- 정답: `basil, butter, mozzarella`
- gold 셀: `4.00 > tablespoon > garlic: butter`
- gold 셀: `4.00 > ounces > garlic: mozzarella`
- gold 셀: `10.00 > leaves > garlic: basil`

**#879** · Numerical Reasoning · OSC=0 · m=1
- Q: Calculate the number of days when the RECVD value exceeds 100,000.
- 정답: `5`
- gold 셀: `15000 > Fri > 11/6: 5`

**#888** · Fact Checking · OSC=0 · m=1
- Q: How many participants belong to the “Barlick Fellrunners” club?
- 정답: `3`
- gold 셀: `3rd Male > Pos: 3`

**#889** · Numerical Reasoning · OSC=0 · m=1
- Q: Calculate the total number of participants across all events with a “Time” below 42 minutes.
- 정답: `5`
- gold 셀: `1st M40 > Pos: 5`

**#909** · Numerical Reasoning · OSC=0 · m=1
- Q: How many transactions were recorded for the “Product” type “Service Work” during the specified date range?
- 정답: `10`
- gold 셀: `February 14, 2015 > 82164 > DRL11 > DRL Construction: 10`

**#963** · Numerical Reasoning · OSC=0 · m=2
- Q: Which product has the highest beginning stock value, and what is that value?
- 정답: `Lamivudine-Zidovudine 150/300mg, 8400`
- gold 셀: `11 > Lamivudine-Zidovudine 150/300mg > 4111309 > 1/1/2016 > 1: 8,400`
- gold 셀: `11 > Lamivudine-Zidovudine 150/300mg > 4111309 > 1/1/2016 > 2014: 8,400`

**#979** · Fact Checking · OSC=1 · m=1
- Q: What is the total number of activations for all departments combined?
- 정답: `13557`
- gold 셀: `Grand Total > Activation Count: 13557`

**#984** · Fact Checking · OSC=0 · m=1
- Q: Compare the local price of a $10 Load Card in Europe to a $10 Load Card in Australia. Which one costs more in local currency?
- 정답: `Australia`
- gold 셀: `MYR > Product Name > North America: Australia`

**#987** · Numerical Reasoning · OSC=1 · m=1
- Q: Compare the local price of a 3-Month Premium Subscription in Canada to that in Japan. Which country has a higher price?
- 정답: `Canada`
- gold 셀: `USD > Product Name > North America: Canada`

**#1064** · Fact Checking · OSC=0 · m=1
- Q: How many players ranked in the top 10 have earned more than 500 points?
- 정답: `2`
- gold 셀: `2 > 544 > Brian Thomas > 10/06/13 > place: 2`

**#1111** · Numerical Reasoning · OSC=0 · m=1
- Q: How many materials have a depth difference (DEPTH DIFF) exceeding 100?
- 정답: `14`
- gold 셀: `982 > 4130 > 4130 > 1 > NORTHING: 14`

**#1234** · Numerical Reasoning · OSC=1 · m=1
- Q: What is the total difference in revenue between the original unit prices and the new unit prices for all items?
- 정답: `-1578`
- gold 셀: `9 > FRU BANN MU > D/P: -1578`

**#1236** · Fact Checking · OSC=1 · m=1
- Q: Which item has the highest difference in total value between the original and new unit prices?
- 정답: `CU CMC VEG`
- gold 셀: `7 > CU CMC VEG > BATCH: CU CMC VEG`

**#1241** · Fact Checking · OSC=0 · m=1
- Q: What is the total balance for all clients at the end date?
- 정답: `5500`
- gold 셀: `TOTAL > CREDIT: 5500`

**#1245** · Fact Checking · OSC=0 · m=1
- Q: What is the total quantity of all items listed?
- 정답: `39`
- gold 셀: `TOTAL > QTY: 39`

**#1302** · Fact Checking · OSC=0 · m=1
- Q: What is the standard deviation of M2 for Day 1 Control?
- 정답: `3.76`
- gold 셀: `st.dev > M2: 3.76`

**#1303** · Fact Checking · OSC=0 · m=1
- Q: What is the average value of M4 for Day 1 IR?
- 정답: `24.68`
- gold 셀: `avg > M3: 24.68`

**#1304** · Fact Checking · OSC=0 · m=1
- Q: What is the average value of M2 for Day 3 IR?
- 정답: `51.65`
- gold 셀: `avg > M1: 51.65`

**#1305** · Fact Checking · OSC=0 · m=1
- Q: How many lives were saved due to contraceptive use in Family Planning?
- 정답: `11.59`
- gold 셀: `st.dev > M1: 11.59`

**#1338** · Fact Checking · OSC=0 · m=1
- Q: What is the average value of M1 for Day 0 Control?
- 정답: `44.85`
- gold 셀: `avg > M1: 44.85`

**#1339** · Fact Checking · OSC=0 · m=1
- Q: What is the standard deviation of M2 for Day 1 Control?
- 정답: `3.76`
- gold 셀: `st.dev > M2: 3.76`

**#1378** · Numerical Reasoning · OSC=0 · m=1
- Q: How many different types of effectiveness measures for health facility mitigation measures are there in the table?
- 정답: `3`
- gold 셀: `Increased risk of inpatient stay vs. ambulatory visit (1 day stay vs. 1 outpatient visit) > h > Mean: 3`

**#1380** · Fact Checking · OSC=0 · m=1
- Q: Which intervention has the highest Benefit-Risk Ratio?
- 정답: `Case management of neonatal sepsis/pneumonia`
- gold 셀: `x > 45/46/47 > 68/69: Case management of neonatal sepsis/pneumonia`

**#1407** · Fact Checking · OSC=0 · m=1
- Q: Which service has the highest number of newborn lives lost due to disruption?
- 정답: `Case management of neonatal sepsis/pneumonia`
- gold 셀: `x > 45/46/47 > 68/69: Case management of neonatal sepsis/pneumonia`

**#1409** · Fact Checking · OSC=0 · m=1
- Q: Which type of Antenatal Care caused the most Maternal Lives Lost?
- 정답: `MgSO4 management of pre-eclampsia`
- gold 셀: `x > 21 > 26: MgSO4 management of pre-eclampsia`

**#1426** · Fact Checking · OSC=0 · m=1
- Q: Which type of Antenatal Care intervention contributed to the most Newborn Lives Saved?
- 정답: `TT - Tetanus toxoid vaccination`
- gold 셀: `x > 11 > 16: TT - Tetanus toxoid vaccination`

**#1427** · Fact Checking · OSC=0 · m=1
- Q: Which Delivery Care intervention led to the highest number of Newborn Lives Saved?
- 정답: `Cesarean delivery`
- gold 셀: `x > 36 > 42: Cesarean delivery`

**#1428** · Fact Checking · OSC=0 · m=1
- Q: Which pentavalent has the highest number of child lives saved?
- 정답: `Hib - Three doses`
- gold 셀: `x > 54 > 58: Hib - Three doses`

**#1435** · Fact Checking · OSC=0 · m=4
- Q: What family planning methods are included in the analysis?
- 정답: `FP - Injectables, FP - Implants, FP - Male Sterilization, FP - Traditional Methods`
- gold 셀: `3 > Include?: FP - Injectables`
- gold 셀: `4 > Include?: FP - Implants`
- gold 셀: `7 > Include?: FP - Male Sterilization`
- gold 셀: `8 > Include?: FP - Traditional Methods`

**#1456** · Fact Checking · OSC=0 · m=1
- Q: Which Pregnancy/ Antenatal intervention had the highest 2019 Pre-COVID coverage rate?
- 정답: `TT - Tetanus toxoid vaccination`
- gold 셀: `x > 11 > 16: TT - Tetanus toxoid vaccination`

**#1461** · Numerical Reasoning · OSC=0 · m=1
- Q: How many types of Effectiveness of Health Facility Mitigation Measures are listed?
- 정답: `3`
- gold 셀: `Increased risk of inpatient stay vs. ambulatory visit (1 day stay vs. 1 outpatient visit) > h > Mean > 200,000: 3`

**#1462** · Numerical Reasoning · OSC=0 · m=1
- Q: How many types of Infection Fatality Rates (IFRs) are listed?
- 정답: `3`
- gold 셀: `Increased risk of inpatient stay vs. ambulatory visit (1 day stay vs. 1 outpatient visit) > h > Mean > 200,000: 3`

**#1465** · Fact Checking · OSC=0 · m=1
- Q: What Antenatal Care has the highest Benefit-Risk Ratio?
- 정답: `MgSO4 management of pre-eclampsia`
- gold 셀: `x > 21 > 26: MgSO4 management of pre-eclampsia`

**#1468** · Fact Checking · OSC=0 · m=1
- Q: Which intervention results in the most lives lost through added COVID infections?
- 정답: `ORS - Oral Rehydration Solution`
- gold 셀: `x > 63 > 71: ORS - Oral Rehydration Solution`

**#1503** · Numerical Reasoning · OSC=1 · m=1
- Q: How many participants in Teaching Group 4 have a total score greater than 40?
- 정답: `5`
- gold 셀: `100 > 4 > CAI > 4: 5`

**#2083** · Numerical Reasoning · OSC=0 · m=1
- Q: How many regions have more than 1,000 schools reporting Algebra I data?
- 정답: `4`
- gold 셀: `Rhode Island > Mathematics  Courses > Geometry > Number: 4`

**#2166** · Fact Checking · OSC=0 · m=1
- Q: What was the percentage of Christians in England and Wales in 2018? (Final answer should be a decimal with one decimal place.)
- 정답: `46.7`
- gold 셀: `Christian > 58 > 57.7 > 58.3: 46.7`

**#2170** · Fact Checking · OSC=0 · m=1
- Q: What was the total population of England and Wales in 2016, according to the table? (Final answer should be an integer.)
- 정답: `36454`
- gold 셀: `Total > 35916 > 35955 > 36076: 36454`

**#2191** · Fact Checking · OSC=1 · m=1
- Q: What was the employment rate for Buddhists in 2018? (Final answer should be an integer.)
- 정답: `69`
- gold 셀: `Buddhist > Estimate > 66.1 > 60 > Contents: 69`

**#2193** · Fact Checking · OSC=0 · m=1
- Q: In which year did the employment rate for Sikhs exceed 70% for the first time? (Final answer should be a specific year.)
- 정답: `2016`
- gold 셀: `2012 > 2013: 2016`

**#2197** · Numerical Reasoning · OSC=0 · m=1
- Q: How many religions had an unemployment rate below 4% in 2018? (Final answer should be an integer.)
- 정답: `6`
- gold 셀: `Sikh > Estimate > 8.5 > 6.1: 6`

**#2204** · Fact Checking · OSC=0 · m=2
- Q: In which year did the estimated percentage of Buddhists working in high-skilled occupations exceed 29%? (Final answer should specify the year(s).)
- 정답: `2015, 2016`
- gold 셀: `2012: 2015`
- gold 셀: `2012: 2016`

**#2206** · Numerical Reasoning · OSC=0 · m=1
- Q: In which year did the Buddhist population show the lowest estimate for employed people in upper-middle-skilled occupations? (Final answer should be a specific year.)
- 정답: `2015`
- gold 셀: `2012: 2015`

**#2210** · Data Analysis · OSC=0 · m=1
- Q: Based on the 2012-2018 trends, project the proportion of employed Muslims in upper-middle-skilled occupations for 2025. (Final answer should be rounded to one decimal place.)
- 정답: `21.1`
- gold 셀: `Muslim > 21.7 > 20.4: 21.1`

**#2317** · Numerical Reasoning · OSC=0 · m=1
- Q: How many different reasons for “Adverse decisions” are listed in the table for the period between October 2012 and September 2013?
- 정답: `18`
- gold 셀: `Low > Month: 18`

**#2320** · Data Analysis · OSC=0 · m=2
- Q: What is the total number of sanctions recorded for “Voluntarily leaves a place on a training scheme” during October 2012, and how does it compare to the sanctions for “Losing through misconduct”?
- 정답: `11972, 10334`
- gold 셀: `Low > Month: 11,972`
- gold 셀: `Low > Month: 10,334`

**#2321** · Data Analysis · OSC=0 · m=1
- Q: Based on the trends from October 2012 to September 2013, can we predict the total sanctions for the “Voluntarily leaves a place on a training scheme” category for October 2013?
- 정답: `11972`
- gold 셀: `Low > Month: 11,972`

**#2356** · Numerical Reasoning · OSC=0 · m=2
- Q: Which player scored more total runs across all their games, and by how many runs?
- 정답: `Rick Bosetti, 33`
- gold 셀: `22 > Rick Bosetti > AB: 33`
- gold 셀: `20 > Al Woods > AB: 33`

**#2470** · Numerical Reasoning · OSC=0 · m=1
- Q: How many tickets are associated with the date “Feb-01-15”?
- 정답: `7`
- gold 셀: `February 6, 2015 > 762482 > DRL11 > DRL Construction: 7`

**#2497** · Visualization · OSC=1 · m=9
- Q: Draw a line chart showing the total number of employees in each country based on the "Total" row for each country in the table. Ensure the x-axis represents the countries and the y-axis represents the total employees.
- 정답: `[[20, 17, 15, 7, 19, 7, 12, 22, 20, 11]]
`
- gold 셀: `Australia Total > Total: 20`
- gold 셀: `Brazil Total > Total: 17`
- gold 셀: `Cannada Total > Total: 15`
- gold 셀: `China Total > Total: 7`
- gold 셀: `Germany Total > Total: 19`
- gold 셀: `India Total > Total: 7`
- gold 셀: `Russia Total > Total: 12`
- gold 셀: `Spain Total > Total: 22`
- gold 셀: `UK Total > Total: 20`

## AIT-QA

| | 건수 | 플래그 | 비율 |
|---|---|---|---|
| 전체 | 451 | 34 | 7.5% |
| 검색 성공 | 398 | 14 | 3.5% |
| 검색 실패 | 53 | 20 | 37.7% |

### 플래그된 34건 전수

**#q-68** · OSC=0 · m=1
- Q: How many flight attendants were employed by Southwest Airlines in November 2018 ?
- 정답: `16,000`
- gold 셀: `Approximate Number of Employees: 16,000`

**#q-69** · OSC=1 · m=1
- Q: how many dispatchers are part of the Southwest dispatchers employee group in 2019?
- 정답: `400`
- gold 셀: `Approximate Number of Employees: 400`

**#q-70** · OSC=1 · m=1
- Q: How many flight crew training instructors Southwest Airlines had in 2019 ?
- 정답: `130`
- gold 셀: `Approximate Number of Employees: 130`

**#q-71** · OSC=0 · m=1
- Q: How many meteorologists were employed by Southwest Airlines in 2019 ?
- 정답: `10`
- gold 셀: `Approximate Number of Employees: 10`

**#q-80** · OSC=1 · m=1
- Q: What was the total share repurchases by Southwest Airlines in 2019?
- 정답: `36.75`
- gold 셀: `Total > Shares received: 36.75`

**#q-115** · OSC=0 · m=1
- Q: how much revenue did Delta Airlines generate from Loyalty programs in 2016 ?
- 정답: `2,184`
- gold 셀: `Contracted services > Year Ended December 31, > 2017: 2,184`

**#q-122** · OSC=0 · m=1
- Q: what was total fuel consumption of United airlines in the year 2019 ?
- 정답: `4,292`
- gold 셀: `Gallons Consumed (in millions): 4,292`

**#q-153** · OSC=0 · m=1
- Q: What is the Q2 operating revenue of United airlines in 2019?
- 정답: `1,472`
- gold 셀: `Income from operations > Quarter Ended > June 30: 1,472`

**#q-160** · OSC=0 · m=1
- Q: how many aircrafts did Southwest Airlines lease in 2018 ?
- 정답: `123`
- gold 셀: `Number Leased: 123`

**#q-161** · OSC=0 · m=1
- Q: As of February 2019, How old was the CEO of Southwest Airlines?
- 정답: `63`
- gold 셀: `Gary C. Kelly > Age: 63`

**#q-203** · OSC=0 · m=1
- Q: How many Fleet and passenger service does American Airlines have for the Piedmont with contract amendable date as 2017?
- 정답: `3,400`
- gold 셀: `Envoy: (3) > TWU > Employees  (1): 3,400`

**#q-207** · OSC=0 · m=1
- Q: In 2017 what was the fuel gallon consumption of American Airlines ?
- 정답: `4,352`
- gold 셀: `Gallons: 4,352`

**#q-220** · OSC=0 · m=1
- Q: How many airplanes did American Airlines have in total as of 2017?
- 정답: `948`
- gold 셀: `Total > Total: 948`

**#q-271** · OSC=1 · m=1
- Q: As of December 2017, how many people worked for United Airlines as security officers?
- 정답: `51`
- gold 셀: `Number of Employees: 51`

**#q-299** · OSC=0 · m=1
- Q: What was American's annual aircraft fuel consumption in 2019?
- 정답: `4,537`
- gold 셀: `Gallons: 4,537`

**#q-301** · OSC=0 · m=1
- Q: How many aircraft had American Airlines agreed to purchase as of the end of 2019?
- 정답: `248`
- gold 셀: `Bombardier > Total > Total: 248`

**#q-315** · OSC=0 · m=1
- Q: What was American Airlines' total cash payment for dividends during 2019?
- 정답: `$178`
- gold 셀: `Total (millions): $178`

**#q-354** · OSC=0 · m=1
- Q: What is the total number of aircraft on Delta airlines' purchase commitments as of the end of 2018?
- 정답: `330`
- gold 셀: `Total > Delivery in Calendar Years Ending > Total: 330`

**#q-400** · OSC=0 · m=1
- Q: How much fuel did American Airlines use in 2018?
- 정답: `4,447`
- gold 셀: `Gallons: 4,447`

**#q-464** · OSC=0 · m=1
- Q: How many planes does Alaska operate as of 2017?
- 정답: `304`
- gold 셀: `Total > Total: 304`

**#q-467** · OSC=0 · m=1
- Q: How old were Alaska's aircraft in 2017?
- 정답: `7.4`
- gold 셀: `Total > Average Age in Years: 7.4`

**#q-472** · OSC=0 · m=1
- Q: How much did Alaska's wage expenses grow in 2017 compared to the previous year?
- 정답: `17.8%`
- gold 셀: `Wages > Change > % Combined: 17.8%`

**#q-481** · OSC=0 · m=1
- Q: How much money was spent by Southwest on fuel in year in 2004?
- 정답: `$1,470`
- gold 셀: `Cost (Millions): $1,470`

**#q-483** · OSC=0 · m=1
- Q: Show Southwest's fuel expenses in Q2 2017.
- 정답: `$990`
- gold 셀: `Cost (Millions): $990`

**#q-485** · OSC=1 · m=1
- Q: what is the total number of pilots working for Southwest Airlines in 2017 ?
- 정답: `8,600`
- gold 셀: `Approximate Number of Employees: 8,600`

**#q-486** · OSC=1 · m=1
- Q: What is the total number of flight attendants working for Southwest airline in 2017 ?
- 정답: `14,500`
- gold 셀: `Approximate Number of Employees: 14,500`

**#q-487** · OSC=1 · m=1
- Q: What is the total number of mechanics working for Southwest Airlines in 2017 ?
- 정답: `2,400`
- gold 셀: `Approximate Number of Employees: 2,400`

**#q-488** · OSC=1 · m=1
- Q: What is the total number of flight simulator technicians working for Southwest Airlines in 2017 ?
- 정답: `50`
- gold 셀: `Approximate Number of Employees: 50`

**#q-489** · OSC=1 · m=1
- Q: What is the total number Meteorologists working for Southwest Airlines in 2017 ?
- 정답: `10`
- gold 셀: `Approximate Number of Employees: 10`

**#q-490** · OSC=1 · m=1
- Q: Which labour union represents Dispatchers of Southwest airlines?
- 정답: `Transportation Workers of America, AFL-CIO, Local 550 ("TWU 550")`
- gold 셀: `Representatives: Transportation Workers of America, AFL-CIO, Local 550 ("TWU 550")`

**#q-492** · OSC=1 · m=1
- Q: When will the contract of Southwest with the labor union representing flight attendants become amendable as of December 31, 2017?
- 정답: `Amendable November 2018`
- gold 셀: `Status of Agreement: Amendable November 2018`

**#q-493** · OSC=1 · m=1
- Q: Which employees of Southwest airlines does IBT represent?
- 정답: `Southwest Flight Simulator Technicians`
- gold 셀: `Employee Group: Southwest Flight Simulator Technicians`

**#q-497** · OSC=1 · m=1
- Q: What is the total number of aircrafts operated by Southwest Airlines as of December 2017 ?
- 정답: `706`
- gold 셀: `Number of Aircraft: 706`

**#q-501** · OSC=1 · m=1
- Q: How many planes did Southwest operate at the end of 2017?
- 정답: `706`
- gold 셀: `Number of Aircraft: 706`
