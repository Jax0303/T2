# 오류 사례 (사전 개선 진단)

모집단: hitab test primary (mode=all, m=1, aggregation=none), query count=991 (results/bottleneck_root_cause/top1_error_detail.csv 그대로 사용, 새로 채점하지 않음). 예시는 query_id 오름차순 처음 20건.

## top1 성공 (전체 570건 중 20건)

### 00803c36f0199f9bce904fac68db66c2

- query: how many percentage points of female veterans were black?
- gold: table=144_69_nscg17-ib19300-tab002 (3,4) path=`race and ethnicity > black, non-hispanic | women > veteran` serialized: In the table 'demographic characteristics of civilian college graduates, by sex and veteran status: 2017', among race and ethnicity > black, non-hispanic, the value of women > veteran is 27.8.
- top1: table=144_69_nscg17-ib19300-tab002 (3,4) path=`race and ethnicity > black, non-hispanic | women > veteran` serialized: In the table 'demographic characteristics of civilian college graduates, by sex and veteran status: 2017', among race and ethnicity > black, non-hispanic, the value of women > veteran is 27.8.
- dense_score gold=0.717796 top1=0.717796 margin=0.0
- hybrid_score gold=0.899908 top1=0.899908 margin=0.0
- gold_rank=1 error_class=exact_match

### 00d47c9aac5050539645dcae34f78570

- query: what percent of canadians with a household income of $150,000 or more said that hockey's influence on the national identity was very important?
- gold: table=2613 (21,4) path=`percent > household income > $150,000 | hockey` serialized: In the table 'perception of national symbols as very important to the canadian identity, by socio-demographic and economic characteristics, 2013', among percent > household income > $150,000, the value of hockey is 49.0.
- top1: table=2613 (21,4) path=`percent > household income > $150,000 | hockey` serialized: In the table 'perception of national symbols as very important to the canadian identity, by socio-demographic and economic characteristics, 2013', among percent > household income > $150,000, the value of hockey is 49.0.
- dense_score gold=0.786861 top1=0.786861 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 01ce71c3b86e4524dab2b41c7af72e6c

- query: 35 states reported expenditures for health-related r&d in fy 2015, how many thousand dollars were state government agency expenditures on health-related r&d in pennsylvania?
- gold: table=209_99_tab3 (6,4) path=`united statesa > pennsylvania | health` serialized: In the table 'state agency expenditures for r&d, by state and function of r&d, for the 10 states with the highest levels of r&d expenditures: fy 2015', among united statesa > pennsylvania, the value of health is 52399.
- top1: table=209_99_tab3 (6,4) path=`united statesa > pennsylvania | health` serialized: In the table 'state agency expenditures for r&d, by state and function of r&d, for the 10 states with the highest levels of r&d expenditures: fy 2015', among united statesa > pennsylvania, the value of health is 52399.
- dense_score gold=0.773801 top1=0.773801 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 022faef43c2d5f631f5d83e9e75f346a

- query: how many touchdown passes did michael bishop finish by playing two seasons?
- gold: table=231_totto26510-4 (2,5) path=`career | passing > td` serialized: In the table 'Michael Bishop (gridiron football): college statistics', among career, the value of passing > td is 36.0.
- top1: table=231_totto26510-4 (2,5) path=`career | passing > td` serialized: In the table 'Michael Bishop (gridiron football): college statistics', among career, the value of passing > td is 36.0.
- dense_score gold=0.667657 top1=0.667657 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 02dee9364e18e5cc338b46a5e992d2ad

- query: what is the percentage of the total respondents who reported that they engaged in utilitarian walking less than an hour a week?
- gold: table=223 (10,0) path=`hours utilitarian walking per week > less than 1 | total > number or %` serialized: In the table 'selected characteristics, by sex, household population aged 20 to 79, canada excluding territories, 2007 to 2011', among hours utilitarian walking per week > less than 1, the value of total > number or % is 34.2.
- top1: table=223 (10,0) path=`hours utilitarian walking per week > less than 1 | total > number or %` serialized: In the table 'selected characteristics, by sex, household population aged 20 to 79, canada excluding territories, 2007 to 2011', among hours utilitarian walking per week > less than 1, the value of total > number or % is 34.2.
- dense_score gold=0.655235 top1=0.655235 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 030333fec48bb883d822295e4e6a18f1

- query: what is the percentage of volunteers sitting on a committee or board in 2004?
- gold: table=1291 (2,3) path=`percentage of volunteers > sitting on a committee or board | 2004` serialized: In the table 'this table displays the results of participation rate by type of volunteer activities. the information is grouped by types of activities (appearing as row headers), 2013 (ref), 2010, 2007 and 2004, calculated using percentage of volunteers units of measure (appearing as column headers)', among percentage of volunteers > sitting on a committee or board, the value of 2004 is 36.
- top1: table=1291 (2,3) path=`percentage of volunteers > sitting on a committee or board | 2004` serialized: In the table 'this table displays the results of participation rate by type of volunteer activities. the information is grouped by types of activities (appearing as row headers), 2013 (ref), 2010, 2007 and 2004, calculated using percentage of volunteers units of measure (appearing as column headers)', among percentage of volunteers > sitting on a committee or board, the value of 2004 is 36.
- dense_score gold=0.738751 top1=0.738751 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 0356b78121f0b07f8c238533de6d8957

- query: the largest gain in representation was observed within the u.s. federal government, where women nearly doubled their share, how many percent of all federally employed seh doctorate holders in 2017?
- gold: table=133_64_sdr17-ib-19307-tab003 (6,9) path=`all employment sectors > federal government | 2017 > female > %` serialized: In the table 'employed u.s.-trained seh doctorate holders residing in the united states, by employment sector and sex: 1997 and 2017', among all employment sectors > federal government, the value of 2017 > female > % is 35.4.
- top1: table=133_64_sdr17-ib-19307-tab003 (6,9) path=`all employment sectors > federal government | 2017 > female > %` serialized: In the table 'employed u.s.-trained seh doctorate holders residing in the united states, by employment sector and sex: 1997 and 2017', among all employment sectors > federal government, the value of 2017 > female > % is 35.4.
- dense_score gold=0.770991 top1=0.770991 margin=0.0
- hybrid_score gold=0.990534 top1=0.990534 margin=0.0
- gold_rank=1 error_class=exact_match

### 03a65e7848f82fcbff091b76cac5afc1

- query: what percent of those with a university degree viewing charter as very important to the national identity.
- gold: table=2613 (13,0) path=`percent > highest level of education completed > university degree | charter of rights and freedoms` serialized: In the table 'perception of national symbols as very important to the canadian identity, by socio-demographic and economic characteristics, 2013', among percent > highest level of education completed > university degree, the value of charter of rights and freedoms is 72.0.
- top1: table=2613 (13,0) path=`percent > highest level of education completed > university degree | charter of rights and freedoms` serialized: In the table 'perception of national symbols as very important to the canadian identity, by socio-demographic and economic characteristics, 2013', among percent > highest level of education completed > university degree, the value of charter of rights and freedoms is 72.0.
- dense_score gold=0.773622 top1=0.773622 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 04161feb4413260cf720dd6196ebee32

- query: in 1994, which league did jones participate for guangdong hongyuan?
- gold: table=502_totto57483-4 (3,0) path=`guangdong hongyuan > 1994 | league > division` serialized: In the table 'Murray Jones: career statistics', among guangdong hongyuan > 1994, the value of league > division is jia-a league.
- top1: table=502_totto57483-4 (3,0) path=`guangdong hongyuan > 1994 | league > division` serialized: In the table 'Murray Jones: career statistics', among guangdong hongyuan > 1994, the value of league > division is jia-a league.
- dense_score gold=0.696573 top1=0.696573 margin=0.0
- hybrid_score gold=0.994161 top1=0.994161 margin=0.0
- gold_rank=1 error_class=exact_match

### 04572d818fcfa7491d9d7f539d40482d

- query: what is the proportion in nunavut in which the reason for not reporting an incident to the police was that it was too minor to be worth reporting?
- gold: table=2769 (0,2) path=`percent > crime was minor and not worth taking the time to report | nunavut` serialized: In the table 'reasons for not reporting victimization incidents to the police, by territory, 2014', among percent > crime was minor and not worth taking the time to report, the value of nunavut is 70.0.
- top1: table=2769 (0,2) path=`percent > crime was minor and not worth taking the time to report | nunavut` serialized: In the table 'reasons for not reporting victimization incidents to the police, by territory, 2014', among percent > crime was minor and not worth taking the time to report, the value of nunavut is 70.0.
- dense_score gold=0.813603 top1=0.813603 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 0465428e164c8dfb486a83a89e5f4f65

- query: among men, what is the percentage for those whose country of ancestry is in africa?
- gold: table=1004 (2,3) path=`region of ancestry > africa | men > university degree only` serialized: In the table 'percentage of canadian-born black immigrants aged 25 to 59 with a postsecondary diploma, by sex and region or country of ancestry, 2016', among region of ancestry > africa, the value of men > university degree only is 35.3.
- top1: table=1004 (2,3) path=`region of ancestry > africa | men > university degree only` serialized: In the table 'percentage of canadian-born black immigrants aged 25 to 59 with a postsecondary diploma, by sex and region or country of ancestry, 2016', among region of ancestry > africa, the value of men > university degree only is 35.3.
- dense_score gold=0.714281 top1=0.714281 margin=0.0
- hybrid_score gold=0.965962 top1=0.965962 margin=0.0
- gold_rank=1 error_class=exact_match

### 04912d50252c088bdef7ddf5ce296846

- query: at two times the national quarterly rate (18.6 per 100,000 population), what was the rate in kingston reported the rate of police-reported sexual assaults after #metoo?
- gold: table=1644 (11,3) path=`total cmas > kingston | post-#metoo > rate` serialized: In the table 'victims of police-reported sexual assault, by quarterly #metoo period and census metropolitan area, canada, 2016 and 2017', among total cmas > kingston, the value of post-#metoo > rate is 29.7.
- top1: table=1644 (11,3) path=`total cmas > kingston | post-#metoo > rate` serialized: In the table 'victims of police-reported sexual assault, by quarterly #metoo period and census metropolitan area, canada, 2016 and 2017', among total cmas > kingston, the value of post-#metoo > rate is 29.7.
- dense_score gold=0.779377 top1=0.779377 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 054a1a4094d78187adc8be002c050889

- query: how many receiving yards did davis get in 2018?
- gold: table=170_totto20194-1 (1,3) path=`2018 > ten | receiving > yds` serialized: In the table 'Corey Davis (wide receiver): regular season', among 2018 > ten, the value of receiving > yds is 891.0.
- top1: table=170_totto20194-1 (1,3) path=`2018 > ten | receiving > yds` serialized: In the table 'Corey Davis (wide receiver): regular season', among 2018 > ten, the value of receiving > yds is 891.0.
- dense_score gold=0.754924 top1=0.754924 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 05961a45ddb856886b4d7151ee4e303c

- query: in 2018, of the $441 million that companies spent on r&d, how many million dollars were spent on basic research?
- gold: table=65_20_nsf20316-tab001 (2,1) path=`domestic r&d performancea > type of r&db > basic research | 2018` serialized: In the table 'funds spent for business r&d performed in the united states, by type of r&d, source of funds, and size of company: 2017-18', among domestic r&d performancea > type of r&db > basic research, the value of 2018 is 28980.0.
- top1: table=65_20_nsf20316-tab001 (2,1) path=`domestic r&d performancea > type of r&db > basic research | 2018` serialized: In the table 'funds spent for business r&d performed in the united states, by type of r&d, source of funds, and size of company: 2017-18', among domestic r&d performancea > type of r&db > basic research, the value of 2018 is 28980.0.
- dense_score gold=0.778105 top1=0.778105 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 0654a77661a60c1aff6387ff4574abab

- query: how many percent of off-reserve first nations women aged 15 and over had been diagnosed with arthritis?
- gold: table=1902 (1,0) path=`percentage > total aboriginal identity female population > off-reserve first nations | chronic conditions > arthritis` serialized: In the table 'prevalence of selected chronic conditions diagnosed by health professional for female population aged 15 and over, by selected aboriginal identity group, canada, 2012', among percentage > total aboriginal identity female population > off-reserve first nations, the value of chronic conditions > arthritis is 23.4.
- top1: table=1902 (1,0) path=`percentage > total aboriginal identity female population > off-reserve first nations | chronic conditions > arthritis` serialized: In the table 'prevalence of selected chronic conditions diagnosed by health professional for female population aged 15 and over, by selected aboriginal identity group, canada, 2012', among percentage > total aboriginal identity female population > off-reserve first nations, the value of chronic conditions > arthritis is 23.4.
- dense_score gold=0.826164 top1=0.826164 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 068e85ed06c5b22bb4aeb913e8e1d4f2

- query: over the past 10 years, how many percentage points did software r&d increase annually compound at average growth rate?
- gold: table=102_53_nsf19315-tab001 (9,1) path=`2006-16 annual growth rate (%)a > all industries | software r&d` serialized: In the table 'total domestic business r&d and software r&d expenditures: 2006 and 2016', among 2006-16 annual growth rate (%)a > all industries, the value of software r&d is 9.6.
- top1: table=102_53_nsf19315-tab001 (9,1) path=`2006-16 annual growth rate (%)a > all industries | software r&d` serialized: In the table 'total domestic business r&d and software r&d expenditures: 2006 and 2016', among 2006-16 annual growth rate (%)a > all industries, the value of software r&d is 9.6.
- dense_score gold=0.688305 top1=0.688305 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 072716b59c63a3c7c38dbe43d8f7dd6b

- query: among metis women aged 15 and over, how many percent reported having a mood disorders?
- gold: table=1902 (2,3) path=`percentage > total aboriginal identity female population > metis | chronic conditions > mood disorder` serialized: In the table 'prevalence of selected chronic conditions diagnosed by health professional for female population aged 15 and over, by selected aboriginal identity group, canada, 2012', among percentage > total aboriginal identity female population > metis, the value of chronic conditions > mood disorder is 16.2.
- top1: table=1902 (2,3) path=`percentage > total aboriginal identity female population > metis | chronic conditions > mood disorder` serialized: In the table 'prevalence of selected chronic conditions diagnosed by health professional for female population aged 15 and over, by selected aboriginal identity group, canada, 2012', among percentage > total aboriginal identity female population > metis, the value of chronic conditions > mood disorder is 16.2.
- dense_score gold=0.778443 top1=0.778443 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 07cba698b92f9be91922e7d551ca40da

- query: how many residents in montreal live in a central municipality?
- gold: table=435 (2,1) path=`percentage > census metropolitan area of residence > montreal | type of municipality > central` serialized: In the table 'population of interest from data of the general social survey on social identity by place of residence, 2013', among percentage > census metropolitan area of residence > montreal, the value of type of municipality > central is 54.0.
- top1: table=435 (2,1) path=`percentage > census metropolitan area of residence > montreal | type of municipality > central` serialized: In the table 'population of interest from data of the general social survey on social identity by place of residence, 2013', among percentage > census metropolitan area of residence > montreal, the value of type of municipality > central is 54.0.
- dense_score gold=0.74029 top1=0.74029 margin=0.0
- hybrid_score gold=0.980529 top1=0.980529 margin=0.0
- gold_rank=1 error_class=exact_match

### 07e810086b40431fb3c2c5a064073471

- query: in 2013, what percent of university educated canadians felt that the rcmp was a very important national symbol?
- gold: table=2613 (13,3) path=`percent > highest level of education completed > university degree | rcmp` serialized: In the table 'perception of national symbols as very important to the canadian identity, by socio-demographic and economic characteristics, 2013', among percent > highest level of education completed > university degree, the value of rcmp is 46.0.
- top1: table=2613 (13,3) path=`percent > highest level of education completed > university degree | rcmp` serialized: In the table 'perception of national symbols as very important to the canadian identity, by socio-demographic and economic characteristics, 2013', among percent > highest level of education completed > university degree, the value of rcmp is 46.0.
- dense_score gold=0.815541 top1=0.815541 margin=0.0
- hybrid_score gold=1.0 top1=1.0 margin=0.0
- gold_rank=1 error_class=exact_match

### 084cc16058303f709a416459bdc30c2e

- query: what was the highest rates for sexual offences did the northwest territories report?
- gold: table=1356 (10,1) path=`territories overall > northwest territories | sexual offences > rate` serialized: In the table 'young female victims of police-reported violent crime in the north, by type of offence and province or territory, canada, 2017', among territories overall > northwest territories, the value of sexual offences > rate is 1827.
- top1: table=1356 (10,1) path=`territories overall > northwest territories | sexual offences > rate` serialized: In the table 'young female victims of police-reported violent crime in the north, by type of offence and province or territory, canada, 2017', among territories overall > northwest territories, the value of sexual offences > rate is 1827.
- dense_score gold=0.790818 top1=0.790818 margin=0.0
- hybrid_score gold=0.921357 top1=0.921357 margin=0.0
- gold_rank=1 error_class=exact_match

## wrong_table (전체 102건 중 20건)

### 00549126c06b99604e8dfb85cc2f7f3f

- query: in fy 2016, how many thousand dollars was r&d support?
- gold: table=151_72_tab3 (0,1) path=`all hbcus | r&d` serialized: In the table 'federal obligations for science and engineering to historically black colleges and universities, ranked by total amount received, by type of activity: fy 2016', among all hbcus, the value of r&d is 257894.2.
- top1: table=106_55_fs17-ib-19314-tab001 (16,1) path=`constant fy 2012 $millions > 2016 | r&d` serialized: In the table 'federal science and engineering obligations, by type of activity: fys 2009-17', among constant fy 2012 $millions > 2016, the value of r&d is 26939.
- dense_score gold=0.607454 top1=0.667131 margin=0.059677
- hybrid_score gold=0.82242 top1=0.9459 margin=0.12348
- gold_rank=180 error_class=wrong_table

### 00b61cc5439989b3092d38892bad9857

- query: what was the rate of police strength in victoria in 2017?
- gold: table=358 (47,2) path=`victoria, b.c | police officers > 2017 police officers per 100,000 population > rate` serialized: In the table 'municipal police services serving a population of 100,000 or more, canada, 2017', among victoria, b.c, the value of police officers > 2017 police officers per 100,000 population > rate is 233.0.
- top1: table=1967 (32,2) path=`victoria | total crime > percent change in rate 2007 to 2017` serialized: In the table 'police-reported crime rate, by census metropolitan area, 2017', among victoria, the value of total crime > percent change in rate 2007 to 2017 is -39.0.
- dense_score gold=0.738707 top1=0.753137 margin=0.014429
- hybrid_score gold=0.975953 top1=0.981232 margin=0.005279
- gold_rank=4 error_class=wrong_table

### 06e47fc12e39a89455f86f74aee140dc

- query: what is the percentage of cases cited for not reporting an incident to the police in the territories were because they would not be able to find the property or identify the offender?
- gold: table=2769 (5,3) path=`percent > police wouldn't have found property/offender | all territories` serialized: In the table 'reasons for not reporting victimization incidents to the police, by territory, 2014', among percent > police wouldn't have found property/offender, the value of all territories is 46.
- top1: table=1510 (13,2) path=`percent > police would not have been able to find or identify the offender | total` serialized: In the table 'reasons for not reporting sexual assault to the police, by sex of victim, canada, 2014', among percent > police would not have been able to find or identify the offender, the value of total is 21.
- dense_score gold=0.793383 top1=0.711993 margin=-0.08139
- hybrid_score gold=0.869139 top1=0.899243 margin=0.030104
- gold_rank=4 error_class=wrong_table

### 07c09d0b2aee8ae8ce94ebb8f5a187a8

- query: what was the changes in the number of completed cases in northwest territories from 2012/2013 to 2013/2014?
- gold: table=1493 (12,4) path=`canada > northwest territories | percent change in number of cases 2012/2013 to 2013/2014 > percent` serialized: In the table 'cases completed in youth court, by province and territory, 2012/2013 and 2013/2014', among canada > northwest territories, the value of percent change in number of cases 2012/2013 to 2013/2014 > percent is 2.
- top1: table=2775 (12,2) path=`canada > northwest territories | 2013/2014 > number` serialized: In the table 'cases completed in adult criminal court, by province and territory, 2012/2013 and 2013/2014', among canada > northwest territories, the value of 2013/2014 > number is 1465.
- dense_score gold=0.79736 top1=0.825122 margin=0.027761
- hybrid_score gold=0.967655 top1=0.980597 margin=0.012942
- gold_rank=5 error_class=wrong_table

### 08147555888ee98eea4891c287470047

- query: how many million dollars did funding for research and development to universities and colleges increase in 2017?
- gold: table=106_55_fs17-ib-19314-tab001 (8,1) path=`2017 | r&d` serialized: In the table 'federal science and engineering obligations, by type of activity: fys 2009-17', among 2017, the value of r&d is 29796.
- top1: table=2547 (5,1) path=`total > universities and colleges | total amount donated > millions of dollars` serialized: In the table 'donor rate and amounts donated to different types of organizations, 2013', among total > universities and colleges, the value of total amount donated > millions of dollars is 161.0.
- dense_score gold=0.599954 top1=0.636316 margin=0.036362
- hybrid_score gold=0.680763 top1=0.93863 margin=0.257867
- gold_rank=448 error_class=wrong_table

### 0a2938b5924faf566b2ccd1061dc1412

- query: how many billion dollars of r&d did companies perform in the united states in 2015?
- gold: table=92_48_nsf19322-tab001 (0,1) path=`all locations | total` serialized: In the table 'domestic r&d performed by companies, by core-based statistical areas with at least $1 billion and source of funds: 2015', among all locations, the value of total is 355821.
- top1: table=16_118_tab1 (1,1) path=`u.s.-located companies > domestic r&d performance | 2009` serialized: In the table 'business r&d performed in the united states, by source of funds and mnc status: 2008-10', among u.s.-located companies > domestic r&d performance, the value of 2009 is 282393.0.
- dense_score gold=0.684337 top1=0.705736 margin=0.021399
- hybrid_score gold=0.919967 top1=0.983904 margin=0.063937
- gold_rank=34 error_class=wrong_table

### 0c04b94b7162d791a417ba4fedd934b5

- query: how many yards did michael bishop finish by playing two seasons?
- gold: table=231_totto26510-4 (2,3) path=`career | passing > yds` serialized: In the table 'Michael Bishop (gridiron football): college statistics', among career, the value of passing > yds is 4401.0.
- top1: table=273_totto31153-4 (13,3) path=`125 | passing > yards` serialized: In the table 'Michael Bishop (gridiron football): nfl, nfle and cfl statistics', among 125, the value of passing > yards is 11772.
- dense_score gold=0.665332 top1=0.644894 margin=-0.020438
- hybrid_score gold=0.904109 top1=0.970201 margin=0.066092
- gold_rank=29 error_class=wrong_table

### 0c863f592b8b2b1612dbded809c4f631

- query: what was the percentage points were they far less likely to be union members?
- gold: table=1676 (4,0) path=`union membership | regression results > coefficient` serialized: In the table 'estimated relationship between teacher job characteristics and private high school employment', among union membership, the value of regression results > coefficient is -0.605.
- top1: table=1602 (10,2) path=`unemployment rate > non-aboriginal population | difference > percentage points` serialized: In the table 'participation, employment and unemployment rates of people aged 25 to 54 by aboriginal group and marital status, 2015', among unemployment rate > non-aboriginal population, the value of difference > percentage points is -3.6.
- dense_score gold=0.557342 top1=0.594693 margin=0.037351
- hybrid_score gold=0.776644 top1=0.881411 margin=0.104767
- gold_rank=40 error_class=wrong_table

### 0cd55f444a3fad2e0eb52342dfa1ba6c

- query: how many thousand dollars did the nation's 42 federally funded research and development centers (ffrdcs) spend on r&d in fy 2015?
- gold: table=196_92_tab1 (5,0) path=`2015 | all r&d expenditures` serialized: In the table 'r&d expenditures at federally funded research and development centers, by detailed source of funds: fys 2010-15', among 2015, the value of all r&d expenditures is 18458257.
- top1: table=55_167_tab03 (0,2) path=`all ffrdcs | all r&d expenditures` serialized: In the table 'total r&d expenditures at federally funded research and development centers, by character of work and ffrdc: fy 2012', among all ffrdcs, the value of all r&d expenditures is 17446036.
- dense_score gold=0.756122 top1=0.758596 margin=0.002474
- hybrid_score gold=0.916744 top1=0.972781 margin=0.056037
- gold_rank=8 error_class=wrong_table

### 145375917385732b4711a4a664c2852b

- query: in 2010, how many percent was the labor force participation rate among seh doctorate holders for women?
- gold: table=50_164_tab3 (3,1) path=`all u.s. seh doctorate holders > sex > female | in labor force > all` serialized: In the table 'employment status of scientists and engineers with u.s. doctoral degrees, by sex, ethnicity, race, and citizenship: 2010', among all u.s. seh doctorate holders > sex > female, the value of in labor force > all is 88.9.
- top1: table=48_164_tab1 (0,4) path=`number in labor force > all seh fields | 2010` serialized: In the table 'number in labor force and unemployment rate for scientists and engineers with u.s. doctoral degrees, by field of doctorate: 2001-10', among number in labor force > all seh fields, the value of 2010 is 709700.0.
- dense_score gold=0.727261 top1=0.761184 margin=0.033923
- hybrid_score gold=0.923342 top1=1.0 margin=0.076658
- gold_rank=19 error_class=wrong_table

### 19940bcad1a9985373e1b831e49b438d

- query: how many rusing yards did alexander get in 2006?
- gold: table=132_totto16111-1 (6,3) path=`seattle seahawks > 2006 | rushing > yds` serialized: In the table 'Shaun Alexander: nfl statistics', among seattle seahawks > 2006, the value of rushing > yds is 896.
- top1: table=402_totto46543-0 (4,7) path=`career total > 41 | receiving > yards` serialized: In the table 'Shaun Alexander: college statistics', among career total > 41, the value of receiving > yards is 792.0.
- dense_score gold=0.59164 top1=0.653695 margin=0.062056
- hybrid_score gold=0.846793 top1=0.985478 margin=0.138686
- gold_rank=13 error_class=wrong_table

### 225abe30f8fd15d7b6a015eb05112e57

- query: what is the percentage of regular force members who were 24 years of age or younger were targeted in the past 12 months?
- gold: table=455 (25,1) path=`age group > 24 years and younger | total sexualized or discriminatory behaviours > percent` serialized: In the table 'canadian armed forces regular force members who experienced targeted sexualized or discriminatory behaviours in the past 12 months, by selected characteristics and types of behaviour, 2018', among age group > 24 years and younger, the value of total sexualized or discriminatory behaviours > percent is 21.1.
- top1: table=449 (21,3) path=`age group > 24 years and younger | 2018 > total victims > percent` serialized: In the table 'canadian armed forces regular force members who were sexually assaulted in the past 12 months, by gender and selected characteristics, 2016 and 2018', among age group > 24 years and younger, the value of 2018 > total victims > percent is 3.9.
- dense_score gold=0.720867 top1=0.752166 margin=0.031299
- hybrid_score gold=0.944312 top1=1.0 margin=0.055688
- gold_rank=9 error_class=wrong_table

### 226e61021f00e07ff8f3a8db1628ca09

- query: what was the percentage of male ecpas who were from southwest asia and africa?
- gold: table=1534 (12,0) path=`% distribution > world region > southwest asia and africa | men > economic class principal applicants > total` serialized: In the table 'selected characteristics of economic class principal applicants in linked immigrant landing file (1980 through 2006)-discharge abstract database (2006/2007 through 2008/2009) and canadian-born in linked 2006 census-discharge abstract database, by sex, population aged 25 to 74, canada excluding quebec and territories', among % distribution > world region > southwest asia and africa, the value of men > economic class principal applicants > total is 12.5.
- top1: table=1404 (23,0) path=`% distribution > source world region > sub-saharan africa | male` serialized: In the table 'percentage distribution of immigrant landing file cohort, by sex, age group, immigration category, landing year, and source world region', among % distribution > source world region > sub-saharan africa, the value of male is 4.8.
- dense_score gold=0.50382 top1=0.634209 margin=0.130389
- hybrid_score gold=0.787648 top1=0.99471 margin=0.207062
- gold_rank=33 error_class=wrong_table

### 281ed5a850dc2a336bedae24f6745d55

- query: what is the percentage of completed possession cases invoving stay or withdrawal, the second common outcome from 2008/2009 to 2011/2012?
- gold: table=933 (7,3) path=`completed cases related to drug possession > total completed cases related to drug possession | stayed/withdrawn > percent` serialized: In the table 'this table displays the results of completed drug-related cases in adult criminal court, by decision, type of offence, and type of drug, canada, 2008/2009 to 2011/2012. the information is grouped by type of offence and type of drug (appearing as row headers), guilty, stayed/withdrawn, acquitted, other and total, calculated using number and percent units of measure (appearing as column headers)', among completed cases related to drug possession > total completed cases related to drug possession, the value of stayed/withdrawn > percent is 48.
- top1: table=1496 (35,3) path=`other federal statute offences > drug possession | stayed/ withdrawn > percent` serialized: In the table 'cases completed in youth court, by type of offence and decision, canada, 2013/2014', among other federal statute offences > drug possession, the value of stayed/ withdrawn > percent is 72.
- dense_score gold=0.665262 top1=0.703274 margin=0.038012
- hybrid_score gold=0.918097 top1=0.930389 margin=0.012292
- gold_rank=2 error_class=wrong_table

### 29d407c8491b4c01c5b30be8334536be

- query: there was a large difference in the proportion of those with a graduate degree, what was the percentage of immigrant owners?
- gold: table=145 (25,1) path=`percent > educational attainment > graduate degree | all private-sector smes > immigrant owned` serialized: In the table 'characteristics of small and medium-sized enterprises and their owners in the study sample', among percent > educational attainment > graduate degree, the value of all private-sector smes > immigrant owned is 21.9.
- top1: table=2656 (3,0) path=`second generation groups > chinese | with a university degree > fathers > percent` serialized: In the table 'university completion rates among immigrant parents of second generation groups', among second generation groups > chinese, the value of with a university degree > fathers > percent is 33.6.
- dense_score gold=0.645109 top1=0.678871 margin=0.033761
- hybrid_score gold=0.942556 top1=0.963397 margin=0.020841
- gold_rank=21 error_class=wrong_table

### 30111d3b2b6ee9afdb8628a42401e76e

- query: what is the averages of income from employment?
- gold: table=460 (11,0) path=`2010 constant dollars > conditional income > labour income | mean` serialized: In the table 'summary statistics of sample', among 2010 constant dollars > conditional income > labour income, the value of mean is 49200.0.
- top1: table=2610 (21,1) path=`2014 dollars > earnings and employment insurance benefits of employees > average employment insurance income received by employees | tercile > middle` serialized: In the table 'profile of economic regions of residence with relatively low and relatively high layoff rates, averages of 2003 to 2013', among 2014 dollars > earnings and employment insurance benefits of employees > average employment insurance income received by employees, the value of tercile > middle is 1117.0.
- dense_score gold=0.651007 top1=0.693117 margin=0.042109
- hybrid_score gold=0.841733 top1=0.968761 margin=0.127028
- gold_rank=23 error_class=wrong_table

### 311f95ac77073b7d720b6fabe1e87365

- query: what was the percentage of male ecpas who were from east asia?
- gold: table=1534 (15,0) path=`% distribution > world region > east asia | men > economic class principal applicants > total` serialized: In the table 'selected characteristics of economic class principal applicants in linked immigrant landing file (1980 through 2006)-discharge abstract database (2006/2007 through 2008/2009) and canadian-born in linked 2006 census-discharge abstract database, by sex, population aged 25 to 74, canada excluding quebec and territories', among % distribution > world region > east asia, the value of men > economic class principal applicants > total is 30.9.
- top1: table=1404 (27,0) path=`% distribution > source world region > east asia | male` serialized: In the table 'percentage distribution of immigrant landing file cohort, by sex, age group, immigration category, landing year, and source world region', among % distribution > source world region > east asia, the value of male is 20.7.
- dense_score gold=0.530162 top1=0.670126 margin=0.139964
- hybrid_score gold=0.654001 top1=0.99674 margin=0.342738
- gold_rank=71 error_class=wrong_table

### 34d78b794062f38ff81f35e4bfea2380

- query: among decedents who had contact with police in 24 months prior to deaths, what is the percentage of those who were employed in each of the 5 years prior to their fatal overdose?
- gold: table=1308 (3,1) path=`percent > years employed in the 5 years prior to death > all five | at least one contact with police in 24 months prior to death` serialized: percent > years employed in the 5 years prior to death > all five > at least one contact with police in 24 months prior to death: 20.0
- top1: table=2720 (8,0) path=`percent > industry > total employed in the five years prior to death | male overdose decedents who died before age 35 > percentage of those with t4 employment` serialized: In the table 'percent of decedents by main industry of employment in the five years prior to death', among percent > industry > total employed in the five years prior to death, the value of male overdose decedents who died before age 35 > percentage of those with t4 employment is 100.0.
- dense_score gold=0.821727 top1=0.813238 margin=-0.008489
- hybrid_score gold=0.970461 top1=0.990131 margin=0.019671
- gold_rank=5 error_class=wrong_table

### 363f9bb3f74c7ec86026ee4e8707dfb8

- query: of these individuals, what was the number in the labor force, which includes those employed full time or part time and those actively seeking work?
- gold: table=48_164_tab1 (0,4) path=`number in labor force > all seh fields | 2010` serialized: In the table 'number in labor force and unemployment rate for scientists and engineers with u.s. doctoral degrees, by field of doctorate: 2001-10', among number in labor force > all seh fields, the value of 2010 is 709700.0.
- top1: table=50_164_tab3 (0,4) path=`all u.s. seh doctorate holders | in labor force > working for pay or profit > part time` serialized: In the table 'employment status of scientists and engineers with u.s. doctoral degrees, by sex, ethnicity, race, and citizenship: 2010', among all u.s. seh doctorate holders, the value of in labor force > working for pay or profit > part time is 9.9.
- dense_score gold=0.565853 top1=0.63958 margin=0.073727
- hybrid_score gold=0.810056 top1=0.993289 margin=0.183233
- gold_rank=86 error_class=wrong_table

### 39d039e3d8d87bddd8f362efeeb62f34

- query: how many receptions did matt asiata get in 2010?
- gold: table=324_totto36813-0 (2,5) path=`2010 | receiving > rec` serialized: In the table 'Matt Asiata: statistics', among 2010, the value of receiving > rec is 32.0.
- top1: table=124_totto14730-2 (2,6) path=`2010 | receiving > yards` serialized: In the table 'Matt Asiata: statistics', among 2010, the value of receiving > yards is 195.0.
- dense_score gold=0.66955 top1=0.703526 margin=0.033976
- hybrid_score gold=0.952464 top1=1.0 margin=0.047536
- gold_rank=4 error_class=wrong_table

## wrong_row (전체 113건 중 20건)

### 03c9e16805ea51611fa700eb5fdd6680

- query: from 2001 to 2016, what was the increasing rate of individuals aged 25 to 64 with a university degree for all groups combined?
- gold: table=729 (5,0) path=`percent > growth from 2001 to 2016 | total` serialized: In the table 'estimated population aged 25 to 64 with at least a bachelor's degree, 2001 to 2016', among percent > growth from 2001 to 2016, the value of total is 66.1.
- top1: table=729 (7,0) path=`percent > individuals with a university degree within each population group > 2001 | total` serialized: In the table 'estimated population aged 25 to 64 with at least a bachelor's degree, 2001 to 2016', among percent > individuals with a university degree within each population group > 2001, the value of total is 19.5.
- dense_score gold=0.705358 top1=0.722251 margin=0.016894
- hybrid_score gold=0.934221 top1=0.991113 margin=0.056892
- gold_rank=7 error_class=wrong_row

### 04587ca45265fc06f3eec539e134096d

- query: what is the percentage of households consisting of one individual to the refugees from all the other countries excluding syria?
- gold: table=246 (6,1) path=`total, excluding syria | household > proportion of households with only one person > percent` serialized: In the table 'household and family characteristics of refugees who resettled in canada between january 1, 2015, and may 10, 2016, by source country, 2016', among total, excluding syria, the value of household > proportion of households with only one person > percent is 12.0.
- top1: table=246 (0,1) path=`syria | household > proportion of households with only one person > percent` serialized: In the table 'household and family characteristics of refugees who resettled in canada between january 1, 2015, and may 10, 2016, by source country, 2016', among syria, the value of household > proportion of households with only one person > percent is 3.6.
- dense_score gold=0.7673 top1=0.784748 margin=0.017448
- hybrid_score gold=0.979901 top1=0.984731 margin=0.00483
- gold_rank=2 error_class=wrong_row

### 0b1861a08ec699c36197279d2a3198a6

- query: what is the proportion of non-vaccinated persons among adults aged 18 to 64 years with no cmc?
- gold: table=2384 (0,0) path=`total | %` serialized: In the table 'proportion of unvaccinated persons and determinants of non-vaccination for seasonal influenza among canadians aged 18 to 64 without a cc, 2013/2014', among total, the value of % is 77.8.
- top1: table=2384 (4,0) path=`age > 45 to 64 years | %` serialized: In the table 'proportion of unvaccinated persons and determinants of non-vaccination for seasonal influenza among canadians aged 18 to 64 without a cc, 2013/2014', among age > 45 to 64 years, the value of % is 71.7.
- dense_score gold=0.765899 top1=0.782068 margin=0.016169
- hybrid_score gold=0.949834 top1=0.997476 margin=0.047642
- gold_rank=32 error_class=wrong_row

### 0d5a890a5fcb7a4d5c8c219219c88ff9

- query: during that year, what was the percentage of women who were not a visible minority reported working mostly or entirely part time?
- gold: table=766 (0,1) path=`percentage > total age groups | women > not a visible minority` serialized: In the table 'this table displays the results of part-time employment among population aged 15 and over who worked in 2010. the information is grouped by age group (appearing as row headers), women, men, visible minority and not a visible minority, calculated using percentage units of measure (appearing as column headers)', among percentage > total age groups, the value of women > not a visible minority is 29.5.
- top1: table=766 (3,1) path=`percentage > total age groups > 55 to 64 | women > not a visible minority` serialized: In the table 'this table displays the results of part-time employment among population aged 15 and over who worked in 2010. the information is grouped by age group (appearing as row headers), women, men, visible minority and not a visible minority, calculated using percentage units of measure (appearing as column headers)', among percentage > total age groups > 55 to 64, the value of women > not a visible minority is 29.7.
- dense_score gold=0.686378 top1=0.692181 margin=0.005803
- hybrid_score gold=0.969687 top1=0.971378 margin=0.001691
- gold_rank=3 error_class=wrong_row

### 0f306293b1b671e908ec7bbc5e3cb4fa

- query: what was the percentage of admissions of female youth to secure custody in youth correctional services in 2014/2015?
- gold: table=347 (9,3) path=`total | custody > secure custody > percent female` serialized: In the table 'admissions to youth correctional services by type of supervision and jurisdiction, 2014/2015', among total, the value of custody > secure custody > percent female is 16.
- top1: table=347 (3,3) path=`ontario | custody > secure custody > percent female` serialized: In the table 'admissions to youth correctional services by type of supervision and jurisdiction, 2014/2015', among ontario, the value of custody > secure custody > percent female is 14.
- dense_score gold=0.798265 top1=0.800346 margin=0.00208
- hybrid_score gold=0.997788 top1=1.0 margin=0.002212
- gold_rank=2 error_class=wrong_row

### 0f72827e8290b01032419840923eed47

- query: list the total number (local, state, and federal) under the adult correctional supervision of the u.s. states.
- gold: table=76_totto8750-1 (53,0) path=`u.s. total | total > none` serialized: In the table 'List of U.S. states by incarceration and correctional supervision rate: correctional supervision rates by state', among u.s. total, the value of total > none is 6582100.0.
- top1: table=76_totto8750-1 (51,0) path=`state or federal district | total > none` serialized: In the table 'List of U.S. states by incarceration and correctional supervision rate: correctional supervision rates by state', among state or federal district, the value of total > none is 6262000.0.
- dense_score gold=0.744784 top1=0.744652 margin=-0.000132
- hybrid_score gold=0.986007 top1=0.999833 margin=0.013826
- gold_rank=4 error_class=wrong_row

### 14280bd73af437e8aa4b9b3ceed16596

- query: in 2013-2014, what was the percentage of younger canadians (12 to 44 years) who got a flu vaccination?
- gold: table=350 (0,1) path=`percent > 12 to 44 | 2013-2014` serialized: In the table 'flu vaccinationnote 1 rates by age group, canada, 2003 and 2013-2014', among percent > 12 to 44, the value of 2013-2014 is 19.
- top1: table=350 (3,1) path=`percent > 12 to 44 > 35 to 44 | 2013-2014` serialized: In the table 'flu vaccinationnote 1 rates by age group, canada, 2003 and 2013-2014', among percent > 12 to 44 > 35 to 44, the value of 2013-2014 is 22.
- dense_score gold=0.809181 top1=0.813 margin=0.003819
- hybrid_score gold=0.984544 top1=1.0 margin=0.015456
- gold_rank=3 error_class=wrong_row

### 15ef07a1e7916da2207b2a0bad45bf22

- query: how many million dollars did total bussiness r&d perform in domestic business r&d in 2016?
- gold: table=102_53_nsf19315-tab001 (5,0) path=`2016 > all industries | total business r&d` serialized: In the table 'total domestic business r&d and software r&d expenditures: 2006 and 2016', among 2016 > all industries, the value of total business r&d is 374685.0.
- top1: table=102_53_nsf19315-tab001 (6,0) path=`2016 > all industries > manufacturing | total business r&d` serialized: In the table 'total domestic business r&d and software r&d expenditures: 2006 and 2016', among 2016 > all industries > manufacturing, the value of total business r&d is 250553.0.
- dense_score gold=0.709512 top1=0.716536 margin=0.007024
- hybrid_score gold=0.990548 top1=0.99632 margin=0.005773
- gold_rank=2 error_class=wrong_row

### 16697a4760740c37d8a302e4e2969a65

- query: how many league goals did hwang score for cerezo osaka?
- gold: table=312_totto35881-3 (10,2) path=`cerezo osaka > 1999 | league > goals` serialized: In the table 'Hwang Sun-hong: club', among cerezo osaka > 1999, the value of league > goals is 24.
- top1: table=312_totto35881-3 (11,2) path=`cerezo osaka > total | league > goals` serialized: In the table 'Hwang Sun-hong: club', among cerezo osaka > total, the value of league > goals is 30.
- dense_score gold=0.72659 top1=0.754415 margin=0.027826
- hybrid_score gold=0.965476 top1=1.0 margin=0.034524
- gold_rank=3 error_class=wrong_row

### 1d8c1a9add04d1d1176a110a16c8d377

- query: what is the number of passengers enplaned and deplaned at canadian airports in 2017?
- gold: table=2560 (4,1) path=`total | 2017 > number` serialized: In the table 'passenger and cargo data', among total, the value of 2017 > number is 149641672.0.
- top1: table=2560 (3,1) path=`enplaned and deplaned passengers > other international segments | 2017 > number` serialized: In the table 'passenger and cargo data', among enplaned and deplaned passengers > other international segments, the value of 2017 > number is 32852551.0.
- dense_score gold=0.61746 top1=0.705781 margin=0.088321
- hybrid_score gold=0.703856 top1=0.994647 margin=0.29079
- gold_rank=10 error_class=wrong_row

### 1ff3c02097b27cd214ab4a4da59d4ec0

- query: list the win% of schottenheimer that got in the regular season?
- gold: table=63_totto7242-2 (24,3) path=`sd total | regular season > win %` serialized: In the table 'Marty Schottenheimer: head coaching record', among sd total, the value of regular season > win % is 0.588.
- top1: table=63_totto7242-2 (25,3) path=`nfl total | regular season > win %` serialized: In the table 'Marty Schottenheimer: head coaching record', among nfl total, the value of regular season > win % is 0.613.
- dense_score gold=0.718441 top1=0.722386 margin=0.003946
- hybrid_score gold=0.938947 top1=0.944283 margin=0.005336
- gold_rank=2 error_class=wrong_row

### 221d3b9d9f3485104601cc2727338bbb

- query: how many percent of male youth did aboriginal male youth account for admissions to custody in the provinces and territories in 2016/2017?
- gold: table=2215 (12,0) path=`percent > total-all jurisdictions | 2016/2017 > male > aboriginal` serialized: In the table 'admissions of youth to custody, by aboriginal identity, sex and jurisdiction, 2016/2017', among percent > total-all jurisdictions, the value of 2016/2017 > male > aboriginal is 47.0.
- top1: table=2215 (10,0) path=`percent > northwest territories | 2016/2017 > male > aboriginal` serialized: In the table 'admissions of youth to custody, by aboriginal identity, sex and jurisdiction, 2016/2017', among percent > northwest territories, the value of 2016/2017 > male > aboriginal is 94.0.
- dense_score gold=0.808151 top1=0.819104 margin=0.010953
- hybrid_score gold=0.971983 top1=1.0 margin=0.028017
- gold_rank=4 error_class=wrong_row

### 25a4efcf51b27b30d9ccec0aea79225f

- query: what was the mean urinary bpa concentration of the sample?
- gold: table=2708 (0,0) path=`total | geometric mean` serialized: In the table 'geometric means of urinary bpa concentrations, by selected characteristics, household population aged 6 to 17, canada, 2007/2009 and 2009/2011', among total, the value of geometric mean is 1.3.
- top1: table=2708 (11,0) path=`time of urine collection > afternoon | geometric mean` serialized: In the table 'geometric means of urinary bpa concentrations, by selected characteristics, household population aged 6 to 17, canada, 2007/2009 and 2009/2011', among time of urine collection > afternoon, the value of geometric mean is 1.5.
- dense_score gold=0.722234 top1=0.743283 margin=0.021048
- hybrid_score gold=0.963708 top1=0.980437 margin=0.01673
- gold_rank=6 error_class=wrong_row

### 269dd1fd590e4d39eadf91213551194e

- query: in 2019, how many total drug offences reported by police?
- gold: table=1340 (0,2) path=`total drug offences | 2019 > number` serialized: In the table 'police-reported crime for selected drug offences, canada, 2018 and 2019', among total drug offences, the value of 2019 > number is 70140.
- top1: table=1340 (11,2) path=`total other drug offences | 2019 > number` serialized: In the table 'police-reported crime for selected drug offences, canada, 2018 and 2019', among total other drug offences, the value of 2019 > number is 53272.
- dense_score gold=0.796817 top1=0.809715 margin=0.012898
- hybrid_score gold=0.972765 top1=0.983364 margin=0.010599
- gold_rank=2 error_class=wrong_row

### 2b84bf96cf8f664b83ec64d36c8dad86

- query: within the population not belonging to a visible minority group, what was the prevalence of low income among women and girls?
- gold: table=771 (2,0) path=`percentage > total age groups > not a visible minority | total > female` serialized: In the table 'this table displays the results of prevalence of low income. the information is grouped by low income (appearing as row headers), total, canadian-born, immigrant, female and male, calculated using percentage units of measure (appearing as column headers)', among percentage > total age groups > not a visible minority, the value of total > female is 14.3.
- top1: table=771 (11,0) path=`percentage > 25 to 54 years > not a visible minority | total > female` serialized: In the table 'this table displays the results of prevalence of low income. the information is grouped by low income (appearing as row headers), total, canadian-born, immigrant, female and male, calculated using percentage units of measure (appearing as column headers)', among percentage > 25 to 54 years > not a visible minority, the value of total > female is 12.7.
- dense_score gold=0.646403 top1=0.654902 margin=0.008498
- hybrid_score gold=0.937757 top1=0.955057 margin=0.017301
- gold_rank=10 error_class=wrong_row

### 2bbc173cf299fada9212f2527aaa887d

- query: what is the percentage of rc-prs from the 1999 cohort collected social assistance one year after initiating their claim?
- gold: table=1219 (1,0) path=`percent > refugee claimants-permanent residents > 1999 | number of years since landing or filing a refugee claim > 1` serialized: In the table 'percentage receiving social assistance income, by cohort', among percent > refugee claimants-permanent residents > 1999, the value of number of years since landing or filing a refugee claim > 1 is 73.8.
- top1: table=1219 (40,0) path=`percent > refugee claimants-non-permanent residents > 1999 | number of years since landing or filing a refugee claim > 1` serialized: In the table 'percentage receiving social assistance income, by cohort', among percent > refugee claimants-non-permanent residents > 1999, the value of number of years since landing or filing a refugee claim > 1 is 74.0.
- dense_score gold=0.69741 top1=0.710643 margin=0.013233
- hybrid_score gold=0.978115 top1=0.995655 margin=0.01754
- gold_rank=6 error_class=wrong_row

### 2f3ca3561f50935f733b10259676a24f

- query: how many goals did sharp score in 125 league appearances?
- gold: table=298_totto34389-3 (10,1) path=`barnsley > total | league > goals` serialized: In the table 'Frank Sharp (footballer, born 1947): statistics', among barnsley > total, the value of league > goals is 7.
- top1: table=298_totto34389-3 (13,1) path=`career total | league > goals` serialized: In the table 'Frank Sharp (footballer, born 1947): statistics', among career total, the value of league > goals is 12.
- dense_score gold=0.64831 top1=0.731167 margin=0.082857
- hybrid_score gold=0.89011 top1=0.994161 margin=0.104051
- gold_rank=8 error_class=wrong_row

### 314ad3e52972a43965d5dc08b47ee45d

- query: how many percentage points did conduct a conversation in inuktut amount to?
- gold: table=1771 (5,1) path=`mitigating factors > knowledge of mother tongue without using it at home | percentage points` serialized: In the table 'devitalizing factors for inuktut and mitigating factors, nunavut, 2016', among mitigating factors > knowledge of mother tongue without using it at home, the value of percentage points is 1.6.
- top1: table=1771 (6,1) path=`mitigating factors > use of inuktut at home among inuit who do not report it as mother tongue | percentage points` serialized: In the table 'devitalizing factors for inuktut and mitigating factors, nunavut, 2016', among mitigating factors > use of inuktut at home among inuit who do not report it as mother tongue, the value of percentage points is 11.4.
- dense_score gold=0.704534 top1=0.724951 margin=0.020416
- hybrid_score gold=0.938084 top1=0.993273 margin=0.055189
- gold_rank=10 error_class=wrong_row

### 31f8ca62248d6196bb3dc215f3ff936a

- query: in 2012, how many percentge point of aboriginal women aged 15 or older reported disabilities that limited their daily activities?
- gold: table=1569 (0,1) path=`percent > 15 or older | women > aboriginal` serialized: In the table 'prevalence of disability among women and men aged 15 or older, by aboriginal identity and age group, canada, 2011', among percent > 15 or older, the value of women > aboriginal is 22.0.
- top1: table=1569 (4,1) path=`percent > 15 or older > 65 or older | women > aboriginal` serialized: In the table 'prevalence of disability among women and men aged 15 or older, by aboriginal identity and age group, canada, 2011', among percent > 15 or older > 65 or older, the value of women > aboriginal is 53.4.
- dense_score gold=0.731413 top1=0.737678 margin=0.006265
- hybrid_score gold=0.987245 top1=0.999061 margin=0.011816
- gold_rank=3 error_class=wrong_row

### 38df3aa1ba261b0946b319dc82c61115

- query: what was the percentage of users not followed the exposure schedule?
- gold: table=557 (32,0) path=`followed exposure schedule in past year > no | %` serialized: In the table 'factors involved in tanning equipment use, household population aged 12 or older, canada excluding territories, 2014', among followed exposure schedule in past year > no, the value of % is 18.4.
- top1: table=557 (35,0) path=`reasons for not following exposure schedule > no exposure schedule | %` serialized: In the table 'factors involved in tanning equipment use, household population aged 12 or older, canada excluding territories, 2014', among reasons for not following exposure schedule > no exposure schedule, the value of % is 22.3.
- dense_score gold=0.637534 top1=0.687858 margin=0.050324
- hybrid_score gold=0.897933 top1=0.991906 margin=0.093973
- gold_rank=9 error_class=wrong_row

## wrong_column (전체 143건 중 20건)

### 00d7a4e4b65ec4c28d61938772ea74cf

- query: what was the amount exports were revised upward for 2015?
- gold: table=1516 (41,2) path=`millions of dollars > 2015 > exports | average revision` serialized: In the table 'revisions to expenditure-based gross domestic product components', among millions of dollars > 2015 > exports, the value of average revision is 1803.0.
- top1: table=1516 (41,0) path=`millions of dollars > 2015 > exports | revised average level` serialized: In the table 'revisions to expenditure-based gross domestic product components', among millions of dollars > 2015 > exports, the value of revised average level is 627234.0.
- dense_score gold=0.702563 top1=0.75869 margin=0.056126
- hybrid_score gold=0.831255 top1=1.0 margin=0.168745
- gold_rank=6 error_class=wrong_column

### 026d4b2395bfabf7a254b5d84b9ee1f0

- query: among all types of birth, what is the percentage of births surviving to age 1?
- gold: table=1312 (2,2) path=`type of birth > surviving to age 1 | linkage rate > %` serialized: In the table 'comparison of cohort and in-scope births, by selected birth and maternal characteristics, canada, 2004 to 2006', among type of birth > surviving to age 1, the value of linkage rate > % is 90.5.
- top1: table=1312 (2,4) path=`type of birth > surviving to age 1 | birth-census cohort > % distribution` serialized: In the table 'comparison of cohort and in-scope births, by selected birth and maternal characteristics, canada, 2004 to 2006', among type of birth > surviving to age 1, the value of birth-census cohort > % distribution is 98.9.
- dense_score gold=0.673812 top1=0.699077 margin=0.025265
- hybrid_score gold=0.961689 top1=0.992373 margin=0.030684
- gold_rank=4 error_class=wrong_column

### 098a10fc3e91f99daffeceeb0a644028

- query: what was the amount business gross fixed capital formation was revised upward for 2014?
- gold: table=1516 (26,2) path=`millions of dollars > 2014 > business gross fixed capital formation | average revision` serialized: In the table 'revisions to expenditure-based gross domestic product components', among millions of dollars > 2014 > business gross fixed capital formation, the value of average revision is 14330.0.
- top1: table=1516 (26,0) path=`millions of dollars > 2014 > business gross fixed capital formation | revised average level` serialized: In the table 'revisions to expenditure-based gross domestic product components', among millions of dollars > 2014 > business gross fixed capital formation, the value of revised average level is 404337.0.
- dense_score gold=0.773739 top1=0.783169 margin=0.00943
- hybrid_score gold=0.954627 top1=1.0 margin=0.045373
- gold_rank=4 error_class=wrong_column

### 0ad8a6a058610ceace633cbd60f25f2e

- query: in 2015, how many interceptions did jacoby brissett pass for?
- gold: table=241_totto27420-4 (3,6) path=`2015 > nc state | passing > int` serialized: In the table 'Jacoby Brissett: statistics', among 2015 > nc state, the value of passing > int is 6.
- top1: table=241_totto27420-4 (3,3) path=`2015 > nc state | passing > yds` serialized: In the table 'Jacoby Brissett: statistics', among 2015 > nc state, the value of passing > yds is 2662.
- dense_score gold=0.646996 top1=0.658713 margin=0.011717
- hybrid_score gold=0.947529 top1=0.963674 margin=0.016145
- gold_rank=5 error_class=wrong_column

### 0c0e67cf0971bb34af10887f30fcea65

- query: what are the percentages of senior women and men living in private households rated their mental health in positive terms in the 2013-2014 canadian community health survey.
- gold: table=1626 (0,0) path=`percentage > positive | total - 65 years and over > women` serialized: In the table 'self-reported mental health of senior women and men, by age group, canada, 2013-2014', among percentage > positive, the value of total - 65 years and over > women is 94.3.
- top1: table=1626 (0,6) path=`percentage > positive | 85 years and over > women` serialized: In the table 'self-reported mental health of senior women and men, by age group, canada, 2013-2014', among percentage > positive, the value of 85 years and over > women is 93.9.
- dense_score gold=0.809339 top1=0.806925 margin=-0.002414
- hybrid_score gold=0.99555 top1=0.997287 margin=0.001737
- gold_rank=3 error_class=wrong_column

### 0c221c1ce1ef03c3f187b4c5b5ab7c1d

- query: what are the percentage of the shares for the goods-only?
- gold: table=549 (0,3) path=`percent > goods only | firm structure > multiple province` serialized: In the table 'operating locations by firm structure and sector', among percent > goods only, the value of firm structure > multiple province is 33.3.
- top1: table=549 (0,0) path=`percent > goods only | firm structure > total` serialized: In the table 'operating locations by firm structure and sector', among percent > goods only, the value of firm structure > total is 100.0.
- dense_score gold=0.615744 top1=0.671174 margin=0.05543
- hybrid_score gold=0.773344 top1=0.859682 margin=0.086338
- gold_rank=103 error_class=wrong_column

### 0e443a1c5103e84d2d05fa0bef5e4ec7

- query: in 2017, how many firearm-related homicides reported in canada?
- gold: table=1663 (13,0) path=`canada | firearm-related > 2017 > number` serialized: In the table 'homicides, by firearm-related status and province or territory, 2016 and 2017', among canada, the value of firearm-related > 2017 > number is 266.0.
- top1: table=1663 (13,3) path=`canada | firearm-related > 2017 > number of total homicides` serialized: In the table 'homicides, by firearm-related status and province or territory, 2016 and 2017', among canada, the value of firearm-related > 2017 > number of total homicides is 660.
- dense_score gold=0.823209 top1=0.845065 margin=0.021856
- hybrid_score gold=0.967991 top1=1.0 margin=0.032009
- gold_rank=3 error_class=wrong_column

### 0edb0c487809c68313f220288ec0ab52

- query: what was the amount that gross operating surplus was adjusted upward by for 2014?
- gold: table=1515 (16,2) path=`millions of dollars > 2014 > gross operating surplus | average revision` serialized: In the table 'revisions to income-based gross domestic product components', among millions of dollars > 2014 > gross operating surplus, the value of average revision is 10446.0.
- top1: table=1515 (16,0) path=`millions of dollars > 2014 > gross operating surplus | revised average level` serialized: In the table 'revisions to income-based gross domestic product components', among millions of dollars > 2014 > gross operating surplus, the value of revised average level is 553497.0.
- dense_score gold=0.707763 top1=0.734952 margin=0.027189
- hybrid_score gold=0.960164 top1=0.993167 margin=0.033003
- gold_rank=3 error_class=wrong_column

### 11f30db17a7c151825da8a03f0fd087d

- query: using employment rates as an example, what was the percent of non-aboriginal people in couples were employed in 2015?
- gold: table=1602 (6,0) path=`employment rate > non-aboriginal population | married or common-law > percent` serialized: In the table 'participation, employment and unemployment rates of people aged 25 to 54 by aboriginal group and marital status, 2015', among employment rate > non-aboriginal population, the value of married or common-law > percent is 84.0.
- top1: table=1602 (6,1) path=`employment rate > non-aboriginal population | single or previously married > percent` serialized: In the table 'participation, employment and unemployment rates of people aged 25 to 54 by aboriginal group and marital status, 2015', among employment rate > non-aboriginal population, the value of single or previously married > percent is 76.9.
- dense_score gold=0.76972 top1=0.779189 margin=0.009469
- hybrid_score gold=0.988027 top1=0.996021 margin=0.007993
- gold_rank=2 error_class=wrong_column

### 16417ec8a3d47e874eeedd059390f099

- query: what are the percentage of the shares for the services-only?
- gold: table=549 (1,3) path=`percent > services only | firm structure > multiple province` serialized: In the table 'operating locations by firm structure and sector', among percent > services only, the value of firm structure > multiple province is 65.7.
- top1: table=549 (1,0) path=`percent > services only | firm structure > total` serialized: In the table 'operating locations by firm structure and sector', among percent > services only, the value of firm structure > total is 100.0.
- dense_score gold=0.578945 top1=0.654389 margin=0.075444
- hybrid_score gold=0.747798 top1=0.859682 margin=0.111883
- gold_rank=74 error_class=wrong_column

### 1bde5a174429e66e66d623e1c0cf66f2

- query: godfrey's best season with brentford came in 1990-91, how many appearances did he make?
- gold: table=204_totto24002-4 (3,9) path=`brentford > 1990-91 | total > apps` serialized: In the table 'Kevin Godfrey (footballer): career statistics', among brentford > 1990-91, the value of total > apps is 46.0.
- top1: table=204_totto24002-4 (3,0) path=`brentford > 1990-91 | league > division` serialized: In the table 'Kevin Godfrey (footballer): career statistics', among brentford > 1990-91, the value of league > division is third division.
- dense_score gold=0.639328 top1=0.675887 margin=0.036559
- hybrid_score gold=0.952083 top1=0.994161 margin=0.042077
- gold_rank=6 error_class=wrong_column

### 1cc3cb53ab7c6e52ab28f18d3bcd307c

- query: what was the percent change of the srr of injury hospitalization for unintentional falls among the aboriginal males?
- gold: table=2441 (13,2) path=`male > unintentional injury > fall | 1991 to 2010 % change` serialized: In the table 'standardized relative risks (ssr) of hospitalization due to injury, by gender and injury cause, aboriginal and total population, british columbia, 1991 to 2010', among male > unintentional injury > fall, the value of 1991 to 2010 % change is -38.9.
- top1: table=2441 (13,9) path=`male > unintentional injury > fall | annual % change` serialized: In the table 'standardized relative risks (ssr) of hospitalization due to injury, by gender and injury cause, aboriginal and total population, british columbia, 1991 to 2010', among male > unintentional injury > fall, the value of annual % change is -3.5.
- dense_score gold=0.811443 top1=0.817738 margin=0.006294
- hybrid_score gold=0.965861 top1=0.979793 margin=0.013931
- gold_rank=11 error_class=wrong_column

### 1f04fd1913de2c6bd1ff58b403506e67

- query: what was the percentage of individuals from households that earn less than $40,000 who said they were a victim of cyberstalking in the last five years?
- gold: table=2701 (41,2) path=`percent > household income > less than $40,000 | cyberstalked, but not cyberbullied` serialized: In the table 'note 1 by various characteristics, 2014', among percent > household income > less than $40,000, the value of cyberstalked, but not cyberbullied is 11.1.
- top1: table=2701 (41,3) path=`percent > household income > less than $40,000 | cyberbullied and cyberstalked` serialized: In the table 'note 1 by various characteristics, 2014', among percent > household income > less than $40,000, the value of cyberbullied and cyberstalked is 7.1.
- dense_score gold=0.720062 top1=0.734888 margin=0.014826
- hybrid_score gold=0.94943 top1=0.972964 margin=0.023534
- gold_rank=2 error_class=wrong_column

### 1f74fb45f3ab2bbd532189825dc92bef

- query: how many touchdowns did davante adams get in 2017?
- gold: table=99_totto11740-4 (3,6) path=`2017 > gb | receiving > td` serialized: In the table 'Davante Adams: regular season', among 2017 > gb, the value of receiving > td is 10.0.
- top1: table=99_totto11740-4 (3,3) path=`2017 > gb | receiving > yds` serialized: In the table 'Davante Adams: regular season', among 2017 > gb, the value of receiving > yds is 885.0.
- dense_score gold=0.690339 top1=0.699416 margin=0.009077
- hybrid_score gold=0.987792 top1=1.0 margin=0.012208
- gold_rank=2 error_class=wrong_column

### 21a5e4a9a813c8aaa602913a5ba013c2

- query: what was the proportion of all workers reported living in lone-parents census family households?
- gold: table=1723 (4,1) path=`total - census family status > lone parent | all workers > percentage` serialized: In the table 'distribution of workers by census family status, for selected industries and occupations, 2016', among total - census family status > lone parent, the value of all workers > percentage is 10.0.
- top1: table=1723 (4,0) path=`total - census family status > lone parent | all workers > count` serialized: In the table 'distribution of workers by census family status, for selected industries and occupations, 2016', among total - census family status > lone parent, the value of all workers > count is 1995250.0.
- dense_score gold=0.737863 top1=0.743434 margin=0.005571
- hybrid_score gold=0.984581 top1=0.99155 margin=0.006969
- gold_rank=2 error_class=wrong_column

### 21bbdbf479cb2b3664ef88a540592992

- query: how many interceptions did kirk cousins make in the 2017 season ?
- gold: table=291_totto33476-4 (5,8) path=`2017 > was | passing > int` serialized: In the table 'Kirk Cousins: regular season', among 2017 > was, the value of passing > int is 13.0.
- top1: table=291_totto33476-4 (5,5) path=`2017 > was | passing > yds` serialized: In the table 'Kirk Cousins: regular season', among 2017 > was, the value of passing > yds is 4093.0.
- dense_score gold=0.677055 top1=0.695149 margin=0.018095
- hybrid_score gold=0.976954 top1=1.0 margin=0.023046
- gold_rank=6 error_class=wrong_column

### 23c3150c94dfb616394283d1beec9ca7

- query: how many did the rate of police strength in toronto, ontario increase in 2016?
- gold: table=358 (0,3) path=`toronto, ont | police officers > percentage change from previous year > percent` serialized: In the table 'municipal police services serving a population of 100,000 or more, canada, 2017', among toronto, ont, the value of police officers > percentage change from previous year > percent is -4.9.
- top1: table=358 (0,0) path=`toronto, ont | 2016 population > number` serialized: In the table 'municipal police services serving a population of 100,000 or more, canada, 2017', among toronto, ont, the value of 2016 population > number is 2876095.0.
- dense_score gold=0.731375 top1=0.745737 margin=0.014362
- hybrid_score gold=0.912475 top1=0.958 margin=0.045525
- gold_rank=11 error_class=wrong_column

### 28f3628abf672ac0297baa46b2a6253c

- query: what is the high boundary of percentage of french-speaking people in quebec is projected to be in 2036 ?
- gold: table=1195 (7,2) path=`percent > quebec > french | 2036 > low immigration` serialized: In the table 'first official language spoken as a percentage of the population, quebec, canada outside quebec and canada, 2011 (estimated) and 2036', among percent > quebec > french, the value of 2036 > low immigration is 83.0.
- top1: table=1195 (7,3) path=`percent > quebec > french | 2036 > high immigration` serialized: In the table 'first official language spoken as a percentage of the population, quebec, canada outside quebec and canada, 2011 (estimated) and 2036', among percent > quebec > french, the value of 2036 > high immigration is 82.0.
- dense_score gold=0.733005 top1=0.771308 margin=0.038303
- hybrid_score gold=0.926688 top1=1.0 margin=0.073312
- gold_rank=26 error_class=wrong_column

### 28fdf715a93fe9d8f0e6a82892333b09

- query: what was the percentage change in foreign control in the science-based sector after 2000?
- gold: table=1979 (5,7) path=`percent > science-based | percentage change in foreign control > between the 1997-to-1999 and 2009-to-2011 periods` serialized: In the table 'average sector output (nominal), average foreign-controlled market shares (nominal), and percentage change in foreign control, by sector', among percent > science-based, the value of percentage change in foreign control > between the 1997-to-1999 and 2009-to-2011 periods is -8.2.
- top1: table=1979 (5,6) path=`percent > science-based | percentage change in foreign control > between the 1973-to-1975 and 1997-to-1999 periods` serialized: In the table 'average sector output (nominal), average foreign-controlled market shares (nominal), and percentage change in foreign control, by sector', among percent > science-based, the value of percentage change in foreign control > between the 1973-to-1975 and 1997-to-1999 periods is -30.0.
- dense_score gold=0.722632 top1=0.723808 margin=0.001176
- hybrid_score gold=0.998412 top1=1.0 margin=0.001588
- gold_rank=2 error_class=wrong_column

### 2a70f77106a674050c9938ff8a97ea3c

- query: as with adults, non-aboriginal females made up a greater proportion of custody admissions among youth relative to their male counterparts, how many percent of admission accounted for?
- gold: table=2215 (12,3) path=`percent > total-all jurisdictions | 2016/2017 > female > non-aboriginal` serialized: In the table 'admissions of youth to custody, by aboriginal identity, sex and jurisdiction, 2016/2017', among percent > total-all jurisdictions, the value of 2016/2017 > female > non-aboriginal is 40.0.
- top1: table=2215 (12,1) path=`percent > total-all jurisdictions | 2016/2017 > male > non-aboriginal` serialized: In the table 'admissions of youth to custody, by aboriginal identity, sex and jurisdiction, 2016/2017', among percent > total-all jurisdictions, the value of 2016/2017 > male > non-aboriginal is 53.0.
- dense_score gold=0.782241 top1=0.756937 margin=-0.025304
- hybrid_score gold=0.957429 top1=0.965578 margin=0.00815
- gold_rank=2 error_class=wrong_column

## gold_rank > 20 (전체 79건, 전부 표시)

### 00549126c06b99604e8dfb85cc2f7f3f (rank=180 (>20))

- query: in fy 2016, how many thousand dollars was r&d support?
- gold: table=151_72_tab3 [0, 1] gold_table_in_context=0 serialized: In the table 'federal obligations for science and engineering to historically black colleges and universities, ranked by total amount received, by type of activity: fy 2016', among all hbcus, the value of r&d is 257894.2.
- top1(this query's own): table=106_55_fs17-ib-19314-tab001 [16, 1] serialized: In the table 'federal science and engineering obligations, by type of activity: fys 2009-17', among constant fy 2012 $millions > 2016, the value of r&d is 26939.
- dense_score gold=0.6074544191360474 top1=0.6671310067176819
- hybrid_score gold=0.8224200010299683 top1=0.9459004998207092

### 08147555888ee98eea4891c287470047 (rank=448 (>20))

- query: how many million dollars did funding for research and development to universities and colleges increase in 2017?
- gold: table=106_55_fs17-ib-19314-tab001 [8, 1] gold_table_in_context=0 serialized: In the table 'federal science and engineering obligations, by type of activity: fys 2009-17', among 2017, the value of r&d is 29796.
- top1(this query's own): table=2547 [5, 1] serialized: In the table 'donor rate and amounts donated to different types of organizations, 2013', among total > universities and colleges, the value of total amount donated > millions of dollars is 161.0.
- dense_score gold=0.5999544858932495 top1=0.636316180229187
- hybrid_score gold=0.6807629466056824 top1=0.938630223274231

### 0a2938b5924faf566b2ccd1061dc1412 (rank=34 (>20))

- query: how many billion dollars of r&d did companies perform in the united states in 2015?
- gold: table=92_48_nsf19322-tab001 [0, 1] gold_table_in_context=0 serialized: In the table 'domestic r&d performed by companies, by core-based statistical areas with at least $1 billion and source of funds: 2015', among all locations, the value of total is 355821.
- top1(this query's own): table=16_118_tab1 [1, 1] serialized: In the table 'business r&d performed in the united states, by source of funds and mnc status: 2008-10', among u.s.-located companies > domestic r&d performance, the value of 2009 is 282393.0.
- dense_score gold=0.6843369007110596 top1=0.7057356834411621
- hybrid_score gold=0.9199666976928711 top1=0.9839035272598267

### 0b1861a08ec699c36197279d2a3198a6 (rank=32 (>20))

- query: what is the proportion of non-vaccinated persons among adults aged 18 to 64 years with no cmc?
- gold: table=2384 [0, 0] gold_table_in_context=1 serialized: In the table 'proportion of unvaccinated persons and determinants of non-vaccination for seasonal influenza among canadians aged 18 to 64 without a cc, 2013/2014', among total, the value of % is 77.8.
- top1(this query's own): table=2384 [4, 0] serialized: In the table 'proportion of unvaccinated persons and determinants of non-vaccination for seasonal influenza among canadians aged 18 to 64 without a cc, 2013/2014', among age > 45 to 64 years, the value of % is 71.7.
- dense_score gold=0.7658994793891907 top1=0.7820684909820557
- hybrid_score gold=0.9498342871665955 top1=0.9974761009216309

### 0c04b94b7162d791a417ba4fedd934b5 (rank=29 (>20))

- query: how many yards did michael bishop finish by playing two seasons?
- gold: table=231_totto26510-4 [2, 3] gold_table_in_context=0 serialized: In the table 'Michael Bishop (gridiron football): college statistics', among career, the value of passing > yds is 4401.0.
- top1(this query's own): table=273_totto31153-4 [13, 3] serialized: In the table 'Michael Bishop (gridiron football): nfl, nfle and cfl statistics', among 125, the value of passing > yards is 11772.
- dense_score gold=0.6653319597244263 top1=0.6448941230773926
- hybrid_score gold=0.9041091203689575 top1=0.9702008962631226

### 0c221c1ce1ef03c3f187b4c5b5ab7c1d (rank=103 (>20))

- query: what are the percentage of the shares for the goods-only?
- gold: table=549 [0, 3] gold_table_in_context=1 serialized: In the table 'operating locations by firm structure and sector', among percent > goods only, the value of firm structure > multiple province is 33.3.
- top1(this query's own): table=549 [0, 0] serialized: In the table 'operating locations by firm structure and sector', among percent > goods only, the value of firm structure > total is 100.0.
- dense_score gold=0.6157441139221191 top1=0.6711742877960205
- hybrid_score gold=0.7733438014984131 top1=0.8596817255020142

### 0c863f592b8b2b1612dbded809c4f631 (rank=40 (>20))

- query: what was the percentage points were they far less likely to be union members?
- gold: table=1676 [4, 0] gold_table_in_context=0 serialized: In the table 'estimated relationship between teacher job characteristics and private high school employment', among union membership, the value of regression results > coefficient is -0.605.
- top1(this query's own): table=1602 [10, 2] serialized: In the table 'participation, employment and unemployment rates of people aged 25 to 54 by aboriginal group and marital status, 2015', among unemployment rate > non-aboriginal population, the value of difference > percentage points is -3.6.
- dense_score gold=0.5573422312736511 top1=0.5946928262710571
- hybrid_score gold=0.7766438722610474 top1=0.8814113140106201

### 16417ec8a3d47e874eeedd059390f099 (rank=74 (>20))

- query: what are the percentage of the shares for the services-only?
- gold: table=549 [1, 3] gold_table_in_context=1 serialized: In the table 'operating locations by firm structure and sector', among percent > services only, the value of firm structure > multiple province is 65.7.
- top1(this query's own): table=549 [1, 0] serialized: In the table 'operating locations by firm structure and sector', among percent > services only, the value of firm structure > total is 100.0.
- dense_score gold=0.5789451599121094 top1=0.6543893814086914
- hybrid_score gold=0.7477983236312866 top1=0.8596817255020142

### 226e61021f00e07ff8f3a8db1628ca09 (rank=33 (>20))

- query: what was the percentage of male ecpas who were from southwest asia and africa?
- gold: table=1534 [12, 0] gold_table_in_context=0 serialized: In the table 'selected characteristics of economic class principal applicants in linked immigrant landing file (1980 through 2006)-discharge abstract database (2006/2007 through 2008/2009) and canadian-born in linked 2006 census-discharge abstract database, by sex, population aged 25 to 74, canada excluding quebec and territories', among % distribution > world region > southwest asia and africa, the value of men > economic class principal applicants > total is 12.5.
- top1(this query's own): table=1404 [23, 0] serialized: In the table 'percentage distribution of immigrant landing file cohort, by sex, age group, immigration category, landing year, and source world region', among % distribution > source world region > sub-saharan africa, the value of male is 4.8.
- dense_score gold=0.5038198828697205 top1=0.6342093348503113
- hybrid_score gold=0.7876480221748352 top1=0.9947097897529602

### 2441b3f323dbce0c90db4ce4ebcc2183 (rank=23 (>20))

- query: how many appearances did sergio velazquez make during the 2013-14 primera b nacional season?
- gold: table=560_totto64057-2 [6, 1] gold_table_in_context=1 serialized: In the table 'Sergio Velázquez (footballer, born 1990): club', among huracan > 2013-14, the value of league > apps is 9.
- top1(this query's own): table=560_totto64057-2 [3, 0] serialized: In the table 'Sergio Velázquez (footballer, born 1990): club', among defensa y justicia > 2013-14, the value of league > division is primera b nacional.
- dense_score gold=0.6688201427459717 top1=0.7398072481155396
- hybrid_score gold=0.7745081782341003 top1=0.989748477935791

### 2604be07ad31af6c65ecfea00713e54c (rank=25 (>20))

- query: in 2017, how many percent of women in the u.s. civilian college educated population?
- gold: table=144_69_nscg17-ib19300-tab002 [2, 4] gold_table_in_context=1 serialized: In the table 'demographic characteristics of civilian college graduates, by sex and veteran status: 2017', among race and ethnicity > white, non-hispanic, the value of women > veteran is 54.4.
- top1(this query's own): table=144_69_nscg17-ib19300-tab002 [0, 5] serialized: In the table 'demographic characteristics of civilian college graduates, by sex and veteran status: 2017', among all civilian college graduates, the value of women > nonveteran is 32204000.0.
- dense_score gold=0.6511656045913696 top1=0.7072745561599731
- hybrid_score gold=0.8852011561393738 top1=1.0

### 28f3628abf672ac0297baa46b2a6253c (rank=26 (>20))

- query: what is the high boundary of percentage of french-speaking people in quebec is projected to be in 2036 ?
- gold: table=1195 [7, 2] gold_table_in_context=1 serialized: In the table 'first official language spoken as a percentage of the population, quebec, canada outside quebec and canada, 2011 (estimated) and 2036', among percent > quebec > french, the value of 2036 > low immigration is 83.0.
- top1(this query's own): table=1195 [7, 3] serialized: In the table 'first official language spoken as a percentage of the population, quebec, canada outside quebec and canada, 2011 (estimated) and 2036', among percent > quebec > french, the value of 2036 > high immigration is 82.0.
- dense_score gold=0.7330045700073242 top1=0.7713075876235962
- hybrid_score gold=0.9266877174377441 top1=1.0

### 29d407c8491b4c01c5b30be8334536be (rank=21 (>20))

- query: there was a large difference in the proportion of those with a graduate degree, what was the percentage of immigrant owners?
- gold: table=145 [25, 1] gold_table_in_context=0 serialized: In the table 'characteristics of small and medium-sized enterprises and their owners in the study sample', among percent > educational attainment > graduate degree, the value of all private-sector smes > immigrant owned is 21.9.
- top1(this query's own): table=2656 [3, 0] serialized: In the table 'university completion rates among immigrant parents of second generation groups', among second generation groups > chinese, the value of with a university degree > fathers > percent is 33.6.
- dense_score gold=0.6451094746589661 top1=0.6788709163665771
- hybrid_score gold=0.9425557851791382 top1=0.9633965492248535

### 30111d3b2b6ee9afdb8628a42401e76e (rank=23 (>20))

- query: what is the averages of income from employment?
- gold: table=460 [11, 0] gold_table_in_context=1 serialized: In the table 'summary statistics of sample', among 2010 constant dollars > conditional income > labour income, the value of mean is 49200.0.
- top1(this query's own): table=2610 [21, 1] serialized: In the table 'profile of economic regions of residence with relatively low and relatively high layoff rates, averages of 2003 to 2013', among 2014 dollars > earnings and employment insurance benefits of employees > average employment insurance income received by employees, the value of tercile > middle is 1117.0.
- dense_score gold=0.6510072946548462 top1=0.6931167840957642
- hybrid_score gold=0.8417333960533142 top1=0.9687609076499939

### 311f95ac77073b7d720b6fabe1e87365 (rank=71 (>20))

- query: what was the percentage of male ecpas who were from east asia?
- gold: table=1534 [15, 0] gold_table_in_context=0 serialized: In the table 'selected characteristics of economic class principal applicants in linked immigrant landing file (1980 through 2006)-discharge abstract database (2006/2007 through 2008/2009) and canadian-born in linked 2006 census-discharge abstract database, by sex, population aged 25 to 74, canada excluding quebec and territories', among % distribution > world region > east asia, the value of men > economic class principal applicants > total is 30.9.
- top1(this query's own): table=1404 [27, 0] serialized: In the table 'percentage distribution of immigrant landing file cohort, by sex, age group, immigration category, landing year, and source world region', among % distribution > source world region > east asia, the value of male is 20.7.
- dense_score gold=0.5301622152328491 top1=0.6701261401176453
- hybrid_score gold=0.6540012955665588 top1=0.9967395663261414

### 35a0ecaadd9c3da2665ed28efe512a50 (rank=260 (>20))

- query: what is the percentage of in-scope birth records to all census records?
- gold: table=1312 [0, 2] gold_table_in_context=1 serialized: In the table 'comparison of cohort and in-scope births, by selected birth and maternal characteristics, canada, 2004 to 2006', among total, the value of linkage rate > % is 90.3.
- top1(this query's own): table=1312 [3, 1] serialized: In the table 'comparison of cohort and in-scope births, by selected birth and maternal characteristics, canada, 2004 to 2006', among type of birth > stillbirth, the value of in-scope population > % distribution is 0.6.
- dense_score gold=0.6041452884674072 top1=0.6811553835868835
- hybrid_score gold=0.8073415160179138 top1=0.9611681699752808

### 363f9bb3f74c7ec86026ee4e8707dfb8 (rank=86 (>20))

- query: of these individuals, what was the number in the labor force, which includes those employed full time or part time and those actively seeking work?
- gold: table=48_164_tab1 [0, 4] gold_table_in_context=0 serialized: In the table 'number in labor force and unemployment rate for scientists and engineers with u.s. doctoral degrees, by field of doctorate: 2001-10', among number in labor force > all seh fields, the value of 2010 is 709700.0.
- top1(this query's own): table=50_164_tab3 [0, 4] serialized: In the table 'employment status of scientists and engineers with u.s. doctoral degrees, by sex, ethnicity, race, and citizenship: 2010', among all u.s. seh doctorate holders, the value of in labor force > working for pay or profit > part time is 9.9.
- dense_score gold=0.5658529996871948 top1=0.639579713344574
- hybrid_score gold=0.8100559711456299 top1=0.9932891726493835

### 38dd34ceca312b80673c5eb605e97dec (rank=51 (>20))

- query: in total, how many games did ian taylor play with 103 goas in league and cup competitions?
- gold: table=6_totto667-0 [20, 8] gold_table_in_context=1 serialized: In the table 'Ian Taylor (footballer, born 1968): career statistics', among career total, the value of total > apps is 577.
- top1(this query's own): table=6_totto667-0 [20, 5] serialized: In the table 'Ian Taylor (footballer, born 1968): career statistics', among career total, the value of league cup > goals is 12.
- dense_score gold=0.6442922949790955 top1=0.7078508138656616
- hybrid_score gold=0.8595028519630432 top1=0.9821941256523132

### 3b7e54939c5ce445505a98259ae60dd8 (rank=141 (>20))

- query: larus sigurðssonwas transferred to west bromwich albion in 1999, how many appearances did he make for stoke?
- gold: table=447_totto51418-3 [12, 9] gold_table_in_context=1 serialized: In the table 'Lárus Sigurðsson: club', among stoke city > total, the value of total > apps is 228.0.
- top1(this query's own): table=447_totto51418-3 [13, 10] serialized: In the table 'Lárus Sigurðsson: club', among west bromwich albion > 1999-2000, the value of total > goals is 0.0.
- dense_score gold=0.6094614267349243 top1=0.7119764089584351
- hybrid_score gold=0.7196378707885742 top1=1.0

### 3fab5ae74a35d4849bcac912f052382b (rank=39 (>20))

- query: completed cases involving other types of drugs resulted in guilty decisions more often, what is the percentage of cocaine from 2008/2009 to 2011/2012?
- gold: table=933 [20, 1] gold_table_in_context=1 serialized: In the table 'this table displays the results of completed drug-related cases in adult criminal court, by decision, type of offence, and type of drug, canada, 2008/2009 to 2011/2012. the information is grouped by type of offence and type of drug (appearing as row headers), guilty, stayed/withdrawn, acquitted, other and total, calculated using number and percent units of measure (appearing as column headers)', among completed drug-related cases > other drugs > methamphetamines, the value of guilty > percent is 62.
- top1(this query's own): table=933 [18, 1] serialized: In the table 'this table displays the results of completed drug-related cases in adult criminal court, by decision, type of offence, and type of drug, canada, 2008/2009 to 2011/2012. the information is grouped by type of offence and type of drug (appearing as row headers), guilty, stayed/withdrawn, acquitted, other and total, calculated using number and percent units of measure (appearing as column headers)', among completed drug-related cases > other drugs > cocaine, the value of guilty > percent is 62.
- dense_score gold=0.7181484699249268 top1=0.7489933967590332
- hybrid_score gold=0.9284534454345703 top1=0.9985369443893433

### 42fae71dc7de2ff8ec037019a41dd13d (rank=51 (>20))

- query: what was the proportion of visible minority men and boys living in low-income situations?
- gold: table=771 [1, 1] gold_table_in_context=1 serialized: In the table 'this table displays the results of prevalence of low income. the information is grouped by low income (appearing as row headers), total, canadian-born, immigrant, female and male, calculated using percentage units of measure (appearing as column headers)', among percentage > total age groups > visible minority, the value of total > male is 21.1.
- top1(this query's own): table=766 [0, 2] serialized: In the table 'this table displays the results of part-time employment among population aged 15 and over who worked in 2010. the information is grouped by age group (appearing as row headers), women, men, visible minority and not a visible minority, calculated using percentage units of measure (appearing as column headers)', among percentage > total age groups, the value of men > visible minority is 16.8.
- dense_score gold=0.6503826975822449 top1=0.6045699119567871
- hybrid_score gold=0.8700284361839294 top1=0.912506639957428

### 5839d6a5a46d6c60a36aae0f3884f5e3 (rank=40 (>20))

- query: what was the percentage of females in the cma of calgary belonged to a visible minority group?
- gold: table=751 [18, 1] gold_table_in_context=1 serialized: In the table 'distribution of visible minority females in selected municipalities in the census metropolitan areas of toronto, vancouver, montreal, and calgary, canada, 2011', among calgary cma, the value of as a percent of total female population in each cma/municipality is 28.4.
- top1(this query's own): table=751 [19, 0] serialized: In the table 'distribution of visible minority females in selected municipalities in the census metropolitan areas of toronto, vancouver, montreal, and calgary, canada, 2011', among calgary cma > calgary, the value of as a percent of visible minority females in each cma is 96.5.
- dense_score gold=0.7931828498840332 top1=0.8239911198616028
- hybrid_score gold=0.9146971702575684 top1=1.0

### 584e4d960971437dba9edf31d69a114a (rank=80 (>20))

- query: what was the number of incidents of sexual assault (including founded and unfounded) reported to the police in 2017?
- gold: table=2352 [41, 0] gold_table_in_context=1 serialized: In the table 'police-reported incidents of sexual assault, by clearance status and province or territory, 2016 and 2017', among canada > 2017, the value of reported > number is 28551.0.
- top1(this query's own): table=2352 [17, 1] serialized: In the table 'police-reported incidents of sexual assault, by clearance status and province or territory, 2016 and 2017', among ontario > 2017, the value of unfounded > number is 1498.0.
- dense_score gold=0.7281620502471924 top1=0.7975345849990845
- hybrid_score gold=0.8968929648399353 top1=1.0

### 5dd3fec581f2e2b614b1c9d792723492 (rank=28 (>20))

- query: what was the rate of police strength in winsor in 2017?
- gold: table=358 [24, 2] gold_table_in_context=1 serialized: In the table 'municipal police services serving a population of 100,000 or more, canada, 2017', among windsor, ont, the value of police officers > 2017 police officers per 100,000 population > rate is 193.0.
- top1(this query's own): table=358 [3, 2] serialized: In the table 'municipal police services serving a population of 100,000 or more, canada, 2017', among calgary, alta, the value of police officers > 2017 police officers per 100,000 population > rate is 168.0.
- dense_score gold=0.5934908390045166 top1=0.6132186651229858
- hybrid_score gold=0.931032657623291 top1=0.9625413417816162

### 5f15bb0bdbd99fee28e3a2c174a9376a (rank=46 (>20))

- query: what is the proportion of individuals who claim a disability tax credit?
- gold: table=460 [9, 0] gold_table_in_context=0 serialized: In the table 'summary statistics of sample', among percent > other characteristics > has disability allowances, the value of mean is 0.4.
- top1(this query's own): table=473 [2, 1] serialized: In the table 'family composition of population aged 25 to 64, by disability status, type and severity class, 2014', among percent > disability status > with a disability, the value of couple > one with a disability is 44.7.
- dense_score gold=0.6147308945655823 top1=0.6681851148605347
- hybrid_score gold=0.8500968217849731 top1=1.0

### 61143c012eadeb4070c9f6ce7072fa8b (rank=43 (>20))

- query: according to brdis, what was the total number of employees at r&d-performing companies in 2015?
- gold: table=105_54_nsf19316-tab003 [0, 7] gold_table_in_context=1 serialized: In the table 'domestic total and r&d employment, by company size: 2008-15', among all companies, the value of 2015 is 18913.0.
- top1(this query's own): table=105_54_nsf19316-tab003 [42, 7] serialized: In the table 'domestic total and r&d employment, by company size: 2008-15', among r&d employment % of total employment in r&d performing companies > large companies > 500-999, the value of 2015 is 9.6.
- dense_score gold=0.6225793957710266 top1=0.6899757981300354
- hybrid_score gold=0.8526554107666016 top1=0.9712739586830139

### 6212e7e804400c635133bec9b7196147 (rank=25 (>20))

- query: in 2010, how many percent was the labor force participation rate among seh doctorate holders for men?
- gold: table=50_164_tab3 [2, 1] gold_table_in_context=0 serialized: In the table 'employment status of scientists and engineers with u.s. doctoral degrees, by sex, ethnicity, race, and citizenship: 2010', among all u.s. seh doctorate holders > sex > male, the value of in labor force > all is 87.7.
- top1(this query's own): table=48_164_tab1 [0, 4] serialized: In the table 'number in labor force and unemployment rate for scientists and engineers with u.s. doctoral degrees, by field of doctorate: 2001-10', among number in labor force > all seh fields, the value of 2010 is 709700.0.
- dense_score gold=0.72807776927948 top1=0.7702213525772095
- hybrid_score gold=0.9110041856765747 top1=1.0

### 63728aa6d540ab921547dde59cd18830 (rank=35 (>20))

- query: in 2014/2015, how many cases did canada's youth courts complet involving 120,907 charges related to criminal code and other federal statute offences, including offences related to the ycja?
- gold: table=1913 [9, 2] gold_table_in_context=1 serialized: In the table 'charges and cases completed in youth court, canada, 2005/2006 to 2014/2015', among 2014/2015, the value of total cases > number is 32835.
- top1(this query's own): table=1496 [37, 8] serialized: In the table 'cases completed in youth court, by type of offence and decision, canada, 2013/2014', among other federal statute offences > youth criminal justice act, the value of total cases > number is 3841.
- dense_score gold=0.7154955863952637 top1=0.7742761373519897
- hybrid_score gold=0.9221044778823853 top1=1.0

### 6575d4efd8d4247b85a1218a67a4ca25 (rank=46 (>20))

- query: what was the percentage of women and girls in the cma of montreal belonged to a visible minority group?
- gold: table=751 [12, 1] gold_table_in_context=1 serialized: In the table 'distribution of visible minority females in selected municipalities in the census metropolitan areas of toronto, vancouver, montreal, and calgary, canada, 2011', among montreal cma, the value of as a percent of total female population in each cma/municipality is 20.2.
- top1(this query's own): table=751 [13, 0] serialized: In the table 'distribution of visible minority females in selected municipalities in the census metropolitan areas of toronto, vancouver, montreal, and calgary, canada, 2011', among montreal cma > montreal, the value of as a percent of visible minority females in each cma is 66.8.
- dense_score gold=0.7924147844314575 top1=0.8197301626205444
- hybrid_score gold=0.9201682209968567 top1=0.995685875415802

### 68eaeef17a05268e1f14d7392447f6a3 (rank=56 (>20))

- query: what was the percent among their counterparts in households with five or more people?
- gold: table=1653 [26, 0] gold_table_in_context=0 serialized: In the table 'proportion of caregivers who received support for caregiving in the past 12 months, by sociodemographic characteristics, 2018', among percent > household size > five or more, the value of received support for caregiving in the past 12 months > any type of support is 80.0.
- top1(this query's own): table=246 [5, 1] serialized: In the table 'household and family characteristics of refugees who resettled in canada between january 1, 2015, and may 10, 2016, by source country, 2016', among other countries, the value of household > proportion of households with only one person > percent is 13.4.
- dense_score gold=0.5638667941093445 top1=0.6589981317520142
- hybrid_score gold=0.7802602648735046 top1=0.897455096244812

### 6939086016c92b318fe710ce09fcaea2 (rank=32 (>20))

- query: what was the percentage of men that had employment in all five years?
- gold: table=2719 [5, 0] gold_table_in_context=0 serialized: In the table 'selected income characteristics of decedents over the five years prior to death - t4 earnings', among 5, the value of men > percentage > percent is 27.9.
- top1(this query's own): table=1835 [9, 2] serialized: In the table 'employment rate of individuals aged 25 to 54, with no bachelor's degree, by region, united states, selected years', among men > all regions, the value of 2011 > percent is 77.5.
- dense_score gold=0.6333358287811279 top1=0.7161492109298706
- hybrid_score gold=0.8493795394897461 top1=0.927197277545929

### 693ec89b19900fffdbcd05cd8eb354fb (rank=21 (>20))

- query: how many million dollars was funding from other sources in 2017?
- gold: table=65_20_nsf20316-tab001 [9, 0] gold_table_in_context=1 serialized: In the table 'funds spent for business r&d performed in the united states, by type of r&d, source of funds, and size of company: 2017-18', among domestic r&d performancea > paid for by others, the value of 2017 is 61065.0.
- top1(this query's own): table=56_173_tab1 [0, 6] serialized: In the table 'higher education r&d expenditures, by source of funds: fys 2010-12', among 2010, the value of all other sources is 1048.
- dense_score gold=0.627364993095398 top1=0.6101892590522766
- hybrid_score gold=0.77535080909729 top1=0.8900200128555298

### 6f6c406a0ed987e0fe507bc434c02a46 (rank=78 (>20))

- query: what's the increasing number of the professionals potentially able to provide services in the minority language in 2011 was greater than expected?
- gold: table=707 [0, 8] gold_table_in_context=1 serialized: In the table 'health care professionals present and expected in new brunswick and its regions, 2011', among numbers > total - health care professionals, the value of south-east of the province > difference is 890.
- top1(this query's own): table=707 [3, 7] serialized: In the table 'health care professionals present and expected in new brunswick and its regions, 2011', among numbers > total - health care professionals > psychologists, the value of south-east of the province > expected in 2011 is 120.
- dense_score gold=0.5180069804191589 top1=0.5661701560020447
- hybrid_score gold=0.8345691561698914 top1=0.9490970969200134

### 6fcebcf499a7f7106400ac2d33cdd048 (rank=257 (>20))

- query: what was the percentage of female ecpas were from eastern europe?
- gold: table=1534 [11, 5] gold_table_in_context=0 serialized: In the table 'selected characteristics of economic class principal applicants in linked immigrant landing file (1980 through 2006)-discharge abstract database (2006/2007 through 2008/2009) and canadian-born in linked 2006 census-discharge abstract database, by sex, population aged 25 to 74, canada excluding quebec and territories', among % distribution > world region > eastern europe, the value of women > economic class principal applicants > total is 10.4.
- top1(this query's own): table=1404 [22, 1] serialized: In the table 'percentage distribution of immigrant landing file cohort, by sex, age group, immigration category, landing year, and source world region', among % distribution > source world region > eastern europe, the value of female is 12.6.
- dense_score gold=0.5185306668281555 top1=0.6677360534667969
- hybrid_score gold=0.640292763710022 top1=0.9674844741821289

### 720e2c9c6caa13f0942e5bc2e444d498 (rank=41 (>20))

- query: in the cma of toronto, what was the percentage of females belonged to a visible minority group?
- gold: table=751 [0, 1] gold_table_in_context=1 serialized: In the table 'distribution of visible minority females in selected municipalities in the census metropolitan areas of toronto, vancouver, montreal, and calgary, canada, 2011', among toronto cma, the value of as a percent of total female population in each cma/municipality is 47.6.
- top1(this query's own): table=751 [1, 0] serialized: In the table 'distribution of visible minority females in selected municipalities in the census metropolitan areas of toronto, vancouver, montreal, and calgary, canada, 2011', among toronto cma > toronto, the value of as a percent of visible minority females in each cma is 49.3.
- dense_score gold=0.7958027124404907 top1=0.8253869414329529
- hybrid_score gold=0.9151023030281067 top1=1.0

### 7380ff5eaeb2557b7a549b68f014e2a3 (rank=340 (>20))

- query: what was the percentage of female skilled workers were university graduates (bachelor's degree or more)?
- gold: table=1534 [20, 6] gold_table_in_context=0 serialized: In the table 'selected characteristics of economic class principal applicants in linked immigrant landing file (1980 through 2006)-discharge abstract database (2006/2007 through 2008/2009) and canadian-born in linked 2006 census-discharge abstract database, by sex, population aged 25 to 74, canada excluding quebec and territories', among % distribution > landing year > university graduation, the value of women > economic class principal applicants > skilled workers is 65.7.
- top1(this query's own): table=2297 [4, 2] serialized: In the table 'percentage of highest certificate, diploma or degree of female lone parents and female parents in couples, aged 25 to 54 with children aged 15 and under in 1991, 2001 and 2011, canada', among percent > total > university degree at the bachelor's level or above, the value of female lone parents > 2011 is 20.
- dense_score gold=0.5897620916366577 top1=0.6557785272598267
- hybrid_score gold=0.7434548139572144 top1=0.974522590637207

### 76f11e0e5ab80386f96fcf2f3ae0e651 (rank=39 (>20))

- query: what was the percentage of manufactures exported in 2011 and 2012?
- gold: table=212 [0, 2] gold_table_in_context=0 serialized: In the table 'import or export participation rates of firms in the manufacturing and wholesale trade sectors', among percent > manufacturing, the value of export participation rate > all firms is 20.04.
- top1(this query's own): table=555 [2, 10] serialized: In the table 'earnings premiums by firm characteristics', among exporters relative to non-exporters, the value of 2012 > number is 1.11.
- dense_score gold=0.6469890475273132 top1=0.6876589059829712
- hybrid_score gold=0.7992991805076599 top1=0.8839818239212036

### 78a19d0e2e5a0ad6398d3389b7d46e98 (rank=155 (>20))

- query: what was the percent of all senior caregivers aged 65 and older providing care primarily for a spouse in the past 12 months?
- gold: table=565 [3, 0] gold_table_in_context=0 serialized: In the table 'relationship between the caregiver and their primary care receiver, by caregiver's age, 2018', among 65 and older, the value of primary care receiver > spouse/partner is 34.1.
- top1(this query's own): table=1653 [6, 0] serialized: In the table 'proportion of caregivers who received support for caregiving in the past 12 months, by sociodemographic characteristics, 2018', among percent > age > 65 and older, the value of received support for caregiving in the past 12 months > any type of support is 67.0.
- dense_score gold=0.6151806712150574 top1=0.694942057132721
- hybrid_score gold=0.7710851430892944 top1=0.9777517914772034

### 7cb39f8e9e1f9282f1b169f390345306 (rank=52 (>20))

- query: what was the percentage of women who did not have employment in any of the previous five years?
- gold: table=2719 [0, 3] gold_table_in_context=0 serialized: In the table 'selected income characteristics of decedents over the five years prior to death - t4 earnings', among 0, the value of women > percentage > percent is 51.0.
- top1(this query's own): table=1835 [16, 1] serialized: In the table 'employment rate of individuals aged 25 to 54, with no bachelor's degree, by region, united states, selected years', among women > south central, the value of 2007 > percent is 65.1.
- dense_score gold=0.6195828914642334 top1=0.7330660820007324
- hybrid_score gold=0.7995513081550598 top1=0.8904762268066406

### 7fb14f32b233aebd6e14ffa0d9c08eff (rank=31 (>20))

- query: among visible minority women aged 15 and over, what was the percentage of women were unemployed in the week prior to the 2011 census?
- gold: table=764 [1, 0] gold_table_in_context=0 serialized: In the table 'this table displays the results of unemployment rate. the information is grouped by population (appearing as row headers), total, canadian-born, immigrant, women and men, calculated using percentage units of measure (appearing as column headers)', among percentage > age 15 and over > visible minority, the value of total > women is 10.6.
- top1(this query's own): table=766 [1, 0] serialized: In the table 'this table displays the results of part-time employment among population aged 15 and over who worked in 2010. the information is grouped by age group (appearing as row headers), women, men, visible minority and not a visible minority, calculated using percentage units of measure (appearing as column headers)', among percentage > total age groups > 15 to 24, the value of women > visible minority is 66.2.
- dense_score gold=0.6510893106460571 top1=0.6999475359916687
- hybrid_score gold=0.8944404125213623 top1=1.0

### 810c706ff53b4354f9608002319683b9 (rank=72 (>20))

- query: overall, what was the percentage of visible minority women and girls living in a low-income situation?
- gold: table=771 [1, 0] gold_table_in_context=0 serialized: In the table 'this table displays the results of prevalence of low income. the information is grouped by low income (appearing as row headers), total, canadian-born, immigrant, female and male, calculated using percentage units of measure (appearing as column headers)', among percentage > total age groups > visible minority, the value of total > female is 21.9.
- top1(this query's own): table=751 [2, 0] serialized: In the table 'distribution of visible minority females in selected municipalities in the census metropolitan areas of toronto, vancouver, montreal, and calgary, canada, 2011', among toronto cma > mississauga, the value of as a percent of visible minority females in each cma is 14.5.
- dense_score gold=0.6474452018737793 top1=0.6648697853088379
- hybrid_score gold=0.8662039041519165 top1=0.9529680609703064

### 88f7b783c4cf4c75f8ae9e12b5ce6d36 (rank=53 (>20))

- query: what was the number of appearances made by dave kitson in his career totally?
- gold: table=181_totto21471-3 [21, 1] gold_table_in_context=1 serialized: In the table 'Dave Kitson: career statistics', among career total, the value of league > apps is 420.
- top1(this query's own): table=181_totto21471-3 [21, 2] serialized: In the table 'Dave Kitson: career statistics', among career total, the value of league > goals is 129.
- dense_score gold=0.4963681399822235 top1=0.5759494304656982
- hybrid_score gold=0.8686145544052124 top1=1.0

### 8e41bf48220d3ec31e7eb2367c58d82d (rank=24 (>20))

- query: what was the percent of seniors aged 65 to 74 cared to a spouse?
- gold: table=565 [4, 0] gold_table_in_context=0 serialized: In the table 'relationship between the caregiver and their primary care receiver, by caregiver's age, 2018', among 65 and older > 65 to 74, the value of primary care receiver > spouse/partner is 27.8.
- top1(this query's own): table=1631 [1, 3] serialized: In the table 'primary caregiver of seniors who received unpaid help in the last 12 months, by age group and sex, canada, 2012', among percentage > total > spouse, the value of 65 to 74 years > men is 71.0.
- dense_score gold=0.6520808339118958 top1=0.7105402946472168
- hybrid_score gold=0.8298449516296387 top1=1.0

### 8ec692fe85b783ce8c4caf34842f4335 (rank=102 (>20))

- query: how many percent of the total number of enterprises in canada has increased from 1,028,397 in 2008 to 1,089,136 in 2014?
- gold: table=1077 [9, 4] gold_table_in_context=1 serialized: In the table 'active enterprises with one or more employees by enterprise size class, 2008 and 2014', among total, the value of variation of number is 5.9.
- top1(this query's own): table=1077 [9, 0] serialized: In the table 'active enterprises with one or more employees by enterprise size class, 2008 and 2014', among total, the value of 2008 > number is 1028397.0.
- dense_score gold=0.5782262086868286 top1=0.6666322946548462
- hybrid_score gold=0.8410353064537048 top1=0.9714421033859253

### 98b5f7cb88c5a1ca069c0b45046153db (rank=47 (>20))

- query: what is the final poistion for tigers in the 2017 - 18 champions cup?
- gold: table=7_totto903-4 [46, 1] gold_table_in_context=1 serialized: In the table 'Leicester Tigers: season summary', among 2017-18, the value of league > position is 5th.
- top1(this query's own): table=7_totto903-4 [46, 6] serialized: In the table 'Leicester Tigers: season summary', among 2017-18, the value of european cup > competition is champions cup.
- dense_score gold=0.6643998026847839 top1=0.7586787343025208
- hybrid_score gold=0.7758700251579285 top1=1.0

### 9f39b6d8cc1cb119e605a419d556b8b1 (rank=22 (>20))

- query: how many ers in the upper-third of the layoff rate distribution?
- gold: table=2610 [0, 2] gold_table_in_context=1 serialized: In the table 'profile of economic regions of residence with relatively low and relatively high layoff rates, averages of 2003 to 2013', among number > economic regions, the value of tercile > upper is 23.0.
- top1(this query's own): table=2610 [2, 2] serialized: In the table 'profile of economic regions of residence with relatively low and relatively high layoff rates, averages of 2003 to 2013', among percent > layoff rate, the value of tercile > upper is 12.6.
- dense_score gold=0.6461714506149292 top1=0.6984584927558899
- hybrid_score gold=0.8609879612922668 top1=1.0

### a8cb77d327fed6ce680ed7ad883b9814 (rank=149 (>20))

- query: what was the percentage of male ecpas who were from south asia?
- gold: table=1534 [13, 0] gold_table_in_context=0 serialized: In the table 'selected characteristics of economic class principal applicants in linked immigrant landing file (1980 through 2006)-discharge abstract database (2006/2007 through 2008/2009) and canadian-born in linked 2006 census-discharge abstract database, by sex, population aged 25 to 74, canada excluding quebec and territories', among % distribution > world region > south asia, the value of men > economic class principal applicants > total is 19.1.
- top1(this query's own): table=1404 [25, 0] serialized: In the table 'percentage distribution of immigrant landing file cohort, by sex, age group, immigration category, landing year, and source world region', among % distribution > source world region > south asia, the value of male is 19.3.
- dense_score gold=0.529620885848999 top1=0.6753660440444946
- hybrid_score gold=0.6530461311340332 top1=1.0

### abaf598dc25817efcf67fcd0bac06a1c (rank=43 (>20))

- query: in the cma of vancouver, what was the percentage of all females belonged to a visible minority group?
- gold: table=751 [6, 1] gold_table_in_context=1 serialized: In the table 'distribution of visible minority females in selected municipalities in the census metropolitan areas of toronto, vancouver, montreal, and calgary, canada, 2011', among vancouver cma, the value of as a percent of total female population in each cma/municipality is 46.1.
- top1(this query's own): table=751 [7, 0] serialized: In the table 'distribution of visible minority females in selected municipalities in the census metropolitan areas of toronto, vancouver, montreal, and calgary, canada, 2011', among vancouver cma > vancouver, the value of as a percent of visible minority females in each cma is 30.2.
- dense_score gold=0.7886776328086853 top1=0.8137898445129395
- hybrid_score gold=0.9110998511314392 top1=0.991443932056427

### b06905508e5b9e1598265fcfb1229906 (rank=35 (>20))

- query: what was the number of appearances made by whelan for stoke?
- gold: table=157_totto18855-3 [20, 9] gold_table_in_context=1 serialized: In the table 'Glenn Whelan: club', among stoke city > total, the value of total > apps is 338.0.
- top1(this query's own): table=157_totto18855-3 [20, 2] serialized: In the table 'Glenn Whelan: club', among stoke city > total, the value of league > goals is 5.
- dense_score gold=0.6377625465393066 top1=0.67303466796875
- hybrid_score gold=0.9372819662094116 top1=0.984430193901062

### b2159bd13b094c55c37ac20154b4f3db (rank=57 (>20))

- query: how many thousand dollars did expenditures funded by business-funded r&d increase to in fy 2012?
- gold: table=56_173_tab1 [2, 4] gold_table_in_context=0 serialized: In the table 'higher education r&d expenditures, by source of funds: fys 2010-12', among 2012, the value of business is 3282.
- top1(this query's own): table=55_167_tab03 [1, 2] serialized: In the table 'total r&d expenditures at federally funded research and development centers, by character of work and ffrdc: fy 2012', among university administered, the value of all r&d expenditures is 5174091.
- dense_score gold=0.7156135439872742 top1=0.6966146230697632
- hybrid_score gold=0.8507596850395203 top1=0.9397889375686646

### b996440119231357e1e3334a12cb2b64 (rank=263 (>20))

- query: what is the overall percentage of perfectly matched postal codes?
- gold: table=626 [6, 4] gold_table_in_context=0 serialized: In the table 'performance of imputation for experiments a and b', among experiment a > total - experiment a, the value of perfect matches > percent is 76.4.
- top1(this query's own): table=2022 [0, 6] serialized: In the table 'percentage of the canadian population matched to one dissemination area or more than one dissemination area in the postal code conversion file plus, by postal code characteristics', among canada, the value of percentage of the population with postal codes, by number of dissemination area links > missing > percent is 0.12.
- dense_score gold=0.5977804064750671 top1=0.7038052082061768
- hybrid_score gold=0.6581695079803467 top1=0.9970598220825195

### ba8d1b2dabb97dc7333783ced6df77bf (rank=108 (>20))

- query: on average, how many dollars did families hold in resps including families who did not hold an resp?
- gold: table=282 [0, 1] gold_table_in_context=1 serialized: In the table 'registered education savings plan holdings of economic families with children', among overall, the value of 1999 > mean value of resps > 2012 constant dollars is 1325.0.
- top1(this query's own): table=282 [12, 2] serialized: In the table 'registered education savings plan holdings of economic families with children', among net worth (less resps) quintile > top, the value of 2012 > have an resp > proportion is 0.732.
- dense_score gold=0.6762675046920776 top1=0.6995913982391357
- hybrid_score gold=0.8898802399635315 top1=0.9908267855644226

### c04336aaa39d30cba448c6a4ae017745 (rank=31 (>20))

- query: what was the percentage of wholesalers exported in 2011 and 2012?
- gold: table=212 [1, 2] gold_table_in_context=0 serialized: In the table 'import or export participation rates of firms in the manufacturing and wholesale trade sectors', among percent > wholesale trade, the value of export participation rate > all firms is 10.42.
- top1(this query's own): table=2439 [1, 3] serialized: In the table 'note 1 of cannabis by wholesalers, october 2018 to september 2019, canada', among total > september 2019, the value of wholesalers proportion of total retail activity > percent is 0.7.
- dense_score gold=0.6664382815361023 top1=0.6738265752792358
- hybrid_score gold=0.8345468640327454 top1=0.9669342041015625

### c16818cb84fdc3c87a1eeec3037e1b4b (rank=56 (>20))

- query: what was the percentage of female correctional workers employed in the criminal justice system in 2011?
- gold: table=349 [4, 9] gold_table_in_context=0 serialized: correctional service officers > 2011 > percent total: 32.0
- top1(this query's own): table=347 [3, 11] serialized: In the table 'admissions to youth correctional services by type of supervision and jurisdiction, 2014/2015', among ontario, the value of total correctional services > percent female is 19.
- dense_score gold=0.7059218287467957 top1=0.6535822153091431
- hybrid_score gold=0.8127390742301941 top1=0.9058201909065247

### c53375d3ec90313b7c70f6b6502769c4 (rank=171 (>20))

- query: what was the national quarterly rate at two times?
- gold: table=1644 [36, 3] gold_table_in_context=0 serialized: In the table 'victims of police-reported sexual assault, by quarterly #metoo period and census metropolitan area, canada, 2016 and 2017', among canada, the value of post-#metoo > rate is 18.6.
- top1(this query's own): table=1553 [21, 2] serialized: In the table 'frequency at which people follow news and current affairs, 2003 and 2013', among percentage > level of education > postsecondary diploma or certificate, the value of several times a month or several times a week > 2003 is 27.0.
- dense_score gold=0.4769749045372009 top1=0.5537033677101135
- hybrid_score gold=0.7122949957847595 top1=0.8856727480888367

### c71c045bac9ce9e5604a95f9b1688676 (rank=223 (>20))

- query: how many games did grogan finished with?
- gold: table=255_totto28913-0 [1, 1] gold_table_in_context=0 serialized: In the table 'List of New England Patriots starting quarterbacks: statistics', among steve grogan > 1975-1990, the value of passing statistics > gs is 135.
- top1(this query's own): table=306_totto35002-0 [2, 16] serialized: In the table 'Steve Grogan: regular season', among 1977 > ne > 14 > 14, the value of notes is final season with 14-game schedule.
- dense_score gold=0.5346595048904419 top1=0.6333068609237671
- hybrid_score gold=0.6817301511764526 top1=0.9279760122299194

### c7e5abd2e9c1c367658a4f424f5e8a65 (rank=23 (>20))

- query: what is the percentage of members who were 24 years of age and younger saw, heard, or experienced sexualized or discriminatory behaviour in the past 12 months?
- gold: table=453 [31, 1] gold_table_in_context=1 serialized: In the table 'canadian armed forces regular force members who witnessed or experienced sexualized or discriminatory behaviours in the past 12 months, by selected characteristics and types of behaviour, 2018', among age group > 50 years and older, the value of total sexualized or discriminatory behaviours > percent is 59.0.
- top1(this query's own): table=453 [25, 1] serialized: In the table 'canadian armed forces regular force members who witnessed or experienced sexualized or discriminatory behaviours in the past 12 months, by selected characteristics and types of behaviour, 2018', among age group > 24 years and younger, the value of total sexualized or discriminatory behaviours > percent is 73.0.
- dense_score gold=0.7315980195999146 top1=0.7626315951347351
- hybrid_score gold=0.9216876029968262 top1=1.0

### c926b3d861ea24989d873c6f781bfce9 (rank=143 (>20))

- query: in fy 2017, how many million dollars did federal agencies obligate to institutions of higher education in support of science and engineering (s&e)?
- gold: table=106_55_fs17-ib-19314-tab001 [8, 0] gold_table_in_context=1 serialized: In the table 'federal science and engineering obligations, by type of activity: fys 2009-17', among 2017, the value of all federal obligations is 32431.
- top1(this query's own): table=106_55_fs17-ib-19314-tab001 [17, 5] serialized: In the table 'federal science and engineering obligations, by type of activity: fys 2009-17', among constant fy 2012 $millions > 2017, the value of general support for s&e is 89.
- dense_score gold=0.625224232673645 top1=0.7279713153839111
- hybrid_score gold=0.7813788652420044 top1=1.0

### cb0de475ffbb8a36ea561a85a4a73cba (rank=44 (>20))

- query: what is the individuals'average age?
- gold: table=460 [1, 0] gold_table_in_context=0 serialized: In the table 'summary statistics of sample', among years > demographics > age, the value of mean is 41.6.
- top1(this query's own): table=2062 [1, 1] serialized: In the table 'descriptive statistics', among years > demographics > individual's age, the value of average is 60.3.
- dense_score gold=0.669280469417572 top1=0.7339557409286499
- hybrid_score gold=0.7814916968345642 top1=0.9497251510620117

### cb8a4498d559394b4617f5775e1315fd (rank=419 (>20))

- query: what was the percentage of male skilled workers were university graduates (bachelor's degree or more)?
- gold: table=1534 [20, 1] gold_table_in_context=0 serialized: In the table 'selected characteristics of economic class principal applicants in linked immigrant landing file (1980 through 2006)-discharge abstract database (2006/2007 through 2008/2009) and canadian-born in linked 2006 census-discharge abstract database, by sex, population aged 25 to 74, canada excluding quebec and territories', among % distribution > landing year > university graduation, the value of men > economic class principal applicants > skilled workers is 73.2.
- top1(this query's own): table=729 [9, 0] serialized: In the table 'estimated population aged 25 to 64 with at least a bachelor's degree, 2001 to 2016', among percent > individuals with a university degree within each population group > 2011, the value of total is 25.5.
- dense_score gold=0.5986771583557129 top1=0.6659137010574341
- hybrid_score gold=0.7486103773117065 top1=0.9190781116485596

### d3b16e0a68b172cf22ba780ddacc0dc0 (rank=87 (>20))

- query: what was the percentage of female ecpas were from east asia?
- gold: table=1534 [15, 5] gold_table_in_context=0 serialized: In the table 'selected characteristics of economic class principal applicants in linked immigrant landing file (1980 through 2006)-discharge abstract database (2006/2007 through 2008/2009) and canadian-born in linked 2006 census-discharge abstract database, by sex, population aged 25 to 74, canada excluding quebec and territories', among % distribution > world region > east asia, the value of women > economic class principal applicants > total is 24.6.
- top1(this query's own): table=1404 [27, 1] serialized: In the table 'percentage distribution of immigrant landing file cohort, by sex, age group, immigration category, landing year, and source world region', among % distribution > source world region > east asia, the value of female is 21.8.
- dense_score gold=0.524380624294281 top1=0.6640433073043823
- hybrid_score gold=0.6645974516868591 top1=0.9853514432907104

### d6d4d3b77a54da32a35ba7b01841d96c (rank=102 (>20))

- query: how many matches did simen wangberg play in tippeligaen before he moved to brann in 2012?
- gold: table=41_totto4556-0 [4, 1] gold_table_in_context=1 serialized: In the table 'Simen Wangberg: club', among rosenborg > total, the value of league > apps is 24.
- top1(this query's own): table=41_totto4556-0 [7, 0] serialized: In the table 'Simen Wangberg: club', among brann > 2012, the value of league > division is tippeligaen.
- dense_score gold=0.5586346387863159 top1=0.7520334124565125
- hybrid_score gold=0.6425896286964417 top1=1.0

### d7fd608ab5aac14a5bb51ede2bff65ac (rank=27 (>20))

- query: what was the percentage points were they less likely to be employed full time?
- gold: table=1676 [1, 0] gold_table_in_context=0 serialized: In the table 'estimated relationship between teacher job characteristics and private high school employment', among employed full time, the value of regression results > coefficient is -0.083.
- top1(this query's own): table=609 [3, 0] serialized: median employment income > employed full year, full time in reference year > level of education > less than a high school diploma > 2014 dollars: 36300.0
- dense_score gold=0.5446290969848633 top1=0.5384318232536316
- hybrid_score gold=0.7296252846717834 top1=0.7819620370864868

### dbc32084535e60e41dd2c9e5b7ec0ddf (rank=105 (>20))

- query: how many appearances did tommy higginson make for his 12-years with brentford?
- gold: table=35_totto4058-2 [11, 7] gold_table_in_context=1 serialized: In the table 'Tommy Higginson: career statistics', among career total, the value of total > apps is 433.
- top1(this query's own): table=35_totto4058-2 [3, 7] serialized: In the table 'Tommy Higginson: career statistics', among brentford > 1962-63, the value of total > apps is 49.
- dense_score gold=0.5444542169570923 top1=0.6915844082832336
- hybrid_score gold=0.720260739326477 top1=1.0

### dcadc31f634e2df848f25119cbf6b1b0 (rank=28 (>20))

- query: what was the infant mortality in british columbia in 2013?
- gold: table=2417 [33, 9] gold_table_in_context=1 serialized: In the table 'this table displays the results of number and rate of infant mortality. the information is grouped by year (appearing as row headers), n.l., p.e.i., n.s., n.b., que., ont., man., sask., alta., b.c., y.t., n.w.t., nvt., unknown and canada, calculated using number and rate per 1,000 units of measure (appearing as column headers)', among rate per 1,000 > 2013, the value of b.c is 3.7.
- top1(this query's own): table=966 [11, 0] serialized: In the table 'thyroid cancer death counts, age-standardized mortality rates (asmrs), and trends in asmrs, by sex, age group and province, canada excluding quebec', among province > british columbia, the value of mortality > number of deaths is 145.0.
- dense_score gold=0.6654179096221924 top1=0.607507586479187
- hybrid_score gold=0.8129143118858337 top1=0.8900315761566162

### dffc23f130e35c1cd12bea73411a878f (rank=82 (>20))

- query: what is the percentage of senior men rated their general health as "good", "very good", or "excellent".
- gold: table=1625 [0, 1] gold_table_in_context=1 serialized: In the table 'self-reported health of senior women and men, by age group, canada, 2013-2014', among percentage > positive, the value of total - 65 years and over > men is 78.6.
- top1(this query's own): table=1625 [2, 5] serialized: In the table 'self-reported health of senior women and men, by age group, canada, 2013-2014', among percentage > positive > very good, the value of 75 to 84 years > men is 28.2.
- dense_score gold=0.6846922636032104 top1=0.7447934150695801
- hybrid_score gold=0.8086583614349365 top1=0.9909110069274902

### e1717106e44ab2792531adc4e4f23a94 (rank=33 (>20))

- query: what was the percent of the national rate of major assault (level 2 and 3) increased in 2019 for the fifth consecutive year due to higher rates of assault with a weapon or causing bodily harm (level 2)?
- gold: table=1337 [13, 8] gold_table_in_context=0 serialized: In the table 'police-reported crime for selected offences, by province or territory, 2019', among canada, the value of major assault > percent change in rate 2018 to 2019 is 8.
- top1(this query's own): table=1336 [11, 4] serialized: In the table 'police-reported crime for selected offences, canada, 2018 and 2019', among total violent crime > assault - level 2 - weapon or bodily harm, the value of change in rate 2018 to 2019 > percent is 8.0.
- dense_score gold=0.6616807579994202 top1=0.7254300117492676
- hybrid_score gold=0.8010562658309937 top1=0.9946975111961365

### e30ae2e681f8526880c0dba2edbba915 (rank=41 (>20))

- query: how many passing yards did detmer get for colorado totally?
- gold: table=127_totto15163-1 [4, 3] gold_table_in_context=1 serialized: In the table 'Koy Detmer: statistics', among career, the value of passing > yds is 5390.0.
- top1(this query's own): table=127_totto15163-1 [1, 3] serialized: In the table 'Koy Detmer: statistics', among 1994 > colorado, the value of passing > yds is 171.0.
- dense_score gold=0.6361625790596008 top1=0.7164206504821777
- hybrid_score gold=0.7870939373970032 top1=1.0

### e457bbfa0f7b82e54322c0c6adde87e9 (rank=68 (>20))

- query: in fy 2015, a total of 39 state governments reported some expenditures for energy-related r&d, how many thousand dollars were state government r&d expenditures for energy-related r&d in oregon?
- gold: table=209_99_tab3 [1, 1] gold_table_in_context=1 serialized: In the table 'state agency expenditures for r&d, by state and function of r&d, for the 10 states with the highest levels of r&d expenditures: fy 2015', among united statesa > california, the value of agriculture is 7567.
- top1(this query's own): table=209_99_tab3 [0, 2] serialized: In the table 'state agency expenditures for r&d, by state and function of r&d, for the 10 states with the highest levels of r&d expenditures: fy 2015', among united statesa, the value of energy is 312114.
- dense_score gold=0.6474284529685974 top1=0.7421942949295044
- hybrid_score gold=0.872834324836731 top1=0.9949506521224976

### e812e27620760570e0c4df5678a40336 (rank=75 (>20))

- query: how many goals did bucur got in sportul studentesc?
- gold: table=55_totto6269-4 [7, 1] gold_table_in_context=1 serialized: In the table 'Gheorghe Bucur: statistics', among total, the value of league > goals is 76.
- top1(this query's own): table=55_totto6269-4 [5, 7] serialized: In the table 'Gheorghe Bucur: statistics', among sportul studentesc > 2003-04, the value of total > goals is 29.
- dense_score gold=0.6111567616462708 top1=0.7222283482551575
- hybrid_score gold=0.6898600459098816 top1=1.0

### ede09a0429b207bcb49ec381eac1bf77 (rank=36 (>20))

- query: how many percent of caregivers living with their care receiver helped a woman?
- gold: table=2481 [7, 3] gold_table_in_context=0 serialized: In the table 'characteristics of primary care receivers care facility (ref.), supportive housing, at home, separate households and at home, shared household, calculated using number (in thousands) and percentage units of measure (appearing as column headers)', among percentage > sex of care receiver > female, the value of at home, shared household is 62.0.
- top1(this query's own): table=565 [0, 0] serialized: In the table 'relationship between the caregiver and their primary care receiver, by caregiver's age, 2018', among caregiver's age, the value of primary care receiver > spouse/partner is percent.
- dense_score gold=0.6462565064430237 top1=0.6724315881729126
- hybrid_score gold=0.8632313013076782 top1=1.0

### ee75c80f04501d76eb87e15f9d95fad6 (rank=175 (>20))

- query: in fy 2016, how many million dollars did federal agencies obligate to institutions of higher education in support of science and engineering (s&e)?
- gold: table=106_55_fs17-ib-19314-tab001 [7, 0] gold_table_in_context=1 serialized: In the table 'federal science and engineering obligations, by type of activity: fys 2009-17', among 2016, the value of all federal obligations is 31647.
- top1(this query's own): table=106_55_fs17-ib-19314-tab001 [16, 5] serialized: In the table 'federal science and engineering obligations, by type of activity: fys 2009-17', among constant fy 2012 $millions > 2016, the value of general support for s&e is 121.
- dense_score gold=0.6194677352905273 top1=0.7141307592391968
- hybrid_score gold=0.788992702960968 top1=0.9979432225227356

### ef879733782be16efc7e0f0712474243 (rank=34 (>20))

- query: which league did bateman appear for brentford team in the 1934-35 season?
- gold: table=364_totto41529-0 [1, 0] gold_table_in_context=1 serialized: In the table 'Arthur Bateman (footballer, born 1908): career statistics', among brentford > 1933-34, the value of league > division is second division.
- top1(this query's own): table=14_totto1734-1 [2, 2] serialized: In the table 'Arthur Bateman (footballer, born 1908): career statistics', among brentford > 1934-35, the value of league > goals is 1.
- dense_score gold=0.6995022296905518 top1=0.7339881658554077
- hybrid_score gold=0.8694005608558655 top1=1.0

### f0567787a554df9acd218b8753a68ac0 (rank=22 (>20))

- query: what is the low boundary of percentage of french-speaking people in quebec is projected to be in 2036 ?
- gold: table=1195 [7, 3] gold_table_in_context=1 serialized: In the table 'first official language spoken as a percentage of the population, quebec, canada outside quebec and canada, 2011 (estimated) and 2036', among percent > quebec > french, the value of 2036 > high immigration is 82.0.
- top1(this query's own): table=1195 [7, 2] serialized: In the table 'first official language spoken as a percentage of the population, quebec, canada outside quebec and canada, 2011 (estimated) and 2036', among percent > quebec > french, the value of 2036 > low immigration is 83.0.
- dense_score gold=0.7270337343215942 top1=0.7774591445922852
- hybrid_score gold=0.9053694009780884 top1=1.0

### f5719a1bbf54e674764f458348ac13ae (rank=84 (>20))

- query: what was the percentage of female ecpas were from southeast asia?
- gold: table=1534 [14, 5] gold_table_in_context=0 serialized: In the table 'selected characteristics of economic class principal applicants in linked immigrant landing file (1980 through 2006)-discharge abstract database (2006/2007 through 2008/2009) and canadian-born in linked 2006 census-discharge abstract database, by sex, population aged 25 to 74, canada excluding quebec and territories', among % distribution > world region > southeast asia, the value of women > economic class principal applicants > total is 31.3.
- top1(this query's own): table=895 [6, 8] serialized: In the table 'voting rates in federal elections by country or region of birth and by sex, 2011 and 2015', among southeast asia, the value of women > change > percentage point is 11.7.
- dense_score gold=0.5284430384635925 top1=0.651740312576294
- hybrid_score gold=0.6861088275909424 top1=0.969260036945343

### faa7c3995f78c7766326c8f404823540 (rank=53 (>20))

- query: how many games did pablo zabaleta play for manchester city over nine seasons?
- gold: table=322_totto36692-0 [17, 9] gold_table_in_context=1 serialized: In the table 'Pablo Zabaleta: career statistics', among manchester city > total, the value of total > apps is 333.0.
- top1(this query's own): table=322_totto36692-0 [17, 2] serialized: In the table 'Pablo Zabaleta: career statistics', among manchester city > total, the value of league > goals is 9.
- dense_score gold=0.6380676031112671 top1=0.7270127534866333
- hybrid_score gold=0.8830559253692627 top1=0.9940448999404907

### fb405c7066103c7a1daa06dafbe395d1 (rank=47 (>20))

- query: how many percentage points did the number of passengers enplaned and deplaned at canadian airports increase in 2017 compared to the previous year?
- gold: table=2560 [4, 2] gold_table_in_context=1 serialized: In the table 'passenger and cargo data', among total, the value of change 2016 to 2017 > percent is 6.2.
- top1(this query's own): table=2560 [2, 2] serialized: In the table 'passenger and cargo data', among enplaned and deplaned passengers > transborder segments, the value of change 2016 to 2017 > percent is 4.9.
- dense_score gold=0.6059278249740601 top1=0.6875426769256592
- hybrid_score gold=0.6984589695930481 top1=0.9805499315261841

### fd0f45f84b9e9ec603acff9ba58b92ff (rank=25 (>20))

- query: what was the total number of deaths attributed to tc in canada from 2012 to 2016?
- gold: table=966 [0, 0] gold_table_in_context=1 serialized: In the table 'thyroid cancer death counts, age-standardized mortality rates (asmrs), and trends in asmrs, by sex, age group and province, canada excluding quebec', among all, the value of mortality > number of deaths is 850.0.
- top1(this query's own): table=1795 [27, 13] serialized: In the table 'number of homicides, by province or territory, 1985 to 2015', among number of victims > 2012, the value of canada is 548.
- dense_score gold=0.6429597735404968 top1=0.7029731273651123
- hybrid_score gold=0.8823676109313965 top1=0.9861541390419006

### fd6347ab41a3e2828894fd180dfee613 (rank=21 (>20))

- query: how many percent of those whose care receiver was living in a private household separate from theirs?
- gold: table=2481 [10, 2] gold_table_in_context=1 serialized: In the table 'characteristics of primary care receivers care facility (ref.), supportive housing, at home, separate households and at home, shared household, calculated using number (in thousands) and percentage units of measure (appearing as column headers)', among percentage > relationship with caregiver > friend or neighbour, the value of at home, separate households is 17.0.
- top1(this query's own): table=2481 [2, 2] serialized: In the table 'characteristics of primary care receivers care facility (ref.), supportive housing, at home, separate households and at home, shared household, calculated using number (in thousands) and percentage units of measure (appearing as column headers)', among percentage > age of care receiver > 65 to 74, the value of at home, separate households is 27.0.
- dense_score gold=0.6785576343536377 top1=0.6917716264724731
- hybrid_score gold=0.9128487706184387 top1=0.9946516752243042

## gold_rank unknown (top-500 스캔 밖, 전체 6건, 전부 표시)

### 4c25190668c859cfffef1f14bd69ffe0 (unknown)

- query: what is the probability that individuals are employed in the pre-reform period?
- gold: table=2062 [8, 1] gold_table_in_context=0 serialized: In the table 'descriptive statistics', among percent > non-pension income > individual has labour income, the value of average is 62.9.
- top1(this query's own): table=399 [3, 0] serialized: In the table 'predicted probability of being employed in a professional or managerial occupation by skill level and selected aboriginal identity group, 2012', among predicted probability > higher skill level > non-aboriginal, the value of literacy > model 1 is 0.71.
- dense_score gold=0.5074079036712646 top1=0.6163744926452637
- hybrid_score gold=0.6375418901443481 top1=0.9334017038345337

### 7c2d4c7010f3d02a4cb5b1da1c4313a4 (unknown)

- query: what is the population comprised of current students and people who had been previously employed but were not permanently retired?
- gold: table=947 [0, 3] gold_table_in_context=1 serialized: In the table 'selected characteristics of persons aged 15 to 64 with disabilities by category of work potentialnote 1, note 2', among number > population, the value of potential workers > other potential workers is 110800.0.
- top1(this query's own): table=947 [0, 6] serialized: In the table 'selected characteristics of persons aged 15 to 64 with disabilities by category of work potentialnote 1, note 2', among number > population, the value of not potential workers > permanently retired is 632600.0.
- dense_score gold=0.550639808177948 top1=0.6435582041740417
- hybrid_score gold=0.6614323258399963 top1=0.9995635747909546

### 89bab78aee8ef8065330506d7128f3f6 (unknown)

- query: what was the percentage of the injuries occurred only once during the previous 12 months?
- gold: table=557 [43, 0] gold_table_in_context=0 serialized: In the table 'factors involved in tanning equipment use, household population aged 12 or older, canada excluding territories, 2014', among number of episodes of discomfort/injury to skin in past year > 1, the value of % is 56.3.
- top1(this query's own): table=1057 [5, 0] serialized: In the table 'health risk behaviours, by sexual orientation and gender, canada, 2018', among used drugs or alcohol to cope > with abuse or violence that occurred in past 12 months, the value of heterosexual > percent is 9.7.
- dense_score gold=0.5512195825576782 top1=0.6009144186973572
- hybrid_score gold=0.7432160377502441 top1=0.9289029836654663

### 8db22461be5d09fe27d62d3b758fad97 (unknown)

- query: what was the peak chart position on the us200 of the album roses?
- gold: table=120_totto14023-2 [1, 0] gold_table_in_context=0 serialized: In the table 'Rufus Wainwright discography: studio albums', among 2001 > poses released: june 5, 2001 label: dreamworks format: cd, the value of peak chart positions > us 200 is 117.0.
- top1(this query's own): table=69_totto7963-4 [0, 0] serialized: In the table 'Against Me! discography: singles', among 2002 > the disco before the breakdown, the value of peak chart positions > us rock is -.
- dense_score gold=0.5226035714149475 top1=0.628366231918335
- hybrid_score gold=0.699243426322937 top1=0.9143199324607849

### b8b332eb07c52c9c44666293f5ea8e2b (unknown)

- query: what percent of individuals aged 60 years old on average are immigrants?
- gold: table=2062 [6, 1] gold_table_in_context=0 serialized: In the table 'descriptive statistics', among percent > immigrant, the value of average is 6.1.
- top1(this query's own): table=729 [7, 2] serialized: In the table 'estimated population aged 25 to 64 with at least a bachelor's degree, 2001 to 2016', among percent > individuals with a university degree within each population group > 2001, the value of long-term immigrants is 21.5.
- dense_score gold=0.625325083732605 top1=0.718929648399353
- hybrid_score gold=0.7579348087310791 top1=0.945180356502533

### d7848bd4a92822affd40df34e647de7d (unknown)

- query: what is the participation rate in 2007?
- gold: table=1852 [1, 0] gold_table_in_context=0 serialized: percent > actual > 2007 > both sexes: 67.4
- top1(this query's own): table=212 [0, 0] serialized: In the table 'import or export participation rates of firms in the manufacturing and wholesale trade sectors', among percent > manufacturing, the value of import participation rate > all firms is 30.64.
- dense_score gold=0.6052795648574829 top1=0.6510086059570312
- hybrid_score gold=0.6966623067855835 top1=0.9700484275817871
