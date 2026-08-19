# 实验15报告：窗口Jacobian与最近词

每行对应一个 `checkpoint / dynamic step`，四个数据列对应四个固定初始token。
所有最近词均由该窗口右端点的同一个hidden state计算。`freq` 是该最近词在
WikiText-2 train tokenization中的出现次数。

## 1. Cosine最近词

- `sim`：hidden state与input embedding的cosine similarity，越大越近；
- `margin`：top1 similarity减top2 similarity，越大表示top1分离越明确。

| checkpoint | dynamic step | ' repetitive' | ' semi' | ' evidence' | ' orientations' |
|---|---:|---|---|---|---|
| step5000 | 256 | 'GAA'<br>sim=0.20129; margin=0.0162; freq=0 | ' maintaining'<br>sim=0.19569; margin=0.0116; freq=42 | 'GAA'<br>sim=0.2054; margin=0.0184; freq=0 | 'GAA'<br>sim=0.20449; margin=0.0162; freq=0 |
| step5000 | 512 | 'GAA'<br>sim=0.20149; margin=0.02; freq=0 | 'GAA'<br>sim=0.19162; margin=0.000602; freq=0 | 'GAA'<br>sim=0.20865; margin=0.00915; freq=0 | 'GAA'<br>sim=0.20968; margin=0.0254; freq=0 |
| step5000 | 768 | 'GAA'<br>sim=0.2078; margin=0.0174; freq=0 | 'GAA'<br>sim=0.21004; margin=0.0215; freq=0 | 'GAA'<br>sim=0.20565; margin=0.0196; freq=0 | 'GAA'<br>sim=0.20274; margin=0.0127; freq=0 |
| step5000 | 1024 | 'GAA'<br>sim=0.20829; margin=0.0161; freq=0 | 'GAA'<br>sim=0.21029; margin=0.0237; freq=0 | 'GAA'<br>sim=0.20398; margin=0.0213; freq=0 | 'GAA'<br>sim=0.19479; margin=0.0143; freq=0 |
| step5000 | 1280 | 'GAA'<br>sim=0.21489; margin=0.0244; freq=0 | 'GAA'<br>sim=0.188; margin=0.00285; freq=0 | 'GAA'<br>sim=0.22136; margin=0.0289; freq=0 | 'GAA'<br>sim=0.19817; margin=0.0126; freq=0 |
| step5000 | 1536 | 'GAA'<br>sim=0.19441; margin=0.00853; freq=0 | 'GAA'<br>sim=0.19977; margin=0.00841; freq=0 | 'GAA'<br>sim=0.22057; margin=0.0245; freq=0 | 'GAA'<br>sim=0.21842; margin=0.0249; freq=0 |
| step5000 | 1792 | 'GAA'<br>sim=0.19386; margin=0.00518; freq=0 | 'GAA'<br>sim=0.22384; margin=0.0311; freq=0 | 'GAA'<br>sim=0.20142; margin=0.0165; freq=0 | 'GAA'<br>sim=0.19124; margin=0.0104; freq=0 |
| step5000 | 2048 | ' maintaining'<br>sim=0.19195; margin=0.00114; freq=42 | 'GAA'<br>sim=0.21996; margin=0.0351; freq=0 | 'GAA'<br>sim=0.20453; margin=0.018; freq=0 | 'GAA'<br>sim=0.20144; margin=0.0191; freq=0 |
| step7000 | 256 | 'ства'<br>sim=0.18173; margin=0.00723; freq=0 | ' ACS'<br>sim=0.16221; margin=0.00127; freq=0 | 'ership'<br>sim=0.17687; margin=2.62e-05; freq=26 | ' smartphone'<br>sim=0.17484; margin=2.76e-05; freq=1 |
| step7000 | 512 | ' supermarket'<br>sim=0.17441; margin=0.002; freq=3 | ' preserv'<br>sim=0.17217; margin=0.00207; freq=2 | ' GDP'<br>sim=0.17726; margin=0.00163; freq=8 | 'GAA'<br>sim=0.18734; margin=0.00935; freq=0 |
| step7000 | 768 | ' preserv'<br>sim=0.18448; margin=0.000263; freq=2 | ' Catalog'<br>sim=0.19266; margin=5.57e-06; freq=0 | 'ード'<br>sim=0.1817; margin=0.0148; freq=0 | 'GAA'<br>sim=0.18866; margin=0.00869; freq=0 |
| step7000 | 1024 | 'umbent'<br>sim=0.17559; margin=0.00302; freq=1 | ' politique'<br>sim=0.18242; margin=0.00837; freq=0 | ' TCR'<br>sim=0.16317; margin=0.000369; freq=0 | 'Tour'<br>sim=0.1765; margin=0.00361; freq=1 |
| step7000 | 1280 | ' destinations'<br>sim=0.18856; margin=0.0134; freq=16 | 'icrobial'<br>sim=0.19053; margin=0.00345; freq=0 | 'cibly'<br>sim=0.17528; margin=0.000314; freq=0 | 'bibr'<br>sim=0.1937; margin=0.0197; freq=0 |
| step7000 | 1536 | ' politique'<br>sim=0.21639; margin=0.0358; freq=0 | ' destinations'<br>sim=0.19916; margin=0.0179; freq=16 | ' Catalog'<br>sim=0.18816; margin=0.00613; freq=0 | 'icrobial'<br>sim=0.19246; margin=0.00735; freq=0 |
| step7000 | 1792 | 'bibr'<br>sim=0.17064; margin=0.00264; freq=0 | 'GAA'<br>sim=0.18069; margin=0.00268; freq=0 | ' *)'<br>sim=0.17361; margin=0.000818; freq=0 | ' GDP'<br>sim=0.18122; margin=0.00406; freq=8 |
| step7000 | 2048 | ' destinations'<br>sim=0.20768; margin=0.0225; freq=16 | 'GAA'<br>sim=0.18255; margin=0.00271; freq=0 | ' foraging'<br>sim=0.18591; margin=0.00814; freq=20 | ' landscape'<br>sim=0.19889; margin=0.0151; freq=33 |
| step9000 | 256 | 'atom'<br>sim=0.16155; margin=0.000216; freq=15 | ' NAC'<br>sim=0.16139; margin=2.19e-05; freq=2 | 'atom'<br>sim=0.16331; margin=0.000719; freq=15 | 'atom'<br>sim=0.16456; margin=8.97e-05; freq=15 |
| step9000 | 512 | 'atom'<br>sim=0.16342; margin=0.00045; freq=15 | 'atom'<br>sim=0.16315; margin=0.000555; freq=15 | 'atom'<br>sim=0.16338; margin=0.000519; freq=15 | 'atom'<br>sim=0.16377; margin=0.000407; freq=15 |
| step9000 | 768 | 'atom'<br>sim=0.16363; margin=0.000412; freq=15 | 'atom'<br>sim=0.16354; margin=0.000438; freq=15 | 'atom'<br>sim=0.16353; margin=0.000453; freq=15 | 'atom'<br>sim=0.1636; margin=0.000439; freq=15 |
| step9000 | 1024 | 'atom'<br>sim=0.16362; margin=0.000424; freq=15 | 'atom'<br>sim=0.1636; margin=0.000427; freq=15 | 'atom'<br>sim=0.16359; margin=0.000434; freq=15 | 'atom'<br>sim=0.16359; margin=0.000436; freq=15 |
| step9000 | 1280 | 'atom'<br>sim=0.16361; margin=0.00043; freq=15 | 'atom'<br>sim=0.16361; margin=0.000429; freq=15 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 | 'atom'<br>sim=0.1636; margin=0.000432; freq=15 |
| step9000 | 1536 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 |
| step9000 | 1792 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 |
| step9000 | 2048 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 | 'atom'<br>sim=0.1636; margin=0.000431; freq=15 |
| step13000 | 256 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step13000 | 512 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step13000 | 768 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step13000 | 1024 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step13000 | 1280 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step13000 | 1536 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step13000 | 1792 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step13000 | 2048 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 | 'escap'<br>sim=0.18877; margin=0.0198; freq=1 |
| step21000 | 256 | ' pathogen'<br>sim=0.19301; margin=0.00761; freq=1 | ' beaches'<br>sim=0.16927; margin=0.00237; freq=23 | 'ña'<br>sim=0.17688; margin=0.00309; freq=9 | '元'<br>sim=0.18288; margin=0.01; freq=0 |
| step21000 | 512 | ' scen'<br>sim=0.19574; margin=0.0298; freq=1 | 'DE'<br>sim=0.1845; margin=0.00156; freq=2 | '性'<br>sim=0.19209; margin=0.0208; freq=0 | 'theme'<br>sim=0.1914; margin=0.0187; freq=0 |
| step21000 | 768 | ' modulation'<br>sim=0.17376; margin=0.00943; freq=2 | ' atherosclerotic'<br>sim=0.18908; margin=0.00273; freq=0 | ' attacks'<br>sim=0.16783; margin=0.00314; freq=173 | 'theme'<br>sim=0.21201; margin=0.0403; freq=0 |
| step21000 | 1024 | '性'<br>sim=0.18772; margin=0.0138; freq=0 | ' sigma'<br>sim=0.16295; margin=0.00424; freq=0 | '）'<br>sim=0.17323; margin=0.00174; freq=0 | '情'<br>sim=0.1898; margin=0.000968; freq=0 |
| step21000 | 1280 | ' intimate'<br>sim=0.18274; margin=0.00684; freq=20 | '被'<br>sim=0.15701; margin=0.00199; freq=0 | 'disc'<br>sim=0.17149; margin=0.006; freq=1 | ' cannabis'<br>sim=0.17727; margin=0.000511; freq=3 |
| step21000 | 1536 | ' shack'<br>sim=0.19194; margin=0.00432; freq=3 | ' ove'<br>sim=0.17227; margin=0.00389; freq=0 | ' nud'<br>sim=0.16382; margin=0.00287; freq=7 | 'mys'<br>sim=0.18694; margin=0.00369; freq=0 |
| step21000 | 1792 | ' files'<br>sim=0.18623; margin=0.00913; freq=5 | ' pathogen'<br>sim=0.18218; margin=0.00835; freq=1 | 'прав'<br>sim=0.1903; margin=0.0241; freq=0 | ' bother'<br>sim=0.16072; margin=0.00261; freq=2 |
| step21000 | 2048 | 'struct'<br>sim=0.18801; margin=0.00868; freq=1 | '）'<br>sim=0.19207; margin=0.00976; freq=0 | ' shack'<br>sim=0.20656; margin=0.0163; freq=3 | ' hanging'<br>sim=0.18636; margin=0.0107; freq=23 |
| step29000 | 256 | ' committed'<br>sim=0.1707; margin=0.00421; freq=64 | 'eth'<br>sim=0.1608; margin=0.000913; freq=95 | ' game'<br>sim=0.16344; margin=0.000403; freq=1850 | ' committed'<br>sim=0.16272; margin=0.000433; freq=64 |
| step29000 | 512 | ' committed'<br>sim=0.18986; margin=0.0143; freq=64 | 'aptic'<br>sim=0.18442; margin=0.00777; freq=0 | ' committed'<br>sim=0.17153; margin=0.00304; freq=64 | 'aptic'<br>sim=0.1852; margin=0.00512; freq=0 |
| step29000 | 768 | ' committed'<br>sim=0.17426; margin=0.00311; freq=64 | ' committed'<br>sim=0.19937; margin=0.0251; freq=64 | ' committed'<br>sim=0.19122; margin=0.0159; freq=64 | ' committed'<br>sim=0.1925; margin=0.0168; freq=64 |
| step29000 | 1024 | ' committed'<br>sim=0.16623; margin=0.00407; freq=64 | ' committed'<br>sim=0.15725; margin=0.000798; freq=64 | ' committed'<br>sim=0.17215; margin=0.00199; freq=64 | 'eth'<br>sim=0.16107; margin=0.000697; freq=95 |
| step29000 | 1280 | ' committed'<br>sim=0.18336; margin=0.00121; freq=64 | 'aptic'<br>sim=0.17949; margin=0.00457; freq=0 | ' committed'<br>sim=0.1672; margin=0.00534; freq=64 | 'aptic'<br>sim=0.18488; margin=0.00789; freq=0 |
| step29000 | 1536 | ' committed'<br>sim=0.18566; margin=0.0108; freq=64 | ' committed'<br>sim=0.19886; margin=0.028; freq=64 | ' committed'<br>sim=0.18448; margin=0.00382; freq=64 | ' committed'<br>sim=0.1989; margin=0.0243; freq=64 |
| step29000 | 1792 | 'με'<br>sim=0.16187; margin=8.12e-05; freq=1 | ' game'<br>sim=0.16161; margin=0.000931; freq=1850 | ' committed'<br>sim=0.1835; margin=0.00915; freq=64 | 'undefined'<br>sim=0.1573; margin=0.000182; freq=0 |
| step29000 | 2048 | 'aptic'<br>sim=0.18564; margin=0.00679; freq=0 | ' committed'<br>sim=0.17266; margin=0.00096; freq=64 | 'με'<br>sim=0.16216; margin=4.71e-05; freq=1 | 'aptic'<br>sim=0.18048; margin=0.00528; freq=0 |
| step37000 | 256 | 'textsf'<br>sim=0.179; margin=0.00675; freq=0 | 'mes'<br>sim=0.17413; margin=0.00889; freq=35 | 'mes'<br>sim=0.18268; margin=0.0142; freq=35 | 'textsf'<br>sim=0.17543; margin=0.0165; freq=0 |
| step37000 | 512 | 'textsf'<br>sim=0.16819; margin=0.00215; freq=0 | 'textsf'<br>sim=0.17254; margin=0.00683; freq=0 | 'textsf'<br>sim=0.17577; margin=0.00497; freq=0 | ' NS'<br>sim=0.17275; margin=0.00103; freq=54 |
| step37000 | 768 | ' NS'<br>sim=0.17979; margin=0.00128; freq=54 | 'textsf'<br>sim=0.18379; margin=0.00376; freq=0 | 'mes'<br>sim=0.18056; margin=0.00247; freq=35 | 'ele'<br>sim=0.16017; margin=0.000611; freq=9 |
| step37000 | 1024 | ' NS'<br>sim=0.17578; margin=0.00786; freq=54 | 'textsf'<br>sim=0.17392; margin=0.00685; freq=0 | 'textsf'<br>sim=0.17389; margin=0.00697; freq=0 | ' NS'<br>sim=0.17587; margin=0.00124; freq=54 |
| step37000 | 1280 | ' NS'<br>sim=0.1937; margin=0.0204; freq=54 | ' NS'<br>sim=0.18658; margin=0.008; freq=54 | ' NS'<br>sim=0.19138; margin=0.0135; freq=54 | 'effects'<br>sim=0.16604; margin=0.0043; freq=0 |
| step37000 | 1536 | ' views'<br>sim=0.15814; margin=0.00106; freq=83 | 'textsf'<br>sim=0.17601; margin=0.0084; freq=0 | ' NS'<br>sim=0.16851; margin=0.00314; freq=54 | 'textsf'<br>sim=0.17522; margin=0.00685; freq=0 |
| step37000 | 1792 | 'fr'<br>sim=0.17076; margin=0.000676; freq=36 | ' NS'<br>sim=0.19412; margin=0.0155; freq=54 | ' NS'<br>sim=0.18809; margin=0.0152; freq=54 | ' pathological'<br>sim=0.16368; margin=0.00179; freq=2 |
| step37000 | 2048 | ' causes'<br>sim=0.15672; margin=0.00333; freq=59 | 'textsf'<br>sim=0.18102; margin=0.00262; freq=0 | ' NS'<br>sim=0.17792; margin=0.00897; freq=54 | 'textsf'<br>sim=0.17614; margin=0.0059; freq=0 |
| step53000 | 256 | 'widget'<br>sim=0.18998; margin=0.00969; freq=0 | '                                     '<br>sim=0.18235; margin=0.00473; freq=0 | 'Context'<br>sim=0.18067; margin=0.00411; freq=0 | 'Context'<br>sim=0.18019; margin=0.00176; freq=0 |
| step53000 | 512 | 'ptic'<br>sim=0.17526; margin=0.000472; freq=7 | 'oker'<br>sim=0.21532; margin=0.0252; freq=16 | 'wx'<br>sim=0.18167; margin=0.00684; freq=0 | 'wx'<br>sim=0.18154; margin=0.00525; freq=0 |
| step53000 | 768 | '                                     '<br>sim=0.1878; margin=0.018; freq=0 | 'ات'<br>sim=0.1765; margin=0.000782; freq=0 | '                                     '<br>sim=0.18257; margin=0.00395; freq=0 | '                                     '<br>sim=0.18316; margin=0.00476; freq=0 |
| step53000 | 1024 | 'oker'<br>sim=0.19897; margin=0.000211; freq=16 | 'SHORT'<br>sim=0.18991; margin=0.00633; freq=0 | 'oker'<br>sim=0.21597; margin=0.0325; freq=16 | 'oker'<br>sim=0.21619; margin=0.0324; freq=16 |
| step53000 | 1280 | 'iterator'<br>sim=0.16563; margin=0.000719; freq=0 | 'speech'<br>sim=0.17238; margin=8.18e-06; freq=0 | '                                     '<br>sim=0.18344; margin=0.00144; freq=0 | '                                     '<br>sim=0.18312; margin=0.00186; freq=0 |
| step53000 | 1536 | 'Context'<br>sim=0.18064; margin=0.00524; freq=0 | 'widget'<br>sim=0.18787; margin=0.00556; freq=0 | 'SHORT'<br>sim=0.19397; margin=0.00687; freq=0 | 'SHORT'<br>sim=0.19403; margin=0.00747; freq=0 |
| step53000 | 1792 | 'wx'<br>sim=0.18215; margin=0.00606; freq=0 | 'oker'<br>sim=0.18804; margin=0.00811; freq=16 | '«'<br>sim=0.1855; margin=0.0093; freq=0 | '«'<br>sim=0.18645; margin=0.00972; freq=0 |
| step53000 | 2048 | '                                     '<br>sim=0.18161; margin=0.00215; freq=0 | '                                     '<br>sim=0.18691; margin=0.0069; freq=0 | 'widget'<br>sim=0.19008; margin=0.00936; freq=0 | 'widget'<br>sim=0.19014; margin=0.0092; freq=0 |
| step61000 | 256 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |
| step61000 | 512 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |
| step61000 | 768 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |
| step61000 | 1024 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |
| step61000 | 1280 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |
| step61000 | 1536 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |
| step61000 | 1792 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |
| step61000 | 2048 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 | ' pastor'<br>sim=0.18636; margin=0.00188; freq=7 |

## 2. Euclidean最近词

- `d`：hidden state到input embedding的欧式距离，越小越近；
- `rel-margin=(d2-d1)/d1`，越大表示top1分离越明确。

| checkpoint | dynamic step | ' repetitive' | ' semi' | ' evidence' | ' orientations' |
|---|---:|---|---|---|---|
| step5000 | 256 | 'GAA'<br>d=37.424; rel-margin=0.000548; freq=0 | ' maintaining'<br>d=37.605; rel-margin=0.000421; freq=42 | 'GAA'<br>d=37.278; rel-margin=0.00057; freq=0 | 'GAA'<br>d=37.388; rel-margin=0.000551; freq=0 |
| step5000 | 512 | 'GAA'<br>d=37.271; rel-margin=0.000603; freq=0 | ' maintaining'<br>d=37.44; rel-margin=2.82e-05; freq=42 | 'GAA'<br>d=37.447; rel-margin=0.000264; freq=0 | 'GAA'<br>d=37.422; rel-margin=0.000834; freq=0 |
| step5000 | 768 | 'GAA'<br>d=37.231; rel-margin=0.000592; freq=0 | 'GAA'<br>d=37.166; rel-margin=0.000728; freq=0 | 'GAA'<br>d=37.373; rel-margin=0.000659; freq=0 | 'GAA'<br>d=37.478; rel-margin=0.000424; freq=0 |
| step5000 | 1024 | 'GAA'<br>d=37.247; rel-margin=0.000551; freq=0 | 'GAA'<br>d=37.208; rel-margin=0.000795; freq=0 | 'GAA'<br>d=37.271; rel-margin=0.000644; freq=0 | 'GAA'<br>d=37.288; rel-margin=0.00042; freq=0 |
| step5000 | 1280 | 'GAA'<br>d=37.437; rel-margin=0.000816; freq=0 | 'GAA'<br>d=37.588; rel-margin=4.58e-05; freq=0 | 'GAA'<br>d=37.169; rel-margin=0.000966; freq=0 | 'GAA'<br>d=37.351; rel-margin=0.000481; freq=0 |
| step5000 | 1536 | 'GAA'<br>d=37.427; rel-margin=0.000229; freq=0 | 'GAA'<br>d=37.358; rel-margin=0.000172; freq=0 | 'GAA'<br>d=37.213; rel-margin=0.000824; freq=0 | 'GAA'<br>d=37.293; rel-margin=0.000835; freq=0 |
| step5000 | 1792 | 'GAA'<br>d=37.362; rel-margin=0.00012; freq=0 | 'GAA'<br>d=37.338; rel-margin=0.001; freq=0 | 'GAA'<br>d=37.456; rel-margin=0.000557; freq=0 | 'GAA'<br>d=37.5; rel-margin=0.000289; freq=0 |
| step5000 | 2048 | ' maintaining'<br>d=37.399; rel-margin=8.46e-05; freq=42 | 'GAA'<br>d=37.354; rel-margin=0.00116; freq=0 | 'GAA'<br>d=37.279; rel-margin=0.000611; freq=0 | 'GAA'<br>d=37.39; rel-margin=0.000631; freq=0 |
| step7000 | 256 | 'ства'<br>d=39.251; rel-margin=0.000211; freq=0 | ' ACS'<br>d=40.143; rel-margin=9.03e-06; freq=0 | 'stance'<br>d=38.661; rel-margin=6.16e-05; freq=4 | ' smartphone'<br>d=37.137; rel-margin=2.28e-05; freq=1 |
| step7000 | 512 | ' supermarket'<br>d=38.603; rel-margin=0.00011; freq=3 | ' preserv'<br>d=37.262; rel-margin=2.48e-05; freq=2 | ' GDP'<br>d=38.613; rel-margin=0.000181; freq=8 | 'GAA'<br>d=39.042; rel-margin=0.000225; freq=0 |
| step7000 | 768 | ' preserv'<br>d=38.428; rel-margin=4.86e-05; freq=2 | ' Catalog'<br>d=40.29; rel-margin=0.000218; freq=0 | 'ード'<br>d=37.558; rel-margin=0.000426; freq=0 | 'GAA'<br>d=39.093; rel-margin=0.000201; freq=0 |
| step7000 | 1024 | 'umbent'<br>d=39.097; rel-margin=4.78e-05; freq=1 | ' politique'<br>d=39.311; rel-margin=0.00015; freq=0 | ' SV'<br>d=39.482; rel-margin=1.93e-07; freq=1 | 'Tour'<br>d=37.769; rel-margin=0.000228; freq=1 |
| step7000 | 1280 | ' destinations'<br>d=37.878; rel-margin=0.000342; freq=16 | 'icrobial'<br>d=39.345; rel-margin=0.0002; freq=0 | ' repertoire'<br>d=38.607; rel-margin=9.39e-05; freq=27 | 'bibr'<br>d=40.142; rel-margin=0.000651; freq=0 |
| step7000 | 1536 | ' politique'<br>d=39.567; rel-margin=0.00115; freq=0 | ' destinations'<br>d=38.831; rel-margin=0.000479; freq=16 | ' Catalog'<br>d=38.319; rel-margin=0.000247; freq=0 | 'icrobial'<br>d=40.043; rel-margin=0.000373; freq=0 |
| step7000 | 1792 | 'bibr'<br>d=38.514; rel-margin=4.04e-05; freq=0 | 'GAA'<br>d=39.349; rel-margin=2.33e-06; freq=0 | ' GDP'<br>d=38.694; rel-margin=1.87e-06; freq=8 | ' GDP'<br>d=40.341; rel-margin=0.000157; freq=8 |
| step7000 | 2048 | ' destinations'<br>d=38.351; rel-margin=0.000642; freq=16 | 'GAA'<br>d=39.291; rel-margin=2.33e-06; freq=0 | ' foraging'<br>d=37.622; rel-margin=0.000231; freq=20 | ' landscape'<br>d=38.904; rel-margin=0.00068; freq=33 |
| step9000 | 256 | 'ervation'<br>d=38.364; rel-margin=4.54e-05; freq=8 | 'ervation'<br>d=38.404; rel-margin=3.65e-05; freq=8 | 'ervation'<br>d=38.652; rel-margin=5.33e-05; freq=8 | 'ervation'<br>d=38.761; rel-margin=7.72e-05; freq=8 |
| step9000 | 512 | 'ervation'<br>d=38.582; rel-margin=6.34e-05; freq=8 | 'ervation'<br>d=38.556; rel-margin=5.93e-05; freq=8 | 'ervation'<br>d=38.598; rel-margin=6.08e-05; freq=8 | 'ervation'<br>d=38.647; rel-margin=6.51e-05; freq=8 |
| step9000 | 768 | 'ervation'<br>d=38.617; rel-margin=6.49e-05; freq=8 | 'ervation'<br>d=38.606; rel-margin=6.38e-05; freq=8 | 'ervation'<br>d=38.608; rel-margin=6.32e-05; freq=8 | 'ervation'<br>d=38.619; rel-margin=6.38e-05; freq=8 |
| step9000 | 1024 | 'ervation'<br>d=38.618; rel-margin=6.44e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.42e-05; freq=8 | 'ervation'<br>d=38.614; rel-margin=6.4e-05; freq=8 | 'ervation'<br>d=38.615; rel-margin=6.39e-05; freq=8 |
| step9000 | 1280 | 'ervation'<br>d=38.617; rel-margin=6.41e-05; freq=8 | 'ervation'<br>d=38.617; rel-margin=6.41e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.41e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.41e-05; freq=8 |
| step9000 | 1536 | 'ervation'<br>d=38.616; rel-margin=6.41e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.42e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.41e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.4e-05; freq=8 |
| step9000 | 1792 | 'ervation'<br>d=38.616; rel-margin=6.41e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.41e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.41e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.4e-05; freq=8 |
| step9000 | 2048 | 'ervation'<br>d=38.616; rel-margin=6.41e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.4e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.4e-05; freq=8 | 'ervation'<br>d=38.616; rel-margin=6.41e-05; freq=8 |
| step13000 | 256 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step13000 | 512 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step13000 | 768 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step13000 | 1024 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step13000 | 1280 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step13000 | 1536 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step13000 | 1792 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step13000 | 2048 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 | 'escap'<br>d=40.965; rel-margin=0.000849; freq=1 |
| step21000 | 256 | ' pathogen'<br>d=45.867; rel-margin=0.000348; freq=1 | ' beaches'<br>d=48.361; rel-margin=8.83e-05; freq=23 | 'ña'<br>d=48.005; rel-margin=0.000131; freq=9 | '元'<br>d=48.293; rel-margin=0.000257; freq=0 |
| step21000 | 512 | ' scen'<br>d=44.861; rel-margin=0.00095; freq=1 | 'DE'<br>d=44.617; rel-margin=0.000119; freq=2 | '性'<br>d=45.596; rel-margin=0.000642; freq=0 | 'theme'<br>d=45.019; rel-margin=0.000557; freq=0 |
| step21000 | 768 | ' modulation'<br>d=44.209; rel-margin=0.000309; freq=2 | ' atherosclerotic'<br>d=45.802; rel-margin=0.000117; freq=0 | ' attacks'<br>d=45.611; rel-margin=0.000176; freq=173 | 'theme'<br>d=46.115; rel-margin=0.00133; freq=0 |
| step21000 | 1024 | '性'<br>d=46.41; rel-margin=0.000466; freq=0 | ' sigma'<br>d=45.681; rel-margin=0.000132; freq=0 | ' overex'<br>d=47.289; rel-margin=4.82e-05; freq=2 | 'UE'<br>d=44.973; rel-margin=6.68e-05; freq=1 |
| step21000 | 1280 | ' intimate'<br>d=47.451; rel-margin=0.000334; freq=20 | '被'<br>d=46.644; rel-margin=7.91e-05; freq=0 | 'disc'<br>d=44.771; rel-margin=0.000158; freq=1 | ' atherosclerotic'<br>d=45.672; rel-margin=1.95e-05; freq=0 |
| step21000 | 1536 | ' shack'<br>d=47.656; rel-margin=8.94e-05; freq=3 | ' ove'<br>d=45.451; rel-margin=0.000234; freq=0 | ' nud'<br>d=48.922; rel-margin=8.53e-05; freq=7 | 'mys'<br>d=47.581; rel-margin=0.000301; freq=0 |
| step21000 | 1792 | ' files'<br>d=45.63; rel-margin=0.000112; freq=5 | ' pathogen'<br>d=47.076; rel-margin=0.000252; freq=1 | 'прав'<br>d=47.558; rel-margin=0.000734; freq=0 | ' bother'<br>d=49.33; rel-margin=8.77e-05; freq=2 |
| step21000 | 2048 | 'struct'<br>d=47.196; rel-margin=8.68e-05; freq=1 | '）'<br>d=46.803; rel-margin=0.000311; freq=0 | ' shack'<br>d=45.744; rel-margin=0.000612; freq=3 | ' hanging'<br>d=45.334; rel-margin=0.000372; freq=23 |
| step29000 | 256 | ' committed'<br>d=51.128; rel-margin=0.00012; freq=64 | 'eth'<br>d=51.068; rel-margin=7.82e-05; freq=95 | ' committed'<br>d=51.206; rel-margin=0.000271; freq=64 | 'eth'<br>d=51.087; rel-margin=4.85e-05; freq=95 |
| step29000 | 512 | ' committed'<br>d=51.241; rel-margin=0.000433; freq=64 | 'aptic'<br>d=51.107; rel-margin=0.000201; freq=0 | ' committed'<br>d=51.13; rel-margin=0.000112; freq=64 | 'aptic'<br>d=51.153; rel-margin=0.000124; freq=0 |
| step29000 | 768 | ' committed'<br>d=51.361; rel-margin=0.000505; freq=64 | ' committed'<br>d=51.287; rel-margin=0.000899; freq=64 | ' committed'<br>d=51.244; rel-margin=0.000462; freq=64 | ' committed'<br>d=51.402; rel-margin=0.000657; freq=64 |
| step29000 | 1024 | ' committed'<br>d=51.107; rel-margin=7.98e-05; freq=64 | 'eth'<br>d=51.097; rel-margin=1.91e-05; freq=95 | ' committed'<br>d=51.338; rel-margin=0.00047; freq=64 | 'eth'<br>d=51.069; rel-margin=7.76e-05; freq=95 |
| step29000 | 1280 | ' committed'<br>d=51.199; rel-margin=6.1e-05; freq=64 | 'aptic'<br>d=51.114; rel-margin=0.000109; freq=0 | ' committed'<br>d=51.112; rel-margin=8.79e-05; freq=64 | 'aptic'<br>d=51.109; rel-margin=0.000205; freq=0 |
| step29000 | 1536 | ' committed'<br>d=51.428; rel-margin=0.000564; freq=64 | ' committed'<br>d=51.242; rel-margin=0.000779; freq=64 | ' committed'<br>d=51.209; rel-margin=0.000137; freq=64 | ' committed'<br>d=51.302; rel-margin=0.000862; freq=64 |
| step29000 | 1792 | 'eth'<br>d=51.079; rel-margin=7.8e-05; freq=95 | ' committed'<br>d=51.171; rel-margin=0.000199; freq=64 | ' committed'<br>d=51.424; rel-margin=0.000545; freq=64 | 'eth'<br>d=51.089; rel-margin=3.79e-05; freq=95 |
| step29000 | 2048 | 'aptic'<br>d=51.134; rel-margin=0.000172; freq=0 | ' committed'<br>d=51.131; rel-margin=5.21e-05; freq=64 | 'eth'<br>d=51.084; rel-margin=6.68e-05; freq=95 | 'aptic'<br>d=51.111; rel-margin=0.000129; freq=0 |
| step37000 | 256 | ' NS'<br>d=51.665; rel-margin=1.85e-05; freq=54 | 'mes'<br>d=51.144; rel-margin=0.00055; freq=35 | 'mes'<br>d=50.651; rel-margin=0.000747; freq=35 | 'textsf'<br>d=53.163; rel-margin=0.000229; freq=0 |
| step37000 | 512 | ' NS'<br>d=55.855; rel-margin=0.000125; freq=54 | 'fr'<br>d=55.882; rel-margin=6.04e-05; freq=36 | 'fr'<br>d=55.269; rel-margin=5.94e-05; freq=36 | ' NS'<br>d=55.356; rel-margin=8.48e-05; freq=54 |
| step37000 | 768 | ' NS'<br>d=51.511; rel-margin=0.000248; freq=54 | ' NS'<br>d=53.122; rel-margin=0.000114; freq=54 | 'mes'<br>d=50.749; rel-margin=0.000425; freq=35 | 'ele'<br>d=53.94; rel-margin=6.46e-05; freq=9 |
| step37000 | 1024 | ' NS'<br>d=55.573; rel-margin=0.000265; freq=54 | 'fr'<br>d=55.774; rel-margin=6.2e-05; freq=36 | 'fr'<br>d=55.762; rel-margin=3.63e-06; freq=36 | ' NS'<br>d=54.892; rel-margin=0.000152; freq=54 |
| step37000 | 1280 | ' NS'<br>d=54.267; rel-margin=0.000652; freq=54 | ' NS'<br>d=54.477; rel-margin=0.000378; freq=54 | ' NS'<br>d=54.186; rel-margin=0.000471; freq=54 | 'effects'<br>d=53.176; rel-margin=0.000198; freq=0 |
| step37000 | 1536 | ' views'<br>d=54.411; rel-margin=1.72e-05; freq=83 | ' NS'<br>d=55.773; rel-margin=1.71e-05; freq=54 | ' NS'<br>d=55.823; rel-margin=0.000138; freq=54 | ' NS'<br>d=55.539; rel-margin=1.44e-05; freq=54 |
| step37000 | 1792 | ' NS'<br>d=55.386; rel-margin=2.93e-05; freq=54 | ' NS'<br>d=53.561; rel-margin=0.000492; freq=54 | ' NS'<br>d=54.875; rel-margin=0.000468; freq=54 | ' pathological'<br>d=52.057; rel-margin=9.82e-06; freq=2 |
| step37000 | 2048 | ' causes'<br>d=53.653; rel-margin=5.62e-05; freq=59 | ' NS'<br>d=52.353; rel-margin=0.000222; freq=54 | ' NS'<br>d=55.315; rel-margin=0.000297; freq=54 | ' NS'<br>d=55.416; rel-margin=6.4e-05; freq=54 |
| step53000 | 256 | 'widget'<br>d=58.773; rel-margin=9.98e-05; freq=0 | ' Games'<br>d=60.17; rel-margin=0.000278; freq=164 | 'Context'<br>d=59.434; rel-margin=5.87e-05; freq=0 | 'Context'<br>d=59.323; rel-margin=5.21e-06; freq=0 |
| step53000 | 512 | 'perm'<br>d=56.642; rel-margin=2.27e-05; freq=6 | 'oker'<br>d=55.535; rel-margin=0.000894; freq=16 | 'wx'<br>d=55.329; rel-margin=0.000125; freq=0 | 'wx'<br>d=55.292; rel-margin=8.56e-05; freq=0 |
| step53000 | 768 | 'ات'<br>d=61.196; rel-margin=5.89e-05; freq=0 | 'ات'<br>d=62.664; rel-margin=0.000229; freq=0 | ' Games'<br>d=59.586; rel-margin=5.76e-06; freq=164 | ' Games'<br>d=59.623; rel-margin=1.57e-05; freq=164 |
| step53000 | 1024 | 'oker'<br>d=56.467; rel-margin=0.000338; freq=16 | 'SHORT'<br>d=59.878; rel-margin=4.49e-05; freq=0 | 'oker'<br>d=55.464; rel-margin=0.00102; freq=16 | 'oker'<br>d=55.456; rel-margin=0.00102; freq=16 |
| step53000 | 1280 | 'iterator'<br>d=62.525; rel-margin=4.96e-05; freq=0 | 'speech'<br>d=57.107; rel-margin=2.2e-06; freq=0 | ' committing'<br>d=62.7; rel-margin=0.000159; freq=10 | ' committing'<br>d=62.709; rel-margin=0.000145; freq=10 |
| step53000 | 1536 | 'Context'<br>d=59.52; rel-margin=8.45e-05; freq=0 | 'widget'<br>d=58.917; rel-margin=2.53e-06; freq=0 | 'SHORT'<br>d=59.374; rel-margin=0.000165; freq=0 | 'SHORT'<br>d=59.412; rel-margin=0.000165; freq=0 |
| step53000 | 1792 | 'wx'<br>d=55.276; rel-margin=0.000106; freq=0 | 'oker'<br>d=56.285; rel-margin=0.000168; freq=16 | '«'<br>d=59.827; rel-margin=0.000209; freq=0 | '«'<br>d=59.657; rel-margin=0.000219; freq=0 |
| step53000 | 2048 | ' Games'<br>d=59.501; rel-margin=5.77e-06; freq=164 | ' committing'<br>d=62.077; rel-margin=0.000199; freq=10 | 'widget'<br>d=58.776; rel-margin=9.18e-05; freq=0 | 'widget'<br>d=58.776; rel-margin=8.8e-05; freq=0 |
| step61000 | 256 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 |
| step61000 | 512 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 |
| step61000 | 768 | ' pastor'<br>d=69.676; rel-margin=5.23e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.23e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.23e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 |
| step61000 | 1024 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.23e-05; freq=7 |
| step61000 | 1280 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 |
| step61000 | 1536 | ' pastor'<br>d=69.676; rel-margin=5.23e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 |
| step61000 | 1792 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 |
| step61000 | 2048 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.23e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.24e-05; freq=7 | ' pastor'<br>d=69.676; rel-margin=5.23e-05; freq=7 |

## 3. LM-head最近词

- `p`：LM-head top1 softmax概率；
- `Δp`：top1与top2概率差；
- `Δlogit`：top1与top2 logit差。

| checkpoint | dynamic step | ' repetitive' | ' semi' | ' evidence' | ' orientations' |
|---|---:|---|---|---|---|
| step5000 | 256 | '\n'<br>p=0.08199; Δp=0.0269; Δlogit=0.397; freq=1 | ' to'<br>p=0.05588; Δp=0.00284; Δlogit=0.0522; freq=39263 | '\n'<br>p=0.06939; Δp=0.000451; Δlogit=0.00652; freq=1 | '\n'<br>p=0.08221; Δp=0.0268; Δlogit=0.394; freq=1 |
| step5000 | 512 | '\n'<br>p=0.06669; Δp=0.00205; Δlogit=0.0312; freq=1 | '\n'<br>p=0.05435; Δp=0.00149; Δlogit=0.0278; freq=1 | '\n'<br>p=0.07373; Δp=0.0242; Δlogit=0.397; freq=1 | '\n'<br>p=0.09307; Δp=0.0305; Δlogit=0.397; freq=1 |
| step5000 | 768 | '\n'<br>p=0.07686; Δp=0.000468; Δlogit=0.00611; freq=1 | '\n'<br>p=0.08392; Δp=0.00108; Δlogit=0.0129; freq=1 | '\n'<br>p=0.08815; Δp=0.0316; Δlogit=0.444; freq=1 | '\n'<br>p=0.05199; Δp=0.00172; Δlogit=0.0337; freq=1 |
| step5000 | 1024 | '\n'<br>p=0.07598; Δp=0.00255; Δlogit=0.0342; freq=1 | '\n'<br>p=0.08477; Δp=0.0104; Δlogit=0.13; freq=1 | '\n'<br>p=0.07568; Δp=0.0202; Δlogit=0.31; freq=1 | ','<br>p=0.06355; Δp=0.00587; Δlogit=0.0969; freq=2711 |
| step5000 | 1280 | '\n'<br>p=0.09826; Δp=0.044; Δlogit=0.594; freq=1 | ','<br>p=0.0535; Δp=0.00575; Δlogit=0.114; freq=2711 | '\n'<br>p=0.09331; Δp=0.00542; Δlogit=0.0598; freq=1 | ' class'<br>p=0.06336; Δp=0.00795; Δlogit=0.134; freq=348 |
| step5000 | 1536 | '\n'<br>p=0.06377; Δp=0.0152; Δlogit=0.271; freq=1 | '\n'<br>p=0.06518; Δp=0.0116; Δlogit=0.196; freq=1 | '\n'<br>p=0.0917; Δp=0.00807; Δlogit=0.0921; freq=1 | '\n'<br>p=0.0679; Δp=0.0104; Δlogit=0.167; freq=1 |
| step5000 | 1792 | '\n'<br>p=0.05846; Δp=0.00785; Δlogit=0.144; freq=1 | '\n'<br>p=0.08281; Δp=0.0192; Δlogit=0.263; freq=1 | '\n'<br>p=0.08228; Δp=0.0329; Δlogit=0.51; freq=1 | '\n'<br>p=0.06097; Δp=0.0135; Δlogit=0.251; freq=1 |
| step5000 | 2048 | '\n'<br>p=0.0553; Δp=0.00421; Δlogit=0.0792; freq=1 | '\n'<br>p=0.08293; Δp=0.0238; Δlogit=0.339; freq=1 | '\n'<br>p=0.082; Δp=0.0167; Δlogit=0.228; freq=1 | '\n'<br>p=0.08242; Δp=0.0275; Δlogit=0.405; freq=1 |
| step7000 | 256 | '\n'<br>p=0.1128; Δp=0.0527; Δlogit=0.63; freq=1 | ' Alexander'<br>p=0.03652; Δp=0.000217; Δlogit=0.00596; freq=84 | ','<br>p=0.03748; Δp=0.0143; Δlogit=0.479; freq=2711 | '\n'<br>p=0.06319; Δp=0.0424; Δlogit=1.11; freq=1 |
| step7000 | 512 | '\n'<br>p=0.06375; Δp=0.014; Δlogit=0.248; freq=1 | '\n'<br>p=0.1112; Δp=0.0788; Δlogit=1.23; freq=1 | ' class'<br>p=0.05165; Δp=0.0087; Δlogit=0.184; freq=348 | '\n'<br>p=0.1024; Δp=0.045; Δlogit=0.579; freq=1 |
| step7000 | 768 | '.'<br>p=0.05334; Δp=0.015; Δlogit=0.329; freq=8666 | '.'<br>p=0.1025; Δp=0.0583; Δlogit=0.84; freq=8666 | '.'<br>p=0.06559; Δp=0.0277; Δlogit=0.549; freq=8666 | '\n'<br>p=0.09763; Δp=0.0421; Δlogit=0.564; freq=1 |
| step7000 | 1024 | ' class'<br>p=0.05304; Δp=0.0266; Δlogit=0.695; freq=348 | ' '<br>p=0.07871; Δp=0.0312; Δlogit=0.504; freq=12180 | '-'<br>p=0.01495; Δp=0.00177; Δlogit=0.126; freq=17019 | '\n'<br>p=0.1429; Δp=0.104; Δlogit=1.31; freq=1 |
| step7000 | 1280 | '\n'<br>p=0.05062; Δp=0.0214; Δlogit=0.551; freq=1 | '.'<br>p=0.04533; Δp=0.0119; Δlogit=0.306; freq=8666 | '.'<br>p=0.03478; Δp=0.00181; Δlogit=0.0535; freq=8666 | 'ी'<br>p=0.04925; Δp=0.0201; Δlogit=0.525; freq=0 |
| step7000 | 1536 | ','<br>p=0.0677; Δp=0.00358; Δlogit=0.0543; freq=2711 | '.'<br>p=0.04943; Δp=0.0112; Δlogit=0.256; freq=8666 | '\n'<br>p=0.07741; Δp=0.0113; Δlogit=0.158; freq=1 | '.'<br>p=0.1081; Δp=0.0854; Δlogit=1.56; freq=8666 |
| step7000 | 1792 | '\n'<br>p=0.01505; Δp=0.00273; Δlogit=0.2; freq=1 | '\n'<br>p=0.1074; Δp=0.0368; Δlogit=0.42; freq=1 | ' class'<br>p=0.04612; Δp=0.00785; Δlogit=0.186; freq=348 | '.'<br>p=0.03593; Δp=0.0156; Δlogit=0.569; freq=8666 |
| step7000 | 2048 | '\n'<br>p=0.07898; Δp=0.0533; Δlogit=1.12; freq=1 | '\n'<br>p=0.1039; Δp=0.038; Δlogit=0.455; freq=1 | ' class'<br>p=0.03049; Δp=0.00383; Δlogit=0.134; freq=348 | '.'<br>p=0.03394; Δp=0.00534; Δlogit=0.171; freq=8666 |
| step9000 | 256 | '\n'<br>p=0.2177; Δp=0.18; Δlogit=1.75; freq=1 | '\n'<br>p=0.2112; Δp=0.172; Δlogit=1.7; freq=1 | '\n'<br>p=0.2133; Δp=0.173; Δlogit=1.66; freq=1 | '\n'<br>p=0.2207; Δp=0.181; Δlogit=1.71; freq=1 |
| step9000 | 512 | '\n'<br>p=0.2207; Δp=0.182; Δlogit=1.74; freq=1 | '\n'<br>p=0.2191; Δp=0.181; Δlogit=1.74; freq=1 | '\n'<br>p=0.2182; Δp=0.179; Δlogit=1.72; freq=1 | '\n'<br>p=0.2189; Δp=0.18; Δlogit=1.72; freq=1 |
| step9000 | 768 | '\n'<br>p=0.22; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2198; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2193; Δp=0.18; Δlogit=1.73; freq=1 | '\n'<br>p=0.2192; Δp=0.18; Δlogit=1.73; freq=1 |
| step9000 | 1024 | '\n'<br>p=0.2196; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2196; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2194; Δp=0.181; Δlogit=1.73; freq=1 |
| step9000 | 1280 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 |
| step9000 | 1536 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 |
| step9000 | 1792 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 |
| step9000 | 2048 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 | '\n'<br>p=0.2195; Δp=0.181; Δlogit=1.73; freq=1 |
| step13000 | 256 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step13000 | 512 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step13000 | 768 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step13000 | 1024 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step13000 | 1280 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step13000 | 1536 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step13000 | 1792 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step13000 | 2048 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 | '\n'<br>p=0.08579; Δp=0.00149; Δlogit=0.0175; freq=1 |
| step21000 | 256 | '.'<br>p=0.1533; Δp=0.0826; Δlogit=0.774; freq=8666 | ','<br>p=0.04546; Δp=0.00373; Δlogit=0.0857; freq=2711 | '\n'<br>p=0.08232; Δp=0.00376; Δlogit=0.0468; freq=1 | '\n'<br>p=0.1043; Δp=0.037; Δlogit=0.438; freq=1 |
| step21000 | 512 | ','<br>p=0.0905; Δp=0.0358; Δlogit=0.503; freq=2711 | '\n'<br>p=0.07088; Δp=0.0341; Δlogit=0.656; freq=1 | '\n'<br>p=0.1106; Δp=0.0383; Δlogit=0.426; freq=1 | ','<br>p=0.03418; Δp=0.00411; Δlogit=0.128; freq=2711 |
| step21000 | 768 | '\n'<br>p=0.07648; Δp=0.0344; Δlogit=0.597; freq=1 | '.'<br>p=0.1192; Δp=0.0354; Δlogit=0.352; freq=8666 | '?'<br>p=0.05261; Δp=0.0119; Δlogit=0.256; freq=0 | '\n'<br>p=0.09298; Δp=0.0271; Δlogit=0.344; freq=1 |
| step21000 | 1024 | '.'<br>p=0.1212; Δp=0.0631; Δlogit=0.736; freq=8666 | 'hen'<br>p=0.0641; Δp=0.0268; Δlogit=0.542; freq=119 | '.'<br>p=0.1507; Δp=0.0612; Δlogit=0.521; freq=8666 | ' %'<br>p=0.1387; Δp=0.107; Δlogit=1.46; freq=881 |
| step21000 | 1280 | ' \|'<br>p=0.08539; Δp=0.0009; Δlogit=0.0106; freq=3 | ','<br>p=0.09048; Δp=0.0379; Δlogit=0.543; freq=2711 | '\n'<br>p=0.06269; Δp=0.0165; Δlogit=0.306; freq=1 | '.'<br>p=0.09817; Δp=0.0363; Δlogit=0.462; freq=8666 |
| step21000 | 1536 | ' '<br>p=0.0548; Δp=0.00374; Δlogit=0.0706; freq=12180 | ' of'<br>p=0.07614; Δp=0.013; Δlogit=0.187; freq=56891 | 'u'<br>p=0.1525; Δp=0.105; Δlogit=1.16; freq=819 | '…'<br>p=0.05307; Δp=0.0212; Δlogit=0.51; freq=0 |
| step21000 | 1792 | '\n'<br>p=0.09849; Δp=0.0424; Δlogit=0.562; freq=1 | ','<br>p=0.07568; Δp=0.00296; Δlogit=0.0399; freq=2711 | '\n'<br>p=0.08299; Δp=0.00179; Δlogit=0.0218; freq=1 | ' I'<br>p=0.05788; Δp=0.0214; Δlogit=0.461; freq=2877 |
| step21000 | 2048 | ','<br>p=0.04552; Δp=0.00742; Δlogit=0.178; freq=2711 | '\n'<br>p=0.1294; Δp=0.0366; Δlogit=0.332; freq=1 | '\n'<br>p=0.1054; Δp=0.0327; Δlogit=0.372; freq=1 | 'hen'<br>p=0.04698; Δp=0.00136; Δlogit=0.0293; freq=119 |
| step29000 | 256 | ')'<br>p=0.05413; Δp=0.0201; Δlogit=0.463; freq=0 | ' I'<br>p=0.08282; Δp=0.0366; Δlogit=0.582; freq=2877 | ' I'<br>p=0.1042; Δp=0.0514; Δlogit=0.68; freq=2877 | ' I'<br>p=0.05986; Δp=0.00456; Δlogit=0.0792; freq=2877 |
| step29000 | 512 | ' '<br>p=0.05302; Δp=0.00577; Δlogit=0.115; freq=12180 | ' STATES'<br>p=0.0586; Δp=0.0195; Δlogit=0.405; freq=0 | ')'<br>p=0.05239; Δp=0.0176; Δlogit=0.41; freq=0 | ' STATES'<br>p=0.05334; Δp=0.00468; Δlogit=0.0918; freq=0 |
| step29000 | 768 | ' I'<br>p=0.09847; Δp=0.0505; Δlogit=0.718; freq=2877 | ' I'<br>p=0.05822; Δp=0.0142; Δlogit=0.28; freq=2877 | ' '<br>p=0.05439; Δp=0.0088; Δlogit=0.177; freq=12180 | ' I'<br>p=0.08133; Δp=0.0456; Δlogit=0.823; freq=2877 |
| step29000 | 1024 | ')'<br>p=0.05745; Δp=0.00975; Δlogit=0.186; freq=0 | ' I'<br>p=0.1006; Δp=0.0515; Δlogit=0.717; freq=2877 | ' I'<br>p=0.09991; Δp=0.0505; Δlogit=0.703; freq=2877 | ' I'<br>p=0.08091; Δp=0.0336; Δlogit=0.536; freq=2877 |
| step29000 | 1280 | '\n'<br>p=0.05129; Δp=0.00602; Δlogit=0.125; freq=1 | ' STATES'<br>p=0.0465; Δp=0.00918; Δlogit=0.22; freq=0 | ')'<br>p=0.05742; Δp=0.0128; Δlogit=0.253; freq=0 | ' STATES'<br>p=0.05947; Δp=0.019; Δlogit=0.386; freq=0 |
| step29000 | 1536 | ' I'<br>p=0.0899; Δp=0.0507; Δlogit=0.83; freq=2877 | ' '<br>p=0.05482; Δp=0.0165; Δlogit=0.358; freq=12180 | '\n'<br>p=0.05126; Δp=0.00461; Δlogit=0.0942; freq=1 | ' I'<br>p=0.0617; Δp=0.0196; Δlogit=0.383; freq=2877 |
| step29000 | 1792 | ' I'<br>p=0.06671; Δp=0.0136; Δlogit=0.228; freq=2877 | ' I'<br>p=0.1062; Δp=0.0537; Δlogit=0.705; freq=2877 | ' I'<br>p=0.09178; Δp=0.0516; Δlogit=0.825; freq=2877 | ' I'<br>p=0.09863; Δp=0.0499; Δlogit=0.705; freq=2877 |
| step29000 | 2048 | ' STATES'<br>p=0.05766; Δp=0.0114; Δlogit=0.22; freq=0 | ')'<br>p=0.04901; Δp=0.0134; Δlogit=0.319; freq=0 | ' I'<br>p=0.06274; Δp=0.00829; Δlogit=0.142; freq=2877 | ' STATES'<br>p=0.04877; Δp=0.0113; Δlogit=0.263; freq=0 |
| step37000 | 256 | '['<br>p=0.06241; Δp=0.0184; Δlogit=0.35; freq=0 | '                        '<br>p=0.1041; Δp=0.0562; Δlogit=0.776; freq=0 | '                        '<br>p=0.07399; Δp=0.0157; Δlogit=0.238; freq=0 | '                        '<br>p=0.1098; Δp=0.0642; Δlogit=0.88; freq=0 |
| step37000 | 512 | '\xa0'<br>p=0.1275; Δp=0.000374; Δlogit=0.00293; freq=0 | '\xa0'<br>p=0.2163; Δp=0.145; Δlogit=1.11; freq=0 | '\xa0'<br>p=0.1986; Δp=0.122; Δlogit=0.955; freq=0 | '\xa0'<br>p=0.1554; Δp=0.0818; Δlogit=0.747; freq=0 |
| step37000 | 768 | '['<br>p=0.07972; Δp=0.0349; Δlogit=0.575; freq=0 | '['<br>p=0.09635; Δp=0.048; Δlogit=0.689; freq=0 | 'ubotu'<br>p=0.06164; Δp=0.0201; Δlogit=0.395; freq=0 | '\n'<br>p=0.1385; Δp=0.0198; Δlogit=0.155; freq=1 |
| step37000 | 1024 | '['<br>p=0.1507; Δp=0.0635; Δlogit=0.548; freq=0 | '\xa0'<br>p=0.189; Δp=0.0983; Δlogit=0.734; freq=0 | '\xa0'<br>p=0.132; Δp=0.0216; Δlogit=0.179; freq=0 | '\xa0'<br>p=0.1696; Δp=0.0954; Δlogit=0.827; freq=0 |
| step37000 | 1280 | '['<br>p=0.08985; Δp=0.0181; Δlogit=0.225; freq=0 | '['<br>p=0.1149; Δp=0.048; Δlogit=0.541; freq=0 | '['<br>p=0.1165; Δp=0.0664; Δlogit=0.843; freq=0 | '\xa0'<br>p=0.05382; Δp=0.000847; Δlogit=0.0159; freq=0 |
| step37000 | 1536 | '\n'<br>p=0.1364; Δp=0.051; Δlogit=0.468; freq=1 | '['<br>p=0.1215; Δp=0.000943; Δlogit=0.00779; freq=0 | '['<br>p=0.1287; Δp=0.0329; Δlogit=0.295; freq=0 | '\xa0'<br>p=0.1386; Δp=0.0321; Δlogit=0.264; freq=0 |
| step37000 | 1792 | '\xa0'<br>p=0.1739; Δp=0.104; Δlogit=0.907; freq=0 | '['<br>p=0.114; Δp=0.0684; Δlogit=0.916; freq=0 | '['<br>p=0.1064; Δp=0.0378; Δlogit=0.438; freq=0 | '.'<br>p=0.08798; Δp=0.000919; Δlogit=0.0105; freq=8666 |
| step37000 | 2048 | '\n'<br>p=0.05576; Δp=0.0159; Δlogit=0.336; freq=1 | '['<br>p=0.0909; Δp=0.0255; Δlogit=0.329; freq=0 | '['<br>p=0.1269; Δp=0.012; Δlogit=0.0991; freq=0 | '\xa0'<br>p=0.1296; Δp=0.0189; Δlogit=0.158; freq=0 |
| step53000 | 256 | '\xa0'<br>p=0.08744; Δp=0.0232; Δlogit=0.308; freq=0 | ' '<br>p=0.09305; Δp=0.022; Δlogit=0.27; freq=12180 | '\xa0'<br>p=0.07303; Δp=0.0187; Δlogit=0.296; freq=0 | '\xa0'<br>p=0.07269; Δp=0.0153; Δlogit=0.237; freq=0 |
| step53000 | 512 | '('<br>p=0.07228; Δp=0.0333; Δlogit=0.618; freq=0 | '\n'<br>p=0.07409; Δp=0.0233; Δlogit=0.378; freq=1 | '\n'<br>p=0.04018; Δp=0.000301; Δlogit=0.00753; freq=1 | '’'<br>p=0.04013; Δp=9.58e-05; Δlogit=0.00239; freq=0 |
| step53000 | 768 | ' '<br>p=0.1234; Δp=0.0509; Δlogit=0.531; freq=12180 | ' '<br>p=0.2403; Δp=0.19; Δlogit=1.57; freq=12180 | ' '<br>p=0.0998; Δp=0.0235; Δlogit=0.269; freq=12180 | ' '<br>p=0.1008; Δp=0.0258; Δlogit=0.295; freq=12180 |
| step53000 | 1024 | '                        '<br>p=0.0666; Δp=0.00173; Δlogit=0.0263; freq=0 | '\xa0'<br>p=0.06857; Δp=0.0295; Δlogit=0.563; freq=0 | '\n'<br>p=0.0788; Δp=0.0182; Δlogit=0.262; freq=1 | '\n'<br>p=0.07855; Δp=0.0187; Δlogit=0.272; freq=1 |
| step53000 | 1280 | '\n'<br>p=0.1548; Δp=0.0197; Δlogit=0.136; freq=1 | '\n'<br>p=0.06023; Δp=0.0213; Δlogit=0.436; freq=1 | ' '<br>p=0.2553; Δp=0.202; Δlogit=1.57; freq=12180 | ' '<br>p=0.258; Δp=0.207; Δlogit=1.61; freq=12180 |
| step53000 | 1536 | '\xa0'<br>p=0.07565; Δp=0.0253; Δlogit=0.407; freq=0 | '\xa0'<br>p=0.08645; Δp=0.0257; Δlogit=0.353; freq=0 | '\xa0'<br>p=0.06379; Δp=0.0159; Δlogit=0.286; freq=0 | '\xa0'<br>p=0.06399; Δp=0.0178; Δlogit=0.325; freq=0 |
| step53000 | 1792 | '’'<br>p=0.0455; Δp=0.00194; Δlogit=0.0436; freq=0 | '('<br>p=0.08875; Δp=0.0373; Δlogit=0.546; freq=0 | '\n'<br>p=0.07738; Δp=0.0289; Δlogit=0.467; freq=1 | '\n'<br>p=0.07802; Δp=0.0336; Δlogit=0.563; freq=1 |
| step53000 | 2048 | ' '<br>p=0.09551; Δp=0.0199; Δlogit=0.234; freq=12180 | ' '<br>p=0.1493; Δp=0.0887; Δlogit=0.901; freq=12180 | '\xa0'<br>p=0.08887; Δp=0.0245; Δlogit=0.323; freq=0 | '\xa0'<br>p=0.0889; Δp=0.0247; Δlogit=0.325; freq=0 |
| step61000 | 256 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |
| step61000 | 512 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |
| step61000 | 768 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |
| step61000 | 1024 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |
| step61000 | 1280 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |
| step61000 | 1536 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |
| step61000 | 1792 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |
| step61000 | 2048 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 | ' Getty'<br>p=0.07738; Δp=0.0449; Δlogit=0.867; freq=2 |

## 4. Jacobian的三个不同尺度

对实Jacobian，特征值仍可能是复数。写作 `λᵢ=aᵢ+bᵢi`，其中
`aᵢ=Re(λᵢ)` 是实部，`bᵢ=Im(λᵢ)` 是虚部，
`|λᵢ|=sqrt(aᵢ²+bᵢ²)` 是模。三个尺度定义如下。

### 4.1 谱半径：最大特征值模

`ρ(J)=maxᵢ|λᵢ|`。它先计算每个复特征值的模，再取最大值；因此
`spectral_radius` 是非负实数，而不是某个特征值的实部。

- `spectral_radius`：最大模；
- `spectral_radius_eigenvalue_real`：达到最大模的那个特征值的实部；
- `spectral_radius_eigenvalue_imag`：同一个特征值的虚部。

离散动力系统的局部渐近稳定性主要看谱半径：`ρ(J)<1` 表示所有
线性特征模态渐近收缩，`ρ(J)>1` 表示至少一个模态局部扩张。

### 4.2 最大特征值：最大实部（谱横坐标）

复数没有天然的大小顺序，因此报告把“最大特征值”明确定义为
`α(J)=maxᵢ Re(λᵢ)`，也称谱横坐标。它按实部排序，不按模排序；
达到最大实部的特征值不一定是达到最大模的同一个特征值。

- `spectral_abscissa`：最大实部，本身就是对应特征值的实部；
- `max_real_eigenvalue_imag`：达到最大实部的那个特征值的虚部；
- `max_real_eigenvalue_abs`：该特征值的模。

虚部不表示扩张强度，而表示线性化模态中的旋转/振荡成分。对当前
离散dynamic step，是否收缩仍应优先看模和谱半径，不能只看实部。

### 4.3 算子二范数：最大奇异值

`||J||₂=σmax(J)`。奇异值是非负实数，所以算子范数没有实部和
虚部。它给出任意单位扰动经过单次线性映射后可能达到的最大长度。

- `operator_norm_2`：最大奇异值 `σ₁`；
- `operator_norm_sigma2`：第二大奇异值 `σ₂`；
- `operator_norm_over_spectral_radius`：`σ₁/ρ(J)`。

即使 `ρ(J)<1`，也可能有 `||J||₂>1`：这表示所有特征模态最终收缩，
但某些方向上的扰动可以先发生单步或短期瞬时放大。这通常来自Jacobian
的非正规性。`||J||₂/ρ(J)` 越大，这种瞬时放大与渐近尺度的差异越明显。

### 4.4 三者对照

| 指标 | 取什么量的最大值 | 模/实部/虚部 | 主要含义 |
|---|---|---|---|
| `spectral_radius` | `max |λᵢ|` | 特征值的**模** | 离散系统渐近收缩或扩张 |
| `spectral_abscissa` | `max Re(λᵢ)` | 特征值的**实部** | 最靠右的复谱位置 |
| `operator_norm_2` | `max σᵢ` | 奇异值；无实部/虚部 | 单步最大扰动放大率 |
| `*_eigenvalue_imag` | 不取最大；记录被选中特征值 | 特征值的**虚部** | 旋转或振荡成分 |

### 4.5 各checkpoint观测范围

下表中每个range都汇总该checkpoint的 `8个dynamic-step窗口端点 × 4个token = 32个Jacobian`。`[a,b]` 表示这32个样本中的最小值为
`a`、最大值为 `b`，不是置信区间，也不是误差条。

- **spectral radius range**：32个 `ρ(J)=max|λᵢ|` 的
  `[最小谱半径, 最大谱半径]`，两端都是特征值模的统计；
- **max Re(eigenvalue) range**：32个 `α(J)=max Re(λᵢ)` 的
  `[最小谱横坐标, 最大谱横坐标]`，两端都是实部的统计；
- **operator norm range**：32个 `||J||₂=σmax(J)` 的
  `[最小算子二范数, 最大算子二范数]`，两端都是奇异值；
- **norm/radius range**：对每个Jacobian先计算
  `||J||₂/ρ(J)`，再报告32个逐样本比值的 `[最小值, 最大值]`；
  它不是 `operator norm range` 的端点除以 `spectral radius range`
  的端点。比值接近1表示接近正规算子的尺度关系；明显大于1表示
  单步最大放大显著强于特征值给出的渐近尺度。

| checkpoint | spectral radius range | max Re(eigenvalue) range | operator norm range | norm/radius range |
|---|---:|---:|---:|---:|
| step5000 | [0.98033, 1.0353] | [0.97653, 1.0331] | [1.7215, 2.0096] | [1.6916, 2.0242] |
| step7000 | [0.97345, 1.0944] | [0.96807, 1.0944] | [1.8073, 2.299] | [1.6979, 2.1738] |
| step9000 | [0.97724, 1.0048] | [0.96293, 0.97494] | [1.9401, 2.0075] | [1.9308, 2.0542] |
| step13000 | [0.95164, 0.95165] | [0.95164, 0.95165] | [1.8362, 1.8363] | [1.9295, 1.9296] |
| step21000 | [0.95588, 1.181] | [0.9533, 1.181] | [1.7267, 3.1224] | [1.6384, 2.8948] |
| step29000 | [0.9379, 1.0453] | [0.93043, 1.0316] | [1.7438, 1.8741] | [1.7719, 1.8632] |
| step37000 | [0.97567, 1.052] | [0.93723, 1.052] | [1.9955, 2.3807] | [1.9234, 2.361] |
| step53000 | [0.91059, 1.1295] | [0.90715, 1.1295] | [1.694, 5.2764] | [1.8267, 4.9821] |
| step61000 | [0.89985, 0.90002] | [0.89985, 0.90002] | [1.9082, 1.9083] | [2.1203, 2.1207] |

### 4.6 最后一个窗口端点的range（t=2048）

最后一个窗口是 `1792–2048`，当前实验的
Jacobian取在其右端点 `t=2048`。下表只汇总该端点的4个
token Jacobian；每个 `[a,b]` 分别是4个token中的最小值和最大值。
列结构与上面的全8端点range表完全一致。

| checkpoint | spectral radius range | max Re(eigenvalue) range | operator norm range | norm/radius range |
|---|---:|---:|---:|---:|
| step5000 | [0.990273, 1.01719] | [0.984373, 1.01026] | [1.73295, 1.74386] | [1.70739, 1.7529] |
| step7000 | [1.00159, 1.08349] | [0.99475, 1.03902] | [1.81892, 2.1143] | [1.80559, 2.03722] |
| step9000 | [0.994397, 0.994399] | [0.96745, 0.967452] | [1.96697, 1.96701] | [1.97805, 1.97809] |
| step13000 | [0.951643, 0.951644] | [0.951643, 0.951644] | [1.83625, 1.83631] | [1.92956, 1.92962] |
| step21000 | [0.955884, 1.09884] | [0.953301, 1.09467] | [1.83285, 3.12239] | [1.73944, 2.89475] |
| step29000 | [0.955372, 1.01114] | [0.937203, 0.996079] | [1.75002, 1.85314] | [1.80303, 1.83272] |
| step37000 | [1.00884, 1.03747] | [1.00884, 1.03747] | [1.99548, 2.25336] | [1.92341, 2.23361] |
| step53000 | [0.96709, 0.995845] | [0.96709, 0.995845] | [1.77056, 2.31429] | [1.83081, 2.36782] |
| step61000 | [0.899868, 0.899994] | [0.899868, 0.899994] | [1.90822, 1.90825] | [2.12026, 2.12059] |

对应图：

- [`jacobian_spectral_radius_detailed.png`](figures/jacobian_spectral_radius_detailed.png)
- [`jacobian_max_real_eigenvalue_detailed.png`](figures/jacobian_max_real_eigenvalue_detailed.png)
- [`jacobian_operator_norm_detailed.png`](figures/jacobian_operator_norm_detailed.png)
- [`jacobian_three_metrics_by_window.png`](figures/jacobian_three_metrics_by_window.png)

