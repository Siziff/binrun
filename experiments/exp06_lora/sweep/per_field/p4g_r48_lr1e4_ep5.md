# Per-field error analysis: p4g_r48_lr1e4_ep5

- n_records: 386
- parse_rate: 1.0000
- mean sec/img: 13.94

## Per-field metrics

| Field | n_gt | precision | recall | F1 | soft_R | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| catalog_number | 386 | 0.987 | 1.000 | 0.993 | 0.995 | 381 | 5 | 0 |
| verbatim_elevation | 147 | 0.980 | 0.993 | 0.986 | 0.982 | 145 | 3 | 1 |
| exsiccata_title | 329 | 0.979 | 0.988 | 0.983 | 0.986 | 323 | 7 | 4 |
| exsiccata_number | 327 | 0.976 | 0.991 | 0.983 | 0.983 | 319 | 8 | 3 |
| country | 385 | 0.961 | 0.997 | 0.979 | 0.969 | 369 | 15 | 1 |
| verbatim_identification | 386 | 0.956 | 1.000 | 0.977 | 0.976 | 369 | 17 | 0 |
| scientific_name | 386 | 0.951 | 1.000 | 0.975 | 0.975 | 367 | 19 | 0 |
| verbatim_substrate | 382 | 0.945 | 0.997 | 0.971 | 0.975 | 363 | 21 | 1 |
| verbatim_latitude | 135 | 0.942 | 1.000 | 0.970 | 0.979 | 130 | 8 | 0 |
| verbatim_longitude | 135 | 0.942 | 1.000 | 0.970 | 0.971 | 130 | 8 | 0 |
| determiner | 379 | 0.935 | 0.997 | 0.965 | 0.964 | 358 | 25 | 1 |
| field_number | 128 | 0.968 | 0.960 | 0.964 | 0.955 | 121 | 4 | 5 |
| verbatim_collector | 378 | 0.930 | 0.997 | 0.962 | 0.966 | 357 | 27 | 1 |
| type_status | 370 | 0.937 | 0.989 | 0.962 | 0.974 | 354 | 24 | 4 |
| verbatim_collection_date | 382 | 0.932 | 0.992 | 0.961 | 0.953 | 355 | 26 | 3 |
| collector | 378 | 0.927 | 0.997 | 0.961 | 0.967 | 355 | 28 | 1 |
| verbatim_locality | 385 | 0.922 | 0.994 | 0.957 | 0.944 | 353 | 30 | 2 |
| verbatim_habitat | 33 | 0.923 | 0.727 | 0.814 | 0.726 | 24 | 2 | 9 |
| organism_remarks | 11 | 0.714 | 0.455 | 0.556 | 0.455 | 5 | 2 | 6 |
| verbatim_identification_date | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 1 |
| identification_remarks | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 4 |

## Error categories per field

| Field | correct | partial | wrong | omission | halluc | true_null |
|---|---|---|---|---|---|---|
| catalog_number | 381 | 5 | 0 | 0 | 0 | 0 |
| verbatim_elevation | 145 | 1 | 0 | 1 | 2 | 237 |
| exsiccata_title | 323 | 2 | 0 | 4 | 5 | 52 |
| exsiccata_number | 319 | 3 | 2 | 3 | 3 | 56 |
| country | 369 | 9 | 6 | 1 | 0 | 1 |
| verbatim_identification | 369 | 13 | 4 | 0 | 0 | 0 |
| scientific_name | 367 | 19 | 0 | 0 | 0 | 0 |
| verbatim_substrate | 363 | 16 | 2 | 1 | 3 | 1 |
| verbatim_latitude | 130 | 5 | 0 | 0 | 3 | 248 |
| verbatim_longitude | 130 | 5 | 0 | 0 | 3 | 248 |
| determiner | 358 | 15 | 5 | 1 | 5 | 2 |
| field_number | 121 | 2 | 0 | 5 | 2 | 256 |
| verbatim_collector | 357 | 16 | 4 | 1 | 7 | 1 |
| type_status | 354 | 10 | 2 | 4 | 12 | 4 |
| verbatim_collection_date | 355 | 17 | 7 | 3 | 2 | 2 |
| collector | 355 | 20 | 2 | 1 | 6 | 2 |
| verbatim_locality | 353 | 20 | 10 | 2 | 0 | 1 |
| verbatim_habitat | 24 | 0 | 0 | 9 | 2 | 351 |
| organism_remarks | 5 | 0 | 0 | 6 | 2 | 373 |
| verbatim_identification_date | 0 | 0 | 0 | 1 | 0 | 385 |
| identification_remarks | 0 | 0 | 0 | 4 | 0 | 382 |

## Top-10 worst cases per field (sorted by similarity asc)


### `catalog_number` (F1=0.993, n_gt=386)

| sim | image | gt | pred |
|---|---|---|---|
| 0.60 | LE F-43621_2.JPG | LE F-43621 | LE F-10183 |
| 0.60 | LE F-64506_2.JPG | LE F-64506 | LE F-10183 |
| 0.60 | LE F-64889_2.JPG | LE F-64889 | LE F-10183 |
| 0.60 | LE F-67371_2.JPG | LE F-67371 | LE F-10686 |
| 0.70 | LE F-60833_2.JPG | LE F-60833 | LE F-11783 |

### `verbatim_elevation` (F1=0.986, n_gt=147)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-61500.JPG | 2430 | <NULL> |
| 0.82 | LE F-214044.JPG | Alt. c. 300 m. | Alt. 10 m. |

### `exsiccata_title` (F1=0.983, n_gt=329)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-201575.JPG | Raciborski Crypt. paras. Javae | <NULL> |
| 0.00 | LE F-43621_2.JPG | Rabenhorst, Fungi europaei | <NULL> |
| 0.00 | LE F-63703.JPG | R. Maire. — Mycotheca boreali-africana | <NULL> |
| 0.00 | LE F-67371_2.JPG | Rabenhorst, Fungi europaei | <NULL> |
| 0.80 | LE F-43532.JPG | J. Smarods, Fungi latvici exsiccati | J. Šmurov. Fungi Lavrovii exsiccati |
| 0.81 | LE F-43627.JPG | Tranzschel et Serebrianikow. Mycotheca rossica | Tranzschel et P. Tranzschel, Mycotheca rossica |

### `exsiccata_number` (F1=0.983, n_gt=327)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | 1997 | <NULL> |
| 0.00 | LE F-63703.JPG | 134 | <NULL> |
| 0.00 | LE F-67371_2.JPG | 1997 | <NULL> |
| 0.29 | LE F-60833_2.JPG | 1698 | 283 |
| 0.29 | LE F-64889_2.JPG | 1698 | 283 |
| 0.50 | LE F-61707.JPG | 42 | 49 |
| 0.75 | LE F-43530.JPG | 3111a | 311 |
| 0.75 | LE F-67371_1.JPG | 1997 | 1992 |

### `country` (F1=0.979, n_gt=385)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43596.JPG | USA | Greece |
| 0.00 | LE F-62024_1.JPG | Dominica | <NULL> |
| 0.11 | LE F-200589.JPG | Saudi Arabia | Morocco |
| 0.14 | LE F-62028_2.JPG | Dominica | Brazil |
| 0.17 | LE F-63703.JPG | Algeria | Egypt |
| 0.25 | LE F-262530.JPG | Tajikistan | Russia |
| 0.25 | LE F-61500.JPG | Kazakhstan | Russia |
| 0.31 | LE F-43594.JPG | Russia | Ukraine |
| 0.31 | LE F-43612.JPG | Poland | Germany |
| 0.31 | LE F-62002.JPG | Germany | Sweden |

### `verbatim_identification` (F1=0.977, n_gt=386)

| sim | image | gt | pred |
|---|---|---|---|
| 0.23 | LE F-64889_2.JPG | Ustilago urceolorum mihi (in sensu latiore). forma nova: Scirpi † | Usil. Montag. |
| 0.25 | LE F-67371_2.JPG | Ustilago Tulasnei Kühn | Sorosporium coroninum bezeichnete (Material stützte) Sorghi Link's |
| 0.26 | LE F-43621_2.JPG | Ustilago Tulasnei Kühn | Sorosporium corinum bezeichnete (von diesem selbst mit der Blüte)teile |
| 0.27 | LE F-43628.JPG | Ustilago turcomanica n. sp. | Exilicum squarrosum Roth |
| 0.35 | LE F-38021.JPG | Exobasidium Vaccini-uliginosi var. Ledi m. | Eriophyes Vasilyi - miyinii |
| 0.41 | LE F-62002.JPG | Entyloma endogenum | Protomyces endogenes gen. nov. Eutypocos endogenes P. Dietel |
| 0.54 | LE F-43531.JPG | Sorosporium Lavrovianum Smarods Sp. nova | Sorospora Lavrovii N. N. Lavrov nom. imp. |
| 0.55 | LE F-67371_1.JPG | Ustilago Tulasnei Kühn | Tilletia Sorghi-vulgarii Kühn |
| 0.56 | LE F-60833_2.JPG | Ustilago urceolorum mihi (in sensu latiore). forma nova: scirpi † | Ustil. urceolorum mihi var. major Desm. |
| 0.66 | LE F-43532.JPG | Sorosporium Lavrovianum Smarods Sp. nova | Sorosporium Lavrovii Lavrov |

### `scientific_name` (F1=0.975, n_gt=386)

| sim | image | gt | pred |
|---|---|---|---|
| 0.30 | LE F-62002.JPG | Entyloma endogenum (Unger) Wünsche | Protomyces dietelii M. A. Lavrov |
| 0.30 | LE F-65375_2.JPG | Ustilago grandis Fr. | Phragmitis communis var. † |
| 0.31 | LE F-43628.JPG | Ustilago turcomanica Tranzschel | Exilicum squarrosum J. Kunze |
| 0.32 | LE F-66285.JPG | Ustilago michnoana Lavrov | Tilletia striaeformis Oudem. ex A.A. Fisch. Waldh. |
| 0.33 | LE F-38021.JPG | Exobasidium nannfeldtii V.A. Dudka | Eriophyes vasilyi miyinii Y. Zinserling |
| 0.50 | LE F-43621_2.JPG | Ustilago tulasnei J.G. Kühn | Sorosporium corinum J.G. Kühn |
| 0.50 | LE F-67371_2.JPG | Ustilago tulasnei J.G. Kühn | Sorosporium coroninum (Ehrenb. ex Link) J.G. Kühn |
| 0.57 | LE F-64889_1.JPG | Ustilago urceolorum f. scirpi J.G. Kühn | Ustilago scripi Wullschlägel |
| 0.62 | LE F-67371_1.JPG | Ustilago tulasnei J.G. Kühn | Tilletia sorghi-vulgarii J.G. Kühn |
| 0.68 | LE F-43531.JPG | Sorosporium lavrovianum Smarods | Sorospora lavrovii Lavrov |

### `verbatim_substrate` (F1=0.971, n_gt=382)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-62037.JPG | Carex glareosa | <NULL> |
| 0.18 | LE F-65375_1.JPG | Phragmitis communis | Dieser seltenen Brandpilz wurde von Herrn Secretär Oertel ... an dem dieser Para |
| 0.26 | LE F-62935.JPG | Secalis cerealis ovaria occupat | bei Rathbor. Geb. des Herrn J. Polomski auf Altendorf |
| 0.31 | LE F-64889_2.JPG | Scirpus caespitosus var. nemorosus Roth | auf dem Moore des Brockenkegels |
| 0.39 | LE F-43628.JPG | Triticum squarrosum Roth | Auf Anoplolepis obscuriceps Mayr |
| 0.40 | LE F-62024_1.JPG | In pedunculis et inflorescentiae Cissi spec. | auf Cyssi DC. eine neue Gattung der Ustilagineen „Mycosyrinx“ zu gründen |
| 0.40 | LE F-64889_1.JPG | Scirpus caespitosus var. nemorosus Roth | In Sorghi vulgaris ovariis |
| 0.42 | LE F-60833_2.JPG | Scirpus caespitosus var. nemorosus Roth | auf Sorghi vulgaris ovariis |
| 0.54 | LE F-43612.JPG | an der cultivirten Form von Panicum sanguinale beobachtet und an mehreren andere | Panicum sanguinalem |
| 0.65 | LE F-65002.JPG | an Sorghum sacharatum | an Sorghum vulgare cultivirt |

### `verbatim_latitude` (F1=0.970, n_gt=135)

| sim | image | gt | pred |
|---|---|---|---|
| 0.45 | LE F-314150.JPG | 43°34'49" N | 16°56'40" S |
| 0.78 | LE F-214028.JPG | 25°47'06'' N | 25°47'06" S |
| 0.80 | LE F-43574.JPG | 45,56 N | 46,56 N. |
| 0.80 | LE F-43575.JPG | 47°28' N | 47,28 N |
| 0.80 | LE F-62040.JPG | 46°17' N | 46,17 N |

### `verbatim_longitude` (F1=0.970, n_gt=135)

| sim | image | gt | pred |
|---|---|---|---|
| 0.43 | LE F-314150.JPG | 141°59'04" E | 26°48'12" E |
| 0.67 | LE F-227018.JPG | 144°03'21" E | 144°51'58" E |
| 0.78 | LE F-214028.JPG | 28°10'38'' W | 28°10'38" E |
| 0.80 | LE F-43575.JPG | 20°54' E | 20,54 E |
| 0.80 | LE F-62040.JPG | 25°09' E | 25,09 E |

### `determiner` (F1=0.965, n_gt=379)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-65375_1.JPG | Kühn J. G. | <NULL> |
| 0.10 | LE F-63641.JPG | Bubák F. | Lagerheim G. |
| 0.24 | LE F-43509.JPG | Saccardo P. A., Paoletti G. | Martyanov N. V. |
| 0.25 | LE F-67159.JPG | Juel O. | J.G. Kühn |
| 0.26 | LE F-65375_2.JPG | Kühn J. G. | Rabenhorst L. |
| 0.29 | LE F-43628.JPG | Tranzschel W. H. | Penner P. G. |
| 0.31 | LE F-61708.JPG | Niessl von Mayendorf G. | Thümen F. |
| 0.32 | LE F-43531.JPG | Smarods J. | Lavrov N. N. |
| 0.32 | LE F-43532.JPG | Smarods J. | Lavrov N. N. |
| 0.32 | LE F-43579.JPG | Fischer von Waldheim A. A. | Ehrenberg C. G. |

### `field_number` (F1=0.964, n_gt=128)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-198021.JPG | 9467 | <NULL> |
| 0.00 | LE F-43494.JPG | N.A.F. 3136 | <NULL> |
| 0.00 | LE F-43579.JPG | 182 | <NULL> |
| 0.00 | LE F-43593.JPG | 182 | <NULL> |
| 0.00 | LE F-43628.JPG | 19.А | <NULL> |
| 0.55 | LE F-262347.JPG | 925/6001 | 925 |
| 0.75 | LE F-214149.JPG | 5164 | 5165 |

### `verbatim_collector` (F1=0.962, n_gt=378)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43628.JPG | Андросовъ | P. Penner |
| 0.00 | LE F-62024_1.JPG | Wullschlägel | <NULL> |
| 0.11 | LE F-62028_2.JPG | Wullschlägel | Poeppig |
| 0.15 | LE F-43614.JPG | Prof. Dr. Julius Kühn | Oertel |
| 0.15 | LE F-43551.JPG | Woronow | H.G.W. |
| 0.32 | LE F-62935.JPG | Herrn J. Polomski | Prof. Dr. Julius Kühn |
| 0.35 | LE F-43621_1.JPG | Julius Kühn | Wullschlägel |
| 0.35 | LE F-60833_1.JPG | Julius Kühn | Wullschlägel |
| 0.35 | LE F-64889_1.JPG | Julius Kühn | Wullschlägel |
| 0.38 | LE F-66949.JPG | Herrn Dr. med. Reil in Cairo | Prof. Dr. Kühn |

### `type_status` (F1=0.962, n_gt=370)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43509.JPG | Type | <NULL> |
| 0.00 | LE F-43628.JPG | Type | <NULL> |
| 0.00 | LE F-65375_1.JPG | Type | <NULL> |
| 0.00 | LE F-66284.JPG | Type | <NULL> |
| 0.19 | LE F-61085.JPG | Original material | Type |
| 0.25 | LE F-62409.JPG | Authentic | Isotype |
| 0.31 | LE F-61708.JPG | Authentic | Type |
| 0.33 | LE F-198017.JPG | Type | Fragment of holotype |
| 0.44 | LE F-214060.JPG | Isoneotype | Paratype |
| 0.53 | LE F-262349.JPG | Paratype | Isotype |

### `verbatim_collection_date` (F1=0.961, n_gt=382)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-201575.JPG | 1899 | <NULL> |
| 0.00 | LE F-43509.JPG | [NULL] | 7.VI |
| 0.00 | LE F-43593.JPG | [NULL] | 182 |
| 0.00 | LE F-43617.JPG | [NULL] | <NULL> |
| 0.00 | LE F-60833_1.JPG | 6. Septbr. 1871 | [NULL] |
| 0.00 | LE F-61801.JPG | [NULL] | 23.VI.25 |
| 0.00 | LE F-62024_1.JPG | [NULL] | <NULL> |
| 0.12 | LE F-198021.JPG | Novemb.-Decemb. | 9467. 21/Ido 8 agra |
| 0.13 | LE F-43613.JPG | [NULL] | Anno 1876 |
| 0.29 | LE F-62244_2.JPG | 30/VIII 26 | 1910. 10.VI |

### `collector` (F1=0.961, n_gt=378)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-62024_1.JPG | Wullschlägel H. R. | <NULL> |
| 0.23 | LE F-43628.JPG | Androsov N. V. | Penner P. G. |
| 0.24 | LE F-66949.JPG | Reil W. | Kühn J. G. |
| 0.32 | LE F-62028_2.JPG | Wullschlägel H. R. | Poeppig H. W. |
| 0.36 | LE F-43621_1.JPG | Kühn J. G. | Wullschlägel H. R. |
| 0.36 | LE F-64889_1.JPG | Kühn J. G. | Wullschlägel F. A. |
| 0.36 | LE F-43614.JPG | Kühn J. G. | Oertel W. G. |
| 0.37 | LE F-67159.JPG | Juel O. | Jules Émile Planchon |
| 0.42 | LE F-43596.JPG | Holway E. W. D. | Dietel P. |
| 0.43 | LE F-60833_1.JPG | Kühn J. G. | Wullschlägel H. G. |

### `verbatim_locality` (F1=0.957, n_gt=385)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43509.JPG | [Минусинск] | Алай, пр.п. |
| 0.00 | LE F-62024_1.JPG | Dominica, insula Antillarum prope Charlottetown | <NULL> |
| 0.00 | LE F-63904.JPG | [Arduenna] | <NULL> |
| 0.14 | LE F-43621_2.JPG | in horto instituti oeconomici Universitatis Halensis | Halle |
| 0.14 | LE F-67371_2.JPG | in horto instituti oeconomici Universitatis Halensis | Halle |
| 0.14 | LE F-66924.JPG | Halle | Auf einem Stoppelfelde des Dorfes Rauscha in der Görlitzer Haide |
| 0.15 | LE F-66949.JPG | bei Saokara auf dem Ruinenfelde des alten Memphis | Halle |
| 0.15 | LE F-43628.JPG | Мервский у., Закасп. обл. Пески блз. ст Репетекъ | Neuenahr, pr. Bassen opp. Mayen |
| 0.18 | LE F-61801.JPG | [Минусинск] | Приморская с.х. оп. ст. |
| 0.18 | LE F-62935.JPG | auf Altendorf bei Ratibor, Oberschlesien | Halle |

### `verbatim_habitat` (F1=0.814, n_gt=33)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-198013.JPG | предгорье, глубокая щель, первый увал, сев. зап. ск. в тени, в зарослях шиповник | <NULL> |
| 0.00 | LE F-198015.JPG | в лесу среди зарослей Filipendula palmata Max. | <NULL> |
| 0.00 | LE F-198021.JPG | locis saxosis | <NULL> |
| 0.00 | LE F-262530.JPG | по болотистым долинным лугам | <NULL> |
| 0.00 | LE F-38021.JPG | Сырой лес | <NULL> |
| 0.00 | LE F-43532.JPG | in agro experimentali | <NULL> |
| 0.00 | LE F-43612.JPG | des Dorfes Rauscha in der Görlitzer Haide | <NULL> |
| 0.00 | LE F-43628.JPG | Пески | <NULL> |
| 0.00 | LE F-63417.JPG | заболоченный луг-алас | <NULL> |

### `organism_remarks` (F1=0.556, n_gt=11)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43558.JPG | amphigena, sed in pagina superiore opulentius evoluta | <NULL> |
| 0.00 | LE F-43559.JPG | amphigena, sed in pagina superiore opulentius evoluta | <NULL> |
| 0.00 | LE F-43560.JPG | amphigena, sed in pagina superiore opulentius evoluta | <NULL> |
| 0.00 | LE F-43561.JPG | amphigena, sed in pagina superiore opulentius evoluta | <NULL> |
| 0.00 | LE F-43562.JPG | amphigena, sed in pagina superiore opulentius evoluta | <NULL> |
| 0.00 | LE F-43563.JPG | amphigena, sed in pagina superiore opulentius evoluta | <NULL> |

### `verbatim_identification_date` (F1=0.000, n_gt=1)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-198015.JPG | 20/VII. 36 | <NULL> |

### `identification_remarks` (F1=0.000, n_gt=4)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-200589.JPG | https://www.biodiversitylibrary.org/page/202160 | <NULL> |
| 0.00 | LE F-201413.JPG | unacum Tilletia eragrostiellae | <NULL> |
| 0.00 | LE F-201416.JPG | unacum Sporisorium polytriadis | <NULL> |
| 0.00 | LE F-343151.JPG | as Urocystis golovinii Kalymb. | <NULL> |
