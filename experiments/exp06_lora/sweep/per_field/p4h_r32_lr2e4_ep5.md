# Per-field error analysis: p4h_r32_lr2e4_ep5

- n_records: 378
- parse_rate: 0.9974
- mean sec/img: 13.97

## Per-field metrics

| Field | n_gt | precision | recall | F1 | soft_R | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| catalog_number | 378 | 0.992 | 0.995 | 0.993 | 0.992 | 373 | 3 | 2 |
| exsiccata_title | 324 | 0.991 | 0.985 | 0.988 | 0.984 | 319 | 3 | 5 |
| verbatim_elevation | 143 | 0.973 | 1.000 | 0.986 | 0.990 | 142 | 4 | 0 |
| country | 377 | 0.979 | 0.992 | 0.985 | 0.978 | 366 | 8 | 3 |
| exsiccata_number | 322 | 0.975 | 0.984 | 0.980 | 0.977 | 312 | 8 | 5 |
| verbatim_identification | 378 | 0.958 | 0.997 | 0.977 | 0.980 | 361 | 16 | 1 |
| verbatim_substrate | 374 | 0.952 | 0.997 | 0.974 | 0.979 | 358 | 18 | 1 |
| verbatim_latitude | 131 | 0.948 | 1.000 | 0.973 | 0.979 | 127 | 7 | 0 |
| scientific_name | 378 | 0.944 | 0.997 | 0.970 | 0.974 | 356 | 21 | 1 |
| verbatim_longitude | 131 | 0.940 | 1.000 | 0.969 | 0.972 | 126 | 8 | 0 |
| determiner | 371 | 0.941 | 0.997 | 0.968 | 0.969 | 352 | 22 | 1 |
| verbatim_collector | 370 | 0.951 | 0.983 | 0.967 | 0.962 | 351 | 18 | 6 |
| type_status | 362 | 0.938 | 0.991 | 0.964 | 0.976 | 347 | 23 | 3 |
| verbatim_collection_date | 374 | 0.946 | 0.980 | 0.963 | 0.952 | 349 | 20 | 7 |
| verbatim_locality | 377 | 0.930 | 0.991 | 0.960 | 0.956 | 348 | 26 | 3 |
| collector | 370 | 0.941 | 0.980 | 0.960 | 0.959 | 348 | 22 | 7 |
| field_number | 124 | 0.958 | 0.943 | 0.950 | 0.938 | 115 | 5 | 7 |
| verbatim_habitat | 32 | 0.897 | 0.812 | 0.852 | 0.805 | 26 | 3 | 6 |
| organism_remarks | 11 | 0.800 | 0.727 | 0.762 | 0.727 | 8 | 2 | 3 |
| identification_remarks | 4 | 1.000 | 0.250 | 0.400 | 0.250 | 1 | 0 | 3 |
| verbatim_identification_date | 1 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 1 |

## Error categories per field

| Field | correct | partial | wrong | omission | halluc | true_null |
|---|---|---|---|---|---|---|
| catalog_number | 373 | 3 | 0 | 2 | 0 | 0 |
| exsiccata_title | 319 | 0 | 0 | 5 | 3 | 51 |
| verbatim_elevation | 142 | 1 | 0 | 0 | 3 | 232 |
| country | 366 | 7 | 1 | 3 | 0 | 1 |
| exsiccata_number | 312 | 3 | 2 | 5 | 3 | 53 |
| verbatim_identification | 361 | 15 | 1 | 1 | 0 | 0 |
| verbatim_substrate | 358 | 14 | 1 | 1 | 3 | 1 |
| verbatim_latitude | 127 | 4 | 0 | 0 | 3 | 244 |
| scientific_name | 356 | 19 | 2 | 1 | 0 | 0 |
| verbatim_longitude | 126 | 5 | 0 | 0 | 3 | 244 |
| determiner | 352 | 13 | 5 | 1 | 4 | 3 |
| verbatim_collector | 351 | 10 | 3 | 6 | 5 | 3 |
| type_status | 347 | 11 | 1 | 3 | 11 | 5 |
| verbatim_collection_date | 349 | 13 | 5 | 7 | 2 | 2 |
| verbatim_locality | 348 | 21 | 5 | 3 | 0 | 1 |
| collector | 348 | 14 | 1 | 7 | 7 | 1 |
| field_number | 115 | 2 | 0 | 7 | 3 | 251 |
| verbatim_habitat | 26 | 0 | 0 | 6 | 3 | 343 |
| organism_remarks | 8 | 0 | 0 | 3 | 2 | 365 |
| identification_remarks | 1 | 0 | 0 | 3 | 0 | 374 |
| verbatim_identification_date | 0 | 0 | 0 | 1 | 0 | 377 |

## Top-10 worst cases per field (sorted by similarity asc)


### `catalog_number` (F1=0.993, n_gt=378)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | LE F-43621 | <NULL> |
| 0.00 | LE F-67371_2.JPG | LE F-67371 | <NULL> |
| 0.60 | LE F-64889_2.JPG | LE F-64889 | LE F-10183 |
| 0.70 | LE F-60833_2.JPG | LE F-60833 | LE F-10186 |
| 0.70 | LE F-64506_2.JPG | LE F-64506 | LE F-10186 |

### `exsiccata_title` (F1=0.988, n_gt=324)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-201575.JPG | Raciborski Crypt. paras. Javae | <NULL> |
| 0.00 | LE F-43516.JPG | Herbarium K. Starcs | <NULL> |
| 0.00 | LE F-43621_2.JPG | Rabenhorst, Fungi europaei | <NULL> |
| 0.00 | LE F-63703.JPG | R. Maire. — Mycotheca boreali-africana | <NULL> |
| 0.00 | LE F-67371_2.JPG | Rabenhorst, Fungi europaei | <NULL> |

### `verbatim_elevation` (F1=0.986, n_gt=143)

| sim | image | gt | pred |
|---|---|---|---|
| 0.82 | LE F-214044.JPG | Alt. c. 300 m. | Alt. 10 m. |

### `country` (F1=0.985, n_gt=377)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | Germany | <NULL> |
| 0.00 | LE F-62024_1.JPG | Dominica | <NULL> |
| 0.00 | LE F-62028_2.JPG | Dominica | <NULL> |
| 0.27 | LE F-64889_1.JPG | Germany | Dominica |
| 0.31 | LE F-43612.JPG | Poland | Germany |
| 0.31 | LE F-62002.JPG | Germany | Sweden |
| 0.31 | LE F-63703.JPG | Algeria | Greece |
| 0.33 | LE F-198021.JPG | South Africa | Russia |
| 0.33 | LE F-43621_1.JPG | Germany | Sudan |
| 0.38 | LE F-68068.JPG | Uzbekistan | Russia |

### `exsiccata_number` (F1=0.980, n_gt=322)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-201575.JPG | 81 | <NULL> |
| 0.00 | LE F-43516.JPG | 218 | <NULL> |
| 0.00 | LE F-43621_2.JPG | 1997 | <NULL> |
| 0.00 | LE F-63703.JPG | 134 | <NULL> |
| 0.00 | LE F-67371_2.JPG | 1997 | <NULL> |
| 0.29 | LE F-60833_2.JPG | 1698 | 283 |
| 0.29 | LE F-64889_2.JPG | 1698 | 283 |
| 0.50 | LE F-61707.JPG | 42 | 49 |
| 0.75 | LE F-43530.JPG | 3111a | 311 |
| 0.75 | LE F-67371_1.JPG | 1997 | 1992 |

### `verbatim_identification` (F1=0.977, n_gt=378)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | Ustilago Tulasnei Kühn | <NULL> |
| 0.12 | LE F-67371_2.JPG | Ustilago Tulasnei Kühn | Sorosporium corinum bezeichnete (von diesemselben Original-Exemplare) Material s |
| 0.38 | LE F-38021.JPG | Exobasidium Vaccini-uliginosi var. Ledi m. | Erysiphe Vankyini - n. sp. |
| 0.53 | LE F-60833_2.JPG | Ustilago urceolorum mihi (in sensu latiore). forma nova: scirpi † | Ustil. urceolorum mihi und entipschript der Ustil. Montagnei var. major Desm. |
| 0.55 | LE F-64889_2.JPG | Ustilago urceolorum mihi (in sensu latiore). forma nova: Scirpi † | Ustil. urceolorum mihi und entipsicht der Usil. Montagnei Sauer |
| 0.57 | LE F-62002.JPG | Entyloma endogenum | Eutylocosceus endogenus Fresen. |
| 0.69 | LE F-43531.JPG | Sorosporium Lavrovianum Smarods Sp. nova | Sorosporium Lavrov nom. imp. nova species |
| 0.75 | LE F-62244_1.JPG | Sorosporium Abramovianum Lavr. | Sorosporium Pekenthanum Lavr. |
| 0.77 | LE F-43622.JPG | Ustilago Tragi Săvulescu | Sphacelotheea Tragi Săvulescu |
| 0.77 | LE F-61500.JPG | Entyloma primulae Murashk. nov. sp. | Entyloma primulae Woronich. |

### `verbatim_substrate` (F1=0.974, n_gt=374)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | In Sorghi vulgaris ovariis | <NULL> |
| 0.29 | LE F-65375_1.JPG | Phragmitis communis | Dieser seltenen Brandpilz wurde von ... aufgenommen |
| 0.31 | LE F-62024_1.JPG | In pedunculis et inflorescentiae Cissi spec. | Uredo Cyssi DC. |
| 0.34 | LE F-64889_1.JPG | Scirpus caespitosus var. nemorosus Roth | In Scirpo latifolio |
| 0.37 | LE F-62935.JPG | Secalis cerealis ovaria occupat | Tilletia Carii auf seinem Oberschlesien |
| 0.54 | LE F-43612.JPG | an der cultivirten Form von Panicum sanguinale beobachtet und an mehreren andere | Panicum sanguinalem |
| 0.59 | LE F-43621_1.JPG | In Sorghi vulgaris ovariis | Sorghum vulgare |
| 0.59 | LE F-64889_2.JPG | Scirpus caespitosus var. nemorosus Roth | Carex nemorosa Roth |
| 0.60 | LE F-61859.JPG | In fol. Ranunculi multifidi | Auf St. Rammurii multifidi |
| 0.65 | LE F-65008.JPG | an Sorghum racharatum | an Sorghum vulgare cultivirt |

### `verbatim_latitude` (F1=0.973, n_gt=131)

| sim | image | gt | pred |
|---|---|---|---|
| 0.45 | LE F-314150.JPG | 43°34'49" N | 16°56'40" S |
| 0.67 | LE F-43574.JPG | 45,56 N | 46°56' N |
| 0.78 | LE F-214028.JPG | 25°47'06'' N | 25°47'06" S |
| 0.80 | LE F-62040.JPG | 46°17' N | 46,17 N |

### `scientific_name` (F1=0.970, n_gt=378)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | Ustilago tulasnei J.G. Kühn | <NULL> |
| 0.25 | LE F-65375_2.JPG | Ustilago grandis Fr. | Phragmitis communis var. oertelii J.G. Kühn |
| 0.26 | LE F-66285.JPG | Ustilago michnoana Lavrov | Tilletia striaeformis (West.) Oudem. & Beij. |
| 0.31 | LE F-67371_2.JPG | Ustilago tulasnei J.G. Kühn | Sorosporium corinum A.A. Fisch. Waldh. |
| 0.39 | LE F-38021.JPG | Exobasidium nannfeldtii V.A. Dudka | Erysiphe vankyinii Zin. |
| 0.52 | LE F-62002.JPG | Entyloma endogenum (Unger) Wünsche | Eutylocosceus endogenus Fresen. |
| 0.56 | LE F-64889_1.JPG | Ustilago urceolorum f. scirpi J.G. Kühn | Ustilago scirpi var. dominica A.A. Fisch. Waldh. |
| 0.58 | LE F-198021.JPG | Ustilago dregeana Tul. & C. Tul. | Ustilago sogdiana Tranzschel |
| 0.60 | LE F-62024_1.JPG | Mycosyrinx cissi (DC.) Beck | Mykosyrinx beckii A.A. Fisch. Waldh. |
| 0.68 | LE F-43523.JPG | Protomyces limosellae J. Kunze | Uredo limosellae J.G. Kühn |

### `verbatim_longitude` (F1=0.969, n_gt=131)

| sim | image | gt | pred |
|---|---|---|---|
| 0.43 | LE F-314150.JPG | 141°59'04" E | 26°48'12" E |
| 0.67 | LE F-227018.JPG | 144°03'21" E | 144°51'58" E |
| 0.78 | LE F-214028.JPG | 28°10'38'' W | 28°10'38" E |
| 0.80 | LE F-43574.JPG | 25,20 E | 25°20' E |
| 0.80 | LE F-62040.JPG | 25°09' E | 25,09 E |

### `determiner` (F1=0.968, n_gt=371)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | Kühn J. G. | <NULL> |
| 0.10 | LE F-63641.JPG | Bubák F. | Lagerheim G. |
| 0.18 | LE F-43535.JPG | Cruchet P. | Zundel K. A. |
| 0.21 | LE F-43525.JPG | Schröter J. | Vánky K. |
| 0.24 | LE F-67159.JPG | Juel O. | Kühn J. G. |
| 0.24 | LE F-64889_1.JPG | Kühn J. G. | Fischer von Waldheim A. A. |
| 0.32 | LE F-43579.JPG | Fischer von Waldheim A. A. | Ehrenberg C. G. |
| 0.39 | LE F-61500.JPG | Murashkinsky K. E. | Woronichin N. N. |
| 0.46 | LE F-198021.JPG | Tulasne E. | Tranzschel W. H. |
| 0.49 | LE F-63181.JPG | Kuznetsova M. N., Schwarzman S. R. | Kusnezowa M. N. |

### `verbatim_collector` (F1=0.967, n_gt=370)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | Julius Kühn | <NULL> |
| 0.00 | LE F-62024_1.JPG | Wullschlägel | <NULL> |
| 0.00 | LE F-62028_2.JPG | Wullschlägel | <NULL> |
| 0.00 | LE F-65005.JPG | Kühn | <NULL> |
| 0.00 | LE F-66905.JPG | Prof. Dr. Julius Kühn | <NULL> |
| 0.00 | LE F-67371_1.JPG | Julius Kühn | <NULL> |
| 0.12 | LE F-43577.JPG | P. Magnus | Herr Ule |
| 0.15 | LE F-43614.JPG | Prof. Dr. Julius Kühn | Oertel |
| 0.22 | LE F-43621_1.JPG | Julius Kühn | Dr. Schweinfurth |
| 0.33 | LE F-43594.JPG | М. К. Хохряков | N. N. Хризант |

### `type_status` (F1=0.964, n_gt=362)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | Type | <NULL> |
| 0.00 | LE F-62935.JPG | Type | <NULL> |
| 0.00 | LE F-67371_2.JPG | Type | <NULL> |
| 0.19 | LE F-61085.JPG | Original material | Type |
| 0.31 | LE F-61708.JPG | Authentic | Type |
| 0.31 | LE F-62409.JPG | Authentic | Type |
| 0.33 | LE F-198017.JPG | Type | Fragment of holotype |
| 0.44 | LE F-214060.JPG | Isoneotype | Paratype |
| 0.53 | LE F-262349.JPG | Paratype | Isotype |
| 0.67 | LE F-198020.JPG | Type | Paratype |

### `verbatim_collection_date` (F1=0.963, n_gt=374)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-198021.JPG | Novemb.-Decemb. | 9467 |
| 0.00 | LE F-201575.JPG | 1899 | <NULL> |
| 0.00 | LE F-43593.JPG | [NULL] | 182 |
| 0.00 | LE F-43621_1.JPG | 1874 | <NULL> |
| 0.00 | LE F-43621_2.JPG | 1874 | <NULL> |
| 0.00 | LE F-61801.JPG | [NULL] | 7.VIII.1895 |
| 0.00 | LE F-62024_1.JPG | [NULL] | <NULL> |
| 0.00 | LE F-62028_1.JPG | [NULL] | <NULL> |
| 0.00 | LE F-62028_2.JPG | [NULL] | <NULL> |
| 0.00 | LE F-64889_1.JPG | 6. Septbr. 1871 | [NULL] |

### `verbatim_locality` (F1=0.960, n_gt=377)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | in horto instituti oeconomici Universitatis Halensis | <NULL> |
| 0.00 | LE F-62024_1.JPG | Dominica, insula Antillarum prope Charlottetown | <NULL> |
| 0.00 | LE F-62028_2.JPG | Dominica, insula Antillarum prope Charlottetown | <NULL> |
| 0.14 | LE F-67371_2.JPG | in horto instituti oeconomici Universitatis Halensis | Halle |
| 0.15 | LE F-66924.JPG | Halle | Auf einem Stoppelfelde des Dorfes Rauscha in der Nähe Halles |
| 0.18 | LE F-66949.JPG | bei Saokara auf dem Ruinenfelde des alten Memphis | in Cairo |
| 0.25 | LE F-62002.JPG | Freiburg | [Suecia] |
| 0.27 | LE F-43621_1.JPG | in horto instituti oeconomici Universitatis Halensis | in dem Goloe-Lande (8° n. B.) |
| 0.30 | LE F-64889_1.JPG | auf dem Brockenfelde, einer weiten Hochmoorfläche in der Nähe des Brockenkegels | Dominica, insula Antillarum prope Charlottetown |
| 0.35 | LE F-43620.JPG | Jamaica | Jamaica. ... prope urbem Westbury |

### `collector` (F1=0.960, n_gt=370)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-22421.JPG | Saccardo D. | <NULL> |
| 0.00 | LE F-43621_2.JPG | Kühn J. G. | <NULL> |
| 0.00 | LE F-62024_1.JPG | Wullschlägel H. R. | <NULL> |
| 0.00 | LE F-62028_2.JPG | Wullschlägel H. R. | <NULL> |
| 0.00 | LE F-65005.JPG | Kühn J. G. | <NULL> |
| 0.00 | LE F-66905.JPG | Kühn J. G. | <NULL> |
| 0.00 | LE F-67371_1.JPG | Kühn J. G. | <NULL> |
| 0.29 | LE F-43577.JPG | Magnus P. W. | Ule H. R. |
| 0.33 | LE F-60833_1.JPG | Kühn J. G. | Wullschlägel G. |
| 0.36 | LE F-64889_1.JPG | Kühn J. G. | Wullschlägel H. R. |

### `field_number` (F1=0.950, n_gt=124)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-198021.JPG | 9467 | <NULL> |
| 0.00 | LE F-227086.JPG | 5417/B | <NULL> |
| 0.00 | LE F-43488.JPG | 4915 | <NULL> |
| 0.00 | LE F-43494.JPG | N.A.F. 3136 | <NULL> |
| 0.00 | LE F-43579.JPG | 182 | <NULL> |
| 0.00 | LE F-43593.JPG | 182 | <NULL> |
| 0.00 | LE F-43628.JPG | 19.А | <NULL> |
| 0.55 | LE F-262347.JPG | 925/6001 | 925 |
| 0.75 | LE F-214149.JPG | 5164 | 5165 |

### `verbatim_habitat` (F1=0.852, n_gt=32)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-198015.JPG | в лесу среди зарослей Filipendula palmata Max. | <NULL> |
| 0.00 | LE F-198021.JPG | locis saxosis | <NULL> |
| 0.00 | LE F-38021.JPG | Сырой лес | <NULL> |
| 0.00 | LE F-43532.JPG | in agro experimentali | <NULL> |
| 0.00 | LE F-43612.JPG | des Dorfes Rauscha in der Görlitzer Haide | <NULL> |
| 0.00 | LE F-43628.JPG | Пески | <NULL> |

### `organism_remarks` (F1=0.762, n_gt=11)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43560.JPG | amphigena, sed in pagina superiore opulentius evoluta | <NULL> |
| 0.00 | LE F-43561.JPG | amphigena, sed in pagina superiore opulentius evoluta | <NULL> |
| 0.00 | LE F-43562.JPG | amphigena, sed in pagina superiore opulentius evoluta | <NULL> |

### `identification_remarks` (F1=0.400, n_gt=4)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-199437.JPG | Unacum Sporisorium tripogonis Gandhe, Sharma & Vánky | <NULL> |
| 0.00 | LE F-201413.JPG | unacum Tilletia eragrostiellae | <NULL> |
| 0.00 | LE F-343151.JPG | as Urocystis golovinii Kalymb. | <NULL> |

### `verbatim_identification_date` (F1=0.000, n_gt=1)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-198015.JPG | 20/VII. 36 | <NULL> |
