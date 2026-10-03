# Per-field error analysis: p4g_r16_lr1e4_ep8

- n_records: 383
- parse_rate: 1.0000
- mean sec/img: 13.71

## Per-field metrics

| Field | n_gt | precision | recall | F1 | soft_R | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|
| catalog_number | 383 | 0.987 | 1.000 | 0.993 | 0.996 | 378 | 5 | 0 |
| exsiccata_title | 327 | 0.982 | 0.994 | 0.988 | 0.991 | 324 | 6 | 2 |
| verbatim_elevation | 143 | 0.972 | 0.993 | 0.983 | 0.984 | 141 | 4 | 1 |
| country | 382 | 0.966 | 0.997 | 0.981 | 0.973 | 368 | 13 | 1 |
| exsiccata_number | 325 | 0.966 | 0.994 | 0.980 | 0.984 | 317 | 11 | 2 |
| verbatim_latitude | 133 | 0.941 | 1.000 | 0.970 | 0.977 | 128 | 8 | 0 |
| verbatim_identification | 383 | 0.940 | 1.000 | 0.969 | 0.971 | 360 | 23 | 0 |
| verbatim_substrate | 380 | 0.942 | 0.997 | 0.969 | 0.971 | 359 | 22 | 1 |
| type_status | 367 | 0.944 | 0.992 | 0.967 | 0.976 | 352 | 21 | 3 |
| scientific_name | 383 | 0.935 | 1.000 | 0.966 | 0.972 | 358 | 25 | 0 |
| verbatim_longitude | 133 | 0.934 | 1.000 | 0.966 | 0.967 | 127 | 9 | 0 |
| determiner | 376 | 0.934 | 0.994 | 0.963 | 0.959 | 351 | 25 | 2 |
| field_number | 126 | 0.952 | 0.968 | 0.960 | 0.962 | 120 | 6 | 4 |
| verbatim_collection_date | 379 | 0.933 | 0.983 | 0.958 | 0.952 | 350 | 25 | 6 |
| verbatim_locality | 382 | 0.916 | 0.994 | 0.953 | 0.947 | 348 | 32 | 2 |
| collector | 375 | 0.925 | 0.980 | 0.952 | 0.953 | 347 | 28 | 7 |
| verbatim_collector | 374 | 0.925 | 0.975 | 0.949 | 0.942 | 344 | 28 | 9 |
| organism_remarks | 11 | 0.833 | 0.909 | 0.870 | 0.909 | 10 | 2 | 1 |
| verbatim_habitat | 32 | 0.857 | 0.774 | 0.814 | 0.771 | 24 | 4 | 7 |
| identification_remarks | 4 | 1.000 | 0.250 | 0.400 | 0.250 | 1 | 0 | 3 |
| verbatim_identification_date | 2 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 0 | 2 |

## Error categories per field

| Field | correct | partial | wrong | omission | halluc | true_null |
|---|---|---|---|---|---|---|
| catalog_number | 378 | 5 | 0 | 0 | 0 | 0 |
| exsiccata_title | 324 | 1 | 0 | 2 | 5 | 51 |
| verbatim_elevation | 141 | 1 | 0 | 1 | 3 | 237 |
| country | 368 | 6 | 7 | 1 | 0 | 1 |
| exsiccata_number | 317 | 3 | 3 | 2 | 5 | 53 |
| verbatim_latitude | 128 | 5 | 0 | 0 | 3 | 247 |
| verbatim_identification | 360 | 20 | 3 | 0 | 0 | 0 |
| verbatim_substrate | 359 | 18 | 2 | 1 | 2 | 1 |
| type_status | 352 | 11 | 1 | 3 | 9 | 7 |
| scientific_name | 358 | 25 | 0 | 0 | 0 | 0 |
| verbatim_longitude | 127 | 6 | 0 | 0 | 3 | 247 |
| determiner | 351 | 17 | 6 | 2 | 2 | 5 |
| field_number | 120 | 2 | 0 | 4 | 4 | 253 |
| verbatim_collection_date | 350 | 18 | 5 | 6 | 2 | 2 |
| verbatim_locality | 348 | 26 | 6 | 2 | 0 | 1 |
| collector | 347 | 17 | 4 | 7 | 7 | 1 |
| verbatim_collector | 344 | 15 | 6 | 9 | 7 | 2 |
| organism_remarks | 10 | 0 | 0 | 1 | 2 | 370 |
| verbatim_habitat | 24 | 1 | 0 | 7 | 3 | 348 |
| identification_remarks | 1 | 0 | 0 | 3 | 0 | 379 |
| verbatim_identification_date | 0 | 0 | 0 | 2 | 0 | 381 |

## Top-10 worst cases per field (sorted by similarity asc)


### `catalog_number` (F1=0.993, n_gt=383)

| sim | image | gt | pred |
|---|---|---|---|
| 0.70 | LE F-43621_2.JPG | LE F-43621 | LE F-60263 |
| 0.70 | LE F-60833_2.JPG | LE F-60833 | LE F-64263 |
| 0.70 | LE F-64506_2.JPG | LE F-64506 | LE F-60823 |
| 0.70 | LE F-67371_2.JPG | LE F-67371 | LE F-60233 |
| 0.80 | LE F-64889_2.JPG | LE F-64889 | LE F-64823 |

### `exsiccata_title` (F1=0.988, n_gt=327)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | Rabenhorst, Fungi europaei | <NULL> |
| 0.00 | LE F-63703.JPG | R. Maire. — Mycotheca boreali-africana | <NULL> |
| 0.38 | LE F-67371_2.JPG | Rabenhorst, Fungi europaei | Ehrenberg's Exsiccatae |

### `verbatim_elevation` (F1=0.983, n_gt=143)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-61500.JPG | 2430 | <NULL> |
| 0.82 | LE F-214044.JPG | Alt. c. 300 m. | Alt. 10 m. |

### `country` (F1=0.981, n_gt=382)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-62244_2.JPG | Russia | <NULL> |
| 0.14 | LE F-62028_2.JPG | Dominica | Brazil |
| 0.17 | LE F-63703.JPG | Algeria | Egypt |
| 0.18 | LE F-62024_1.JPG | Dominica | Czech Republic |
| 0.25 | LE F-262530.JPG | Tajikistan | Russia |
| 0.25 | LE F-43533_2.JPG | Kazakhstan | Russia |
| 0.25 | LE F-43596.JPG | USA | Italy |
| 0.27 | LE F-62002.JPG | Germany | Dominica |
| 0.31 | LE F-43612.JPG | Poland | Germany |
| 0.31 | LE F-43621_2.JPG | Germany | Russia |

### `exsiccata_number` (F1=0.980, n_gt=325)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_2.JPG | 1997 | <NULL> |
| 0.00 | LE F-63703.JPG | 134 | <NULL> |
| 0.29 | LE F-60833_2.JPG | 1698 | 283 |
| 0.29 | LE F-64889_2.JPG | 1698 | 283 |
| 0.29 | LE F-67371_2.JPG | 1997 | 116 |
| 0.50 | LE F-61707.JPG | 42 | 49 |
| 0.67 | LE F-43595.JPG | 3111a | 3114 |
| 0.75 | LE F-67371_1.JPG | 1997 | 1992 |

### `verbatim_latitude` (F1=0.970, n_gt=133)

| sim | image | gt | pred |
|---|---|---|---|
| 0.45 | LE F-314150.JPG | 43°34'49" N | 16°56'40" S |
| 0.75 | LE F-62040.JPG | 46°17' N | 46,17 N. |
| 0.78 | LE F-214028.JPG | 25°47'06'' N | 25°47'06" S |
| 0.80 | LE F-43574.JPG | 45,56 N | 46,56 N. |
| 0.80 | LE F-43575.JPG | 47°28' N | 47,28 N |

### `verbatim_identification` (F1=0.969, n_gt=383)

| sim | image | gt | pred |
|---|---|---|---|
| 0.25 | LE F-43621_2.JPG | Ustilago Tulasnei Kühn | Sorosporium corinum bezichtete (von diesem selbst mit der Blüte)te Bräune |
| 0.26 | LE F-38021.JPG | Exobasidium Vaccini-uliginosi var. Ledi m. | Eutypella Vankyae † |
| 0.30 | LE F-67371_1.JPG | Ustilago Tulasnei Kühn | Tilletia Sorghi-vulgaris Tul. in Ann. d. Sc. Nat. III. T. 9 |
| 0.30 | LE F-67371_2.JPG | Ustilago Tulasnei Kühn | Sorosporium coronatum (in sensu latiore) nov. spec. |
| 0.33 | LE F-43621_1.JPG | Ustilago Tulasnei Kühn | Tilletia Sorghi-vulgarii Tul. & C. Tul. 1874 |
| 0.35 | LE F-43628.JPG | Ustilago turcomanica n. sp. | Triticum squarrosum Roth |
| 0.37 | LE F-64889_1.JPG | Ustilago urceolorum mihi (in sensu latiore). forma nova: Scirpi † | Usilago Scripi † |
| 0.40 | LE F-60833_2.JPG | Ustilago urceolorum mihi (in sensu latiore). forma nova: scirpi † | Ustil. Montagn. nov. spec. |
| 0.48 | LE F-62002.JPG | Entyloma endogenum | Protomyces endogenus Emt. conf. nov. |
| 0.49 | LE F-64889_2.JPG | Ustilago urceolorum mihi (in sensu latiore). forma nova: Scirpi † | Ustil. urceolorum Tul. et Usil. & O. Montagn. |

### `verbatim_substrate` (F1=0.969, n_gt=380)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-62244_2.JPG | на Spodiopogon sibiricus Trin. | <NULL> |
| 0.05 | LE F-43628.JPG | Triticum squarrosum Roth | Ангреканск. пшеница |
| 0.28 | LE F-62024_1.JPG | In pedunculis et inflorescentiae Cissi spec. | auf Cyssi DC. |
| 0.34 | LE F-64889_2.JPG | Scirpus caespitosus var. nemorosus Roth | auf dem Brandpilze heimisch. Carex nemorosa |
| 0.36 | LE F-60833_1.JPG | Scirpus caespitosus var. nemorosus Roth | Uredos Careis Pers. — Uredos urceolorum DC. — Uredos urceolorum mihi (in sensu l |
| 0.40 | LE F-67371_2.JPG | In Sorghi vulgari ovariis | auf Sorgho |
| 0.44 | LE F-60833_2.JPG | Scirpus caespitosus var. nemorosus Roth | auf Sorghum vulgare Pers. |
| 0.45 | LE F-64889_1.JPG | Scirpus caespitosus var. nemorosus Roth | U. soris heteroideis, in oppositis lateribus appianatis |
| 0.45 | LE F-63693.JPG | Melica nutans | Antarctica: Antipolitica mutans |
| 0.54 | LE F-63181.JPG | на Poa nemoralis L. | на Elymus sibiricus L. |

### `type_status` (F1=0.967, n_gt=367)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43628.JPG | Type | <NULL> |
| 0.00 | LE F-65375_1.JPG | Type | <NULL> |
| 0.00 | LE F-66284.JPG | Type | <NULL> |
| 0.19 | LE F-61085.JPG | Original material | Type |
| 0.31 | LE F-61708.JPG | Authentic | Type |
| 0.31 | LE F-62409.JPG | Authentic | Type |
| 0.33 | LE F-198017.JPG | Type | Fragment of holotype |
| 0.44 | LE F-214060.JPG | Isoneotype | Paratype |
| 0.53 | LE F-262349.JPG | Paratype | Isotype |
| 0.67 | LE F-198020.JPG | Type | Paratype |

### `scientific_name` (F1=0.966, n_gt=383)

| sim | image | gt | pred |
|---|---|---|---|
| 0.31 | LE F-38021.JPG | Exobasidium nannfeldtii V.A. Dudka | Eutypella vankyana Y.S. Chen & Z.Y. Li |
| 0.35 | LE F-66285.JPG | Ustilago michnoana Lavrov | Tilletia striaeformis West. |
| 0.36 | LE F-43628.JPG | Ustilago turcomanica Tranzschel | Triticum squarrosum Roth |
| 0.39 | LE F-43621_1.JPG | Ustilago tulasnei J.G. Kühn | Tilletia sorghi-vulgarii Tul. & C. Tul. |
| 0.39 | LE F-67371_1.JPG | Ustilago tulasnei J.G. Kühn | Tilletia sorghi-vulgaris Tul. & C. Tul. |
| 0.46 | LE F-62002.JPG | Entyloma endogenum (Unger) Wünsche | Protomyces endogenus Holway |
| 0.50 | LE F-43621_2.JPG | Ustilago tulasnei J.G. Kühn | Sorosporium corinum J.G. Kühn |
| 0.52 | LE F-67371_2.JPG | Ustilago tulasnei J.G. Kühn | Sorosporium coronatum J.G. Kühn |
| 0.60 | LE F-43531.JPG | Sorosporium lavrovianum Smarods | Sorospora lavrovii Šķipsna |
| 0.62 | LE F-63654.JPG | Urocystis leimbachii Oertel | Urocystis cremnatheri Bertel |

### `verbatim_longitude` (F1=0.966, n_gt=133)

| sim | image | gt | pred |
|---|---|---|---|
| 0.43 | LE F-314150.JPG | 141°59'04" E | 26°48'12" E |
| 0.59 | LE F-62039.JPG | 20°00' E | 20° 00' E |
| 0.67 | LE F-227018.JPG | 144°03'21" E | 144°51'58" E |
| 0.75 | LE F-62040.JPG | 25°09' E | 25,09 E. |
| 0.78 | LE F-214028.JPG | 28°10'38'' W | 28°10'38" E |
| 0.80 | LE F-43575.JPG | 20°54' E | 20,54 E |

### `determiner` (F1=0.963, n_gt=376)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43628.JPG | Tranzschel W. H. | <NULL> |
| 0.00 | LE F-65375_1.JPG | Kühn J. G. | <NULL> |
| 0.21 | LE F-43525.JPG | Schröter J. | Vánky K. |
| 0.22 | LE F-65375_2.JPG | Kühn J. G. | Fries A. |
| 0.22 | LE F-67159.JPG | Juel O. | Johann Gottlieb Kühn |
| 0.27 | LE F-63641.JPG | Bubák F. | Duby A. |
| 0.28 | LE F-43621_1.JPG | Kühn J. G. | Tul. Ges. z. Pilzkrk. 1874 |
| 0.29 | LE F-64889_1.JPG | Kühn J. G. | Montagne F. |
| 0.30 | LE F-64889_2.JPG | Kühn J. G. | Tulasne E. |
| 0.32 | LE F-43532.JPG | Smarods J. | Lavrov N. N. |

### `field_number` (F1=0.960, n_gt=126)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43494.JPG | N.A.F. 3136 | <NULL> |
| 0.00 | LE F-43579.JPG | 182 | <NULL> |
| 0.00 | LE F-43593.JPG | 182 | <NULL> |
| 0.00 | LE F-43628.JPG | 19.А | <NULL> |
| 0.47 | LE F-262347.JPG | 925/6001 | 925, 6001 |
| 0.75 | LE F-214149.JPG | 5164 | 5165 |

### `verbatim_collection_date` (F1=0.958, n_gt=379)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-201575.JPG | 1899 | <NULL> |
| 0.00 | LE F-43509.JPG | [NULL] | 11/VIII 24 |
| 0.00 | LE F-43585.JPG | [NULL] | <NULL> |
| 0.00 | LE F-43593.JPG | [NULL] | 182 |
| 0.00 | LE F-43621_2.JPG | 1874 | <NULL> |
| 0.00 | LE F-60833_1.JPG | 6. Septbr. 1871 | <NULL> |
| 0.00 | LE F-61801.JPG | [NULL] | 1906 |
| 0.00 | LE F-62244_2.JPG | 30/VIII 26 | <NULL> |
| 0.00 | LE F-64889_1.JPG | 6. Septbr. 1871 | [NULL] |
| 0.00 | LE F-66949.JPG | Juni 1868 | <NULL> |

### `verbatim_locality` (F1=0.953, n_gt=382)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-60833_1.JPG | auf dem Brockenfelde, einer weiten Hochmoorfläche in der Nähe des Brockenkegels | <NULL> |
| 0.00 | LE F-62244_2.JPG | Приморская с.х. оп. ст. | <NULL> |
| 0.14 | LE F-66924.JPG | Halle | Auf einem Stoppelfelde des Dorfes Rauscha in der Görlitzer Haide |
| 0.18 | LE F-62935.JPG | auf Altendorf bei Ratibor, Oberschlesien | Halle |
| 0.18 | LE F-62002.JPG | Freiburg | Dominica, insula Antillarum prope Charlottetown |
| 0.23 | LE F-67371_2.JPG | in horto instituti oeconomici Universitatis Halensis | in dem Golö-Lande |
| 0.23 | LE F-62024_1.JPG | Dominica, insula Antillarum prope Charlottetown | auf Cyssi DC. |
| 0.30 | LE F-43620.JPG | Jamaica | Jamaica. Hab. in hort. Soc. Cult. Charl. |
| 0.33 | LE F-62244_1.JPG | Приморская с.х. оп. ст. | по х. Буранды |
| 0.34 | LE F-43621_2.JPG | in horto instituti oeconomici Universitatis Halensis | einem von Dr. Schwinfurth in dem Golö-Lande (80° n. B.) |

### `collector` (F1=0.952, n_gt=375)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_1.JPG | Kühn J. G. | <NULL> |
| 0.00 | LE F-62028_2.JPG | Wullschlägel H. R. | <NULL> |
| 0.00 | LE F-62244_2.JPG | Abramov I. N. | <NULL> |
| 0.00 | LE F-64889_1.JPG | Kühn J. G. | <NULL> |
| 0.00 | LE F-66923.JPG | Kühn J. G. | <NULL> |
| 0.00 | LE F-66949.JPG | Reil W. | <NULL> |
| 0.00 | LE F-67371_1.JPG | Kühn J. G. | <NULL> |
| 0.18 | LE F-67159.JPG | Juel O. | Oğuz |
| 0.22 | LE F-43621_2.JPG | Kühn J. G. | Braun F. |
| 0.22 | LE F-62024_1.JPG | Wullschlägel H. R. | Presl von |

### `verbatim_collector` (F1=0.949, n_gt=374)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43621_1.JPG | Julius Kühn | <NULL> |
| 0.00 | LE F-60833_1.JPG | Julius Kühn | <NULL> |
| 0.00 | LE F-62028_2.JPG | Wullschlägel | <NULL> |
| 0.00 | LE F-62244_2.JPG | Абрамов | <NULL> |
| 0.00 | LE F-64889_1.JPG | Julius Kühn | <NULL> |
| 0.00 | LE F-66904.JPG | Prof. Dr. Julius Kühn | <NULL> |
| 0.00 | LE F-66923.JPG | Prof. Dr. Julius Kühn | <NULL> |
| 0.00 | LE F-66949.JPG | Herrn Dr. med. Reil in Cairo | <NULL> |
| 0.00 | LE F-67371_1.JPG | Julius Kühn | <NULL> |
| 0.12 | LE F-43577.JPG | P. Magnus | Herr Ule |

### `organism_remarks` (F1=0.870, n_gt=11)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-43561.JPG | amphigena, sed in pagina superiore opulentius evoluta | <NULL> |

### `verbatim_habitat` (F1=0.814, n_gt=32)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-198013.JPG | предгорье, глубокая щель, первый увал, сев. зап. ск. в тени, в зарослях шиповник | <NULL> |
| 0.00 | LE F-198015.JPG | в лесу среди зарослей Filipendula palmata Max. | <NULL> |
| 0.00 | LE F-38021.JPG | Сырой лес | <NULL> |
| 0.00 | LE F-43532.JPG | in agro experimentali | <NULL> |
| 0.00 | LE F-43612.JPG | des Dorfes Rauscha in der Görlitzer Haide | <NULL> |
| 0.00 | LE F-43628.JPG | Пески | <NULL> |
| 0.00 | LE F-63417.JPG | заболоченный луг-алас | <NULL> |
| 0.72 | LE F-61978.JPG | сорняк в посевах | по полям, едва ли не в посевах |

### `identification_remarks` (F1=0.400, n_gt=4)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-199437.JPG | Unacum Sporisorium tripogonis Gandhe, Sharma & Vánky | <NULL> |
| 0.00 | LE F-201413.JPG | unacum Tilletia eragrostiellae | <NULL> |
| 0.00 | LE F-343151.JPG | as Urocystis golovinii Kalymb. | <NULL> |

### `verbatim_identification_date` (F1=0.000, n_gt=2)

| sim | image | gt | pred |
|---|---|---|---|
| 0.00 | LE F-198015.JPG | 20/VII. 36 | <NULL> |
| 0.00 | LE F-198016.JPG | 20. VII. 1936 г. | <NULL> |
