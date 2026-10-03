# Эксперимент 1 — top-5 best / worst примеры


## Top-5 best (самые «лёгкие» — все движки справились)

### 1. `LE F-341598.JPG`  (mean sim = 0.923, std = 0.024, min = 0.895, max = 0.955)

**Ground truth (выборка полей):**
- `scientific_name` = Ustilago panici-proliferi Henn.
- `verbatim_identification` = Ustilago Panici-proliferi P. Henn. n. sp.
- `type_status` = Type
- `verbatim_collector` = E. W. D. Holway
- `verbatim_locality` = Mexico: pr. urbem Mexico
- `country` = Mexico
- `verbatim_collection_date` = 10. 10. 1898
- `verbatim_substrate` = In spicis Panici proliferi var. acuminati
- `exsiccata_title` = Sydow, Ustilagineen
- `exsiccata_number` = 212

**Per-engine similarity:**

| engine | sim |
|---|---:|
| Tesseract | 0.942 |
| PaddleOCR | 0.907 |
| EasyOCR | 0.895 |
| docTR | 0.955 |
| Surya | 0.906 |
| TrOCR-pr | 0.955 |
| TrOCR-hw | 0.903 |

**OCR-выдача (top + worst движков):**

- **docTR** (sim=0.95): `LE F-341598 LE F-341598 Sydow, Ustilagineen. 212. Ustilago Panici-proliferi P. Henn. n. sp. Mexico: pr. urbem Mexico. In spicis Panici proliferi var. acuminati. 10. 10. 1898. leg. E. W. D. Holway.`
- **TrOCR-pr** (sim=0.95): `LE F-341598 LE F-341598 SYDOW, USTILAGINEEN. 212. USTILAGO PANICI-PROLIFERI P. HENN. N. SP. MEXICO: PR. URBEM MEXICO. IN SPICIS PANICI PROLIFERI VAR. ACUMINATI. 10. 10. 1898. LEG. E. W. D. HOLWAY.`
- **EasyOCR** (sim=0.90): `LE F-341598 LE F-341598 Sydow, Ustilagineen. 212. Ustilago Panici-proliferi Р. Henn n sp . Mexico : pr. urbem Mexico. In spicis Panici proliferi var: acuminati: 10. 10. 1898. leg. Е W. D: Holway.`

---
### 2. `LE F-22421.JPG`  (mean sim = 0.922, std = 0.036, min = 0.861, max = 0.964)

**Ground truth (выборка полей):**
- `scientific_name` = Exobasidium patavinum D. Sacc.
- `verbatim_identification` = Exobasidium patavinum D. Sacc. sp. n.
- `type_status` = Type
- `verbatim_locality` = Padova — in horto botanico
- `country` = Italy
- `verbatim_collection_date` = Nov. 1897
- `verbatim_substrate` = in foliis Ilicis Aquifolii
- `exsiccata_title` = D. Saccardo — Mycotheca italica
- `exsiccata_number` = 8

**Per-engine similarity:**

| engine | sim |
|---|---:|
| Tesseract | 0.956 |
| PaddleOCR | 0.891 |
| EasyOCR | 0.900 |
| docTR | 0.949 |
| Surya | 0.964 |
| TrOCR-pr | 0.861 |
| TrOCR-hw | 0.937 |

**OCR-выдача (top + worst движков):**

- **Surya** (sim=0.96): `16 33431 D. SACCARDO - Mycotheca italica anno acco does anno secon cons anno 2002. THELEPHORACEAE 8 * Exobasidium patavinum D. Sacc. sp. n. D. Sacc. Fung. Ven. Nov. (mox edendi) Padova — in horto botanico — in foliis Ilicis Aquifolii - Nov. 1897. ....................................... TII I KU EI KE EI KU HEKE E HEREHE KE HEI KE EIK LE F-22421 1. Exobasidium patavinum D. Sacc. sp. n. Myc. Ital. …`
- **Tesseract** (sim=0.96): `5 f | É Е : q 3 eR sini i. — — Dk OA UE coo iin RN : 290 i x LE 22421 99 0. SACCARDO — Mycotheca italica 7^ "" ” | | 8. THELEPHORACEAE | | 2 *Exobasidium patavinum D. Sacc. sp. n. | ee D. Sacc. Fung. Ven. Nov. (mox edendi) | | | Padova — in horto botanico — in foliis Ilicis Aquifolii | | uu c | — Noy. 1897. | и т | Ebo = m а | & Sa E | LE F-22421 | | P 1. Exobasidium patavinum D. Sacc. sp. n. …`
- **TrOCR-pr** (sim=0.86): `D. SACCARDO - MYCOTHECA ITALICA LE 22421 8. *EXOBASIDIUM PATAVINUM D. SACC. SP. N. D. SACC. FUNG.VEN. NOV. (MOX EDENDI) PADORA - IN HOTTO BOTANICO - IN FOLLS LICIS AQUIFOLII NOV. 1897. LE F-22421 L. EXOBASIDIUM PATAVINUM D. SACC. SP. N. MYC. ITAL. N. 8. HYPOOPBALLAM, ALBAN 7, DILATISSIME CHERENN, MARNAS INREPHALES 2-6 MIN. IATAS, PETEMLOSS-VELLA- TINAS FORMANS; BASILIS CREBRE PARAILLELELE …`

---
### 3. `LE F-43584.JPG`  (mean sim = 0.922, std = 0.038, min = 0.844, max = 0.962)

**Ground truth (выборка полей):**
- `scientific_name` = Ustilago cilinodis Savile
- `verbatim_identification` = Ustilago cilinodis Savile sp. nov.
- `type_status` = Type
- `verbatim_collector` = W.A. Wojtas
- `verbatim_locality` = 1.1 Km. along the road to the Jet Trail, Riding Mountain National Park, MANITOBA
- `country` = Canada
- `verbatim_collection_date` = 1 Aug., 1979
- `verbatim_substrate` = on Polygonum cilinode Michx.

**Per-engine similarity:**

| engine | sim |
|---|---:|
| Tesseract | 0.945 |
| PaddleOCR | 0.916 |
| EasyOCR | 0.844 |
| docTR | 0.962 |
| Surya | 0.934 |
| TrOCR-pr | 0.957 |
| TrOCR-hw | 0.897 |

**OCR-выдача (top + worst движков):**

- **docTR** (sim=0.96): `LE-43584. EX NATIONAL MYCOLOGICAL HERBARIUM OF CANADA HERBIER NATIONAL DE MYCOLOGIE DU CANADA DAOM 183263 DATE 1 Aug., 1979 Ustilago cilinodis Savile sp. nov. on Polygonum cilinode Michx. In shaded area of a Populus sp. woods 1.1 Km. along the road to the Jet Trail Riding Mountain National Park, MANITOBA Coll.: W.A. Wojtas (No. 918) Det.: J.A. Parmelee LE F-43584`
- **TrOCR-pr** (sim=0.96): `LE-43584 - EX NATIONAL MYCOLOGICAL HERBARIUM DU CANADA DAOM 183263 DAREL AUG., 1979 USTILAGO CILINODIS SAVILE SP. NOV. ON POLYGONUM CILINODE MICHX. IN SHADED AREA OF A POPULUS SP. WOODS 1.1 KM. ALONG THE ROAD TO THE JET TRAIL RIDING MOUNTAIN NATIONAL PARK, MANITOBA COLL.: W.A. WOJTAS (NO. 918) DET.: J.A. PARMELEE LE F-43584`
- **EasyOCR** (sim=0.84): `LE-43584 . EX NATIONAL MYCOLOGICAL HERBARIUN OF CANADA HERBIER NATIONAL DE MYCOLOGIE DU CANADA D A ОМ 183263 0 4 r Е ] 1979 Ustilago cilinodis Savile sp . nov . on Polygonum cilinode Michx In shaded area of a Populus sp . woods 1.1 Km . along the road to the Je t Trail Riding Mountain National Park , MANITOBA : Coll.: W. A . Wojtas (No . 918) Det. : J. A . Parmelee Е LE F-43584 Aug. ,`

---
### 4. `LE F-43511.JPG`  (mean sim = 0.894, std = 0.032, min = 0.836, max = 0.926)

**Ground truth (выборка полей):**
- `scientific_name` = Entyloma ludwigianum Syd.
- `verbatim_identification` = Entyloma Ludwigianum Syd. nov. spec.
- `type_status` = Type
- `verbatim_collector` = A. Ludwig
- `verbatim_locality` = Westfalen: bei Dirlenbach, Kreis Siegen
- `country` = Germany
- `verbatim_collection_date` = 8. 8. 1931
- `verbatim_substrate` = Auf Blättern von Chrysanthemum segetum
- `exsiccata_title` = Sydow, Mycotheca germanica
- `exsiccata_number` = 2508

**Per-engine similarity:**

| engine | sim |
|---|---:|
| Tesseract | 0.836 |
| PaddleOCR | 0.859 |
| EasyOCR | 0.908 |
| docTR | 0.926 |
| Surya | 0.917 |
| TrOCR-pr | 0.923 |
| TrOCR-hw | 0.891 |

**OCR-выдача (top + worst движков):**

- **docTR** (sim=0.93): `LE F-43511 LE-43511. Sydow, Mycotheca germanica. 2508. Entyloma Ludwigianum Syd. nov. spec. Auf Blattern von Chrysanthemum segetum. Westfalen: bei Dirlenbach, Kreis Siegen. 8. 8. 1931. leg. A. Ludwig.`
- **TrOCR-pr** (sim=0.92): `LE F-43511 LE-43511. SYDOW, MYCOTHECA GERMANICA. 2508. ENTYLOMA LUDWIGIANUM SYD. NOV. SPEC. AUF BLATTERN VON CHRYSANTHEMUM SEGETUM. WESTFALEN: BEI DIRLENBACH, KREIS SIEGEN. 8.8. 1931.`
- **Tesseract** (sim=0.84): `NIMM LE F-43511 on eT | os LE»43511. ES и | | 2508. Entyloma Ludwigianum Syd. nov. spec. | | | Auf Blattern von Chrysanthemum segetum. | | : Westfalen: bei Dirlenbach, Kreis Siegen. | | 8. 8. 1931. | leg. A. Ludwig.`

---
### 5. `LE F-63132.JPG`  (mean sim = 0.888, std = 0.031, min = 0.854, max = 0.936)

**Ground truth (выборка полей):**
- `scientific_name` = Tolyposporium leptideum Syd. & P. Syd.
- `verbatim_identification` = Tolyposporium leptideum Syd. nov. spec.
- `type_status` = Type
- `verbatim_collector` = A. Ludwig
- `verbatim_locality` = Lothringen: bei Forbach
- `country` = Germany
- `verbatim_collection_date` = August—Oktober 1912/13
- `verbatim_substrate` = In den Ovarien von Chenopodium album
- `exsiccata_title` = Sydow, Mycotheca germanica
- `exsiccata_number` = 1165

**Per-engine similarity:**

| engine | sim |
|---|---:|
| Tesseract | 0.920 |
| PaddleOCR | 0.861 |
| EasyOCR | 0.856 |
| docTR | 0.936 |
| Surya | 0.854 |
| TrOCR-pr | 0.908 |
| TrOCR-hw | 0.881 |

**OCR-выдача (top + worst движков):**

- **docTR** (sim=0.94): `LE F-63132 LE 63132 Sydow, Mycotheca germanica. 1165. Tolyposporium leptideum Syd. nov. spec. In den Ovarien von Chenopodium album. Lothringen: bei Forbach. August-Oktober 1912/13. leg. A. Ludwig.`
- **Tesseract** (sim=0.92): `LE F-63132 pet | | | L£: 63:35 Sydow, Mycotheca germanica. : EN 1165. Tolyposporium leptideum Syd. nov. spec. In den Ovarien von Chenopodium album. Lothringen: bei Forbach. August—Oktober 1912/13. leg. A. Ludwig.`
- **Surya** (sim=0.85): `<b>THE LEADER IN THE LEADER IN THE REAL PROPERTY OF REAL PROPERTY OF REAL PROPERTY OF REAL PROPERTY OF REAL</b> LE F-63132 163132 Sydow, Mycotheca germanica. 1165. Tolyposporium leptideum Syd. nov. spec. In den Ovarien von Chenopodium album. Lothringen: bei Forbach. leg. A. Ludwig. August–Oktober 1912/13.`

---

## Top-5 worst (самые «трудные» — никто не справился)

### 1. `LE F-43615.JPG`  (mean sim = 0.176, std = 0.175, min = 0.000, max = 0.441)

**Ground truth (выборка полей):**
- `scientific_name` = Ustilago reessiana f. cardui-acanthoides J.G. Kühn
- `verbatim_identification` = Ustilago Reesiana Jul. Kühn. Forma: Cardui acanthoidis
- `type_status` = Type
- `verbatim_collector` = Joh. Kunze
- `verbatim_locality` = Eisleben (Prov. Sachsen)
- `country` = Germany
- `verbatim_collection_date` = August 1874
- `verbatim_substrate` = Carduus acanthoides L.
- `exsiccata_title` = Rabenhorst, Fungi europaei
- `exsiccata_number` = 1995

**Per-engine similarity:**

| engine | sim |
|---|---:|
| Tesseract | 0.171 |
| PaddleOCR | 0.394 |
| EasyOCR | 0.226 |
| docTR | 0.000 |
| Surya | 0.441 |
| TrOCR-pr | 0.000 |
| TrOCR-hw | 0.000 |

**OCR-выдача (top + worst движков):**

- **Surya** (sim=0.44): `<math>\mathbf{\mathbb{C}}</math> <b>S</b> <math>\mathbf \omega</math> <math>\boldsymbol{\omega}</math> <math>\bullet</math> <math>\blacksquare</math> <math>\mathbf \omega</math> E S 2 <math>\boldsymbol{\omega}</math> <math>\bullet</math> <b>SO</b> <math>\bullet</math> <math>\boldsymbol{\omega}</math> 9 <math>\bullet</math> un <math>\mathbf \omega</math> N SUSPICIE <math>\alpha</math> Noesiges. …`
- **PaddleOCR** (sim=0.39): `November. Acanthium), ubrig LE F-43615 die Mansfelder Substratform in Nordostufer bei dies mein Stunden o. in fast Seebur Prof. vollstandig Wormsleben Cardui Fisch. dass alle Distelpllanzen Ustilag 180 Ustilago Ein, dass yon pun spaten schon und 80 Forma Anfang (nutans und von ca Pilze Am Wormsleben da sich diese mit gsdes in derselben wie 1995 den Auge brandigen terrissdorf Excursion Die dem …`
- **TrOCR-hw** (sim=0.00): ``

---
### 2. `LE F-67004.JPG`  (mean sim = 0.193, std = 0.264, min = 0.000, max = 0.803)

**Ground truth (выборка полей):**
- `scientific_name` = Ustilago reiliana J.G. Kühn
- `verbatim_identification` = Ustilago Reiliana Kühn in litt.
- `type_status` = Type
- `verbatim_collector` = Herrn Dr. med. Reil in Cairo
- `verbatim_locality` = bei Saokara auf dem Ruinenfelde des alten Memphis
- `country` = Egypt
- `verbatim_collection_date` = Juni 1868
- `verbatim_substrate` = Crescit in Sorgho vulgari
- `exsiccata_title` = Rabenhorst, Fungi europaei
- `exsiccata_number` = 1998

**Per-engine similarity:**

| engine | sim |
|---|---:|
| Tesseract | 0.128 |
| PaddleOCR | 0.803 |
| EasyOCR | 0.237 |
| docTR | 0.000 |
| Surya | 0.182 |
| TrOCR-pr | 0.000 |
| TrOCR-hw | 0.000 |

**OCR-выдача (top + worst движков):**

- **PaddleOCR** (sim=0.80): `67004 die Spelzen sammt Bluthentheilen in cruenta beobachtete ich bisher von Sorghum- ,Ihrem Wunsche nachkommend sende im Arabischen kirzt, verdickt und mannichfach verkrimpft, bleiben aber Vorkommen der Brand- der Rispe, die Rispenaste und Treten diese an den Rispenasten Ich babe ihn selbst bei raue und asch- Herr Dr. Reil schrieb mir die Aufmerksamkeit auf einer weisslichen Haut umgebenen …`
- **EasyOCR** (sim=0.24): `' = 3 Дева # ё= ё '$ = $ Ё Ё 5 = ё g Ё #i я ё [ Зё=дай = g 0p ё # с = s ~ Ii ! Ё а '9 5 = Iliil ыр ! # $Ё € [ 5 1 Ё Il $ 1 = Е= [ = # # 1 а > [ = u 5 sf8 2 # 1 = 1 ёв 3 ₽ ё Iil ( 1 $ о ё # I р = [ =# = i I 6 5 1 3 = 5 Ё я ё 5 69 [ # [ de 1 g = F $ ё $ 1 2 d э [ I # а 1 2в 6V ; 1 ё [ [lill 2 ё & 2 1 5 Ii Ш4 # 5 ( 5 1 gа ~ Ш p Vl 5o Ё 2"а ' g $ ₽ : Ё 1 E # 5 I4 [i ё F g 2 7 1 igg ШMШll 5 > @8 = Е …`
- **TrOCR-hw** (sim=0.00): ``

---
### 3. `LE F-66949.JPG`  (mean sim = 0.209, std = 0.287, min = 0.000, max = 0.871)

**Ground truth (выборка полей):**
- `scientific_name` = Ustilago reiliana J.G. Kühn
- `verbatim_identification` = Ustilago Reiliana Kühn in litt.
- `type_status` = Type
- `verbatim_collector` = Herrn Dr. med. Reil in Cairo
- `verbatim_locality` = bei Saokara auf dem Ruinenfelde des alten Memphis
- `country` = Egypt
- `verbatim_collection_date` = Juni 1868
- `verbatim_substrate` = Crescit in Sorgho vulgari
- `exsiccata_title` = Rabenhorst, Fungi europaei
- `exsiccata_number` = 1998

**Per-engine similarity:**

| engine | sim |
|---|---:|
| Tesseract | 0.139 |
| PaddleOCR | 0.871 |
| EasyOCR | 0.240 |
| docTR | 0.000 |
| Surya | 0.215 |
| TrOCR-pr | 0.000 |
| TrOCR-hw | 0.000 |

**OCR-выдача (top + worst движков):**

- **PaddleOCR** (sim=0.87): `66949 Dieser Parasit bildet die Rispe der Mohrhirse An einer solchen Umhullung ist et obvolutam et Dicke Diesem steht Ust. Reiliana Dergleichen Fasern finden sich bekanntich bei subglobosis, crassiusculis (10,4 Mikr. Ust. Tulasnei m. (conf.v. Nr.) ist auch dadurch abwei- semipellucidis, mihi (Hamburger Treten diese an den Rispenasten cruenta beobachtete ich bisher gebildet der Rispe, die …`
- **EasyOCR** (sim=0.24): `C c ю = 3 # @Esiie= $з ё Ё '# = ё # Ё 2 : 2 Ё = #==лad $ о ё 8р = 1 = s Ii | ё Ез а = 80 2 Ili = o9 1 !ае [ 1 5 в 1 ё I $ 1 Ё [ 2s [ = а [Ш 5 ; j = # 5 3 ~$a8 # 1 [ I ё в [ з 2 ё I ( 1 Еg ё Ё 7 & 3е 60 = [ 22аg 1 I а ii Е Ёg 2 E = 5 1 # 1 5 [ 1 # 1 3 @ ё 1 Ш # pE € 1 а [ # 6 ва 1 ё Е 1 | 2 Ё 1 e8 # 1 < 5 Шllll $ 4 1Ё Ёg G $ 1 = 7 # 2 ы9 = # # ё > а 60 $ = [ 5 ё 0 Шf = = 8 5 € Г g в a9 g 1 ~ 35 [ …`
- **TrOCR-hw** (sim=0.00): ``

---
### 4. `LE F-62935.JPG`  (mean sim = 0.240, std = 0.209, min = 0.046, max = 0.715)

**Ground truth (выборка полей):**
- `scientific_name` = Tilletia secalis (Corda) J.G. Kühn
- `verbatim_identification` = Tilletia Secalis Jul. Kühn
- `type_status` = Type
- `verbatim_collector` = Herrn J. Polomski
- `verbatim_locality` = auf Altendorf bei Ratibor, Oberschlesien
- `country` = Poland
- `verbatim_collection_date` = Juli [1876]
- `verbatim_substrate` = Secalis cerealis ovaria occupat
- `exsiccata_title` = Rabenhorst, Fungi europaei
- `exsiccata_number` = 2191

**Per-engine similarity:**

| engine | sim |
|---|---:|
| Tesseract | 0.158 |
| PaddleOCR | 0.715 |
| EasyOCR | 0.224 |
| docTR | 0.046 |
| Surya | 0.302 |
| TrOCR-pr | 0.090 |
| TrOCR-hw | 0.150 |

**OCR-выдача (top + worst движков):**

- **PaddleOCR** (sim=0.71): `62935 (1876) costis limbo Exemplare verdanke ich der Polomski auf Altendorf Tilletia ergab der sie auf seiner und T. controversa mihi zunachst, hat aber Brand- Zugehorig- n Faden Secales Rabenb. (Flora, 1848, S. 2091) weicht engere Tal. Von ergleich dass die Ustilago konnte, unterscheidet er sich ahnlich, Be- Mik. Prof. Dr. Julius Kuhn. wie Caries geruchlosc 12,2 calis Jul. Kihn. xix. Jahrg. …`
- **Surya** (sim=0.30): `<math>\mathbb{C}</math> E europaei.<br>Als Jul. Kühn.<br>mg, XIX. Jahrg. (18 Exemplare verdanke ich den Bornia occupat.<br> 1. Polomski auf Altende<br> sammelle. Die Zugehörinner Gattung Tilletia erg<br> thung einer jüngeren Brar<br> thung einer jüngeren Brar<br> tr sa mihi zunächst, hat ab<br> her schärfere Zeichnu The als lettere. V<br> Tischeidet er sich ähnlic<br> schou dadurch, dass d<br> …`
- **docTR** (sim=0.05): `n 0 A`

---
### 5. `LE F-43612.JPG`  (mean sim = 0.243, std = 0.130, min = 0.120, max = 0.455)

**Ground truth (выборка полей):**
- `scientific_name` = Ustilago rabenhorstiana f. agrestis J.G. Kühn
- `verbatim_identification` = Ustilago Rabenhorstiana Jul. Kühn. Forma: agrestis †
- `type_status` = Type
- `verbatim_collector` = Prof. Dr. Julius Kühn
- `verbatim_locality` = des Dorfes Rauscha in der Görlitzer Haide
- `country` = Poland
- `verbatim_collection_date` = 14. September 1878
- `verbatim_substrate` = an der cultivirten Form von Panicum sanguinale beobachtet und an mehreren anderen Digitariaarten erzogen worden
- `exsiccata_title` = Rabenhorst-Winter, Fungi europaei
- `exsiccata_number` = 2604

**Per-engine similarity:**

| engine | sim |
|---|---:|
| Tesseract | 0.190 |
| PaddleOCR | 0.418 |
| EasyOCR | 0.255 |
| docTR | 0.126 |
| Surya | 0.141 |
| TrOCR-pr | 0.455 |
| TrOCR-hw | 0.120 |

**OCR-выдача (top + worst движков):**

- **TrOCR-pr** (sim=0.45): `8.00 00 : P 0.0 III II. LE F-43612`
- **PaddleOCR** (sim=0.42): `LE-43612 Ustilago Rabenhorstiana. Ja. Rihn. s landwirthschaftlichen Instituts der Universitat Halle an Dieser Pilz war von mir bisher nur in dem Garten Gorlitzer Haide mir verschafft hatte, so vermuthete ich dort Die Bluthirse wird in den Walddorfern der Gorlitzer Haide ich den Samen der Bluthirse aus einer Ortschaft der das urspringliche Vorkommen dieses Parasiten und einc theils waren es bis …`
- **TrOCR-hw** (sim=0.12): `1962 63 000 # # Today # 30 e ii . 2194-1/`

---