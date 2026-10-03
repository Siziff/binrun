# Эксп. 2 — top-5 best / worst для тройки PaddleOCR + Surya + TrOCR-pr

Картинки — препроцессед версии (после orientation + resize).
Worst — среди single-файлов (без `_1`/`_2`-разворотов).


## PaddleOCR


### Top-5 best

**1. `LE F-65008.JPG`**  sim=0.906  CER=5.62  runtime=2.65s

GT:
- `scientific_name` = Ustilago cruenta J.G. Kühn f. culmicola
- `verbatim_identification` = Ustilago cruenta Jul. Kühn. Forma: culmicola †
- `type_status` = Type
- `verbatim_collector` = Prof. Dr. Julius Kühn
- `verbatim_locality` = auf der zur Herrschaft Schwusen (Niederschlesien) gehörigen Feldflur Attendorf
- `country` = Germany
- `verbatim_collection_date` = Jahre 1859
- `verbatim_substrate` = an Sorghum racharatum
- `exsiccata_title` = Rabenhorst, Fungi europaei
- `exsiccata_number` = 2594

OCR:
  `Rabenhorst, Fungi europaei. 2594 Ustilago cruenta Jul. Rihn. Hamburg. Gart u. Blum. Ztg. Jahrg 1872 B. XXVIll. S. 177 u. f. (Conf. Rabenhorst, fungi europaei XX. Cent. Nr. 1998). 1 Hedwigia,Jahrgang 1878.Nr.1,S.11. Forma: culmicola ". Im Jahre 1878 auf dem Versuchsfelde des landwirth- schaftlichen Instituts der Universitat Halle an Sorghum vul- gare cultivirt. Ob s.Dieser von mir im Jahre 1859 auf der zur Herrschaft Schwusen (Niederschlesien) gehorigen Feldflur Attendorf an Sorghum racharatum …`

---
**2. `LE F-113259.JPG`**  sim=0.900  CER=0.52  runtime=2.51s

GT:
- `scientific_name` = Ustilago andropogonis-hirtifolii Henn.
- `verbatim_identification` = Ustilago Andropogonis-hirtifolii P. Henn. n. sp.
- `type_status` = Type
- `verbatim_collector` = E. W. D. Holway
- `verbatim_locality` = Mexico: Patzcuaro
- `country` = Mexico
- `verbatim_collection_date` = 20. 10. 1898
- `verbatim_substrate` = In spicis Andropogonis hirtifolii var. pubiflori
- `exsiccata_title` = Sydow, Ustilagineen
- `exsiccata_number` = 201

OCR:
  `Sydow, Ustilagineen. EA113259 20l. Ustilago Andropogonis-hirtifoli P. Henn. n. sp. Mexico:Patzcuaro. In spicis Andropogonis hirtifoli var. pubiflori. 20. 10. 1898. leg. E. w.D. Holway. LEF-113259`

---
**3. `LE F-43624.JPG`**  sim=0.900  CER=0.69  runtime=2.03s

GT:
- `scientific_name` = Ustilago turcomanica Tranzschel
- `verbatim_identification` = Ustilago turcomanica W. Tranzsch. sp. n.
- `type_status` = Type
- `verbatim_collector` = Androssow
- `verbatim_locality` = prope st. Repetek, Transcaspiae
- `country` = Turkmenistan
- `verbatim_collection_date` = 1911. 4. V.
- `verbatim_substrate` = In ovariis Tritici squarrosi Roth
- `exsiccata_title` = Tranzschel et Serebrianikow. Mycotheca rossica
- `exsiccata_number` = 302

OCR:
  `LE43624. TRANZSCHEL ET SEREBRIANIKOW. MYCOTHECA ROSSICA 302. Ustilago turcomanica w.Tranzsch.sp.n. LEF-43624 lu ovaris Tritici squarrosi Roth. In arenosis prope st. Repetek, Transcaspiae. 1911.4.V. Legit Androssow.`

---
**4. `LE F-341598.JPG`**  sim=0.891  CER=0.63  runtime=1.51s

GT:
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

OCR:
  `LEF-341598 LEF-341598 Sydow, Ustilagineen 212. Ustilago Panici-proliferi P. Henn. n. sp. Mexico:pr. urbem Mexico. In spicis Panici proliferi var. acuminati. 10.10.1898 leg. E.W.D. Holway.`

---
**5. `LE F-201413.JPG`**  sim=0.891  CER=0.70  runtime=1.49s

GT:
- `scientific_name` = Macalpinomyces eragrostiellae Vánky & C. Vánky
- `verbatim_identification` = Macalpinomyces eragrostiellae K. & C. Vánky
- `type_status` = Isotype
- `verbatim_collector` = N.D. Sharma, S.R. Singh, R.P. Mishra, M.B. Nagar, C. et K. Vánky
- `verbatim_locality` = INDIA: Madhya Pradesh, 9 km S.E. Jabalpur, Tropical Forest Res. Inst.
- `country` = India
- `verbatim_collection_date` = 13.X.1992
- `verbatim_substrate` = Matrix: Eragrostiella bifaria (Vahl) Bor (Pradhan)
- `exsiccata_title` = K. Vánky: Ustilaginales
- `exsiccata_number` = 960

OCR:
  `LE201413 K.VANKY:USTILAGINALES 96O. MACALPINOMYCES ERAGROSTIELLAE K. & C. Vanky Isotypus Matrix: Eragrostiella bifaria (Vahl) Bor (Pradhan) LEF-201413 INDIA: Madhya Pradesh, 9 km S.E. Jabalpur, Tropical Forest Res. Inst. 23°07' N, 80°0' E. Alt. 390 m. 13.X.1992 leg. N.D. sharma, S.R. Singh, R.P. Mishra, M.B. Nagar, C. et K. Vanky det. K. Vanky (4074) (unacum Tilletia eragrostiellae)`

---

### Top-5 worst

**1. `LE F-63167.JPG`**  sim=0.134  CER=0.89  runtime=1.19s

GT:
- `scientific_name` = Tracya hydrocharidis Lagerh.
- `verbatim_identification` = Tracya Hydrocharidis Lagerh.
- `type_status` = Type
- `verbatim_collector` = G. Lagerheim
- `verbatim_locality` = Suec. Stockholm, Järla
- `country` = Sweden
- `verbatim_collection_date` = sept. 1898
- `verbatim_substrate` = H[ydrocharis] morsus rana

OCR:
  `LE63167 LEF-63167`

---
**2. `LE F-64716.JPG`**  sim=0.136  CER=0.91  runtime=1.60s

GT:
- `scientific_name` = Cintractia arctica (Rostr.) Lagerh.
- `verbatim_identification` = Cintractia arctica Lagh. nov. spec.
- `type_status` = Type
- `verbatim_collector` = G. v. Lagerheim
- `verbatim_locality` = Arkt. Norwegen, pr. Tromsö
- `country` = Norway
- `verbatim_collection_date` = 6. 1893
- `verbatim_substrate` = Auf Carex glareosa
- `exsiccata_title` = Sydow, Ustilagineen
- `exsiccata_number` = 72

OCR:
  `LE/54716 LEF-64716`

---
**3. `LE F-64712.JPG`**  sim=0.149  CER=0.90  runtime=1.45s

GT:
- `scientific_name` = Cintractia arctica (Rostr.) Lagerh.
- `verbatim_identification` = Cintractia arctica Lagh. nov. spec.
- `type_status` = Type
- `verbatim_collector` = G. v. Lagerheim
- `verbatim_locality` = Arkt. Norwegwn, pr. Tromsö
- `country` = Norway
- `verbatim_collection_date` = 6. 1893
- `verbatim_substrate` = Auf Carex glareosa
- `exsiccata_title` = Sydow, Ustilagineen
- `exsiccata_number` = 72

OCR:
  `LE64712 LEF-64712`

---
**4. `LE F-61707.JPG`**  sim=0.155  CER=0.89  runtime=1.40s

GT:
- `scientific_name` = Urocystis corydalis Niessl
- `verbatim_identification` = Urocystis corydalis Niessl
- `verbatim_collector` = O. Pazschke
- `verbatim_locality` = Connewitz bei Leipzig
- `country` = Germany
- `verbatim_collection_date` = 5. 1886
- `verbatim_substrate` = Auf Corydalis cavae
- `exsiccata_title` = Sydow, Ustilagineen
- `exsiccata_number` = 42

OCR:
  `LE61707 LEF-61707`

---
**5. `LE F-43493.JPG`**  sim=0.161  CER=0.93  runtime=1.11s

GT:
- `scientific_name` = Doassansia rhinanthi Lagerh. ex Zundel
- `verbatim_identification` = Doassansia Rhinanthi Lagh. nov. spec.
- `type_status` = Type
- `verbatim_collector` = G. v. Lagerheim
- `verbatim_locality` = Arkt. Norwegen, Storsteinnes pr. Tromsö
- `country` = Norway
- `verbatim_collection_date` = 9.1895
- `verbatim_substrate` = Auf Rhinanthus minor
- `exsiccata_title` = Sydow, Ustilagineen
- `exsiccata_number` = 95

OCR:
  `LEF-43493 LE-43493. 9.493.`

---

## Surya


### Top-5 best

**1. `LE F-43584.JPG`**  sim=0.962  CER=1.26  runtime=0.98s

GT:
- `scientific_name` = Ustilago cilinodis Savile
- `verbatim_identification` = Ustilago cilinodis Savile sp. nov.
- `type_status` = Type
- `verbatim_collector` = W.A. Wojtas
- `verbatim_locality` = 1.1 Km. along the road to the Jet Trail, Riding Mountain National Park, MANITOBA
- `country` = Canada
- `verbatim_collection_date` = 1 Aug., 1979
- `verbatim_substrate` = on Polygonum cilinode Michx.

OCR:
  `LE-43584. NATIONAL MYCOLOGICAL HERBARIUM OF CANADA EX HERBIER NATIONAL DE MYCOLOGIE DU CANADA DAOM 183263 DATE 1 Aug., 1979 Ustilago cilinodis Savile sp. nov. on Polygonum cilinode Michx. In shaded area of a Populus sp. woods 1.1 Km. along the road to the Jet Trail <math display="inline">\circ</math> Riding Mountain National Park, MANITOBA Coll.: W.A. Wojtas (No. 918) Det.: J.A. Parmelee <math>\tilde{\epsilon}</math> LE F-43584`

---
**2. `LE F-341598.JPG`**  sim=0.955  CER=0.64  runtime=0.89s

GT:
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

OCR:
  `LE F-341598 LE F-341598 Sydow, Ustilagineen. 212. Ustilago Panici-proliferi P. Henn. n. sp. Mexico: pr. urbem Mexico. In spicis Panici proliferi var. acuminati. 10. 10. 1898. leg. E. W. D. Holway.`

---
**3. `LE F-65008.JPG`**  sim=0.954  CER=5.67  runtime=1.65s

GT:
- `scientific_name` = Ustilago cruenta J.G. Kühn f. culmicola
- `verbatim_identification` = Ustilago cruenta Jul. Kühn. Forma: culmicola †
- `type_status` = Type
- `verbatim_collector` = Prof. Dr. Julius Kühn
- `verbatim_locality` = auf der zur Herrschaft Schwusen (Niederschlesien) gehörigen Feldflur Attendorf
- `country` = Germany
- `verbatim_collection_date` = Jahre 1859
- `verbatim_substrate` = an Sorghum racharatum
- `exsiccata_title` = Rabenhorst, Fungi europaei
- `exsiccata_number` = 2594

OCR:
  `Rabenhorst, Fungi europaei. 2594 Ustilago cruenta Jul. Kühn. Hamburg. Gart u. Blum. Ztg. Jahrg 1872 B. XXVIII. S. 177 u. f. (Conf. Rabenhorst, fungi europaei XX. Cent. Nr. 1998). Hedwigia, Jahrgang 1878. Nr. 1, S. 11. Forma: culmicola <math>\ddagger</math>. Im Jahre 1878 auf dem Versuchsfelde des landwirth- schaftlichen Instituts der Universität Halle an Sorghum vul- gare cultivirt. Obs. Dieser von mir im Jahre 1859 auf der zur Herrschaft Schwusen (Niederschlesien) gehörigen Feldflur Attendorf …`

---
**4. `LE F-43623.JPG`**  sim=0.947  CER=0.67  runtime=0.91s

GT:
- `scientific_name` = Ustilago turcomanica Tranzschel
- `verbatim_identification` = Ustilago turcomanica W. Tranzsch. sp. n.
- `type_status` = Type
- `verbatim_collector` = Androssow
- `verbatim_locality` = prope st. Repetek, Transcaspiae
- `country` = Turkmenistan
- `verbatim_collection_date` = 1911. 4. V.
- `verbatim_substrate` = In ovariis Tritici squarrosi Roth
- `exsiccata_title` = Tranzschel et Serebrianikow. Mycotheca rossica
- `exsiccata_number` = 302

OCR:
  `LE-43623. E TRANZSCHEL ET SEREBRIANIKOW. MYCOTHECA ROSSICA. 302. Ustilago turcomanica w. Tranzsch. sp. n. Iu ovariis Tritici squarrosi Roth. In arenosis prope st. Repetek, Transcaspiae. 1911. 4. V. Legit Androssow.`

---
**5. `LE F-43625.JPG`**  sim=0.947  CER=0.66  runtime=0.95s

GT:
- `scientific_name` = Ustilago turcomanica Tranzschel
- `verbatim_identification` = Ustilago turcomanica W. Tranzsch. sp. n.
- `type_status` = Type
- `verbatim_collector` = Androssow
- `verbatim_locality` = prope st. Repetek, Transcaspiae
- `country` = Turkmenistan
- `verbatim_collection_date` = 1911. 4. V.
- `verbatim_substrate` = In ovariis Tritici squarrosi Roth
- `exsiccata_title` = Tranzschel et Serebrianikow. Mycotheca rossica
- `exsiccata_number` = 302

OCR:
  `LE-43625. TRANZSCHEL ET SEREBRIANIKOW. MYCOTHECA ROSSICA. 802. Ustilago turcomanica w. Tranzsch. sp. n. Iu ovariis Tritici squarrosi Roth. In arenosis prope st. Repetek, Transcaspiae. 1911. 4. V. Legit Androssow.`

---

### Top-5 worst

**1. `LE F-62002.JPG`**  sim=0.230  CER=2.69  runtime=0.77s

GT:
- `scientific_name` = Entyloma endogenum (Unger) Wünsche
- `verbatim_identification` = Entyloma endogenum
- `type_status` = Type
- `verbatim_locality` = Freiburg
- `country` = Germany
- `verbatim_collection_date` = [NULL]

OCR:
  `LE 62002 Пожертвовано way ces ends AHAN. M. C. BOPOHNNELMEN ence <math>\mathcal{Q}</math> endogenis <math>LEF-62002</math> <b>Vo</b> <math>\sim</math> <math>\sim</math>`

---
**2. `LE F-43628.JPG`**  sim=0.271  CER=2.87  runtime=2.48s

GT:
- `scientific_name` = Ustilago turcomanica Tranzschel
- `verbatim_identification` = Ustilago turcomanica n. sp.
- `type_status` = Type
- `verbatim_collector` = Андросовъ
- `verbatim_locality` = Мервский у., Закасп. обл. Пески блз. ст Репетекъ
- `country` = Turkmenistan
- `verbatim_collection_date` = 7/V 1911
- `verbatim_substrate` = Triticum squarrosum Roth

OCR:
  `N19 A Uchlajo Ausconsence nap 69 Fritiann Synassofum Roth Ellepterin y. Baraen ods. Neeve duy <math display="block">\begin{array}{c}\n\overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ \overline{111} \\ …`

---
**3. `LE F-43594.JPG`**  sim=0.306  CER=1.31  runtime=1.15s

GT:
- `scientific_name` = Ustilago hierochloae Jacz.
- `verbatim_identification` = Ustilago hierochloae Jacz. nov. sp.
- `type_status` = Type
- `verbatim_collector` = М. К. Хохряков
- `verbatim_locality` = Буерак ... бл. хут. Попова, Вешенск. р. Донецк. окр., Сев. Кавк. Края
- `country` = Russia
- `verbatim_collection_date` = 21-V-1928
- `verbatim_substrate` = На Hierochloe odorata

OCR:
  `LF-43594. INSTITUTUM CRYPTOGAMICUM HORTI BOTANICI REIPUBLICAE ROSSICAE (ante Petropolitani). Elstilago hierochloac Jacq nov. s. Na Hiervehloa odorata Statio: Byggan Parcemikin 81. rue Honoba, Benieuen <math>21 - v</math> Anno <math>128</math> det. Ju. K. Kox, and. Anno.`

---
**4. `LE F-262530.JPG`**  sim=0.310  CER=1.24  runtime=1.25s

GT:
- `scientific_name` = Sorosporium pedicularis Golovin
- `verbatim_identification` = Sorosporium pedicularis sp. nov.
- `type_status` = Type
- `verbatim_collector` = Н. Никифорова
- `verbatim_locality` = Восточный Памир, уроч. Чатыкой, у р. Мургаб
- `country` = Tajikistan
- `verbatim_collection_date` = 1987-IX-9
- `verbatim_substrate` = Pedicularis (dubia)

OCR:
  `Sorosporieur pedicularies sp. nod. LE 262530 S/ Pedicularis (dubia) Bocjonnen having, ypar. Tafhuon, yp. Myprad, no doutsuchen gouvernemen <b>THE REACH</b> vynam.<br>1937 - 14 - J. codp. H. Hukuquyota Souchen Ill administrar Clos Topus-Badaxmancher abmonscena aviacunt`

---
**5. `LE F-43532.JPG`**  sim=0.365  CER=5.09  runtime=1.71s

GT:
- `scientific_name` = Sorosporium lavrovianum Smarods
- `verbatim_identification` = Sorosporium Lavrovianum Smarods Sp. nova
- `type_status` = Type
- `verbatim_collector` = J. Škipsna
- `verbatim_locality` = URSS, Republica Latvia: Stende
- `country` = Latvia
- `verbatim_collection_date` = 24.VI 54
- `verbatim_substrate` = In ovariis Clinelymi sibirici (L.) Nevski
- `exsiccata_title` = J. Smarods, Fungi latvici exsiccati
- `exsiccata_number` = 1308

OCR:
  `. onogait menimon vorval. N. N Notate . Solooym . Salas. Wycolog. Sibiric. experimentali, cult. 24.VI 54, J.Skipsna legit. Neveki, URSS, Republica batria: Stephe, in agro (.T) rotatdis tmylentio sitisvo ni .oitsitdsh preae Aeronyoso stangins , cenundoscatio ovail murriogs circqaiq <math>008 = 77</math>, <math>1000</math>, <math>1000</math>, <math>1000</math>, <math>1000</math>, <math>1000</math> snx <math>-900</math> uox 8 a 7 de 18 a 9 a 70 a 71 a 75 a 75 a 76 a 76 a 76 a 76 a 77 de 18 a 19 …`

---

## TrOCR-pr


### Top-5 best

**1. `LE F-43584.JPG`**  sim=0.957  CER=0.96  runtime=0.85s

GT:
- `scientific_name` = Ustilago cilinodis Savile
- `verbatim_identification` = Ustilago cilinodis Savile sp. nov.
- `type_status` = Type
- `verbatim_collector` = W.A. Wojtas
- `verbatim_locality` = 1.1 Km. along the road to the Jet Trail, Riding Mountain National Park, MANITOBA
- `country` = Canada
- `verbatim_collection_date` = 1 Aug., 1979
- `verbatim_substrate` = on Polygonum cilinode Michx.

OCR:
  `LE-43584 - EX NATIONAL MYCOLOGICAL HERBARIUM OF CANADA DAOM 183263 DAREL AUG., 1979 USTILAGO CILINODIS SAVILE SP. NOV. ON POLYGONUM CILINODE MICHX. IN SHADED AREA OF A POPULUS SP. WOODS 1.1 KM. ALONG THE ROAD TO THE JET TRAIL RIDING MOUNTAIN NATIONAL PARK, MANITOBA COLL.: W.A. WOJTAS (NO. 918) DET.: J.A. PARMELEE LE F-43584`

---
**2. `LE F-341598.JPG`**  sim=0.934  CER=0.63  runtime=0.84s

GT:
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

OCR:
  `LE F-341598 LE F-341598 SYDOW, USTILAGINEEN. 212. USTILAGO PANICI-PROLIFERI P. HENN. N. SP. MEXICO: PR. URBEM MEXICO. IN SPICIS PANICI PROLIFERI VAR. ACUMINATI. 10. 10. 1898.`

---
**3. `LE F-201420.JPG`**  sim=0.927  CER=0.59  runtime=1.02s

GT:
- `scientific_name` = Anthracoidea altiphila Vánky & M. Piepenbr.
- `verbatim_identification` = Anthracoidea altiphila K. Vánky & M. Piepenbring
- `type_status` = Isotype
- `verbatim_collector` = P. Döbbeler, J. Gómez-Laurito, T. et K. Vánky
- `verbatim_locality` = COSTA RICA: Prov. Cartago, Mts. Cordillera de Talamanca, 38 km S. Cartago, inter pagg. Cañon et Trinidad
- `country` = Costa Rica
- `verbatim_collection_date` = 5.III.1991
- `verbatim_substrate` = Matrix: Carex jamesonii Boott
- `exsiccata_title` = K. Vánky: Ustilaginales
- `exsiccata_number` = 901

OCR:
  `ISOTYPUS K. VANKY : USTILAGINALES 901. ANTHRACOIDEA ALTIPHILA K. VANKY & M. PIEPENBRING ISOTYPUS MATRIX: CAREX JAMESONII BOOTT COSTA RICA: PROV. CARTAGO, MTS. CORDILLERA DE TALAMANCA, 38 KM S. CARTAGO, INTER PAGG. CARON ET TRINDAD. 9041' N, 83055' W. ALT. C. 2630 M. 5.III.1991 LEG. P. DOBBELER, J. G6MEZ-LAURITO, T. ET K. VANKY, DET. K. VANKY (3854)`

---
**4. `LE F-43511.JPG`**  sim=0.923  CER=0.69  runtime=0.72s

GT:
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

OCR:
  `LE F-43511 LE-43511. SYDOW, MYCOTHECA GERMANICA. 2508. ENTYLOMA LUDWIGIANUM SYD. NOV. SPEC. AUF BLATTERN VON CHRYSANTHEMUM SEGETUM. WESTFALEN: BEI DIRLENBACH, KREIS SIEGEN. 8.8. 1931.`

---
**5. `LE F-262344.JPG`**  sim=0.923  CER=0.69  runtime=0.84s

GT:
- `scientific_name` = Jamesdicksonia ischaemiana (Thirum. & Pavgi) R. Bauer, Begerow, A. Nagler & Oberw.
- `verbatim_identification` = Jamesdicksonia ischaemiana (Thirum. & Pavgi) R. Bauer, Begerow, A. Nagler & Oberw.
- `type_status` = Topotype
- `verbatim_collector` = N.D. Sharma, C. & K. Vánky
- `verbatim_locality` = INDIA: Maharashtra State, Kandala
- `country` = India
- `verbatim_collection_date` = 19.X.1995
- `verbatim_substrate` = Matrix: Ischaemum semisagittatum Roxb.
- `exsiccata_title` = K. Vánky: Ustilaginales
- `exsiccata_number` = 1344

OCR:
  `TOPOTYPUS K. VANKY: USTILAGINALES 1344. JAMESDICKSONIA ISCHAEMIANA (THIRUM. & PAVGI) R. BAUER, BEGEROW, A. NAGLER & OBERW. TOPOTYPUS MATRIX: ISCHAEMUM SEMISAGITTATUM ROXB. INDIA: MAHARASHTRA STATE, KANDALA, 18'45' N, 73'25' E, ALT. 580 M. 19.X.1995 LEG. N.D. SHARMA, C. & K. VANKY DET. K. VANKY (4178)`

---

### Top-5 worst

**1. `LE F-43577.JPG`**  sim=0.191  CER=12.01  runtime=2.71s

GT:
- `scientific_name` = Urocystis ulei Magnus
- `verbatim_identification` = Urocystis Ulii P. Magn. nov. sp.
- `type_status` = Type
- `verbatim_collector` = P. Magnus
- `verbatim_locality` = Auf der Insel Pehlitz-Werder im Parsteiner See. Mrk. Brdbg.
- `country` = Germany
- `verbatim_collection_date` = 27. Sept. 1877
- `verbatim_substrate` = Auf. Poa pratensis var. angustifolia
- `exsiccata_title` = Rabenhorst, Fungi europaei
- `exsiccata_number` = 2390

OCR:
  `BELIN, 27, Sep1, 1877. P. MAGNUS. 12-20 MMM. DETREST. OCCULIA DUR 18-24 MINIM, JER BEI UY, AGROPYRI NUR INF DURCHNESSER 24 - 30 MINN., WABENU der bei UPDE. MESSER, ALS HEI DEN DETEN andEREN AYTER UND ZWAR ISE SO HAHER SUCH AIR GANZEN GLOMERULI GINEN GRUSSERON DUROH- AN Dr. ULL WILL NOTER, ALS ARE ARE ADVERTISE AGAIN, SAND, SCHOR ODEN BETYORgeHoben, die RANDZELEN der GLOMERULI ZUM THEIL YOU DEN STERIES RANDZELEN UNAHILL. DA, WIE ALLEN AREI AFTER SIDE RISKIMENEN CERTRIZELLEA DUR SIE BEI UF. …`

---
**2. `LE F-66924.JPG`**  sim=0.245  CER=7.53  runtime=2.12s

GT:
- `scientific_name` = Ustilago rabenhorstiana J.G. Kühn
- `verbatim_identification` = Ustilago rabenhorstiana J.G. Kühn
- `type_status` = Type
- `verbatim_locality` = Halle
- `country` = Germany
- `verbatim_collection_date` = 23. Febr. 1879
- `verbatim_substrate` = Panicum sanguinalem
- `exsiccata_title` = Rabenhorst-Winter, Fungi europaei
- `exsiccata_number` = 2604

OCR:
  `Prof. Dr. Jullus Kühn. HALLE, den 23. Febr. 1879. MAN BEI der Ernte alS UNDENUTZAR STELEN JELASSEN HATE. THEILS WARDIES IS DISSIDE, TOTAL FAMILYE BISCHE, DETAIL SPROSSTRIBLE PER MILDER SICHEL ABSERMITTENES, STICKET STOPPER PRICE, DISPLAY ADJUSTED WITH NOT WITH RECEIPT IN NEED OF RECEIPT LOH TAMPING RIE DETAILS ADPERITELY Jedoch, GLOICE HAS ESTE RECEIPTAINABLE, ADER DUR IN DESCHENKTEN UNAHAGE ENGEBALL. DIE BLUTHITSE WITH IN EN WALDORIENT der GORITZER HAICE PLEASE INCLUSIVE AND FORMULATE …`

---
**3. `LE F-63417.JPG`**  sim=0.249  CER=1.37  runtime=1.65s

GT:
- `scientific_name` = Tuburcinia atropidis Lavrov
- `verbatim_identification` = Tuburcinia atropidis Lavr.
- `type_status` = Type
- `verbatim_collector` = К. Бенуа
- `verbatim_locality` = Якутская респ. Якутский округ. Охотский тракт (от г. Якутска до р. Амги). уроч. Тючал-алас близ. станция Лебягинская
- `country` = Russia
- `verbatim_collection_date` = 13.VII.1925
- `verbatim_substrate` = Atropis Hauptiana (Trin.) V. Krecz. Determ. R. Roshevitz, IX.1935

OCR:
  `E 63417 AKVTCKAN 3KCHEANINA AKAKEMAN HAYK CCCP. LOOOPYLL'S AGROPPER / PRICE/SCHANGLEA ALA AKYTCKAN PECH, AKYTCKHI OXPYR. OXOTCKHIL TPART (OT R. AKYTCKA PLEASE ALL PAYT FOR ALL FACE IN NEED OF RECEIPT FOR AN OFFERING. BADOUD WAWSAN AM-AVAC ATAPLE PHOPPERS, KEEPY, JALAN MALANAL, POY, HAYORAN TEL/135. DETERM. R. ROSHEVITZ. NOTA CRITICA PUBLICINIA ATIO PIGIS LAVE HA ATTOPLE HAUPLE ANA RICE IS . VU 18862. MOONWOOD. LALPO APTEMB ,HOBAN KHHRA" 2247-1000`

---
**4. `LE F-43594.JPG`**  sim=0.253  CER=1.52  runtime=1.03s

GT:
- `scientific_name` = Ustilago hierochloae Jacz.
- `verbatim_identification` = Ustilago hierochloae Jacz. nov. sp.
- `type_status` = Type
- `verbatim_collector` = М. К. Хохряков
- `verbatim_locality` = Буерак ... бл. хут. Попова, Вешенск. р. Донецк. окр., Сев. Кавк. Края
- `country` = Russia
- `verbatim_collection_date` = 21-V-1928
- `verbatim_substrate` = На Hierochloe odorata

OCR:
  `LE-43594. INSTITUTUM CRYPTOGAMICUM HORTI BOTANICI REIPUBLICAE ROSSICAE (ANTE PETROPOLITANI). MYILAGO HIESELLONE HACE NO. NO HIERUCHLSA ODOROTA STATION: PAYMENT PAYMENT SUMMARY AND PAY FACE AGAURABLE, BELLULANCE DENTING. MY. CEL KOLK. KRAX. SOLL TASCO. MY ANNOURS LEGAT ANNO DETRICK. COM/AMPL. S.A. AMELERUN.`

---
**5. `LE F-198016.JPG`**  sim=0.258  CER=1.63  runtime=1.07s

GT:
- `scientific_name` = Tuburcinia polygonati Lavrov
- `verbatim_identification` = Tuburcinia polygonati Lavrov
- `type_status` = Type
- `verbatim_collector` = П. Михно
- `verbatim_locality` = Забайкальская область. Окрестности села Преображенскаго (Усть-Киранъ), въ 30 в. отъ г. Троицкосавска
- `country` = Russia
- `verbatim_collection_date` = 1913. 3 VII
- `verbatim_substrate` = на Polygonatum officinale L.

OCR:
  `TOTAL INCLUSIVE AND RETURNS AND RETURNS AND CLASSIVE IN NEED ALL RETURNS. -C MUSEUM BOTANICUM ACADEMIAE SDIENTIARUM PETROPOLITANAE. 4/ THURHAND HEAVYERS, COME AGAINABLE LABERS LOTALYONSLINE, PRICE(PROHIBER) EXCLUDES, REGISTER OF OBJECTIVE OF RECEIPT FOR AN OFFER LE 1913.34 C06.11. MARYHO. EXLE NOTA CRITICA PUBLICING POLY NATE HAVOV KA POLICONA OFFICINIVE . 7/1 19862. MOONUTS LALA APTEMB .HOBAN KHHRA' 2247-1000`

---